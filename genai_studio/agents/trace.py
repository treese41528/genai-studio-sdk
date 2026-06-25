"""
Tracing — the pedagogical centerpiece.

Every step of a run emits a structured ``TraceEvent``. A ``Tracer`` decides what
to do with them. Crucially, ``LLMCall`` carries the **actual messages sent to the
model** — the "show me the prompt" transparency that lets students (and Postyl's
UI) see exactly what the model received, not a sanitized summary.

Shipped tracers:
- ``ConsoleTracer`` — pretty, step-numbered, colorized (interactive default).
- ``NullTracer``    — silent (production default for non-UI use).
- ``JsonlTracer``   — newline-delimited events; the substrate for eval harnesses.

Event names use the ``*Event`` suffix for tool steps (``ToolCallEvent`` /
``ToolResultEvent``) to avoid colliding with the wire ``ToolCall`` / ``ToolResult``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ════════════════════════════════════════════════════════════════════════════
# Event types
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceEvent:
    t: float = field(default_factory=time.time)
    step: int | None = None


@dataclass
class AgentStart(TraceEvent):
    prompt: Any = None
    tools: list = field(default_factory=list)
    model: str | None = None


@dataclass
class LLMCall(TraceEvent):
    messages: list = field(default_factory=list)  # the literal prompt sent
    tools: list = field(default_factory=list)


@dataclass
class LLMResponse(TraceEvent):
    response: Any = None


@dataclass
class LLMRetry(TraceEvent):
    attempt: int = 0
    delay: float = 0.0
    status: int | None = None
    error: str | None = None


@dataclass
class ToolCallEvent(TraceEvent):
    call: Any = None


@dataclass
class ToolResultEvent(TraceEvent):
    call: Any = None
    result: Any = None
    elapsed: float | None = None


@dataclass
class StepEnd(TraceEvent):
    usage: Any = None


@dataclass
class AgentEnd(TraceEvent):
    result: Any = None
    stopped: str = "final"


# ════════════════════════════════════════════════════════════════════════════
# Protocol + tracers
# ════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Tracer(Protocol):
    def on_event(self, event: TraceEvent) -> None: ...


class NullTracer:
    """Silent tracer (production default)."""

    def on_event(self, event: TraceEvent) -> None:
        pass


class _C:
    """Minimal ANSI palette, auto-disabled off a TTY (mirrors the SDK style)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.enabled else s

    def dim(self, s):
        return self._w("2", s)

    def bold(self, s):
        return self._w("1", s)

    def cyan(self, s):
        return self._w("36", s)

    def green(self, s):
        return self._w("32", s)

    def yellow(self, s):
        return self._w("33", s)

    def red(self, s):
        return self._w("31", s)


def _truncate(s: str, n: int = 80) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        parts.append(f"{k}={_truncate(json.dumps(v, default=str), 40)}")
    return ", ".join(parts)


class ConsoleTracer:
    """Pretty, step-numbered console output (the interactive default).

    Set ``show_prompts=True`` to dump the literal messages of every ``LLMCall``
    — the strongest form of "show me the prompt".
    """

    def __init__(self, *, stream=None, color: bool | None = None, show_prompts: bool = False):
        self.stream = stream or sys.stderr
        if color is None:
            color = getattr(self.stream, "isatty", lambda: False)()
        self.c = _C(color)
        self.show_prompts = show_prompts

    def _print(self, *parts: str) -> None:
        self.stream.write(" ".join(parts) + "\n")
        self.stream.flush()

    def on_event(self, event: TraceEvent) -> None:
        c = self.c
        if isinstance(event, AgentStart):
            names = ", ".join(getattr(t, "name", "?") for t in (event.tools or []))
            self._print(c.bold("agent"), c.dim(f"model={event.model or '-'} tools=[{names}]"))
        elif isinstance(event, LLMCall):
            self._print(c.cyan(f"▸ step {(_si(event.step))}"))
            if self.show_prompts:
                for m in event.messages:
                    role = getattr(m, "role", "?")
                    content = getattr(m, "content", "") or ""
                    self._print(c.dim(f"    [{role}] {_truncate(content, 200)}"))
        elif isinstance(event, LLMRetry):
            self._print(c.yellow(
                f"  ↻ provider busy ({event.status or '?'}), retrying in {event.delay:.1f}s"
            ))
        elif isinstance(event, LLMResponse):
            resp = event.response
            calls = getattr(resp, "tool_calls", None) or []
            if calls:
                for call in calls:
                    self._print("    " + c.dim("llm →"),
                                f"calls {call.name}({_fmt_args(call.arguments)})")
            else:
                usage = getattr(resp, "usage", None)
                toks = getattr(usage, "total_tokens", None)
                tail = f"({toks} tokens, $0.00 — Purdue)" if toks else ""
                self._print("    " + c.dim("llm →"), c.green("final answer"), c.dim(tail))
        elif isinstance(event, ToolResultEvent):
            res = event.result
            elapsed = f"({event.elapsed:.1f}s)" if event.elapsed is not None else ""
            if getattr(res, "error", None):
                self._print("    " + c.dim("tool ←"), c.red(_truncate(res.error, 80)), c.dim(elapsed))
            else:
                content = getattr(res, "content", "") or ""
                self._print("    " + c.dim("tool ←"), _truncate(content, 80), c.dim(elapsed))
        elif isinstance(event, AgentEnd):
            self._print(c.dim(f"agent end ({event.stopped})"))


def _si(step) -> str:
    return str((step if step is not None else 0) + 1)


class JsonlTracer:
    """Append one JSON object per event to a file. Eval-harness substrate."""

    def __init__(self, path, *, mode: str = "a"):
        self.path = path
        self._f = open(path, mode)

    def on_event(self, event: TraceEvent) -> None:
        self._f.write(json.dumps(_event_to_dict(event), default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlTracer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _event_to_dict(event: TraceEvent) -> dict:
    d = _jsonable(event)
    d["type"] = type(event).__name__
    return d


def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/containers to JSON-able values, dropping
    the heavy ``raw`` provider payload."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if f.name == "raw":  # untouched provider object — never serialize
                continue
            out[f.name] = _jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
