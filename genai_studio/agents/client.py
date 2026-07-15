"""
``ModelClient`` — the reuse seam: "chat, optionally with tools".

The agent loop never talks to an API directly; it talks to a ``ModelClient`` and
always receives the same ``ModelResponse`` shape. Two implementations ship here:

- :class:`GenAIStudioClient` — wraps the repo's ``GenAIStudio`` client, sends
  native ``tools=`` to the OpenAI-compatible gateway, and parses ``tool_calls``.
- :class:`ReActClient` — a *decorator* that gives tool-calling to any backend
  lacking native support, by prompting for a JSON action and parsing it back into
  the **same** ``ModelResponse``. The ``Agent`` is byte-identical either way — so a
  lesson can swap ``GenAIStudioClient`` <-> ``ReActClient(GenAIStudioClient)`` and
  watch the mechanics of tool-calling appear.

Retry/backoff for transient provider failures lives *here* (the client seam) so
every consumer inherits it. Async + streaming methods are declared now and filled
in by the production layer (M3).
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

from .errors import TransientError
from .tool import ToolSpec

# Status codes worth retrying: rate-limit, gateway/overload, and 5xx.
# 529 = "overloaded" (Anthropic-style), surfaced through the gateway.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


# ════════════════════════════════════════════════════════════════════════════
# Rate limiting — pace requests so a capacity-limited gateway never drops them
# ════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Global request pacer: guarantees ≥ ``60/rpm`` seconds between requests.

    A capacity-limited gateway (e.g. Purdue GenAI Studio, ~20 req/min) silently
    *drops* responses under burst load rather than returning 429. Spacing requests
    keeps every consumer — sync, async, and concurrent agents — under the limit so
    nothing is dropped. Thread- and asyncio-safe; share ONE across all clients.
    """

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0  # earliest monotonic time the next request may start

    def _reserve(self) -> float:
        if self.min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
            return max(0.0, start - now)

    def acquire(self) -> None:
        wait = self._reserve()
        if wait:
            time.sleep(wait)

    async def aacquire(self) -> None:
        wait = self._reserve()
        if wait:
            await asyncio.sleep(wait)


class _NullLimiter:
    def acquire(self) -> None: ...
    async def aacquire(self) -> None: ...


_DEFAULT_LIMITER = None


def _default_limiter():
    """Process-wide limiter from ``GENAI_STUDIO_RPM`` (unset/0 → no limiting)."""
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        try:
            rpm = float(os.getenv("GENAI_STUDIO_RPM", "0"))
        except ValueError:
            rpm = 0.0
        _DEFAULT_LIMITER = RateLimiter(rpm) if rpm > 0 else _NullLimiter()
    return _DEFAULT_LIMITER


# ════════════════════════════════════════════════════════════════════════════
# Wire dataclasses — identical in sync and async, native and ReAct paths
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """A model's request to call one tool."""

    id: str
    name: str
    arguments: dict
    raw_arguments: str | None = None  # original JSON string (debugging)


@dataclass
class Usage:
    """Token accounting; every field may be ``None`` (gateway may omit usage)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def zero(cls) -> "Usage":
        return cls(0, 0, 0)

    def __add__(self, other: "Usage") -> "Usage":
        def _s(a, b):
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            _s(self.prompt_tokens, other.prompt_tokens),
            _s(self.completion_tokens, other.completion_tokens),
            _s(self.total_tokens, other.total_tokens),
        )


@dataclass
class Message:
    """One conversation message. Knows how to render itself to the OpenAI wire."""

    role: str  # 'system' | 'user' | 'assistant' | 'tool'
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role == 'tool'
    name: str | None = None  # tool name on role == 'tool'

    def to_openai(self) -> dict:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content or "",
            }
        msg: dict = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments
                        if tc.raw_arguments is not None
                        else json.dumps(tc.arguments or {}),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall] = ()) -> "Message":
        return cls(role="assistant", content=content, tool_calls=list(tool_calls))


@dataclass
class ModelResponse:
    """A model turn: free text, and/or tool calls, plus usage/metadata."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None  # untouched provider payload (debugging/teaching)


def _as_openai_messages(messages) -> list[dict]:
    """Accept ``Message`` objects or raw dicts; emit OpenAI wire dicts."""
    out = []
    for m in messages:
        out.append(m.to_openai() if isinstance(m, Message) else m)
    return out


def _loads_args(s: str | None) -> dict:
    """Tolerant JSON-args parse: ``None``/empty -> ``{}``, malformed -> ``{}``."""
    if not s:
        return {}
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, ValueError):
        return {}


# Machine wrappers models put around text-emitted tool calls. Hermes/Qwen-style
# tag blocks, llama-3-style <function=name>{args}</function>, the llama
# <|python_tag|> prefix, and whole-message ```json fences.
_CALL_TAG_RE = re.compile(
    r"<\s*(?:tool_call|function_call|toolcall)\s*>\s*(.*?)\s*"
    r"(?:<\s*/\s*(?:tool_call|function_call|toolcall)\s*>|\Z)", re.S | re.I)
_FN_TAG_RE = re.compile(
    r"<\s*function\s*=\s*([\w.\-]+)\s*>\s*(.*?)\s*(?:<\s*/\s*function\s*>|\Z)", re.S | re.I)
_FENCE_WRAP_RE = re.compile(r"\A```[\w+-]*[ \t]*\n(.*?)\n?\s*```\s*\Z", re.S)
_TRAIL_FENCE_RE = re.compile(r"\n```[\w+-]*[ \t]*\n(.*?)\n?\s*```\s*\Z", re.S)
_ARG_KEYS = ("arguments", "parameters", "args", "tool_input")
# How much prose may precede a trailing call payload before we refuse to treat it
# as a call (a long answer that merely ENDS with example JSON stays an answer).
_MAX_PREAMBLE = 300


def _calls_from_json_payload(s: str, *, require_args: bool = False) -> list["ToolCall"]:
    """Parse one JSON payload (object or array) into tool calls; ``[]`` if it
    isn't call-shaped. With ``require_args`` an item must carry an explicit
    arguments-ish key — used for payloads found AFTER prose, where a bare
    ``{"name": ...}`` is far more likely data than a call."""
    obj = None
    try:
        obj = json.loads(s)
    except (TypeError, ValueError):
        span = _balanced_object(s, s.find("{"))
        if span is not None:
            try:
                obj = json.loads(span)
            except (TypeError, ValueError):
                obj = None
    if obj is None:
        return []
    if isinstance(obj, dict):
        tc = obj.get("tool_calls", obj.get("toolcalls", obj.get("tool_call")))
        if isinstance(tc, list):
            items = tc
        elif isinstance(tc, dict):
            items = [tc]
        else:
            items = [obj]
    else:
        items = obj if isinstance(obj, list) else [obj]
    calls: list[ToolCall] = []
    for c in items:
        if not isinstance(c, dict):
            continue
        f = c.get("function")
        if isinstance(f, dict):
            fn = f                                    # {"function": {"name", "arguments"}}
        elif isinstance(f, str):                      # {"function": "<name>", "arguments": {...}}
            fn = {"name": f, "arguments": c.get("arguments", c.get("parameters", c.get("args", {})))}
        else:
            fn = c                                    # {"name", "arguments"} flat
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            # {"tool": "<name>", "tool_input": {...}} — accepted only WITH an args
            # key, so data that merely has a "tool" field can't become a call.
            t = fn.get("tool", fn.get("tool_name"))
            if isinstance(t, str) and t and any(k in fn for k in _ARG_KEYS):
                name = t
            else:
                continue
        if require_args and not any(k in fn for k in _ARG_KEYS):
            continue
        args = fn.get("arguments", fn.get("parameters", fn.get("args", fn.get("tool_input", {}))))
        if isinstance(args, str):
            args = _loads_args(args)
        calls.append(ToolCall(id=c.get("id") or f"text-{uuid.uuid4().hex[:6]}",
                              name=name, arguments=args if isinstance(args, dict) else {}))
    return calls


def _calls_from_pythonic(s: str) -> list["ToolCall"]:
    """Parse llama-style pythonic call listings — ``[read_file(path="x")]`` or a
    bare ``read_file(path="x")`` — into tool calls. The WHOLE string must be the
    listing, every element a call with keyword-only literal arguments; anything
    else returns ``[]`` (prose and ordinary code never qualify)."""
    s = s.strip()
    if not re.match(r"\[?\s*[A-Za-z_][\w.]*\s*\(", s):
        return []
    try:
        tree = ast.parse(s, mode="eval").body
    except (SyntaxError, ValueError):
        return []
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls: list[ToolCall] = []
    for n in nodes:
        if not isinstance(n, ast.Call) or n.args:      # positional args -> not a tool call
            return []
        func, parts = n.func, []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name):
            return []
        name = ".".join([func.id, *reversed(parts)])
        args = {}
        for kw in n.keywords:
            if kw.arg is None:                         # **kwargs -> bail
                return []
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                return []
        calls.append(ToolCall(id=f"text-{uuid.uuid4().hex[:6]}", name=name, arguments=args))
    return calls


def _tool_calls_from_text(content: str | None) -> list["ToolCall"]:
    """Recover a tool call that a model emitted in the message CONTENT instead of
    the native ``tool_calls`` field.

    Some OpenAI-compatible gateways / models (observed on the Purdue gateway with
    qwen and llama) emit the call as text. Recognized shapes, most-specific first:

    - ``<tool_call>{...}</tool_call>`` tag blocks (Hermes/Qwen; multiple allowed)
      and llama's ``<function=name>{args}</function>`` / ``<|python_tag|>`` prefix;
    - the whole message as (optionally ```json-fenced) call JSON —
      ``{"function": {...}}`` / ``{"name", "arguments"}`` / ``{"tool_calls": [...]}``
      / ``{"tool": ..., "tool_input": ...}`` — or a pythonic ``[f(a=1)]`` listing;
    - a SHORT prose preamble ("Let's read the file.") followed by nothing but the
      call payload (bare or fenced JSON with an explicit arguments key).

    Without this, the loop would mistake the text for a final answer. Deliberately
    conservative everywhere: a long answer that merely contains or ends with
    JSON-looking data is left alone.
    """
    if not content:
        return []
    s = content.strip()

    # 1) Unambiguous tag wrappers — safe to honor anywhere in the text.
    tag_bodies = _CALL_TAG_RE.findall(s)
    if tag_bodies:
        calls = [c for body in tag_bodies for c in
                 (_calls_from_json_payload(body) or _calls_from_pythonic(body))]
        if calls:
            return calls
    m = _FN_TAG_RE.search(s)
    if m:
        raw = m.group(2).strip()
        try:
            parsed = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return [ToolCall(id=f"text-{uuid.uuid4().hex[:6]}",
                             name=m.group(1), arguments=parsed)]
    if s.startswith("<|python_tag|>"):
        s = s[len("<|python_tag|>"):].strip()

    # 2) The whole message is the payload (optionally fenced).
    m = _FENCE_WRAP_RE.match(s)
    if m:
        s = m.group(1).strip()
    if s.startswith("{") or s.startswith("["):
        calls = _calls_from_json_payload(s)
        if calls:
            return calls
    calls = _calls_from_pythonic(s)
    if calls:
        return calls

    # 3) Short prose preamble + trailing call payload, nothing after it.
    m = _TRAIL_FENCE_RE.search(s)
    if m and m.start() <= _MAX_PREAMBLE:
        return _calls_from_json_payload(m.group(1).strip(), require_args=True)
    stripped = s.rstrip()
    if stripped.endswith("}"):
        idx = stripped.find("{")
        while 0 <= idx:
            span = _balanced_object(stripped, idx)
            if span is not None and idx + len(span) == len(stripped):
                if 0 < idx <= _MAX_PREAMBLE:            # idx == 0 was case (2)
                    return _calls_from_json_payload(span, require_args=True)
                break
            idx = stripped.find("{", idx + 1)
    return []


# ════════════════════════════════════════════════════════════════════════════
# The protocol + a base with deferred async/stream defaults
# ════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ModelClient(Protocol):
    @property
    def supports_native_tools(self) -> bool: ...

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_retry: Callable | None = None,
        **opts,
    ) -> ModelResponse: ...

    async def acomplete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_retry: Callable | None = None,
        **opts,
    ) -> ModelResponse: ...

    def stream(self, messages, *, tools=None, **opts): ...  # -> Iterator[StreamEvent]

    async def astream(self, messages, *, tools=None, **opts): ...  # -> AsyncIterator


class BaseModelClient:
    """Mixin providing deferred async/stream methods + the shared retry helpers."""

    supports_streaming = False  # overridden by clients with real token streaming

    async def acomplete(self, *args, **kwargs) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} does not implement async; use a client that does."
        )

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} does not implement streaming."
        )

    async def astream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} does not implement async streaming."
        )


@dataclass
class RetryPolicy:
    """Capped exponential backoff with full jitter for transient failures."""

    max_retries: int = 4
    base: float = 0.5
    cap: float = 8.0

    def backoff(self, attempt: int) -> float:
        raw = min(self.cap, self.base * (2 ** attempt))
        return raw * (0.5 + 0.5 * random.random())  # full-jitter in [raw/2, raw]


def _classify(exc: BaseException) -> TransientError | None:
    """Map an OpenAI/httpx/SDK exception to a ``TransientError`` (or ``None``)."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        openai = None

    if openai is not None:
        if isinstance(exc, getattr(openai, "RateLimitError", ())):
            return TransientError(str(exc), status_code=429, retry_after=_retry_after(exc))
        status = getattr(exc, "status_code", None)
        if isinstance(exc, getattr(openai, "APIStatusError", ())) and status in _RETRYABLE_STATUS:
            return TransientError(str(exc), status_code=status, retry_after=_retry_after(exc))
        if isinstance(exc, (getattr(openai, "APITimeoutError", ()),
                            getattr(openai, "APIConnectionError", ()))):
            return TransientError(str(exc))

    # The SDK's own ConnectionError (raised by _chat_create on empty completions).
    for base in type(exc).__mro__:
        if base.__name__ == "ConnectionError" and base.__module__.startswith("genai_studio"):
            return TransientError(str(exc))
    return None


def _retry_after(exc) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    val = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _with_retry(op: Callable, policy: RetryPolicy, on_retry: Callable | None):
    """Run ``op`` with backoff on transient failures (sync)."""
    for attempt in range(policy.max_retries + 1):
        try:
            return op()
        except Exception as exc:
            te = _classify(exc)
            if te is None or attempt == policy.max_retries:
                raise te if te is not None else exc
            delay = te.retry_after if te.retry_after is not None else policy.backoff(attempt)
            if on_retry is not None:
                on_retry(attempt, delay, te)
            time.sleep(delay)


async def _awith_retry(op: Callable, policy: RetryPolicy, on_retry: Callable | None):
    """Async counterpart of ``_with_retry`` — ``await op()`` with backoff."""
    for attempt in range(policy.max_retries + 1):
        try:
            return await op()
        except Exception as exc:
            te = _classify(exc)
            if te is None or attempt == policy.max_retries:
                raise te if te is not None else exc
            delay = te.retry_after if te.retry_after is not None else policy.backoff(attempt)
            if on_retry is not None:
                on_retry(attempt, delay, te)
            await asyncio.sleep(delay)


# ── low-level streaming chunks (below AgentEvent; assembled into ModelResponse) ─
@dataclass
class _TextChunk:
    delta: str


@dataclass
class _ToolChunk:
    index: int
    id: str | None
    name: str | None
    args_fragment: str


@dataclass
class _StreamDone:
    finish_reason: str | None = None
    usage: Usage | None = None


def _assemble_tool_calls(buf: dict) -> tuple[list[ToolCall], bool]:
    """Assemble streamed tool-call fragments (keyed by index) into ToolCalls.

    Returns ``(calls, ok)``; ``ok`` is False if any fragment stream failed to
    parse as JSON — the caller then degrades to a non-streamed completion.
    """
    calls: list[ToolCall] = []
    ok = True
    for idx in sorted(buf):
        b = buf[idx]
        if not b.get("name"):
            continue
        raw = b.get("args") or ""
        try:
            args = json.loads(raw) if raw.strip() else {}
            if not isinstance(args, dict):
                args = {"value": args}
        except (TypeError, ValueError):
            args, ok = {}, False
        calls.append(ToolCall(id=b.get("id") or f"stream-{idx}",
                              name=b["name"], arguments=args, raw_arguments=raw))
    return calls, ok


def _usage_from(u) -> Usage | None:
    if u is None:
        return None
    return Usage(getattr(u, "prompt_tokens", None),
                 getattr(u, "completion_tokens", None),
                 getattr(u, "total_tokens", None))


def _emit_chunk(chunk):
    """Translate one OpenAI stream chunk into low-level ``_TextChunk`` /
    ``_ToolChunk`` / ``_StreamDone`` events."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        u = _usage_from(getattr(chunk, "usage", None))
        if u is not None:
            yield _StreamDone(usage=u)
        return
    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if content:
            yield _TextChunk(delta=content)
        for tc in (getattr(delta, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            yield _ToolChunk(
                index=getattr(tc, "index", 0) or 0,
                id=getattr(tc, "id", None),
                name=getattr(fn, "name", None) if fn else None,
                args_fragment=(getattr(fn, "arguments", None) if fn else None) or "",
            )
    if getattr(choice, "finish_reason", None):
        yield _StreamDone(finish_reason=choice.finish_reason,
                          usage=_usage_from(getattr(chunk, "usage", None)))


# ════════════════════════════════════════════════════════════════════════════
# GenAIStudioClient — native tool-calling over the Purdue gateway
# ════════════════════════════════════════════════════════════════════════════

class GenAIStudioClient(BaseModelClient):
    """A ``ModelClient`` over the repo's ``GenAIStudio`` client.

    Routes through the public ``GenAIStudio.chat_raw(...)`` so it can read
    ``choices[0].message.tool_calls`` (which ``ChatResponse`` drops) and inherit
    the empty-completion retry — never coupling to a private method.
    """

    def __init__(
        self,
        studio,
        *,
        default_model: str | None = None,
        native_tools: bool | None = None,
        retry: RetryPolicy | None = None,
        rate_limiter=None,
    ):
        self._studio = studio
        self._default_model = default_model
        self._retry = retry or RetryPolicy()
        # None => probe lazily on first tool-bearing call; True/False => override.
        self._native_cached: bool | None = native_tools
        self._async = None  # lazily-built AsyncOpenAI (M3)
        # Shared pacer (default from GENAI_STUDIO_RPM) so bursts never exceed the
        # gateway's capacity and get dropped.
        self._rl = rate_limiter if rate_limiter is not None else _default_limiter()

    # ── capability probe ────────────────────────────────────────────────────
    @property
    def supports_native_tools(self) -> bool:
        if self._native_cached is None:
            self._native_cached = self.probe_native_tools()
        return self._native_cached

    def probe_native_tools(self, *, force: bool = False) -> bool:
        """One-time, fail-safe probe: does the gateway return ``tool_calls``?

        Sends a tiny forced-tool request. Any failure (gateway rejects ``tools``,
        400, timeout) -> ``False`` so we transparently degrade to ReAct. Public so
        a lesson can demonstrate the capability check explicitly.
        """
        if not force and self._native_cached is not None:
            return self._native_cached
        probe_tool = {
            "type": "function",
            "function": {
                "name": "_probe",
                "description": "Probe tool. Call it with done=true.",
                "parameters": {
                    "type": "object",
                    "properties": {"done": {"type": "boolean"}},
                    "required": ["done"],
                },
            },
        }
        try:
            self._rl.acquire()
            raw = self._studio.chat_raw(
                [{"role": "user", "content": "Call the _probe tool with done=true."}],
                model=self._default_model,
                tools=[probe_tool],
                tool_choice="auto",
                max_tokens=64,
            )
            tcs = getattr(raw.choices[0].message, "tool_calls", None)
            return bool(tcs)
        except Exception:
            return False

    # ── parsing ─────────────────────────────────────────────────────────────
    def _parse_completion(self, raw) -> ModelResponse:
        choice = raw.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = tc.function
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", None) or uuid.uuid4().hex[:8],
                    name=fn.name,
                    arguments=_loads_args(fn.arguments),
                    raw_arguments=fn.arguments,
                )
            )
        u = getattr(raw, "usage", None)
        usage = Usage(
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            getattr(u, "total_tokens", None),
        )
        text = getattr(msg, "content", None)
        # Robustness: if the model emitted a tool call as JSON in the content
        # (no native tool_calls), recover it instead of treating it as the answer.
        if not tool_calls and text:
            recovered = _tool_calls_from_text(text)
            if recovered:
                tool_calls, text = recovered, None
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=raw,
        )

    # ── sync completion ─────────────────────────────────────────────────────
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_retry: Callable | None = None,
        **opts,
    ) -> ModelResponse:
        kwargs = dict(opts)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools and self.supports_native_tools:
            kwargs["tools"] = [s.to_openai() for s in tools]
            kwargs.setdefault("tool_choice", "auto")

        openai_messages = _as_openai_messages(messages)

        def op():
            self._rl.acquire()  # pace under the gateway limit (incl. retries)
            return self._studio.chat_raw(
                openai_messages, model=model or self._default_model, **kwargs
            )

        raw = _with_retry(op, self._retry, on_retry)
        return self._parse_completion(raw)

    # ── async + streaming (production layer) ─────────────────────────────────
    @property
    def supports_streaming(self) -> bool:
        return True

    def _aclient(self):
        """Lazily build an ``AsyncOpenAI`` from the same config as the sync client."""
        if self._async is None:
            from openai import AsyncOpenAI
            self._async = AsyncOpenAI(
                api_key=self._studio.api_key,
                base_url=f"{self._studio.base_url}/api",
                timeout=self._studio.timeout,
            )
        return self._async

    def _build_kwargs(self, tools, temperature, opts) -> dict:
        kwargs = dict(opts)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools and self.supports_native_tools:
            kwargs["tools"] = [s.to_openai() for s in tools]
            kwargs.setdefault("tool_choice", "auto")
        return kwargs

    async def acomplete(
        self, messages, *, tools=None, model=None, temperature=None, on_retry=None, **opts
    ) -> ModelResponse:
        kwargs = self._build_kwargs(tools, temperature, opts)
        openai_messages = _as_openai_messages(messages)

        async def op():
            await self._rl.aacquire()
            return await self._aclient().chat.completions.create(
                model=model or self._default_model, messages=openai_messages, **kwargs
            )

        raw = await _awith_retry(op, self._retry, on_retry)
        return self._parse_completion(raw)

    def stream(self, messages, *, tools=None, model=None, temperature=None, **opts):
        kwargs = self._build_kwargs(tools, temperature, opts)
        openai_messages = _as_openai_messages(messages)
        # Stream via the raw OpenAI client directly — chat_raw's empty-completion
        # guard assumes a non-stream response, so it must be bypassed here.
        self._rl.acquire()
        gen = self._studio.client.chat.completions.create(
            model=model or self._default_model, messages=openai_messages,
            stream=True, **kwargs,
        )
        for chunk in gen:
            yield from _emit_chunk(chunk)

    async def astream(self, messages, *, tools=None, model=None, temperature=None, **opts):
        kwargs = self._build_kwargs(tools, temperature, opts)
        openai_messages = _as_openai_messages(messages)
        await self._rl.aacquire()
        gen = await self._aclient().chat.completions.create(
            model=model or self._default_model, messages=openai_messages,
            stream=True, **kwargs,
        )
        async for chunk in gen:
            for ev in _emit_chunk(chunk):
                yield ev


# ════════════════════════════════════════════════════════════════════════════
# ReActClient — synthesize tool-calling via a JSON-action protocol
# ════════════════════════════════════════════════════════════════════════════

_REACT_INSTRUCTIONS = """\
You can use tools to help answer. To call a tool, respond with ONLY a single \
JSON object on one line — no prose, no code fences:
{{"action": "<tool_name>", "action_input": {{<arguments>}}}}

When you have the final answer, respond with ONLY:
{{"action": "final", "action_input": "<your answer>"}}

Available tools:
{tools}

Rules:
- Output exactly one JSON object and nothing else.
- Use only the tools listed above.
- action_input must match the tool's parameters."""

_REACT_REPAIR = (
    "Your last message was not a single valid JSON object. "
    "Respond with ONLY the JSON object described above."
)


class ReActClient(BaseModelClient):
    """Give tool-calling to any backend by prompting for a JSON action.

    Produces the same ``ModelResponse`` as native tool-calling, so the ``Agent``
    is unchanged. When no tools are passed, it is a transparent pass-through.
    """

    def __init__(self, inner: ModelClient, *, max_json_repair: int = 1):
        self._inner = inner
        self._max_json_repair = max_json_repair

    @property
    def supports_native_tools(self) -> bool:
        return True  # it *synthesizes* tool-calling regardless of the inner backend

    # ── prompt construction ─────────────────────────────────────────────────
    @staticmethod
    def _render_tools(tools: list[ToolSpec]) -> str:
        lines = []
        for s in tools:
            props = s.parameters.get("properties", {})
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in props.items()
            )
            desc = (s.description or "").splitlines()[0] if s.description else ""
            lines.append(f"- {s.name}({params}): {desc}")
        return "\n".join(lines)

    def _augment(self, messages: list[Message], tools: list[ToolSpec]) -> list[Message]:
        """Inject the ReAct protocol into the system message and make tool
        replies legible to a non-native backend (role 'tool' -> 'Observation')."""
        block = _REACT_INSTRUCTIONS.format(tools=self._render_tools(tools))
        out: list[Message] = []
        injected = False
        for m in messages:
            if m.role == "system" and not injected:
                out.append(Message.system((m.content or "") + "\n\n" + block))
                injected = True
            elif m.role == "tool":
                out.append(Message.user("Observation: " + (m.content or "")))
            elif m.role == "assistant" and m.tool_calls:
                # Echo the prior action as text the backend can read back.
                actions = "; ".join(f"{tc.name}({json.dumps(tc.arguments)})" for tc in m.tool_calls)
                out.append(Message.assistant((m.content or "") + f"\n[called: {actions}]"))
            else:
                out.append(m)
        if not injected:
            out.insert(0, Message.system(block))
        return out

    # ── parsing ─────────────────────────────────────────────────────────────
    @staticmethod
    def _react_response(resp: ModelResponse) -> ModelResponse | None:
        """Map a raw inner response to a ModelResponse, or None if unparseable."""
        parsed = _parse_react(resp.text or "")
        if parsed is None:
            return None
        action, value = parsed
        if action == "final":
            return ModelResponse(text=str(value), tool_calls=[],
                                 usage=resp.usage, finish_reason="stop", raw=resp.raw)
        call = ToolCall(
            id=f"react-{uuid.uuid4().hex[:6]}",
            name=action,
            arguments=value if isinstance(value, dict) else {"input": value},
        )
        return ModelResponse(text=None, tool_calls=[call],
                             usage=resp.usage, finish_reason="tool_calls", raw=resp.raw)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_retry: Callable | None = None,
        **opts,
    ) -> ModelResponse:
        if not tools:
            return self._inner.complete(
                messages, model=model, temperature=temperature, on_retry=on_retry, **opts
            )
        convo = self._augment(list(messages), tools)
        resp = None
        for attempt in range(self._max_json_repair + 1):
            resp = self._inner.complete(
                convo, model=model, temperature=temperature, on_retry=on_retry, **opts
            )
            out = self._react_response(resp)
            if out is not None:
                return out
            if attempt < self._max_json_repair:
                convo = convo + [Message.user(_REACT_REPAIR)]
        # Unparseable after repair: hand back the model's prose as the final answer.
        return ModelResponse(text=resp.text, tool_calls=[],
                             usage=resp.usage, finish_reason="stop", raw=resp.raw)

    async def acomplete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        on_retry: Callable | None = None,
        **opts,
    ) -> ModelResponse:
        if not tools:
            return await self._inner.acomplete(
                messages, model=model, temperature=temperature, on_retry=on_retry, **opts
            )
        convo = self._augment(list(messages), tools)
        resp = None
        for attempt in range(self._max_json_repair + 1):
            resp = await self._inner.acomplete(
                convo, model=model, temperature=temperature, on_retry=on_retry, **opts
            )
            out = self._react_response(resp)
            if out is not None:
                return out
            if attempt < self._max_json_repair:
                convo = convo + [Message.user(_REACT_REPAIR)]
        return ModelResponse(text=resp.text, tool_calls=[],
                             usage=resp.usage, finish_reason="stop", raw=resp.raw)


def _parse_react(text: str):
    """Extract a ReAct action from model text. Returns ``(action, action_input)``
    or ``None`` if no JSON object is found."""
    if not text:
        return None
    # Strip code fences and any leading prose before the first '{'.
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    # Try a direct parse, then a balanced-brace extraction.
    candidates = [cleaned[start:]]
    span = _balanced_object(cleaned, start)
    if span is not None:
        candidates.insert(0, span)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj.get("action"), obj.get("action_input")
    return None


def _balanced_object(s: str, start: int) -> str | None:
    """Return the substring of the first balanced ``{...}`` starting at ``start``."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None
