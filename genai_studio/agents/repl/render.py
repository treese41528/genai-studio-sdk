"""``StreamRenderer`` — turns ``Agent.stream`` events into live terminal output.

Drives off the AgentEvent stream (NOT the tracer, which can't prompt). Assistant text is
BUFFERED per segment and flushed on the next tool call / final — and a segment that is
actually a tool-call JSON (some gateway models emit tool calls as text, recovered by the
Agent) is DROPPED rather than shown. A spinner covers model calls; it is never started
during tool execution (a tool may block on an approval ``input()`` prompt). ``StepFinished``
stub fields are ignored.
"""

from __future__ import annotations

import json
import sys

from ..client import _tool_calls_from_text
from ..events import Final, StepFinished, TextDelta, ToolCallFinished, ToolCallStarted
from .prettify import prettify

_DIM, _RED, _RESET = "\033[2m", "\033[31m", "\033[0m"


def _short(args: dict, n: int = 60) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return s if len(s) <= n else s[: n - 1] + "…"


def _oneline(s, n: int = 100) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _is_toolcall_text(text: str) -> bool:
    """True if the text is really a tool-call emitted as JSON (so we drop, not show it)."""
    try:
        return bool(_tool_calls_from_text(text))
    except Exception:
        return False


def _prose_preamble(text: str) -> str:
    """The human prose before a machine tool-call payload ("Let's read the file.
    {json}") — shown so a say-then-do preamble isn't swallowed with the payload."""
    cut = len(text)
    for marker in ("{", "```", "<tool_call", "<function", "<|python_tag|>", "["):
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


_ANSWER_KEYS = ("answer", "final_answer", "response", "result", "output", "text", "content", "code", "message")


def _dig_answer(obj, depth: int = 0):
    """Recursively pull the human answer out of a (possibly nested) JSON envelope like
    ``{"id":…, "response":{"result":"…"}}`` — follow answer-ish keys to the first non-empty string."""
    if isinstance(obj, str):
        return obj.strip() or None
    if depth > 6 or not isinstance(obj, dict):
        return None
    for key in _ANSWER_KEYS:                              # priority: descend named answer fields
        if key in obj:
            r = _dig_answer(obj[key], depth + 1)
            if r:
                return r
    if len(obj) == 1:                                    # single-key wrapper -> descend it
        return _dig_answer(next(iter(obj.values())), depth + 1)
    return None


def _unwrap_answer(text: str) -> str:
    """Some models wrap their whole final answer in a JSON envelope (``{"answer":"…"}`` or nested
    ``{"id":…,"response":{"result":"…"}}``) instead of plain prose — extract the inner text so the user
    sees clean output. Only fires when the entire message parses as one JSON object."""
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return text
    try:
        obj = json.loads(s)
    except (TypeError, ValueError):
        return text
    return _dig_answer(obj) or text


class StreamRenderer:
    def __init__(self, *, color=None, stream=None, pretty=True):
        self.out = stream or sys.stdout
        self.color = (self.out.isatty() if color is None else color)
        self.pretty = pretty                # LaTeX→Unicode + light markdown on assistant text
        self._spinner = None
        self._seg: list[str] = []          # assistant text buffered since the last flush
        self._printed = False              # did we print any real assistant text this turn?
        self.result = None

    def _c(self, s, code):
        return f"{code}{s}{_RESET}" if self.color else s

    def _spin(self, msg):
        self._stop_spin()
        try:
            from genai_studio import Spinner
            self._spinner = Spinner(msg)
            self._spinner.start()
        except Exception:
            self._spinner = None

    def _stop_spin(self):
        if self._spinner is not None:
            try:
                self._spinner.stop()
            except Exception:
                pass
            self._spinner = None

    def _flush_text(self) -> None:
        """Print the buffered assistant text, unless it's actually a tool-call JSON
        (then only its prose preamble, if any, is shown)."""
        text = "".join(self._seg).strip()
        self._seg = []
        if not text:
            return
        if _is_toolcall_text(text):
            pre = _prose_preamble(text)
            if pre:
                self.out.write(self._pretty(pre) + "\n")
                self.out.flush()
            return
        self.out.write(self._pretty(_unwrap_answer(text)) + "\n")
        self.out.flush()
        self._printed = True

    def _pretty(self, text: str) -> str:
        return prettify(text, color=self.color) if self.pretty else text

    def start(self):
        self._spin("thinking…")

    def handle(self, ev):
        if isinstance(ev, TextDelta):
            self._seg.append(ev.text)              # buffer (keep the spinner running)
        elif isinstance(ev, ToolCallStarted):
            self._stop_spin()
            self._flush_text()                     # show any real preamble; drop tool-JSON
            self.out.write(self._c(f"  → {ev.name}({_short(ev.arguments)})\n", _DIM))
            self.out.flush()
            # NB: no spinner here — the tool may block on an approval input() prompt.
        elif isinstance(ev, ToolCallFinished):
            self._stop_spin()
            r = ev.result
            if getattr(r, "error", None):
                self.out.write(self._c(f"  ✗ {_oneline(r.error)}\n", _RED))
            else:
                self.out.write(self._c(f"  ← {_oneline(getattr(r, 'content', ''))}\n", _DIM))
            self.out.flush()
            self._spin("thinking…")
        elif isinstance(ev, StepFinished):
            if self._seg:                      # a text-only step continued (e.g. after an
                self._seg.append("\n\n")       # intent nudge) — keep the segments apart
            self._spin("thinking…")
        elif isinstance(ev, Final):
            self._stop_spin()
            res = ev.result
            self.result = res
            self._flush_text()                     # the final answer (if streamed as text)
            text = (getattr(res, "text", "") or "").strip()
            if not self._printed and text and not _is_toolcall_text(text):
                self.out.write(self._pretty(_unwrap_answer(text)) + "\n")   # finish-tool / non-streamed answer
            stopped = getattr(res, "stopped", "final")
            if stopped and stopped != "final":
                note = {"cancelled": "(interrupted)", "max_steps": "(stopped: step limit)",
                        "budget": "(stopped: budget)",
                        "error": f"(error: {getattr(res, 'error', '')})"}.get(stopped, f"({stopped})")
                self.out.write(self._c(note + "\n", _RED if stopped == "error" else _DIM))
            self.out.flush()

    def abort(self):
        self._stop_spin()
        self.out.write(self._c("\n(aborted)\n", _RED))
        self.out.flush()
