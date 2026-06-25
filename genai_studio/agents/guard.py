"""
``Guard`` — a deterministic before/after-tool seam (the hook pattern).

A guard runs DETERMINISTIC code around every tool dispatch, the complement to
probabilistic prompting. It is the single chokepoint where cross-cutting policy
lives, so several concerns collapse into one place instead of being sprinkled
through the loop:

- **policy** — allow/deny a tool call (e.g. block ``python_exec`` on untrusted input);
- **tree-wide budget** — count tool calls across a whole agent team (one shared
  ``BudgetGuard``), denying once a cap is hit;
- **cancellation / pacing** — halt or rate-limit before a call;
- **redaction / post-processing** — rewrite a tool's result after it runs.

Two hooks, both optional (subclass :class:`Guard` and override what you need, or
use the :func:`guard` factory over plain callables):

    before_tool(call) -> Decision | None   # None == allow; deny(reason) / modify(args)
    after_tool(call, result) -> ToolResult | None   # None == keep result

Wired into the agent loop at the tool-dispatch chokepoint (``Agent(guards=[...])``).
**Never crashes a run:** a ``before_tool`` that raises fails CLOSED (the call is
blocked with a guard-error message); an ``after_tool`` that raises is ignored
(the original result stands).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from .tool import ToolResult


# ── the decision a before_tool hook returns ──────────────────────────────────
@dataclass(frozen=True)
class Decision:
    """What a ``before_tool`` guard decides. Use :data:`ALLOW` / :func:`deny` /
    :func:`modify` rather than constructing directly."""

    action: str = "allow"            # "allow" | "deny" | "modify"
    reason: str | None = None        # for deny -> fed back to the model
    arguments: dict | None = None    # for modify -> the rewritten tool arguments


ALLOW = Decision("allow")


def deny(reason: str) -> Decision:
    """Block the tool call; ``reason`` is fed back to the model as the tool result."""
    return Decision("deny", reason=reason)


def modify(arguments: dict) -> Decision:
    """Run the tool with ``arguments`` replaced (override what the model asked for)."""
    return Decision("modify", arguments=arguments)


# ── the guard protocol ───────────────────────────────────────────────────────
class Guard:
    """Base class for a tool guard. Override ``before_tool`` and/or ``after_tool``."""

    def before_tool(self, call) -> Decision | None:  # noqa: D401
        """Run before a tool executes. Return ``None``/``ALLOW`` to proceed,
        ``deny(reason)`` to block, or ``modify(args)`` to rewrite the call."""
        return None

    def after_tool(self, call, result: ToolResult) -> ToolResult | None:
        """Run after a tool executes. Return a (new) ``ToolResult`` to replace it,
        or ``None`` to keep the original."""
        return None


class _FunctionGuard(Guard):
    def __init__(self, before=None, after=None):
        self._before, self._after = before, after

    def before_tool(self, call):
        return self._before(call) if self._before is not None else None

    def after_tool(self, call, result):
        return self._after(call, result) if self._after is not None else None


def guard(*, before: Callable | None = None, after: Callable | None = None) -> Guard:
    """Wrap plain callables into a :class:`Guard`.

    ``before(call) -> Decision | None`` and ``after(call, result) -> ToolResult | None``.
    """
    return _FunctionGuard(before, after)


# ── shipped guards ───────────────────────────────────────────────────────────
class BudgetGuard(Guard):
    """A **tree-wide** tool-call budget enforced as a ``before_tool`` gate.

    Differs from the per-run :class:`Budget`: ``Budget`` caps ONE run (tokens / steps
    / tool-calls) and *raises* on overflow → the run ends with ``stopped="budget"``.
    ``BudgetGuard`` caps tool calls across a WHOLE team and *denies the one call*
    (feeding an error back) → the run continues and the model may adapt.

    Share ONE instance across every agent in a team — pass it to each ``Agent``'s
    ``guards=``. The framework does NOT propagate guards to sub-agents for you; a
    ``BudgetGuard`` on only the manager bounds only the manager's own calls. (A
    future ``Team`` will inject it for you.) Thread-safe, so it is correct even when
    sub-agents run off-thread under ``arun``. ``tool_calls`` counts ALLOWED calls
    (denied attempts don't increment), so it never exceeds ``max_tool_calls``.
    """

    def __init__(self, *, max_tool_calls: int | None = None):
        self.max_tool_calls = max_tool_calls
        self._tool_calls = 0
        self._lock = threading.Lock()

    @property
    def tool_calls(self) -> int:
        """Number of tool calls this guard has ALLOWED (≤ ``max_tool_calls``)."""
        return self._tool_calls

    def before_tool(self, call) -> Decision | None:
        if self.max_tool_calls is None:
            return None
        with self._lock:
            if self._tool_calls >= self.max_tool_calls:  # check BEFORE counting
                return deny(f"tree tool-call budget exhausted ({self.max_tool_calls})")
            self._tool_calls += 1
        return None


class ToolFilterGuard(Guard):
    """Allow/deny tool calls by name — a simple deterministic policy.

    If ``allow`` is given, ONLY those tool names may run; any name in ``block`` is
    always denied (``block`` takes precedence).
    """

    def __init__(self, *, allow: Any = None, block: Any = None):
        self.allow = set(allow) if allow is not None else None
        self.block = set(block or ())

    def before_tool(self, call) -> Decision | None:
        if call.name in self.block:
            return deny(f"tool {call.name!r} is blocked by policy")
        if self.allow is not None and call.name not in self.allow:
            return deny(f"tool {call.name!r} is not in the allowed set")
        return None
