"""
Typed errors for the agent framework.

Kept deliberately small and self-contained — the agent layer does NOT reuse the
SDK's ``genai_studio.GenAIStudioError`` hierarchy, so there are no name clashes
(e.g. the SDK already defines its own ``ConnectionError``). Where the underlying
client raises an SDK/OpenAI/httpx error, the model-client layer *translates* it
into one of these at the boundary (see ``client._classify``).

Exception map
-------------
- ``TransientError``  — a retryable provider failure (429/529/5xx, timeouts).
  Raised *inside* the model client after its own backoff is exhausted. The agent
  loop does NOT swallow it: production wants to see "provider busy, gave up".
- ``ToolError``       — a tool raised while executing. Normally NOT propagated;
  the registry captures it into ``ToolResult.error`` and the loop feeds it back
  so the model can recover. Raised only when ``Agent(on_tool_error="abort")``.
- ``BudgetExceeded``  — a ``Budget`` cap was hit. Raised by ``Budget`` checks and
  converted by the loop into a graceful ``AgentResult(stopped="budget")``.
- ``Cancelled``       — a cancellation token was tripped. Converted by the loop
  into ``AgentResult(stopped="cancelled")``.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for every error raised by the agent framework."""


class TransientError(AgentError):
    """A retryable provider failure (rate limit / overloaded / 5xx / timeout).

    Raised by the model client after its internal capped-exponential-backoff
    retries are exhausted. Carries the originating HTTP status (if any) and a
    ``retry_after`` hint parsed from a ``Retry-After`` header.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ToolError(AgentError):
    """A tool raised during execution.

    By default the loop captures the underlying exception into
    ``ToolResult.error`` and feeds it back to the model rather than raising —
    this is the design's "a tool raising is caught -> recover" behaviour. This
    exception is only raised when an ``Agent`` is configured with
    ``on_tool_error="abort"``.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        call_id: str | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.call_id = call_id
        self.cause = cause


class BudgetExceeded(AgentError):
    """A :class:`~genai_studio.agents.agent.Budget` limit was exceeded.

    ``kind`` is one of ``"tokens" | "steps" | "tool_calls"``. The agent loop
    catches this and returns a partial ``AgentResult(stopped="budget")`` rather
    than letting it propagate out of ``run()``.
    """

    def __init__(self, kind: str, limit, used):
        super().__init__(
            f"Budget exceeded: {kind} limit={limit} used={used}"
        )
        self.kind = kind
        self.limit = limit
        self.used = used


class Cancelled(AgentError):
    """A cancellation token was tripped (cooperative cancellation).

    The agent loop catches this and returns ``AgentResult(stopped="cancelled")``.
    """
