"""
Streaming events — the ``AgentEvent`` union yielded by ``Agent.stream`` /
``Agent.astream`` for live UIs (Postyl renders these over SSE).

    async for ev in agent.astream("..."):
        match ev:
            case TextDelta(text=t):                 ui.append(t)
            case ToolCallStarted(name=n):           ui.status(f"Calling {n}…")
            case ToolCallFinished(result=r):        ui.cite(r.sources)
            case Final(result=res):                 ui.done(res)

Every event carries ``step`` so a UI can group deltas under the right step.
Because an async generator cannot ``return`` a value, the terminal ``Final``
event carries the complete ``AgentResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextDelta:
    text: str
    step: int


@dataclass(frozen=True)
class ToolCallStarted:
    id: str
    name: str
    arguments: dict
    step: int


@dataclass(frozen=True)
class ToolCallFinished:
    id: str
    name: str
    result: Any  # ToolResult — carries .content, .sources (citations), .data
    step: int


@dataclass(frozen=True)
class StepFinished:
    step: int
    had_text: bool
    tool_calls: int
    usage: Any


@dataclass(frozen=True)
class Final:
    result: Any  # AgentResult


AgentEvent = (
    "TextDelta | ToolCallStarted | ToolCallFinished | StepFinished | Final"
)
