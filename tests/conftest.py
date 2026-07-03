"""Shared fixtures for the agent-framework unit tests.

The ``ModelClient`` protocol is the deterministic seam: ``ScriptedClient``
returns canned ``ModelResponse``s (including scripted tool calls) so the agent
loop, schema derivation, ReAct parsing, budgets, structured output, streaming
and tracing can all be tested with ZERO network.
"""

from __future__ import annotations

import json

import pytest

from genai_studio.agents import ModelResponse, ToolCall, Usage, tool
from genai_studio.agents.client import _StreamDone, _TextChunk, _ToolChunk


# ── response builders ────────────────────────────────────────────────────────
def says(text: str, **usage) -> ModelResponse:
    """A final text answer."""
    return ModelResponse(text=text, tool_calls=[],
                         usage=Usage(**usage) if usage else Usage(0, 0, 5),
                         finish_reason="stop")


def calls_tool(name: str, arguments: dict, *, id: str = "call_1") -> ModelResponse:
    """A turn that calls one tool."""
    return ModelResponse(text=None,
                         tool_calls=[ToolCall(id=id, name=name, arguments=arguments)],
                         usage=Usage(0, 0, 3), finish_reason="tool_calls")


class ScriptedClient:
    """A ``ModelClient`` that replays a fixed list of ``ModelResponse``s.

    ``calls`` records the (messages, tools) it saw each turn, so tests can assert
    the loop fed prior tool results back. ``supports_native_tools`` toggles the
    native vs ReAct path. Running past the script raises (catches over-stepping).
    """

    def __init__(self, script, *, native: bool = True, streaming: bool = True):
        self.script = list(script)
        self.native = native
        self.streaming = streaming
        self.i = 0
        self.calls: list[dict] = []

    @property
    def supports_native_tools(self) -> bool:
        return self.native

    @property
    def supports_streaming(self) -> bool:
        return self.streaming

    def _next(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        if self.i >= len(self.script):
            raise AssertionError("ScriptedClient ran out of scripted responses")
        resp = self.script[self.i]
        self.i += 1
        return resp

    def complete(self, messages, *, tools=None, model=None, temperature=None, on_retry=None, **o):
        return self._next(messages, tools)

    async def acomplete(self, messages, *, tools=None, model=None, temperature=None, on_retry=None, **o):
        return self._next(messages, tools)

    def stream(self, messages, *, tools=None, model=None, temperature=None, **o):
        yield from _chunks(self._next(messages, tools))

    async def astream(self, messages, *, tools=None, model=None, temperature=None, **o):
        for ch in _chunks(self._next(messages, tools)):
            yield ch


def _chunks(resp: ModelResponse):
    """Turn a ModelResponse into a plausible low-level chunk stream."""
    if resp.tool_calls:
        for j, tc in enumerate(resp.tool_calls):
            yield _ToolChunk(index=j, id=tc.id, name=tc.name,
                             args_fragment=json.dumps(tc.arguments))
        yield _StreamDone(finish_reason="tool_calls", usage=resp.usage)
    else:
        # Stream word-by-word but preserve exact content (no trailing space).
        words = (resp.text or "").split(" ")
        for k, w in enumerate(words):
            yield _TextChunk(delta=(w if k == 0 else " " + w))
        yield _StreamDone(finish_reason="stop", usage=resp.usage)


# ── sample tools ─────────────────────────────────────────────────────────────
@pytest.fixture
def add_tool():
    @tool
    def add(a: int, b: int = 0) -> str:
        """Add two integers.

        Args:
            a: first addend.
            b: second addend.
        """
        return str(a + b)

    return add
