"""Streaming: arun parity, AgentEvents, and the tool-delta degrade path."""

from __future__ import annotations

import asyncio

from genai_studio.agents import (
    Agent,
    Final,
    NullTracer,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    Usage,
)
from genai_studio.agents.client import _StreamDone, _ToolChunk
from conftest import ScriptedClient, calls_tool, says


def test_arun_mirrors_run(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 2, "b": 3}), says("Sum is 5.")])
    res = asyncio.run(Agent(client=c, tools=[add_tool], tracer=NullTracer()).arun("add"))
    assert res.text == "Sum is 5." and res.stopped == "final"


def test_astream_event_sequence(add_tool):
    async def collect():
        c = ScriptedClient([calls_tool("add", {"a": 4, "b": 1}), says("Result is 5.")])
        return [ev async for ev in Agent(client=c, tools=[add_tool], tracer=NullTracer()).astream("go")]

    evs = asyncio.run(collect())
    kinds = [type(e).__name__ for e in evs]
    assert "ToolCallStarted" in kinds and "ToolCallFinished" in kinds
    assert any(isinstance(e, TextDelta) for e in evs)
    assert isinstance(evs[-1], Final) and evs[-1].result.stopped == "final"
    text = "".join(e.text for e in evs if isinstance(e, TextDelta))
    assert "Result is 5" in text


def test_sync_stream_final(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 1, "b": 1}), says("Two total.")])
    evs = list(Agent(client=c, tools=[add_tool], tracer=NullTracer()).stream("go"))
    assert isinstance(evs[-1], Final)
    assert evs[-1].result.text == "Two total."


def test_streaming_degrades_on_bad_tool_json(add_tool):
    """If streamed tool-call fragments don't assemble, fall back to complete()."""

    class Degrade(ScriptedClient):
        def stream(self, messages, *, tools=None, model=None, temperature=None, **o):
            resp = self._next(messages, tools)
            if resp.tool_calls:
                yield _ToolChunk(index=0, id="x", name="add", args_fragment='{"a": 4, "b":')  # broken
                yield _StreamDone(finish_reason="tool_calls", usage=Usage(0, 0, 1))
            else:
                yield _StreamDone(finish_reason="stop", usage=resp.usage)

    # script[0] streamed (broken) -> degrade calls complete() -> script[1]
    c = Degrade([calls_tool("add", {"a": 4, "b": 1}), says("after degrade")])
    evs = list(Agent(client=c, tools=[add_tool], tracer=NullTracer()).stream("go"))
    assert isinstance(evs[-1], Final)
    assert evs[-1].result.text == "after degrade"
