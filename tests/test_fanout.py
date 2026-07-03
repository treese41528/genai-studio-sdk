"""Dynamic + parallel sub-agent fan-out — parallel_agents + the model-facing fan_out tool.

Uses a deterministic EchoClient (answer derived from the subtask, not a fixed script order) so a
parallel run is testable regardless of completion order."""

from __future__ import annotations

from genai_studio.agents import make_fanout_tool, parallel_agents
from genai_studio.agents.client import Message, ModelResponse, Usage


class EchoClient:
    """A ModelClient whose answer echoes the last user message — deterministic under concurrency."""

    supports_native_tools = False
    supports_streaming = False

    def complete(self, messages, *, tools=None, model=None, temperature=None, on_retry=None, **o):
        last = next((m.content for m in reversed(messages) if getattr(m, "role", None) == "user"), "")
        return ModelResponse(text=f"done: {last}", tool_calls=[], usage=Usage())

    async def acomplete(self, messages, *, tools=None, model=None, temperature=None, on_retry=None, **o):
        return self.complete(messages)


def test_parallel_agents_preserves_order():
    res = parallel_agents(EchoClient(), ["alpha", "beta", "gamma"], max_steps=2)
    assert [r.text for r in res] == ["done: alpha", "done: beta", "done: gamma"]


def test_parallel_agents_empty():
    assert parallel_agents(EchoClient(), []) == []


def test_fan_out_tool_runs_subtasks_in_parallel():
    fan = make_fanout_tool(EchoClient(), max_agents=5)
    out = fan.run({"subtasks": ["research X", "research Y"]})
    assert out.error is None and out.data["n"] == 2
    assert "research X" in out.content and "done: research X" in out.content
    assert "research Y" in out.content


def test_fan_out_caps_at_max_agents():
    fan = make_fanout_tool(EchoClient(), max_agents=2)
    out = fan.run({"subtasks": ["a", "b", "c", "d"]})     # 4 given, cap 2
    assert out.data["n"] == 2


def test_fan_out_empty_errors():
    assert make_fanout_tool(EchoClient()).run({"subtasks": []}).error


def test_fan_out_isolates_worker_failure():
    class HalfBroken(EchoClient):
        def complete(self, messages, **kw):
            last = next((m.content for m in reversed(messages) if getattr(m, "role", None) == "user"), "")
            if "boom" in last:
                raise RuntimeError("worker exploded")
            return ModelResponse(text=f"done: {last}", tool_calls=[], usage=Usage())

        async def acomplete(self, messages, **kw):
            return self.complete(messages)

    out = make_fanout_tool(HalfBroken()).run({"subtasks": ["ok task", "boom task"]})
    assert "done: ok task" in out.content and "ERROR" in out.content     # one failed, one succeeded
