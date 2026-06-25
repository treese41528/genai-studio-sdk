"""Unit tests for ``Agent.as_tool()`` — the minimal multi-agent primitive.

All zero-network via the ``ScriptedClient`` fixture: a sub-agent is wrapped as a
tool and exercised both directly and through a manager's loop.
"""

from __future__ import annotations

import asyncio

import pytest

from genai_studio.agents import Agent, Source, ToolResult, tool
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


def _agent(script, *, native=True, **kw):
    return Agent(client=ScriptedClient(script, native=native),
                 tracer=NullTracer(), **kw)


# ── spec / naming ────────────────────────────────────────────────────────────
def test_spec_shape_and_name_default():
    a = _agent([], name="researcher")
    t = a.as_tool()
    assert t.name == "researcher"
    assert t.spec.parameters["required"] == ["task"]
    assert t.spec.parameters["properties"]["task"]["type"] == "string"
    assert t.spec.parameters["additionalProperties"] is False
    assert t.is_async is False


def test_name_precedence_and_custom_input_field():
    assert _agent([]).as_tool().name == "agent"             # no name -> 'agent'
    assert _agent([], name="x").as_tool(name="y").name == "y"  # explicit wins
    t = _agent([]).as_tool(input_field="question")
    assert t.spec.parameters["required"] == ["question"]


# ── result packing ───────────────────────────────────────────────────────────
def test_returns_subagent_final_text():
    sub = _agent([says("the answer")], name="sub")
    res = sub.as_tool().run({"task": "anything"})
    assert isinstance(res, ToolResult)
    assert res.content == "the answer"
    assert res.error is None


def test_sources_propagate_up():
    @tool
    def cite() -> ToolResult:
        "Return a cited fact."
        return ToolResult(content="fact", sources=[Source(id="s1", title="Src", url="http://x")])

    sub = _agent([calls_tool("cite", {}), says("answer with cite")],
                 tools=[cite], name="sub")
    res = sub.as_tool().run({"task": "q"})
    assert res.content == "answer with cite"
    assert [s.id for s in res.sources] == ["s1"]


def test_structured_output_rides_in_data():
    # A finish/final_answer-style terminal call sets text; output stays None here,
    # but we assert data is the AgentResult.output (None unless output_schema set).
    sub = _agent([says("plain")], name="sub")
    res = sub.as_tool().run({"task": "q"})
    assert res.data is None  # no output_schema -> output is None


def test_non_final_stop_is_surfaced():
    # max_steps with force_final_answer off -> stopped='max_steps', empty text.
    sub = _agent([calls_tool("noop", {})], name="sub",
                 max_steps=1, force_final_answer=False)
    res = sub.as_tool().run({"task": "q"})
    assert "stopped early" in res.content and "max_steps" in res.content
    assert res.error is None  # not an error, just truncated


# ── runs through a manager loop (sync + async) ───────────────────────────────
def test_manager_delegates_via_run():
    sub = _agent([says("42")], name="calc")
    manager = _agent([calls_tool("calc", {"task": "what is 6*7"}), says("the total is 42")],
                     tools=[sub.as_tool()])
    result = manager.run("compute the total")
    assert result.text == "the total is 42"


def test_sync_wrapper_runs_under_arun_offthread():
    sub = _agent([says("async-ok")], name="calc")
    manager = _agent([calls_tool("calc", {"task": "x"}), says("done")],
                     tools=[sub.as_tool()])  # sync wrapper, parent uses arun
    result = asyncio.run(manager.arun("go"))
    assert result.text == "done"


def test_async_wrapper_is_async_and_runs():
    sub = _agent([says("the answer")], name="sub")
    t = sub.as_tool(use_async=True)
    assert t.is_async is True
    res = asyncio.run(t.arun({"task": "q"}))
    assert res.content == "the answer"


# ── shared cancellation forwards into the sub-run ────────────────────────────
def test_shared_cancel_stops_subagent():
    from genai_studio.agents import Cancel
    c = Cancel()
    c.cancel()  # pre-cancelled
    sub = _agent([says("never reached")], name="sub")
    res = sub.as_tool(cancel=c).run({"task": "q"})
    # cancelled before the first model call -> surfaced as a non-final stop
    assert "cancelled" in res.content


# ── review-driven fixes: positional call, missing-arg guard, shared budget ───
def test_positional_name_and_description():
    # name/description are positional-or-keyword (examples/docstrings call them so).
    t = _agent([], name="x").as_tool("verify_claims", "Fact-check the claims.")
    assert t.name == "verify_claims"
    assert t.spec.description == "Fact-check the claims."


def test_missing_or_empty_arg_is_surfaced_not_run():
    sub = _agent([says("should not run")], name="sub")
    t = sub.as_tool()
    assert "missing required argument" in (t.run({}).error or "")
    assert "missing required argument" in (t.run({"wrong": "x"}).error or "")
    assert "missing required argument" in (t.run({"task": ""}).error or "")
    assert sub.client.i == 0  # never burned a (rate-limited) sub-agent call


def test_shared_budget_does_not_abort_manager():
    from genai_studio.agents import Budget

    @tool
    def noop() -> str:
        "A no-op tool."
        return "ok"

    # max_tool_calls=1 would abort the manager if the child SHARED the instance
    # (child tick -> 1, then the manager's tick -> 2 > 1). With a per-run copy it
    # doesn't.
    B = Budget(max_tool_calls=1)
    sub = _agent([calls_tool("noop", {}), says("sub done")], tools=[noop], name="sub")
    manager = _agent([calls_tool("sub", {"task": "x"}), says("manager done")],
                     tools=[sub.as_tool(budget=B)])
    result = manager.run("go", budget=B)
    assert result.stopped == "final"      # the child's tool call did NOT abort the manager
    assert result.text == "manager done"
