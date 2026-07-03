"""Unit tests for the multi-agent topologies: ``supervisor`` + ``pipeline``.

All zero-network via ``ScriptedClient``. For the supervisor happy path, the
manager and worker share ONE scripted client (matching the real rate-limiter
invariant), so a single script serves the interleaved manager/worker calls.
"""

from __future__ import annotations

import pytest

from genai_studio.agents import (
    Agent, DELEGATION_GUIDE, ROUTING_GUIDE, Source, ToolResult, pipeline, supervisor, tool,
)
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


class _Dummy:  # minimal client for build-only (no run) tests
    supports_native_tools = True


def _agent(client, **kw):
    return Agent(client=client, tracer=NullTracer(), **kw)


# ── supervisor ───────────────────────────────────────────────────────────────
def test_supervisor_delegates_and_synthesizes():
    # one shared client serves: manager delegates -> worker answers -> manager finalizes
    client = ScriptedClient([
        calls_tool("researcher", {"task": "find X"}),
        says("X is 42"),
        says("The answer is 42."),
    ])
    worker = _agent(client, name="researcher")
    mgr = supervisor(client, "Coordinate the researcher.", [worker], tracer=NullTracer())
    result = mgr.run("What is X?")
    assert result.text == "The answer is 42."


def test_supervisor_builds_unique_worker_tool_names():
    c = _Dummy()
    w1, w2 = _agent(c, name="analyst"), _agent(c, name="analyst")  # same name
    mgr = supervisor(c, "sys", [w1, w2])
    names = [t.name for t in mgr.tools]
    assert "analyst" in names and "analyst_2" in names   # auto-suffixed
    assert "final_answer" in names


def test_supervisor_appends_delegation_guide():
    c = _Dummy()
    w = _agent(c, name="w")
    assert DELEGATION_GUIDE in supervisor(c, "Base prompt.", [w]).system
    assert "Base prompt." in supervisor(c, "Base prompt.", [w]).system
    assert DELEGATION_GUIDE not in supervisor(c, "Base.", [w], delegation_guide=False).system


def test_supervisor_appends_routing_guide_by_default():
    # the orchestrator is GIVEN the measured decision knowledge (models/sampling/tools) to route well
    c = _Dummy()
    w = _agent(c, name="w")
    sysprompt = supervisor(c, "Base.", [w]).system
    assert ROUTING_GUIDE in sysprompt
    assert "greedy" in sysprompt.lower() and "verify_math" in sysprompt   # key knowledge present
    assert ROUTING_GUIDE not in supervisor(c, "Base.", [w], routing_guide=False).system


def test_routed_team_wires_specialists_sharing_one_client():
    from genai_studio.agents import routed_team
    c = _Dummy()
    mgr = routed_team(c, tracer=NullTracer())
    names = {t.name for t in mgr.tools}
    assert {"math_specialist", "reasoning_specialist", "research_specialist"} <= names
    assert "final_answer" in names
    assert ROUTING_GUIDE in mgr.system and mgr.client is c        # manager shares the client + has the guide


def test_routed_team_worker_models_and_greedy_reasoning():
    from genai_studio.agents.orchestrate import _math_worker, _reasoning_worker
    c = _Dummy()
    r = _reasoning_worker(c, "deepseek-r1:32b", NullTracer())
    assert r.model == "deepseek-r1:32b" and r.temperature == 0.0 and r.client is c   # greedy, shared client
    m = _math_worker(c, "qwen2.5:72b", NullTracer())
    assert m.model == "qwen2.5:72b" and any(t.name == "verify_math" for t in m.tools)


def test_routed_team_include_and_model_override():
    from genai_studio.agents import routed_team
    c = _Dummy()
    mgr = routed_team(c, include=("math",), models={"math": "llama4:latest"}, tracer=NullTracer())
    names = {t.name for t in mgr.tools}
    assert "math_specialist" in names and "reasoning_specialist" not in names


def test_routed_team_optin_critic_specialist():
    from genai_studio.agents import routed_team
    from genai_studio.agents.orchestrate import _critic_worker
    c = _Dummy()
    mgr = routed_team(c, include=("math", "critic"), tracer=NullTracer())
    assert "critic_specialist" in {t.name for t in mgr.tools}
    cr = _critic_worker(c, "gpt-oss:120b", NullTracer())
    assert cr.model == "gpt-oss:120b" and cr.name == "critic_specialist"


def test_supervisor_warns_on_mismatched_client():
    with pytest.warns(UserWarning, match="different ModelClient"):
        supervisor(_Dummy(), "s", [_agent(_Dummy(), name="x")])  # different clients


def test_supervisor_no_warning_when_client_shared(recwarn):
    c = _Dummy()
    supervisor(c, "s", [_agent(c, name="x")])
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_supervisor_empty_workers_raises():
    with pytest.raises(ValueError):
        supervisor(_Dummy(), "s", [])


def test_supervisor_worker_cannot_shadow_final_answer():
    c = _Dummy()
    w = _agent(c, name="final_answer")   # collides with the appended terminal tool
    mgr = supervisor(c, "sys", [w])      # must NOT crash at construction
    names = [t.name for t in mgr.tools]
    assert names.count("final_answer") == 1   # the real terminal tool, exactly once
    assert "final_answer_2" in names          # the worker, auto-suffixed away


def test_supervisor_rejects_non_agent_worker():
    with pytest.raises(TypeError):
        supervisor(_Dummy(), "sys", ["not an agent"])


# ── pipeline ─────────────────────────────────────────────────────────────────
def test_pipeline_threads_text_between_stages():
    c1, c2 = ScriptedClient([says("STAGE1_OUT")]), ScriptedClient([says("STAGE2_OUT")])
    s1, s2 = _agent(c1, name="s1"), _agent(c2, name="s2")
    result = pipeline([s1, s2])("input")
    assert result.text == "STAGE2_OUT"
    assert c2.calls[0]["messages"][-1].content == "STAGE1_OUT"  # s2 saw s1's output


def test_pipeline_gate_aborts_early():
    c = ScriptedClient([says("bad"), says("never runs")])
    s1, s2 = _agent(c, name="s1"), _agent(c, name="s2")
    result = pipeline([s1, s2], gate=lambda r: "bad" not in r.text)("in")
    assert result.text == "bad"
    assert c.i == 1  # second stage never ran


def test_pipeline_unions_sources_across_stages():
    @tool
    def cite1() -> ToolResult:
        "cite a source."
        return ToolResult(content="c1", sources=[Source(id="src1", url="http://1")])

    @tool
    def cite2() -> ToolResult:
        "cite a source."
        return ToolResult(content="c2", sources=[Source(id="src2", url="http://2")])

    c1 = ScriptedClient([calls_tool("cite1", {}), says("out1")])
    c2 = ScriptedClient([calls_tool("cite2", {}), says("out2")])
    s1 = _agent(c1, tools=[cite1], name="s1")
    s2 = _agent(c2, tools=[cite2], name="s2")
    res = pipeline([s1, s2])("in")
    assert {s.id for s in res.sources} == {"src1", "src2"}  # union of both stages


def test_pipeline_stops_when_stage_produces_no_output():
    @tool
    def noop() -> str:
        "a no-op."
        return "ok"

    # stage1 burns its one step on a tool call -> stops with empty text (no rescue)
    c1 = ScriptedClient([calls_tool("noop", {})])
    s1 = _agent(c1, tools=[noop], name="s1", max_steps=1, force_final_answer=False)
    c2 = ScriptedClient([says("should not run")])
    s2 = _agent(c2, name="s2")
    res = pipeline([s1, s2])("in")
    assert res.text == ""               # stage1 produced nothing
    assert res.stopped == "max_steps"
    assert c2.i == 0                    # stage2 never ran on empty input


def test_pipeline_per_stage_budget_is_independent():
    from genai_studio.agents import Budget

    @tool
    def noop() -> str:
        "a no-op."
        return "ok"

    # each stage makes ONE tool call; a SHARED max_tool_calls=1 would abort stage 2,
    # but per-stage budget copies let both finish cleanly.
    c1 = ScriptedClient([calls_tool("noop", {}), says("out1")])
    c2 = ScriptedClient([calls_tool("noop", {}), says("out2")])
    s1, s2 = _agent(c1, tools=[noop], name="s1"), _agent(c2, tools=[noop], name="s2")
    res = pipeline([s1, s2])("in", budget=Budget(max_tool_calls=1))
    assert res.stopped == "final" and res.text == "out2"


def test_pipeline_halts_on_truncated_stage_but_returns_its_text():
    @tool
    def noop() -> str:
        "a no-op."
        return "ok"

    # stage1 hits max_steps; force_final_answer rescues text -> stopped='max_steps'.
    c1 = ScriptedClient([calls_tool("noop", {}), says("rescued answer")])
    s1 = _agent(c1, tools=[noop], name="s1", max_steps=1)
    c2 = ScriptedClient([says("should not run")])
    s2 = _agent(c2, name="s2")
    res = pipeline([s1, s2])("in")
    assert res.stopped == "max_steps"
    assert res.text == "rescued answer"   # the rescued text IS returned
    assert c2.i == 0                      # but the pipeline halted; stage2 never ran


def test_pipeline_empty_raises():
    with pytest.raises(ValueError):
        pipeline([])
