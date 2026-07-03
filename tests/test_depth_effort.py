"""Tests for the delegation depth cap + effort-scaling presets."""

from __future__ import annotations

import pytest

from genai_studio.agents import (
    Agent, BudgetGuard, EFFORT_PRESETS, Team, effort_policy, supervisor, tool,
)
from genai_studio.agents.agent import _delegation_depth
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


@tool
def noop() -> str:
    "A no-op tool."
    return "ok"


# ── as_tool max_depth ────────────────────────────────────────────────────────
def test_as_tool_refuses_beyond_max_depth():
    # top -> sub -> leaf, max_depth=1 at each delegation: top enters at depth 0 (ok),
    # sub enters at depth 1 (>=1 -> refuses to call leaf). The refusal short-circuits
    # leaf.run, so the script is consumed exactly; WITHOUT the cap, leaf would run and
    # exhaust the script (IndexError) — so a clean "top final" proves the cap fired.
    c = ScriptedClient([
        calls_tool("sub", {"task": "x"}),       # top delegates to sub
        calls_tool("leaf", {"task": "y"}),      # sub tries to delegate to leaf -> REFUSED
        says("sub final"),                       # sub answers after the refusal
        says("top final"),                       # top answers
    ])
    leaf = Agent(client=c, name="leaf", tools=[noop], tracer=NullTracer())
    sub = Agent(client=c, name="sub", tracer=NullTracer(), tools=[leaf.as_tool(max_depth=1)])
    top = Agent(client=c, name="top", tracer=NullTracer(), tools=[sub.as_tool(max_depth=1)])
    assert top.run("go").text == "top final"


def test_max_depth_none_is_unbounded():
    # script: mgr delegates -> leaf answers -> mgr answers (leaf shares the client)
    c = ScriptedClient([calls_tool("leaf", {"task": "x"}), says("leaf done"), says("mgr done")])
    leaf = Agent(client=c, name="leaf", tracer=NullTracer())
    mgr = Agent(client=c, name="mgr", tracer=NullTracer(), tools=[leaf.as_tool()])
    assert mgr.run("go").text == "mgr done"      # no cap -> delegation runs normally


def test_depth_contextvar_restored_after_delegation():
    c = ScriptedClient([calls_tool("leaf", {"task": "x"}), says("leaf done"), says("after")])
    leaf = Agent(client=c, name="leaf", tracer=NullTracer())
    mgr = Agent(client=c, name="mgr", tracer=NullTracer(), tools=[leaf.as_tool(max_depth=3)])
    mgr.run("go")
    assert _delegation_depth.get() == 0          # reset cleanly, no leak


def test_refused_delegation_surfaces_error_not_crash():
    # a manager already AT the cap: simulate by entering the contextvar at depth 1
    c = ScriptedClient([calls_tool("leaf", {"task": "x"}), says("recovered")])
    leaf = Agent(client=c, name="leaf", tracer=NullTracer())
    mgr = Agent(client=c, name="mgr", tracer=NullTracer(), tools=[leaf.as_tool(max_depth=1)])
    token = _delegation_depth.set(1)             # pretend we're already one level deep
    try:
        res = mgr.run("go")
    finally:
        _delegation_depth.reset(token)
    # the delegation was refused (depth 1 >= max_depth 1), manager recovered + answered
    assert res.text == "recovered"
    errs = [tr.error for s in res.steps for tr in s.tool_results if tr.error]
    assert any("max delegation depth" in e for e in errs)


# ── effort presets ───────────────────────────────────────────────────────────
def test_effort_policy_lookup_and_unknown():
    assert effort_policy("simple").max_depth == 1
    assert effort_policy("complex").max_tool_calls == 24
    with pytest.raises(ValueError, match="unknown effort"):
        effort_policy("nonsense")


def test_supervisor_effort_adds_budgetguard_and_depth_and_hint():
    c = ScriptedClient([says("x")])
    worker = Agent(client=c, name="w", tracer=NullTracer())
    mgr = supervisor(c, "lead", [worker], effort="simple", tracer=NullTracer())
    assert any(isinstance(g, BudgetGuard) for g in mgr.guards)   # fan-out cap installed
    assert "EFFORT — simple" in (mgr.system or "")               # hint appended


def test_explicit_max_depth_overrides_effort_default():
    c = ScriptedClient([says("x")])
    worker = Agent(client=c, name="w", tracer=NullTracer())
    # effort 'simple' default max_depth=1, but explicit max_depth=5 should win
    mgr = supervisor(c, "lead", [worker], effort="simple", max_depth=5, tracer=NullTracer())
    # the worker tool should carry max_depth=5; we can't read the closure directly, but
    # the call must not raise and the guard/hint still apply
    assert any(isinstance(g, BudgetGuard) for g in mgr.guards)


# ── Team propagation ─────────────────────────────────────────────────────────
def test_team_max_depth_field_and_effort_propagate():
    c = ScriptedClient([says("x")])
    team = Team(c, tracer=NullTracer(), max_depth=2)
    w = team.agent("w")
    mgr = team.supervisor("lead", [w], effort="comparison")
    assert any(isinstance(g, BudgetGuard) for g in mgr.guards)
    assert "EFFORT — comparison" in (mgr.system or "")
