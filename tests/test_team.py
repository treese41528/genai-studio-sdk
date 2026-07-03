"""Tests for Team — correct-by-construction multi-agent wiring."""

from __future__ import annotations

import warnings

import pytest

from genai_studio.agents import (
    Agent, BudgetGuard, ConsoleTracer, ScopedTracer, Team, tool,
)
from genai_studio.agents.client import RateLimiter, _NullLimiter
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


class _CappedClient(ScriptedClient):
    """A ScriptedClient that also exposes a limiter, like the real gateway client."""
    def __init__(self, script, rl):
        super().__init__(script)
        self._rl = rl


@tool
def noop() -> str:
    "A no-op tool."
    return "ok"


def _client(script):
    return ScriptedClient(script)


# ── the four guarantees on team.agent() ──────────────────────────────────────
def test_agent_shares_client_model_and_scoped_tracer():
    c = _client([says("hi")])
    inner = ConsoleTracer(color=False)
    team = Team(c, model="qwen2.5:72b", tracer=inner)
    a = team.agent("researcher", depth=2)
    assert a.client is c
    assert a.model == "qwen2.5:72b"
    assert isinstance(a.tracer, ScopedTracer)
    assert a.tracer.inner is inner and a.tracer.agent == "researcher" and a.tracer.depth == 2


def test_agent_override_model_wins():
    team = Team(_client([says("x")]), model="qwen2.5:72b")
    assert team.agent("w", model="llama3.3:70b").model == "llama3.3:70b"


def test_team_guards_propagate_and_compose_with_per_agent_guards():
    g_team, g_local = BudgetGuard(max_tool_calls=5), BudgetGuard(max_tool_calls=2)
    team = Team(_client([says("x")]), guards=[g_team])
    a = team.agent("w", guards=[g_local])
    assert list(a.guards) == [g_team, g_local]            # team first, then local


def test_system_prefix_is_prepended():
    team = Team(_client([says("x")]), system_prefix="HOUSE RULES")
    assert team.agent("w", system="Do the thing.").system == "HOUSE RULES\n\nDo the thing."
    assert team.agent("w2").system == "HOUSE RULES"        # prefix alone when no system
    assert Team(_client([says("x")])).agent("w3").system is None   # neither → None


def test_no_tracer_is_silent_nulltracer():
    team = Team(_client([says("x")]))
    assert isinstance(team.agent("w").tracer.inner, NullTracer)


def test_agent_requires_a_name():
    with pytest.raises(ValueError):
        Team(_client([says("x")])).agent("")


# ── enforce-by-construction: shared client ───────────────────────────────────
def test_supervisor_raises_on_foreign_client():
    team = Team(_client([says("x")]))
    stranger = Agent(client=_client([says("y")]), name="stranger")
    with pytest.raises(ValueError, match="different"):
        team.supervisor("coordinate", [team.agent("w"), stranger])


def test_pipeline_raises_on_foreign_client():
    team = Team(_client([says("x")]))
    stranger = Agent(client=_client([says("y")]), name="stranger")
    with pytest.raises(ValueError, match="different"):
        team.pipeline([stranger])


def test_compositions_reject_empty():
    team = Team(_client([says("x")]))
    with pytest.raises(ValueError):
        team.supervisor("c", [])
    with pytest.raises(ValueError):
        team.pipeline([])


# ── supervisor re-scopes workers one level deeper ────────────────────────────
def test_supervisor_rescopes_workers_below_manager():
    c = _client([says("x")])
    team = Team(c, tracer=ConsoleTracer(color=False))
    worker = team.agent("researcher", depth=0)            # built at depth 0
    mgr = team.supervisor("coordinate", [worker], depth=0)
    assert isinstance(mgr.tracer, ScopedTracer) and mgr.tracer.depth == 0
    # the manager's worker tool wraps a depth-1 copy of the worker (original untouched)
    assert worker.tracer.depth == 0


# ── end-to-end: a team supervisor actually delegates, traces are attributable ─
def test_team_supervisor_runs_and_trace_is_attributable():
    events = []

    class Rec:
        def on_event(self, e):
            events.append(e)

    team = Team(
        _client([
            calls_tool("researcher", {"task": "look it up"}),  # manager delegates
            calls_tool("noop", {}),                             # worker acts
            says("worker done"),                                 # worker final
            says("manager done"),                                # manager final
        ]),
        tracer=Rec(),
    )
    worker = team.agent("researcher", tools=[noop])
    mgr = team.supervisor("Use the researcher.", [worker])
    res = mgr.run("go")
    assert res.text == "manager done"
    assert {e.agent for e in events} == {"supervisor", "researcher"}
    assert {e.depth for e in events} == {0, 1}              # manager 0, worker re-scoped to 1


def test_team_pipeline_sequences_stages_and_defaults_budget():
    c = _client([says("stage-one out"), says("stage-two out")])
    team = Team(c, tracer=NullTracer())
    s1 = team.agent("first")
    s2 = team.agent("second")
    run = team.pipeline([s1, s2])
    assert run("start").text == "stage-two out"
    assert run.stages[0].name == "first" and run.stages[1].name == "second"


# ── honesty: warn when the shared client has no rate cap ─────────────────────
def test_uncapped_client_warns_unpaced():
    c = _CappedClient([says("x")], _NullLimiter())          # no pacing
    with pytest.warns(UserWarning, match="UNPACED"):
        Team(c)


def test_capped_client_does_not_warn():
    c = _CappedClient([says("x")], RateLimiter(20))         # real cap
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # any warning -> failure
        Team(c)


def test_client_without_limiter_attr_is_not_warned():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # ScriptedClient has no _rl
        Team(_client([says("x")]))


# ── name-aware re-scope: attribute even a non-team-built same-client agent ────
def test_rescope_attributes_a_plain_same_client_stage():
    events = []

    class Rec:
        def on_event(self, e):
            events.append(e)

    c = _client([says("done")])
    team = Team(c, tracer=Rec())
    plain = Agent(client=c, name="solo", tracer=NullTracer())   # built OUTSIDE team.agent()
    team.pipeline([plain])("go")
    assert any(getattr(e, "agent", None) == "solo" for e in events)   # still attributed


# ── idempotent system prefix ─────────────────────────────────────────────────
def test_system_prefix_is_idempotent():
    team = Team(_client([says("x")]), system_prefix="HOUSE RULES")
    once = team.agent("a", system="do x").system
    assert once == "HOUSE RULES\n\ndo x"
    assert team._system(once) == once                        # no double-stamp


# ── documented limitation: depth is single-level, names always correct ───────
def test_nested_supervisor_keeps_names_but_flattens_depth():
    """Pins the documented contract: names never collide at any nesting (the real
    anti-collision guarantee), but depth indentation does not recurse — a
    grandchild shares its parent's depth."""
    events = []

    class Rec:
        def on_event(self, e):
            events.append((getattr(e, "agent", None), getattr(e, "depth", None)))

    c = _client([
        calls_tool("sub", {"task": "x"}),      # parent -> sub-supervisor
        calls_tool("leaf", {"task": "y"}),     # sub -> leaf
        calls_tool("noop", {}),                 # leaf acts
        says("leaf done"), says("sub done"), says("parent done"),
    ])
    team = Team(c, tracer=Rec())
    leaf = team.agent("leaf", tools=[noop])
    sub = team.supervisor("use leaf", [leaf], name="sub")
    parent = team.supervisor("use sub", [sub], name="parent")
    parent.run("go")
    depth_of = {a: d for a, d in events}
    assert set(depth_of) == {"parent", "sub", "leaf"}        # no NAME collision
    assert depth_of["parent"] == 0 and depth_of["sub"] == 1
    assert depth_of["leaf"] == depth_of["sub"]               # documented flattening


# ── reserved kwargs raise a Team-scoped error, not a deep TypeError ──────────
def test_reserved_kwargs_raise_team_scoped_error():
    team = Team(_client([says("x")]))
    with pytest.raises(TypeError, match="controls"):
        team.agent("a", tracer=NullTracer())
    with pytest.raises(TypeError, match="controls"):
        team.supervisor("s", [team.agent("w")], tracer=NullTracer())
