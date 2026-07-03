"""Unit tests for the Guard seam (before/after-tool hooks), zero-network."""

from __future__ import annotations

from genai_studio.agents import (
    Agent, BudgetGuard, ToolFilterGuard, ToolResult, deny, guard, modify,
    supervisor, tool,
)
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


@tool
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: first addend.
        b: second addend.
    """
    return str(a + b)


def _agent(script, **kw):
    return Agent(client=ScriptedClient(script), tools=[add], tracer=NullTracer(), **kw)


def _tool_msgs(res):
    return [m.content for m in res.messages if getattr(m, "role", None) == "tool"]


# ── before_tool ──────────────────────────────────────────────────────────────
def test_before_tool_deny_blocks_and_feeds_back_reason():
    g = guard(before=lambda c: deny("not allowed") if c.name == "add" else None)
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("done")], guards=[g]).run("go")
    assert res.text == "done"
    assert any("not allowed" in m for m in _tool_msgs(res))   # error fed back
    assert "5" not in _tool_msgs(res)                          # add never ran


def test_before_tool_modify_rewrites_arguments():
    g = guard(before=lambda c: modify({"a": 10, "b": 10}) if c.name == "add" else None)
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")], guards=[g]).run("go")
    assert "20" in _tool_msgs(res) and "5" not in _tool_msgs(res)


def test_first_deny_wins_across_multiple_guards():
    g1 = guard(before=lambda c: None)                  # allow
    g2 = guard(before=lambda c: deny("blocked by #2"))
    res = _agent([calls_tool("add", {"a": 1, "b": 1}), says("x")], guards=[g1, g2]).run("go")
    assert any("blocked by #2" in m for m in _tool_msgs(res))


def test_before_tool_guard_error_fails_closed():
    def boom(call):
        raise RuntimeError("kaboom")
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")],
                 guards=[guard(before=boom)]).run("go")
    assert any("guard error" in m for m in _tool_msgs(res))
    assert "5" not in _tool_msgs(res)                          # failed closed -> not run


# ── after_tool ───────────────────────────────────────────────────────────────
def test_after_tool_replaces_result():
    g = guard(after=lambda c, r: ToolResult(content="[redacted]") if r.content == "5" else None)
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")], guards=[g]).run("go")
    assert "[redacted]" in _tool_msgs(res)


def test_after_tool_error_keeps_original_result():
    def boom(call, result):
        raise RuntimeError("x")
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")],
                 guards=[guard(after=boom)]).run("go")
    assert "5" in _tool_msgs(res)                              # original result stands


def test_after_tool_non_toolresult_return_is_ignored():
    # a non-ToolResult return must NOT crash the run (the cardinal "never crash" rule)
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")],
                 guards=[guard(after=lambda c, r: "i am a string")]).run("go")
    assert res.text == "ok" and "5" in _tool_msgs(res)        # original result stands


def test_before_tool_non_decision_fails_closed():
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")],
                 guards=[guard(before=lambda c: True)]).run("go")  # True, not a Decision
    assert any("expected None or a" in m for m in _tool_msgs(res))
    assert "5" not in _tool_msgs(res)                          # blocked


def test_before_tool_unknown_action_fails_closed():
    from genai_studio.agents import Decision
    res = _agent([calls_tool("add", {"a": 2, "b": 3}), says("ok")],
                 guards=[guard(before=lambda c: Decision("Allow"))]).run("go")  # typo'd action
    assert any("invalid decision" in m for m in _tool_msgs(res))
    assert "5" not in _tool_msgs(res)                          # fail closed on a typo


# ── shipped guards ───────────────────────────────────────────────────────────
def test_tool_filter_block_and_allow():
    blocked = _agent([calls_tool("add", {"a": 1, "b": 1}), says("x")],
                     guards=[ToolFilterGuard(block={"add"})]).run("go")
    assert any("blocked by policy" in m for m in _tool_msgs(blocked))

    not_allowed = _agent([calls_tool("add", {"a": 1, "b": 1}), says("x")],
                         guards=[ToolFilterGuard(allow={"other"})]).run("go")
    assert any("not in the allowed set" in m for m in _tool_msgs(not_allowed))


def test_budget_guard_caps_tool_calls_within_one_agent():
    budget = BudgetGuard(max_tool_calls=1)
    res = _agent([calls_tool("add", {"a": 1, "b": 1}),
                  calls_tool("add", {"a": 2, "b": 2}), says("done")],
                 guards=[budget], max_steps=5).run("go")
    msgs = _tool_msgs(res)
    assert "2" in msgs                                         # 1st call ran (1+1)
    assert any("budget" in m.lower() for m in msgs)           # 2nd denied
    assert budget.tool_calls == 1                             # counts ALLOWED calls only


def test_budget_guard_is_tree_wide_across_a_team():
    budget = BudgetGuard(max_tool_calls=1)
    # one shared client: manager delegates (tool call #1) -> worker calls add (#2 -> denied)
    client = ScriptedClient([
        calls_tool("worker", {"task": "add 2 and 3"}),   # manager: delegate (counts as 1)
        calls_tool("add", {"a": 2, "b": 3}),             # worker: add (2 > cap -> denied)
        says("could not add"),                            # worker gives up
        says("the worker could not complete it"),         # manager finalizes
    ])
    worker = Agent(client=client, name="worker", tools=[add], guards=[budget], tracer=NullTracer())
    mgr = supervisor(client, "Delegate to the worker.", [worker],
                     guards=[budget], tracer=NullTracer())
    res = mgr.run("add 2 and 3")
    assert res.stopped == "final"
    assert budget.tool_calls == 1          # delegation allowed; the worker's add was denied (not counted)
