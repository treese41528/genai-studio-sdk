"""The agent loop: tool dispatch, error feed-back, budgets, guards."""

from __future__ import annotations

from genai_studio.agents import Agent, Budget, Cancel, NullTracer, tool
from conftest import ScriptedClient, calls_tool, says


def _agent(client, tools, **kw):
    return Agent(client=client, tools=tools, tracer=NullTracer(), **kw)


def test_calls_tool_then_finalizes(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 2, "b": 3}), says("The answer is 5.")])
    res = _agent(c, [add_tool]).run("add 2 and 3")
    assert res.text == "The answer is 5."
    assert res.stopped == "final"
    # the second LLM call saw the tool result fed back
    fed_back = any("5" in str(m.to_openai()) for m in c.calls[1]["messages"])
    assert fed_back


def test_tool_error_captured_and_recovered():
    @tool
    def boom(x: int) -> str:
        "Boom.\n\nArgs:\n    x: n."
        raise ValueError("nope")

    c = ScriptedClient([calls_tool("boom", {"x": 1}), says("Recovered.")])
    res = _agent(c, [boom]).run("go")
    assert res.stopped == "final"
    assert any("nope" in str(m.to_openai()) for m in c.calls[1]["messages"])


def test_unknown_tool_lists_available(add_tool):
    c = ScriptedClient([calls_tool("missing", {}), says("ok")])
    res = _agent(c, [add_tool]).run("go")
    # the error message naming valid tools was fed back
    assert any("add" in str(m.to_openai()) and "Unknown tool" in str(m.to_openai())
               for m in c.calls[1]["messages"])
    assert res.stopped == "final"


def test_budget_stops_on_tool_calls(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 1}) for _ in range(5)])
    res = _agent(c, [add_tool]).run("loop", budget=Budget(max_tool_calls=2))
    assert res.stopped == "budget"


def test_budget_stops_on_steps(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 1}) for _ in range(5)])
    res = _agent(c, [add_tool]).run("loop", budget=Budget(max_steps=2))
    assert res.stopped == "budget"


def test_budget_stops_on_tokens(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 1}, id=str(i)) for i in range(5)])
    # each turn reports total_tokens=3; cap at 5 -> trips on the 2nd turn
    res = _agent(c, [add_tool]).run("loop", budget=Budget(max_tokens=5))
    assert res.stopped == "budget"


def test_max_steps_stop(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 1}) for _ in range(5)])
    res = _agent(c, [add_tool], max_steps=2).run("loop")
    assert res.stopped == "max_steps"


def test_cancel_token(add_tool):
    cancel = Cancel()
    cancel.cancel()  # pre-tripped
    c = ScriptedClient([says("never reached")])
    res = _agent(c, [add_tool]).run("go", cancel=cancel)
    assert res.stopped == "cancelled"


def test_on_tool_error_abort():
    @tool
    def boom(x: int) -> str:
        "Boom.\n\nArgs:\n    x: n."
        raise RuntimeError("kaboom")

    c = ScriptedClient([calls_tool("boom", {"x": 1}), says("unreached")])
    res = _agent(c, [boom], on_tool_error="abort").run("go")
    assert res.stopped == "error"


def test_force_final_answer_on_max_steps(add_tool):
    # Model loops on tool calls; the tool-less forced final call yields text.
    c = ScriptedClient([calls_tool("add", {"a": 1})] * 3 + [says("Final: it is 1.")])
    res = _agent(c, [add_tool], max_steps=3).run("loop")
    assert res.stopped == "max_steps"
    assert res.text == "Final: it is 1."  # rescued from the forced tool-less call
    # the forced call was made with tools=None
    assert c.calls[-1]["tools"] is None


def test_no_force_final_answer_leaves_empty(add_tool):
    c = ScriptedClient([calls_tool("add", {"a": 1})] * 3)
    res = _agent(c, [add_tool], max_steps=3, force_final_answer=False).run("loop")
    assert res.stopped == "max_steps" and res.text == ""


def test_no_tools_direct_answer():
    c = ScriptedClient([says("Hello there.")])
    res = _agent(c, []).run("hi")
    assert res.text == "Hello there." and res.stopped == "final"
    assert len(res.steps) == 1


# ── intent nudge: narrated-but-not-taken actions get a "do it" nudge ──────────

NARRATION = "Let's read the file to see what it contains."


def test_intent_nudge_turns_narration_into_action(add_tool):
    c = ScriptedClient([says(NARRATION), calls_tool("add", {"a": 2, "b": 3}), says("It is 5.")])
    res = _agent(c, [add_tool], intent_nudges=2).run("add the numbers in the file")
    assert res.stopped == "final" and res.text == "It is 5."
    assert len(c.calls) == 3                                  # narrate -> nudged -> tool -> answer
    # the nudge went back as a user message and the narration stayed in the transcript
    assert any(m.role == "user" and "did not call any tool" in (m.content or "")
               for m in res.messages)
    assert any(m.role == "assistant" and m.content == NARRATION for m in res.messages)


def test_intent_nudge_default_off_is_unchanged(add_tool):
    c = ScriptedClient([says(NARRATION)])
    res = _agent(c, [add_tool]).run("go")
    assert res.stopped == "final" and res.text == NARRATION   # old behavior, byte-identical
    assert len(c.calls) == 1


def test_intent_nudge_is_bounded(add_tool):
    # keeps narrating: one nudge allowed, then the second narration is final
    c = ScriptedClient([says(NARRATION), says(NARRATION)])
    res = _agent(c, [add_tool], intent_nudges=1).run("go")
    assert res.stopped == "final" and res.text == NARRATION
    assert len(c.calls) == 2


def test_intent_nudge_ignores_real_answers(add_tool):
    c = ScriptedClient([says("The file contains a Flask app. Let me know if you need more.")])
    res = _agent(c, [add_tool], intent_nudges=2).run("go")
    assert len(c.calls) == 1 and res.stopped == "final"


def test_intent_nudge_needs_tools():
    c = ScriptedClient([says(NARRATION)])
    res = _agent(c, [], intent_nudges=2).run("go")            # nothing to call -> no nudge
    assert len(c.calls) == 1 and res.text == NARRATION


def test_narration_detector():
    from genai_studio.agents.agent import _narrates_action
    yes = ["Let's read the file.",
           "I'll start by running the tests:",
           "Now, I will search the codebase for the function.",
           "First, I am going to open main.py",
           "We will now examine the data.",
           "Sure! Let me grep for the handler…"]
    no = ["Let me know if you need anything else.",
          "The answer is 42.",
          "Would you like me to read the file?",
          "If you want, I can check the logs.",
          "I read the file. It contains a Flask app.",
          "You can read the file with open().",
          "I checked everything. Let me know your thoughts.",
          ""]
    assert all(_narrates_action(t) for t in yes)
    assert not any(_narrates_action(t) for t in no)
