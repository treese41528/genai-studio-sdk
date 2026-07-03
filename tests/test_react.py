"""ReActClient — synthesizing tool-calling via a JSON-action protocol."""

from __future__ import annotations

from genai_studio.agents import Agent, Message, NullTracer, ReActClient, tool
from conftest import ScriptedClient, says


@tool
def add(a: int, b: int) -> str:
    "Add.\n\nArgs:\n    a: x.\n    b: y."
    return str(a + b)


def test_parses_clean_json_action():
    inner = ScriptedClient([says('{"action":"add","action_input":{"a":1,"b":2}}')], native=False)
    resp = ReActClient(inner).complete([Message.user("add")], tools=[add.spec])
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].arguments == {"a": 1, "b": 2}


def test_parses_fenced_and_prefixed():
    text = 'Thought: I will add.\n```json\n{"action":"add","action_input":{"a":4,"b":5}}\n```'
    inner = ScriptedClient([says(text)], native=False)
    resp = ReActClient(inner).complete([Message.user("add")], tools=[add.spec])
    assert resp.tool_calls[0].arguments == {"a": 4, "b": 5}


def test_final_action_is_answer():
    inner = ScriptedClient([says('{"action":"final","action_input":"the answer is 42"}')], native=False)
    resp = ReActClient(inner).complete([Message.user("q")], tools=[add.spec])
    assert resp.tool_calls == []
    assert resp.text == "the answer is 42"


def test_passthrough_when_no_tools():
    inner = ScriptedClient([says("plain answer")], native=False)
    resp = ReActClient(inner).complete([Message.user("hi")], tools=None)
    assert resp.text == "plain answer"


def test_unparseable_repairs_then_falls_back():
    inner = ScriptedClient([says("no json here"), says("still no json")], native=False)
    resp = ReActClient(inner, max_json_repair=1).complete([Message.user("q")], tools=[add.spec])
    # after one repair attempt fails, prose is handed back as the final answer
    assert resp.tool_calls == []
    assert resp.text == "still no json"


def test_end_to_end_react_agent():
    inner = ScriptedClient([
        says('{"action":"add","action_input":{"a":2,"b":3}}'),
        says('{"action":"final","action_input":"sum is 5"}'),
    ], native=False)
    res = Agent(client=ReActClient(inner), tools=[add], tracer=NullTracer()).run("add 2 and 3")
    assert res.text == "sum is 5"
    assert res.stopped == "final"
