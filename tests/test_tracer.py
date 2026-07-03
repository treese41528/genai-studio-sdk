"""Tracers: JsonlTracer event log, ConsoleTracer robustness, NullTracer."""

from __future__ import annotations

import json

from genai_studio.agents import Agent, ConsoleTracer, NullTracer, tool
from genai_studio.agents.trace import JsonlTracer
from conftest import ScriptedClient, calls_tool, says


@tool
def add(a: int, b: int) -> str:
    "Add.\n\nArgs:\n    a: x.\n    b: y."
    return str(a + b)


def test_jsonl_one_line_per_event(tmp_path):
    path = tmp_path / "trace.jsonl"
    c = ScriptedClient([calls_tool("add", {"a": 2, "b": 3}), says("five")])
    with JsonlTracer(path) as tr:
        Agent(client=c, tools=[add], tracer=tr).run("go")

    lines = path.read_text().splitlines()
    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]
    assert types[0] == "AgentStart"
    assert "LLMCall" in types and "LLMResponse" in types
    assert "ToolCallEvent" in types and "ToolResultEvent" in types
    assert types[-1] == "AgentEnd"
    # every line is valid JSON and carries no raw provider payload
    assert all("raw" not in e for e in events)


def test_llmcall_carries_actual_prompt(tmp_path):
    path = tmp_path / "t.jsonl"
    c = ScriptedClient([says("hi")])
    with JsonlTracer(path) as tr:
        Agent(client=c, tools=[], system="be terse", tracer=tr).run("hello world")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    llmcall = next(e for e in events if e["type"] == "LLMCall")
    blob = json.dumps(llmcall)
    assert "be terse" in blob and "hello world" in blob  # transparency


def test_console_tracer_does_not_crash():
    import io

    buf = io.StringIO()
    c = ScriptedClient([calls_tool("add", {"a": 1, "b": 1}), says("two")])
    Agent(client=c, tools=[add], tracer=ConsoleTracer(stream=buf, color=False)).run("go")
    out = buf.getvalue()
    assert "step" in out and "add" in out


def test_null_tracer_silent(capsys):
    c = ScriptedClient([says("quiet")])
    Agent(client=c, tools=[], tracer=NullTracer()).run("go")
    assert capsys.readouterr().out == ""
