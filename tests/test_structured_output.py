"""Structured output via output_schema (native return_result + JSON fallback)."""

from __future__ import annotations

import pytest

from genai_studio.agents import Agent, NullTracer
from conftest import ScriptedClient, calls_tool, says

pydantic = pytest.importorskip("pydantic")


class Findings(pydantic.BaseModel):
    summary: str
    n: int


def test_native_return_result_validates_and_retries():
    c = ScriptedClient([
        calls_tool("return_result", {"summary": "x"}),            # invalid: missing n
        calls_tool("return_result", {"summary": "x", "n": 3}),    # valid on retry
    ])
    res = Agent(client=c, tools=[], output_schema=Findings, tracer=NullTracer()).run("go")
    assert isinstance(res.output, Findings)
    assert res.output.n == 3 and res.stopped == "final"


def test_native_return_result_first_try():
    c = ScriptedClient([calls_tool("return_result", {"summary": "ok", "n": 7})])
    res = Agent(client=c, tools=[], output_schema=Findings, tracer=NullTracer()).run("go")
    assert res.output.summary == "ok" and res.output.n == 7


def test_second_failure_stops_error():
    c = ScriptedClient([
        calls_tool("return_result", {"summary": "x"}),       # invalid
        calls_tool("return_result", {"summary": "y"}),       # still invalid
    ])
    res = Agent(client=c, tools=[], output_schema=Findings, tracer=NullTracer()).run("go")
    assert res.output is None and res.stopped == "error"


def test_json_mode_fallback_non_native():
    # A non-native client never gets a return_result tool; it answers with JSON text.
    c = ScriptedClient([says('{"summary": "via json", "n": 9}')], native=False)
    res = Agent(client=c, tools=[], output_schema=Findings, tracer=NullTracer()).run("go")
    assert isinstance(res.output, Findings) and res.output.n == 9


def test_missing_pydantic_raises_at_construction(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pydantic":
            raise ImportError("no pydantic")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        Agent(client=ScriptedClient([]), output_schema=Findings, tracer=NullTracer())
