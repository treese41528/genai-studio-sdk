"""Offline tests for the REPL streaming renderer + the stream turn-loop (no network)."""

from __future__ import annotations

import io
import types

from genai_studio.agents import Agent, ToolResult, tool
from genai_studio.agents.events import Final, ToolCallFinished, ToolCallStarted, TextDelta
from genai_studio.agents.repl.render import StreamRenderer
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


def _result(text, stopped="final", messages=None):
    return types.SimpleNamespace(text=text, stopped=stopped, error=None, messages=messages or [])


def _render(events):
    buf = io.StringIO()
    r = StreamRenderer(color=False, stream=buf)
    r.start()
    for ev in events:
        r.handle(ev)
    return r, buf.getvalue()


def test_no_double_final_when_streamed():
    _, out = _render([TextDelta("Hello ", 0), TextDelta("world", 0), Final(_result("Hello world"))])
    assert out.count("Hello world") == 1            # streamed once, not re-printed


def test_prints_final_when_not_streamed():
    _, out = _render([
        ToolCallStarted("c1", "read_file", {"path": "x"}, 0),
        ToolCallFinished("c1", "read_file", ToolResult(content="file data"), 0),
        Final(_result("the answer")),
    ])
    assert "→ read_file" in out and "← file data" in out and "the answer" in out


def test_tool_error_rendered():
    _, out = _render([
        ToolCallStarted("c1", "run_shell", {"command": "boom"}, 0),
        ToolCallFinished("c1", "run_shell", ToolResult(content="", error="exited with status 1"), 0),
        Final(_result("handled")),
    ])
    assert "✗" in out and "exited with status 1" in out


def test_cancelled_note():
    _, out = _render([Final(_result("", stopped="cancelled"))])
    assert "interrupted" in out


# ── full stream loop with a real tool via ScriptedClient ──────────────────────
@tool
def echo(x: str) -> str:
    """Echo text back.

    Args:
        x: text to echo.
    """
    return x


def test_stream_loop_end_to_end():
    client = ScriptedClient([calls_tool("echo", {"x": "hi"}), says("done")])
    agent = Agent(client=client, tools=[echo], tracer=NullTracer())
    r, out = _render(list(agent.stream("go")))
    res = r.result
    assert res is not None and res.text == "done"
    history = [m for m in res.messages if getattr(m, "role", None) != "system"]
    assert history and all(getattr(m, "role", None) != "system" for m in history)
    assert "→ echo" in out and "done" in out


def test_stream_recovers_text_emitted_tool_call():
    # Regression: some gateway models emit a tool call as JSON TEXT (not native tool_calls)
    # in the STREAMING path; the Agent must recover and run it (parity with complete()).
    script = [says('{"name": "echo", "arguments": {"x": "hi"}}'), says("done")]
    agent = Agent(client=ScriptedClient(script), tools=[echo], tracer=NullTracer())
    evs = list(agent.stream("go"))
    started = [e.name for e in evs if type(e).__name__ == "ToolCallStarted"]
    assert "echo" in started                       # the text tool-call was recovered + executed
    # and the renderer must NOT show the raw tool-call JSON
    _, out = _render(evs)
    assert '{"name": "echo"' not in out


# ── say-then-do: prose preamble of a text-emitted tool call is shown, JSON dropped ─

def test_preamble_shown_json_payload_dropped():
    _, out = _render([
        TextDelta("Let me check the file. ", 0),
        TextDelta('{"name": "read_file", "arguments": {"path": "x.py"}}', 0),
        ToolCallStarted("c1", "read_file", {"path": "x.py"}, 0),
        ToolCallFinished("c1", "read_file", ToolResult(content="data"), 0),
        Final(_result("done")),
    ])
    assert "Let me check the file." in out
    assert '"arguments"' not in out                 # the machine payload never shows


def test_nudged_segments_are_separated():
    from genai_studio.agents.events import StepFinished
    _, out = _render([
        TextDelta("Let's read the file.", 0),
        StepFinished(0, True, 0, None),             # intent-nudge step boundary
        TextDelta("The file defines a Flask app.", 1),
        Final(_result("The file defines a Flask app.")),
    ])
    assert "Let's read the file.\n\nThe file defines a Flask app." in out
