"""Tests for ScopedTracer — nested, attributable multi-agent tracing."""

from __future__ import annotations

import io
import json
import time

from genai_studio.agents import Agent, ConsoleTracer, JsonlTracer, ScopedTracer, tool
from genai_studio.agents.trace import LLMCall

from conftest import ScriptedClient, calls_tool, says


class Recorder:
    def __init__(self):
        self.events = []

    def on_event(self, e):
        self.events.append(e)


@tool
def noop() -> str:
    "A no-op tool."
    return "ok"


# ── stamping ─────────────────────────────────────────────────────────────────
def test_stamps_agent_and_depth():
    rec = Recorder()
    ScopedTracer(rec, "worker", depth=2).on_event(LLMCall(step=0))
    assert rec.events[0].agent == "worker" and rec.events[0].depth == 2


def test_stamp_is_idempotent_first_scope_wins():
    rec = Recorder()
    e = LLMCall(step=0, agent="inner", depth=1)
    ScopedTracer(rec, "outer", depth=0).on_event(e)
    assert e.agent == "inner" and e.depth == 1


def test_never_crashes_on_broken_inner():
    class Boom:
        def on_event(self, e):
            raise RuntimeError("x")
    ScopedTracer(Boom(), "a").on_event(LLMCall())   # must not raise


# ── consumer tracers respect the scope ───────────────────────────────────────
def test_console_tags_and_indents_by_scope():
    buf = io.StringIO()
    ScopedTracer(ConsoleTracer(stream=buf, color=False), "researcher", depth=1).on_event(
        LLMCall(step=0))
    out = buf.getvalue()
    assert "[researcher]" in out and out.startswith("  ")    # depth-1 indent + tag


def test_console_unscoped_output_is_unchanged():
    buf = io.StringIO()
    ConsoleTracer(stream=buf, color=False).on_event(LLMCall(step=0))
    out = buf.getvalue()
    assert "[" not in out and not out.startswith(" ")        # no tag, no indent


def test_console_scope_is_not_shared_mutable_state_across_threads():
    """One shared ConsoleTracer fed by two agents on different threads must keep
    every line attributed to its OWN agent — scope is a per-event local, not
    instance state (regresses the M1 mis-tag where 'A' lines printed as '[B]')."""
    import threading

    class SlowStream:
        """A stream whose write() yields the GIL mid-line to force interleaving."""
        def __init__(self):
            self.lines = []
        def write(self, s):
            if s.strip():
                time.sleep(0)            # cooperative preemption point
                self.lines.append(s)
        def flush(self):
            pass

    buf = SlowStream()
    shared = ConsoleTracer(stream=buf, color=False)
    a = ScopedTracer(shared, "alpha", depth=0)
    b = ScopedTracer(shared, "bravo", depth=1)

    def hammer(tr, n):
        for i in range(n):
            tr.on_event(LLMCall(step=i))

    t1 = threading.Thread(target=hammer, args=(a, 200))
    t2 = threading.Thread(target=hammer, args=(b, 200))
    t1.start(); t2.start(); t1.join(); t2.join()

    # every emitted line must carry exactly its own agent's tag — no cross-tagging
    for line in buf.lines:
        assert ("[alpha]" in line) ^ ("[bravo]" in line)
        if "[bravo]" in line:
            assert line.startswith("  ")     # depth-1 indent stayed with bravo


def test_jsonl_rows_carry_agent_and_depth(tmp_path):
    p = tmp_path / "t.jsonl"
    jt = JsonlTracer(str(p))
    ScopedTracer(jt, "w", depth=1).on_event(LLMCall(step=0))
    jt.close()
    row = json.loads(p.read_text().splitlines()[0])
    assert row["agent"] == "w" and row["depth"] == 1 and row["type"] == "LLMCall"


# ── end-to-end multi-agent attribution ───────────────────────────────────────
def test_multi_agent_trace_is_attributable():
    inner = Recorder()
    client = ScriptedClient([
        calls_tool("researcher", {"task": "x"}),   # manager delegates
        calls_tool("noop", {}),                     # worker acts
        says("worker done"),                         # worker final
        says("manager done"),                        # manager final
    ])
    worker = Agent(client=client, name="researcher", tools=[noop],
                   tracer=ScopedTracer(inner, "researcher", depth=1))
    manager = Agent(client=client, name="manager", tools=[worker.as_tool()],
                    tracer=ScopedTracer(inner, "manager", depth=0))
    manager.run("go")
    assert {e.agent for e in inner.events} == {"manager", "researcher"}
    assert {e.depth for e in inner.events} == {0, 1}
