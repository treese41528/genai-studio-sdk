"""The Agent.sampling passthrough forwards extra sampling opts (e.g. top_p) to every
client call — used to run reasoning models (DeepSeek-R1/QwQ/Qwen3) at their documented
recipe (temperature=0.6, top_p=0.95). Default sampling={} must be a pure no-op."""

from __future__ import annotations

from genai_studio.agents import Agent
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, says


class CapturingScripted(ScriptedClient):
    """ScriptedClient that also records temperature + extra opts seen per call."""

    def __init__(self, script, **kw):
        super().__init__(script, **kw)
        self.kw: list[dict] = []

    def complete(self, messages, *, tools=None, model=None, temperature=None, on_retry=None, **o):
        self.kw.append({"temperature": temperature, **o})
        return super().complete(messages, tools=tools)

    def stream(self, messages, *, tools=None, model=None, temperature=None, **o):
        self.kw.append({"temperature": temperature, **o})
        yield from super().stream(messages, tools=tools)


def test_sampling_forwarded_nonstreaming():
    c = CapturingScripted([says("done")], streaming=False)
    Agent(client=c, tools=[], temperature=0.6, sampling={"top_p": 0.95},
          tracer=NullTracer()).run("hi")
    assert c.kw[0]["temperature"] == 0.6 and c.kw[0]["top_p"] == 0.95


def test_sampling_forwarded_streaming():
    c = CapturingScripted([says("done")], streaming=True)
    list(Agent(client=c, tools=[], temperature=0.6, sampling={"top_p": 0.95},
               tracer=NullTracer()).stream("hi"))
    assert c.kw[0]["temperature"] == 0.6 and c.kw[0]["top_p"] == 0.95


def test_sampling_default_is_noop():
    c = CapturingScripted([says("done")], streaming=False)
    Agent(client=c, tools=[], tracer=NullTracer()).run("hi")
    assert "top_p" not in c.kw[0] and c.kw[0]["temperature"] is None
