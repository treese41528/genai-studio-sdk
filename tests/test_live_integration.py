"""
Live integration tests against the Purdue GenAI Studio gateway.

Gated on ``GENAI_STUDIO_API_KEY``. The headline test answers design **open
question #1**: does OpenWebUI -> Ollama/LiteLLM forward ``tools`` and return
``tool_calls``? The answer sets the framework default (native vs ReActClient).

Run:  pytest tests/test_live_integration.py -q -s
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("GENAI_STUDIO_API_KEY"),
    reason="live: set GENAI_STUDIO_API_KEY to run",
)

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _pick_model(studio) -> str:
    override = os.getenv("GENAI_STUDIO_MODEL")
    if override:
        return override
    models = studio.models
    for pref in ("qwen2.5:72b", "qwen", "llama3.1", "llama3", "llama", "gemma3"):
        for m in models:
            if pref in m:
                return m
    return models[0]


@pytest.fixture(scope="module")
def studio():
    from genai_studio import GenAIStudio

    return GenAIStudio(validate_model=False)


def test_native_tool_calling_probe(studio):
    """OPEN QUESTION #1 — does the gateway return tool_calls? (informational)."""
    model = _pick_model(studio)
    resp = studio.chat_raw(
        [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        model=model, tools=[_WEATHER_TOOL], tool_choice="auto", max_tokens=128,
    )
    tcs = getattr(resp.choices[0].message, "tool_calls", None)
    if tcs:
        print(f"\nNATIVE TOOLS SUPPORTED on {model}: {tcs[0].function.name}")
        assert tcs[0].function.name == "get_weather"
    else:
        pytest.skip(f"gateway did NOT return tool_calls on {model} -> ReActClient is the default")


def test_genaistudioclient_probe_matches(studio):
    from genai_studio.agents import GenAIStudioClient

    client = GenAIStudioClient(studio, default_model=_pick_model(studio))
    supported = client.probe_native_tools(force=True)
    print(f"\nGenAIStudioClient.probe_native_tools() -> {supported}")
    assert isinstance(supported, bool)


def test_end_to_end_agent(studio):
    """A real agent run; uses native tools if supported, else falls back to ReAct."""
    from genai_studio.agents import (
        Agent, GenAIStudioClient, NullTracer, ReActClient, tool,
    )

    @tool
    def add(a: int, b: int) -> str:
        "Add two integers.\n\nArgs:\n    a: first.\n    b: second."
        return str(a + b)

    model = _pick_model(studio)
    base = GenAIStudioClient(studio, default_model=model)
    client = base if base.probe_native_tools(force=True) else ReActClient(base)
    agent = Agent(client=client, tools=[add], tracer=NullTracer(), max_steps=4,
                  system="Use the add tool to compute sums.")
    res = agent.run("What is 21 plus 21? Use the add tool.")
    print(f"\nfinal: {res.text!r} | stopped: {res.stopped} | tokens: {res.usage.total_tokens}")
    assert res.stopped in ("final", "max_steps")
    assert "42" in res.text or any(
        tc.arguments.get("a") == 21 for st in res.steps for tc in st.tool_calls
    )
