"""The loop: the SAME Agent over native tool-calling vs ReAct.

Both clients return an identical ModelResponse, so the agent code is unchanged —
only the mechanism differs. Watch the two traces.

Run: python examples/02_agent_loop.py
"""

from __future__ import annotations

from genai_studio.agents import Agent, ConsoleTracer, ReActClient, tool
from _common import make_client


@tool
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: first addend.
        b: second addend.
    """
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two integers.

    Args:
        a: first factor.
        b: second factor.
    """
    return str(a * b)


PROMPT = "What is (21 + 21) then multiplied by 2? Use the tools, then give the number."


def run_with(label, client):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    agent = Agent(client=client, tools=[add, multiply],
                  system="Use the tools to compute; then state the final number.",
                  tracer=ConsoleTracer(), max_steps=6)
    result = agent.run(PROMPT)
    print("FINAL:", result.text, "| stopped:", result.stopped)


if __name__ == "__main__":
    base = make_client()
    run_with("NATIVE tool-calling (GenAIStudioClient)", base)
    run_with("ReAct (ReActClient wrapping the same client)", ReActClient(base))
