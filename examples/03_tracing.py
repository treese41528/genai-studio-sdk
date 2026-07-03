"""Tracing & debugging: observe every step.

ConsoleTracer prints the loop; JsonlTracer writes structured events to a file
you can analyze later (the eval-harness substrate, see 07).

Run: python examples/03_tracing.py
"""

from __future__ import annotations

import json

from genai_studio.agents import Agent, ConsoleTracer, tool
from genai_studio.agents.trace import JsonlTracer
from _common import make_client


@tool
def word_count(text: str) -> str:
    """Count the words in a string.

    Args:
        text: the text to count.
    """
    return str(len(text.split()))


if __name__ == "__main__":
    client = make_client()

    print("=== ConsoleTracer (show_prompts=True shows the literal prompt sent) ===")
    Agent(client=client, tools=[word_count],
          system="Use word_count, then report the number.",
          tracer=ConsoleTracer(show_prompts=True)).run(
        "How many words are in 'the quick brown fox jumps'?")

    print("\n=== JsonlTracer -> run.jsonl ===")
    path = "run.jsonl"
    with JsonlTracer(path) as tracer:
        Agent(client=client, tools=[word_count],
              system="Use word_count, then report the number.",
              tracer=tracer).run("Count words in 'data science is fun'.")

    print(f"wrote {path}; event types:")
    for line in open(path):
        print("  -", json.loads(line)["type"])
