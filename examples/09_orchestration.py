"""Orchestration: supervisor (dynamic delegation) + pipeline (fixed stages).

Two minimal multi-agent topologies, both thin factories over Agent.as_tool():
- supervisor — one coordinator LLM delegates to specialist workers-as-tools.
- pipeline   — a fixed sequence of agents; each stage's answer feeds the next.

Build every agent from the SAME client so the whole team shares one rate-limiter.

Run: python examples/09_orchestration.py
"""

from __future__ import annotations

from genai_studio.agents import Agent, ConsoleTracer, NullTracer, pipeline, supervisor
from _common import make_client

client = make_client()  # ONE shared client across the whole team

# ── supervisor: a lead agent routes each sub-question to a specialist ────────
mathematician = Agent(client=client, name="mathematician", tracer=NullTracer(), max_steps=3,
                      system="You do arithmetic only. Answer with just the number.")
historian = Agent(client=client, name="historian", tracer=NullTracer(), max_steps=3,
                  system="You answer history questions in one short sentence.")

lead = supervisor(
    client,
    "You lead a team: a 'mathematician' (arithmetic) and a 'historian' (history facts). "
    "Delegate each part of the question to the right specialist, then combine the answers.",
    [mathematician, historian], tracer=ConsoleTracer(), max_steps=6)

# ── pipeline: a fixed extract -> summarize workflow (no manager LLM) ─────────
extractor = Agent(client=client, name="extractor", tracer=NullTracer(), max_steps=2,
                  system="List the key facts in the text as short bullets.")
summarizer = Agent(client=client, name="summarizer", tracer=NullTracer(), max_steps=2,
                   system="Write ONE concise sentence summarizing the bullet points.")
summarize_doc = pipeline([extractor, summarizer])

if __name__ == "__main__":
    print("=== supervisor (delegates to specialists) ===")
    print(lead.run("Who painted the Mona Lisa, and what is 12 * 11?").text)

    print("\n=== pipeline (extract -> summarize) ===")
    print(summarize_doc(
        "The Eiffel Tower, built in 1889 for the World's Fair, is 330 metres tall and stands in Paris."
    ).text)
