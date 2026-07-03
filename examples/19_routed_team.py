"""Routed team — an orchestrator with the decision knowledge AND the specialists to act on it.

``routed_team`` builds a coordinator whose workers are pre-wired to the benchmark-optimal model +
sampling + tools for each role: ``math_specialist`` (CAS/prove tools), ``reasoning_specialist`` (a
reasoning model at greedy), grounded ``research_specialist``, and an opt-in ``critic_specialist``
(a low-hallucination fact-checker). The manager's prompt carries ``ROUTING_GUIDE`` — the measured
routing knowledge — so it delegates each subtask to the right specialist itself. Every worker and the
manager share ONE client (one rate-limiter); each worker overrides only its model.

Model picks are DATA-BACKED (benchmarks/): the math role uses a tool-disciplined 70B (a bake-off found
the tool-FREE GSM8K "champion" never calls the CAS tools), reasoning runs greedy, etc.

Run: GENAI_STUDIO_API_KEY=... python examples/19_routed_team.py
"""

from __future__ import annotations

from genai_studio.agents import ConsoleTracer, routed_team
from _common import make_client

if __name__ == "__main__":
    client = make_client()
    # include the opt-in critic too; override any model per gateway via models={...}
    team = routed_team(client, include=("math", "reasoning", "research", "critic"),
                       tracer=ConsoleTracer(), max_steps=8)

    for task in (
        "Compute the exact value of the integral of sin(x) from 0 to pi, and confirm it.",
        "What is the capital of Australia, and in what year did it become the capital?",
    ):
        print(f"\n{'='*70}\nTASK: {task}\n{'='*70}")
        print(team.run(task).text)

    # The manager routes: the integral -> math_specialist (computes + verify_math);
    # the fact -> research_specialist (grounded). Add a critic pass for high-stakes claims.
