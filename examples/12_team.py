"""Team: a multi-agent tree, correct by construction.

Example 09 wires a supervisor/pipeline by hand and has to repeat the same client,
tracer, and conventions on every agent — miss one and the failure is quiet (a
dropped request, a mis-attributed trace line). A Team holds those once and stamps
them onto every agent it builds:

- ONE shared client  -> the whole tree shares one rate-limiter (cap it on the
  client: GENAI_STUDIO_RPM=20, or RateLimiter(20); a Team warns if it's uncapped).
- ONE inner tracer    -> each agent gets a ScopedTracer, so the console output is
  attributed ([researcher] / [writer]) and nested, never a step-number collision.
- tree-wide guards    -> e.g. one BudgetGuard shared across the tree caps total
  tool calls no matter which agent spends them.
- a shared system_prefix (house rules) prepended to every agent.

team.supervisor()/team.pipeline() additionally ENFORCE the shared client (they
raise on a stranger), where the standalone factories can only warn.

Run: python examples/12_team.py
"""

from __future__ import annotations

from genai_studio.agents import BudgetGuard, ConsoleTracer, Team, tool
from _common import make_client


@tool
def wordcount(text: str) -> str:
    "Count the words in some text.\n\nArgs:\n    text: the text to count."
    return str(len(text.split()))


if __name__ == "__main__":
    team = Team(
        make_client(),                       # cap RPM on the client; the tree shares it
        tracer=ConsoleTracer(),              # ONE inner tracer, auto-scoped per agent
        guards=[BudgetGuard(max_tool_calls=8)],   # tree-wide tool-call ceiling
        system_prefix="Be concise and state your reasoning in one line.",
    )

    researcher = team.agent(
        "researcher", max_steps=3,
        system="Pull out the key facts from the prompt as short bullets.")
    writer = team.agent(
        "writer", tools=[wordcount], max_steps=3,
        system="Write ONE tweet-length sentence from the bullets, then call "
               "wordcount on it to confirm it is under 30 words.")

    lead = team.supervisor(
        "You lead a 'researcher' (extracts facts) and a 'writer' (drafts the "
        "sentence). Delegate to each, then return the writer's final sentence.",
        [researcher, writer], max_steps=6)

    print(lead.run(
        "The James Webb Space Telescope, launched in December 2021, observes in "
        "infrared from the Sun-Earth L2 point, 1.5 million km from Earth."
    ).text)
