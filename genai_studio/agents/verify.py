"""
``verifier`` — a grounded fact-checking sub-agent.

A thin factory over :class:`Agent`: it wires grounding tools (a knowledge-base
``kb_search`` and/or a ``datacommons_lookup``) behind a fact-checker system
prompt. Exposed to a finalizer/supervisor via :meth:`Agent.as_tool`, it becomes
a ``verify_claims`` step that checks quantitative claims against real evidence
*before* the system commits to an answer.

Why it works as a hallucination gate: the verifier's tools are execute-then-read
(the model only ever sees the real retrieved value, never a fabricated one), and
the verifier is itself execute-then-read to its parent — so a parent cannot
finalize on an invented statistic without the chain having had a real chance to
contradict it. The verifier's PASS/FAIL + grounded numbers + ``[n]`` sources flow
up through ``ToolResult.sources`` -> ``AgentResult.sources``.

    from genai_studio.agents import verifier
    from genai_studio.agents.tools import make_kb_search_tool

    kb = make_kb_search_tool(studio, "<collection-id>")
    v = verifier(client, kb_search=kb)
    finalizer = Agent(client=client, tools=[v.as_tool("verify_claims",
        "Fact-check every quantitative claim against the knowledge base before answering.")])
"""

from __future__ import annotations

from .agent import Agent

VERIFY_PROMPT = """\
You are a careful fact-checker. You are given text containing factual or
quantitative claims. Your job is to verify them against real evidence — never
from memory.

- For EACH claim, call a tool to find the real value:
  - use kb_search for claims that should be supported by the knowledge base;
  - use datacommons_lookup for public statistics (population, income, rates, …).
- Report one line per claim: `PASS` / `FAIL` / `UNVERIFIABLE`, the claim, the
  grounded value you retrieved, and its source.
- NEVER assert a number you did not retrieve from a tool. If a tool returns no
  data, mark the claim `UNVERIFIABLE` (not FAIL) — absence of data is not
  disproof.
- When done, call final_answer with a short verdict (or simply reply in text):
  which claims are supported, which are contradicted, and which could not be checked.
"""


def verifier(client, *, kb_search=None, datacommons=None, model: str | None = None,
             extra_tools=(), name: str = "verifier", system: str | None = None,
             **agent_kwargs) -> Agent:
    """Build a grounded fact-checking :class:`Agent`.

    Pass at least one grounding tool. Build it from the SAME ``client`` as the
    parent so the whole tree shares one rate-limiter (the gateway drops bursts).

    Args:
        client: a ``ModelClient`` (shared across the agent tree).
        kb_search: a knowledge-base retrieval tool (e.g. from ``make_kb_search_tool``).
        datacommons: a Data Commons grounding tool (e.g. from ``make_datacommons_tool``).
        model: model id (defaults to the client's default).
        extra_tools: any additional grounding tools.
        name: the agent name (its default ``as_tool`` name).
        system: override the fact-checker system prompt.
        **agent_kwargs: forwarded to :class:`Agent` (e.g. ``tracer``, ``max_steps``).
    """
    from .tools.general import final_answer

    grounding = [t for t in (kb_search, datacommons, *extra_tools) if t is not None]
    if not grounding:
        raise ValueError(
            "verifier() needs at least one grounding tool — pass kb_search= "
            "and/or datacommons= (a verifier with nothing to check against is "
            "just an ungrounded model).")
    return Agent(client=client, model=model, name=name,
                 tools=[*grounding, final_answer],
                 system=system or VERIFY_PROMPT, **agent_kwargs)
