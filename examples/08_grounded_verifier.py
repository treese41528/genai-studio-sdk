"""Grounded verifier — fact-check claims against evidence before answering.

A ``verifier`` sub-agent (grounded by ``kb_search``) is exposed via
``Agent.as_tool()`` as a ``verify_claims`` step. A finalizer must call it to
check claims against the knowledge base before committing to an answer — so it
can't finalize on a fabricated statistic without the chain having had a chance to
contradict it. Add ``make_datacommons_tool(...)`` to also ground public stats.

Run: python examples/08_grounded_verifier.py   (needs a knowledge base on the gateway)
"""

from __future__ import annotations

from genai_studio import GenAIStudio
from genai_studio.agents import (
    Agent, ConsoleTracer, GenAIStudioClient, NullTracer, verifier,
)
from genai_studio.agents.tools import make_kb_search_tool
from _common import DEFAULT_MODEL

studio = GenAIStudio(validate_model=False)
client = GenAIStudioClient(studio, default_model=DEFAULT_MODEL)  # ONE shared client

# Pick a knowledge base to ground against.
kbs = studio.list_knowledge_bases()
if not kbs:
    raise SystemExit("No knowledge base on the gateway — create one first (RAG docs).")
collection_id = kbs[0].id
print("Grounding against KB:", kbs[0].name, f"({collection_id})")

kb_search = make_kb_search_tool(studio, collection_id)

# A grounded fact-checker, exposed to a finalizer as one 'verify_claims' tool.
checker = verifier(client, kb_search=kb_search, tracer=NullTracer())
verify_claims = checker.as_tool(
    "verify_claims",
    "Fact-check every factual/quantitative claim against the knowledge base; "
    "returns PASS/FAIL/UNVERIFIABLE per claim with the grounded value + source.")

finalizer = Agent(
    client=client, tracer=ConsoleTracer(), max_steps=5, tools=[verify_claims],
    system="Before answering, call verify_claims to check any factual claims "
           "against the knowledge base. Only state claims the verifier supports.")

if __name__ == "__main__":
    result = finalizer.run(
        "Summarize the key facts in the knowledge base, and verify any figures you cite.")
    print("\nFINAL:\n", result.text)
    print("\nSOURCES:", [s.title for s in result.sources])
