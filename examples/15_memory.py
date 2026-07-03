"""Recall memory: the agent writes durable facts and recalls them by relevance.

``make_memory_tools`` gives the agent ``write_memory`` / ``recall_memory`` backed by
a JSONL store (dedup -> supersede, a keyword floor, and an optional embedding rerank
that fails open to keyword). Facts persist to disk, so a LATER run can recall them —
pass ``studio=`` to enable semantic recall.

Run: python examples/15_memory.py
"""

from __future__ import annotations

import pathlib
import tempfile

from genai_studio.agents import Agent, ConsoleTracer
from genai_studio.agents.memory import MemoryStore, make_memory_tools
from _common import make_client

if __name__ == "__main__":
    store = MemoryStore(pathlib.Path(tempfile.mkdtemp()) / "memory.jsonl")
    client = make_client()
    # studio=GenAIStudio() would enable the embedding rerank; keyword-only here.
    agent = Agent(client=client, tools=make_memory_tools(store), tracer=ConsoleTracer(),
                  system="Save durable facts about the user with write_memory; look them "
                         "up with recall_memory. Keep answers short.")

    agent.run("Please remember that I prefer metric units and ISO-8601 dates.")
    print("\n--- new turn: the fact is recalled ---")
    print(agent.run("What formatting preferences have I told you about?").text)

    print("\nfacts on disk:", [f.text for f in store.live()])
