"""Recall-based agent memory — the agent WRITES durable facts and RECALLS them by relevance
across runs (P0: keyword-only). Distinct from session persistence (the CONVERSATION) and OKF
bundles (CURATED knowledge): this is the agent's self-maintained durable-fact store.

Public surface:
- ``MemoryStore`` / ``MemoryFact`` / ``open_store`` — the JSONL truth + dedup/supersede/forget.
- ``recall`` — keyword relevance retrieval (embedding rerank is P1, fail-open to keyword).
- ``make_memory_tools`` — the model-facing ``write_memory`` / ``recall_memory`` tools.
- ``memory_index_text`` / ``inject_memory`` — the capped always-on index for the system prompt.
"""

from __future__ import annotations

from .index import inject_memory, memory_index_text
from .retrieval import recall
from .store import MemoryFact, MemoryStore, open_store
from .tools import make_memory_tools

__all__ = ["MemoryStore", "MemoryFact", "open_store", "recall",
           "make_memory_tools", "memory_index_text", "inject_memory"]
