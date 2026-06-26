"""Knowledge-authoring conventions for grounded agents.

``okf`` implements the Open Knowledge Format v0.1 loader — a portable, git-friendly
way to author a knowledge bundle on disk and push it into a gateway knowledge base
for server-side RAG (consumed by ``kb_search`` and the ``verifier`` sub-agent).
"""

from __future__ import annotations

from .okf import OKFDoc, ingest_bundle, load_bundle

__all__ = ["OKFDoc", "load_bundle", "ingest_bundle"]
