"""The model-facing memory tools: ``write_memory`` + ``recall_memory``.

These are the ONLY memory operations on the model surface. Destructive ops (forget / compact) are
library + REPL only (``/forget``, ``/memory``), so the model can record and retrieve but never bulk-
erase. P0 is keyword-only; ``studio``/``embed_model`` are accepted so the P1 embedding rerank
(write-time vectors + query-time cosine, shared rate-limiter, fail-open) drops in without an API change.
"""

from __future__ import annotations

from ..tool import ToolResult, tool
from .retrieval import recall


def make_memory_tools(store, *, studio=None, embed_model: str = "llama3.2:latest",
                      rate_limiter=None) -> list:
    """Return ``[write_memory, recall_memory]`` bound to ``store``. Passing ``studio`` enables the
    embedding rerank (each fact is embedded ONCE at write; recall embeds the query and reranks by
    cosine); it FAILS OPEN to the keyword floor on any embed failure. Keyword-only when ``studio``
    is ``None``."""
    from ..embed import make_embedder
    embedder = make_embedder(studio, model=embed_model, limiter=rate_limiter)   # None if studio is None

    @tool(name="write_memory",
          description=("Save a durable fact to remember across sessions — a stable, reusable fact "
                       "(a user preference, a project convention, a decision). NOT for transient "
                       "details of the current task. Near-duplicate facts are merged automatically."))
    def write_memory(fact: str, tags: list[str] | None = None) -> ToolResult:
        vec = embedder(fact) if embedder else None                  # embed once at write (cached in the record)
        f = store.add(fact, tags or [], source="agent", vec=vec,
                      embed_model=(embed_model if vec else None))
        return ToolResult(content=f"saved memory [{f.id}]: {f.text}")

    @tool(name="recall_memory",
          description="Search durable memory for saved facts relevant to a query.")
    def recall_memory(query: str, k: int = 5) -> ToolResult:
        hits = recall(store.live(), query, k, embedder=embedder)
        if not hits:
            return ToolResult(content="(no relevant memory)")
        lines = [f"- [{f.id}] {f.text}" + (f"  (tags: {', '.join(map(str, f.tags))})" if f.tags else "")
                 for f, _ in hits]
        return ToolResult(content="\n".join(lines))

    return [write_memory, recall_memory]
