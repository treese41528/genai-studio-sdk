"""
``kb_search`` — server-side RAG retrieval over a GenAI Studio knowledge base.

The gateway does the embedding + vector search; this tool just POSTs a query to
``/api/v1/retrieval/query/collection`` and maps the Chroma-style parallel arrays
(``documents`` / ``distances`` / ``metadatas``) into a :class:`ToolResult`. The
model only ever sees the retrieved passage text (``ToolResult.content``) — it
cannot fabricate the result, because the loop feeds back the *real* tool reply.

Use the factory so the model-visible schema stays just ``query``/``k`` and the
collection plumbing is closed over::

    from genai_studio.agents.tools import make_kb_search_tool
    kb_search = make_kb_search_tool(studio, "aec63dc0-...")   # a KB id
    agent = Agent(client=..., tools=[kb_search])

Defaults to **dense-only** retrieval (``hybrid=False``): Open WebUI's hybrid
(BM25+dense) mode 400s on some builds (open-webui#14432).
"""

from __future__ import annotations

from genai_studio.agents import Source, ToolResult, tool
from genai_studio.agents.client import _default_limiter

_ENDPOINT = "/api/v1/retrieval/query/collection"


def make_kb_search_tool(studio, collection_names, *, k: int = 4,
                        name: str = "kb_search", hybrid: bool = False,
                        rate_limiter=None):
    """Build a ``kb_search`` tool bound to one or more knowledge-base collections.

    Args:
        studio: a ``GenAIStudio`` client (supplies auth + ``_http_post``).
        collection_names: a knowledge-base id, or a list of ids, to search.
        k: default number of passages to return.
        name: the tool name the model sees.
        hybrid: enable BM25+dense hybrid retrieval (off by default; see module note).
        rate_limiter: paces retrieval through the SAME limiter as LLM calls
            (defaults to the process-wide one from ``GENAI_STUDIO_RPM``) — the
            gateway silently drops bursts, so retrieval must be paced too.
    """
    cols = [collection_names] if isinstance(collection_names, str) else list(collection_names)
    default_k = k
    limiter = rate_limiter if rate_limiter is not None else _default_limiter()

    @tool(name=name,
          description="Search the knowledge base for passages relevant to a query "
                      "and return the most similar ones with their sources.")
    def kb_search(query: str, k: int = default_k) -> ToolResult:
        """Retrieve the most relevant passages from the knowledge base.

        Args:
            query: what to search for.
            k: maximum number of passages to return.
        """
        k = max(1, min(int(k), 20))  # clamp a model-supplied k to a sane range
        body = {"collection_names": cols, "query": query, "k": k, "hybrid": hybrid}
        try:
            limiter.acquire()  # pace retrieval through the SAME limiter as LLM calls
            resp = studio._http_post(_ENDPOINT, json=body)
            data = resp.json()
        except Exception as exc:
            return ToolResult(content="", error=f"kb_search failed: {type(exc).__name__}: {exc}")

        # Chroma-style parallel arrays; flatten across ALL collection rows so a
        # multi-collection search doesn't silently drop all but the first.
        docs = _flat(data.get("documents"))
        dists = _flat(data.get("distances"))
        metas = _flat(data.get("metadatas"))
        if not docs:
            return ToolResult(
                content="No matching passages found in the knowledge base.",
                data=data)

        lines, sources = [], []
        for i, chunk in enumerate(docs):
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            dist = dists[i] if i < len(dists) else None
            title = meta.get("name") or meta.get("title") or meta.get("source")
            url = meta.get("resource") or meta.get("url")
            lines.append(f"[{i + 1}] {str(chunk).strip()}")
            sources.append(Source(
                id=str(meta.get("file_id") or meta.get("id") or f"{name}:{i}"),
                title=title, url=url, snippet=str(chunk)[:200],
                metadata={**meta, "distance": dist}))
        return ToolResult(content="\n\n".join(lines), sources=sources, data=data)

    return kb_search


def _flat(arr):
    """Flatten one level of a Chroma-style ``[[...]]`` array (all rows), tolerating
    ``None`` and an already-flat array."""
    if not arr:
        return []
    if isinstance(arr[0], list):
        return [x for row in arr for x in row]
    return list(arr)
