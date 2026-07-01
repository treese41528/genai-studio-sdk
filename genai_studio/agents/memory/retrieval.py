"""Relevance retrieval over memory facts.

P0 is a deterministic KEYWORD floor: token overlap + a tag boost, recency as the tiebreak, and a
strict ``score > 0`` floor so a query with no overlap returns nothing (an honest "no memory" rather
than a recency-padded false hit). An optional ``embedder`` (P1) reranks the keyword candidates and
FAILS OPEN to keyword order on any error — embeddings are never a hard dependency.
"""

from __future__ import annotations

from .store import _tokens


def _keyword_score(qtok: set, fact) -> int:
    overlap = len(qtok & _tokens(fact.text))
    tag_boost = sum(1 for t in fact.tags if str(t).lower() in qtok)
    return overlap + 2 * tag_boost


def recall(facts, query, k: int = 5, *, embedder=None, cosine=None):
    """Return ``[(fact, score)]`` for the top-``k`` facts relevant to ``query``, best first.

    Keyword floor: facts with zero overlap are dropped (honest no-memory). With an ``embedder``
    (P1), the keyword candidates are reranked by cosine of ``embedder(query)`` vs each fact's
    cached ``vec``; any failure falls back to the keyword ranking.
    """
    qtok = _tokens(query)
    cands = []
    for f in facts:
        s = _keyword_score(qtok, f)
        if s > 0:
            cands.append((f, s))
    cands.sort(key=lambda fs: (fs[1], fs[0].ts), reverse=True)   # score, then recency
    if embedder is not None and cands:
        try:
            cands = _embed_rerank(cands, query, embedder, cosine)
        except Exception:
            pass                                                  # fail-open to keyword order
    return cands[:k]


def _embed_rerank(cands, query, embedder, cosine):
    """Rerank keyword candidates (those with cached vectors) by cosine similarity to the query
    embedding; candidates lacking a vector keep their keyword position at the tail. If the query
    embedding fails (None), keep the keyword order (fail-open)."""
    qv = embedder(query)
    if not qv:
        return cands
    if cosine is None:
        cosine = _cosine
    with_vec = [(f, s) for f, s in cands if f.vec]
    without = [(f, s) for f, s in cands if not f.vec]
    with_vec.sort(key=lambda fs: cosine(qv, fs[0].vec), reverse=True)
    return with_vec + without


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0
