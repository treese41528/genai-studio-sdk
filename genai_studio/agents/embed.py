"""Shared embedding helper — a paced, fail-safe embedder + the chat-vs-embed model split.

The gateway's ``/api/embeddings`` works for only ~15/35 models (gemma3, llama4, qwen3, gpt-oss,
devstral, medgemma do NOT embed). Embedding is always OPTIONAL in this framework: recall-memory and
deferred-tool search have a deterministic keyword floor and fall back to it on ANY embed failure.
Embed calls are paced through the ONE process-wide ``RateLimiter`` so they share the gateway's RPM
budget with completions (the gateway silently drops bursts).
"""

from __future__ import annotations

# Model families WITHOUT a working /api/embeddings endpoint (probed live 2026-06-17). Everything
# else is assumed embed-capable. Default embed model is llama3.2:latest (3072-dim, reliable).
_NO_EMBED = ("gemma3", "llama4", "qwen3", "gpt-oss", "devstral", "medgemma")
DEFAULT_EMBED_MODEL = "llama3.2:latest"


def can_embed(model) -> bool:
    """True if ``model`` is expected to expose a working embeddings endpoint (blocklist-based)."""
    m = (model or "").lower()
    return bool(m) and not any(m.startswith(p) for p in _NO_EMBED)


def make_embedder(studio, *, model: str = DEFAULT_EMBED_MODEL, limiter=None):
    """Return ``embed(text) -> vector|None`` (``text`` may be a list -> list of vectors), paced
    through the shared ``RateLimiter`` and returning ``None`` on ANY failure so callers fail open
    to keyword. ``studio=None`` -> ``None`` (no embedder; pure keyword mode)."""
    if studio is None:
        return None
    from .client import _default_limiter
    lim = limiter if limiter is not None else _default_limiter()

    def embed(text):
        try:
            lim.acquire()                       # share the gateway RPM budget with completions
            return studio.embed(text, model=model)
        except Exception:
            return None                         # fail open — the caller falls back to keyword
    return embed


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0
