"""Deferred (searchable) tools — carry hundreds of tools as an always-on name+1-line catalog,
load full JSON schemas only on demand.

When ``Agent.tool_search`` is set, deferred tools' schemas are withheld; the model calls the
``search_tools(query)`` meta-tool, which ranks the catalog and returns the matching names in
``ToolResult.data['unlock']``. The Agent's loop keeps a run-local LRU set of unlocked names and
sends only the eager + unlocked schemas on the next model call. Deferral hides SCHEMAS, not
capability — a name-guessed deferred tool still executes (and auto-unlocks for the next turn);
access control stays with the Guard layer. Default ``tool_search=None`` ⇒ behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tool import ToolResult, tool


@dataclass(frozen=True)
class ToolSearch:
    """Opt-in deferral config (a single ``Agent.tool_search`` field).

    ``deferred``: explicit tool names to defer, or ``None`` / ``("*",)`` to defer every tool
    except the eager set, the finish tools, and the meta-tool. ``eager``: names to keep always
    resident. ``max_active``: LRU cap on simultaneously-unlocked schemas. ``catalog_limit``:
    cap on the always-on listing length (0 = no cap; ranking still spans the full catalog).
    """
    deferred: tuple | None = None
    eager: tuple = ()
    max_active: int = 16
    catalog_limit: int = 0
    tool_name: str = "search_tools"
    search_limit: int = 5
    rank: object = None            # optional ranker(query, catalog, limit) -> [names]; None = keyword


def _tokens(s: str) -> set:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 1}


def keyword_rank(query: str, catalog, limit: int) -> list:
    """Rank ``(name, description)`` catalog entries by token overlap with ``query``; return the
    top-``limit`` names (a zero-overlap entry is never returned)."""
    qtok = _tokens(query)
    scored = []
    for name, desc in catalog:
        s = len(qtok & _tokens(f"{name} {desc}"))
        if s > 0:
            scored.append((s, name))
    scored.sort(key=lambda sn: sn[0], reverse=True)
    return [name for _, name in scored[:limit]]


def render_catalog(catalog, *, limit: int = 0) -> str:
    """The always-on lightweight tool listing (one ``- name: desc`` line each), optionally capped
    with a '…and N more' tail so the prompt stays bounded while search still spans everything."""
    if not catalog:
        return ""
    head = ["# Available tools (call search_tools(query) to load a tool's full schema before using it)"]
    shown = catalog if (not limit or len(catalog) <= limit) else catalog[:limit]
    head += [f"- {name}: {desc}" for name, desc in shown]
    if limit and len(catalog) > limit:
        head.append(f"- … and {len(catalog) - limit} more; use search_tools to discover them")
    return "\n".join(head)


def make_search_tool(catalog, *, rank=keyword_rank, tool_name: str = "search_tools",
                     search_limit: int = 5):
    """Build the ``search_tools`` meta-tool over ``catalog`` (a list of ``(name, description)``).
    It returns the matching names in ``ToolResult.data['unlock']`` for the loop to act on."""
    desc_by_name = {name: desc for name, desc in catalog}

    @tool(name=tool_name,
          description=("Search the available tools by capability and unlock the matching ones so "
                       "their full schemas become callable on your next step. Call this BEFORE "
                       "using a tool whose full schema you do not yet see."))
    def search_tools(query: str) -> ToolResult:
        names = rank(query, catalog, search_limit)
        if not names:
            return ToolResult(content=f"No tools matched {query!r}. Try different terms.")
        listing = "\n".join(f"- {n}: {desc_by_name.get(n, '')}" for n in names)
        return ToolResult(content=f"Unlocked {len(names)} tool(s) — now callable:\n{listing}",
                          data={"unlock": names})

    return search_tools


def embedding_rank(studio, *, limiter=None, model: str | None = None):
    """A ranker(query, catalog, limit) -> [names] ranking by cosine of the query vs each tool's
    'name: desc' embedding. The catalog is embedded ONCE (cached in-closure, one batched call);
    on ANY embed failure it FAILS OPEN to ``keyword_rank`` — embeddings are never a hard dependency."""
    from .embed import DEFAULT_EMBED_MODEL, cosine, make_embedder
    embed = make_embedder(studio, model=model or DEFAULT_EMBED_MODEL, limiter=limiter)
    cache: dict = {}                                            # names-tuple -> {name: vec}

    def rank(query, catalog, limit):
        if embed is None:
            return keyword_rank(query, catalog, limit)
        names = tuple(n for n, _ in catalog)
        vecs = cache.get(names)
        if vecs is None:
            batch = embed([f"{n}: {d}" for n, d in catalog])   # one batched embed for the whole catalog
            if not isinstance(batch, list) or len(batch) != len(names) or not all(batch):
                return keyword_rank(query, catalog, limit)     # fail open
            vecs = dict(zip(names, batch))
            cache[names] = vecs
        qv = embed(query)
        if not qv:
            return keyword_rank(query, catalog, limit)         # fail open
        ranked = sorted(names, key=lambda n: cosine(qv, vecs[n]), reverse=True)
        return list(ranked[:limit])

    return rank
