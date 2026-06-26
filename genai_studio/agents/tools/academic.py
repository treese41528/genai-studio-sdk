"""
Academic search tools (core — ``httpx`` + stdlib XML; no API keys).

Scholarly look-up for a research agent, each populating ``ToolResult.sources`` so
papers flow through as ``[n]`` citations:

- ``arxiv_search``     — arXiv preprints (Atom API; title/authors/abstract/PDF).
- ``openalex_search``  — OpenAlex works (240M+ scholarly records; DOIs, citations,
  open-access links). Keyless; sends a ``mailto`` for the polite pool when given.

Both degrade to ``ToolResult.error`` when offline so the agent can recover. Add
more providers (Semantic Scholar, Crossref, PubMed) by following the same shape:
hit the API, map records to ``Source`` + a ``[n]`` content block.
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from genai_studio.agents import Source, ToolResult, tool

_UA = {"User-Agent": "genai-studio-sdk/1.0 (educational; +https://github.com/treese41528/genai-studio-sdk)"}
_ATOM = "{http://www.w3.org/2005/Atom}"


def _clean(s: str) -> str:
    return " ".join((s or "").split())


def _clamp_n(n, default: int = 5) -> int:
    """Coerce a (possibly model-stringified) count into [1, 25]; never raises."""
    try:
        return max(1, min(int(n), 25))
    except (TypeError, ValueError):
        return default


def _block(records: list[dict]) -> ToolResult:
    """Render parsed paper records into a cited ToolResult (one Source each)."""
    if not records:
        return ToolResult(content="No results.", data=records)
    sources, lines = [], []
    for i, r in enumerate(records):
        title = r.get("title") or "(untitled)"
        who = ", ".join(r.get("authors", [])[:3]) + ("  et al." if len(r.get("authors", [])) > 3 else "")
        meta = {k: r[k] for k in ("year", "venue", "doi", "citations") if r.get(k) is not None}
        sources.append(Source(title=title, url=r.get("url"),
                              snippet=_clean(r.get("abstract", ""))[:300] or None, metadata=meta))
        head = f"[{i + 1}] {title}"
        if r.get("year"):
            head += f" ({r['year']})"
        lines.append(f"{head}\n    {who}\n    {_clean(r.get('abstract', ''))[:280]}\n    {r.get('url', '')}")
    return ToolResult(content="\n".join(lines), sources=sources, data=records)


@tool
def arxiv_search(query: str, max_results: int = 5) -> ToolResult:
    """Search arXiv for preprints and return titles, authors, abstracts, and links.

    Args:
        query: search terms (matched across title/abstract/authors).
        max_results: maximum number of papers to return (default 5).
    """
    params = {"search_query": f"all:{query}", "start": 0,
              "max_results": _clamp_n(max_results),
              "sortBy": "relevance", "sortOrder": "descending"}
    try:
        with httpx.Client(timeout=20, headers=_UA, follow_redirects=True) as c:
            r = c.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params))
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as exc:
        return ToolResult(content="", error=f"arxiv_search unavailable: {exc}")

    records = []
    for e in root.findall(f"{_ATOM}entry"):
        title = _clean((e.findtext(f"{_ATOM}title") or ""))
        if not title:
            continue
        authors = [_clean(a.findtext(f"{_ATOM}name") or "") for a in e.findall(f"{_ATOM}author")]
        url = _clean(e.findtext(f"{_ATOM}id") or "")            # the abs/ page
        for link in e.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                url = link.get("href") or url
        published = _clean(e.findtext(f"{_ATOM}published") or "")
        records.append({"title": title, "authors": authors, "url": url,
                        "abstract": e.findtext(f"{_ATOM}summary") or "",
                        "year": published[:4] or None, "venue": "arXiv"})
    return _block(records)


def _openalex_abstract(inv: dict | None) -> str:
    """Reconstruct OpenAlex's inverted-index abstract into plain text."""
    if not isinstance(inv, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs or []:
            positions.append((i, word))
    return " ".join(w for _, w in sorted(positions))


@tool
def openalex_search(query: str, max_results: int = 5, mailto: str | None = None) -> ToolResult:
    """Search OpenAlex for scholarly works (papers, with DOIs and citation counts).

    Args:
        query: search terms.
        max_results: maximum number of works to return (default 5).
        mailto: optional contact email for OpenAlex's faster "polite pool".
    """
    params = {"search": query, "per_page": _clamp_n(max_results)}
    if mailto:
        params["mailto"] = mailto
    try:
        with httpx.Client(timeout=20, headers=_UA, follow_redirects=True) as c:
            r = c.get("https://api.openalex.org/works", params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as exc:
        return ToolResult(content="", error=f"openalex_search unavailable: {exc}")

    records = []
    for w in results:
        title = _clean(w.get("title") or w.get("display_name") or "")
        if not title:
            continue
        authors = [_clean((a.get("author") or {}).get("display_name") or "")
                   for a in (w.get("authorships") or [])]
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        url = w.get("doi") or (w.get("primary_location") or {}).get("landing_page_url") or w.get("id")
        records.append({"title": title, "authors": [a for a in authors if a], "url": url,
                        "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
                        "year": w.get("publication_year"), "doi": doi,
                        "citations": w.get("cited_by_count"),
                        "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")})
    return _block(records)
