"""
Web tools (core — uses ``httpx``, already a dependency; no API keys).

- ``web_search`` — DuckDuckGo HTML results (title + snippet + URL), with a
  Wikipedia fallback. Populates ``ToolResult.sources`` so results flow through to
  ``AgentResult.sources`` as ``[n]`` citations.
- ``wikipedia_search`` — Wikipedia search (reliable, no key).

Both degrade to ``ToolResult.error`` when offline, so the agent can recover.
"""

from __future__ import annotations

import html
import re
import urllib.parse

import httpx

from genai_studio.agents import Source, ToolResult, tool

_UA = {"User-Agent": "genai-studio-sdk/1.0 (educational; +https://github.com/treese41528/genai-studio-sdk)"}


def _strip(s: str) -> str:
    return html.unescape(re.sub("<.*?>", "", s or "")).strip()


def _parse_ddg(html_text: str, n: int) -> list[tuple[str, str, str]]:
    results = []
    anchors = re.finditer(r'<a ([^>]*class="result__a"[^>]*)>(.*?)</a>', html_text, re.S)
    for m in anchors:
        attrs, title = m.group(1), _strip(m.group(2))
        href_m = re.search(r'href="([^"]+)"', attrs)
        href = href_m.group(1) if href_m else ""
        if "uddg=" in href:  # DDG wraps the real URL in a redirect
            href = urllib.parse.unquote(re.search(r"uddg=([^&]+)", href).group(1))
        elif href.startswith("//"):
            href = "https:" + href
        if title:
            results.append((title, href, ""))
        if len(results) >= n:
            break
    snippets = [_strip(s) for s in
                re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_text, re.S)]
    return [(t, u, snippets[i] if i < len(snippets) else "")
            for i, (t, u, _) in enumerate(results)]


def _format(results: list[tuple[str, str, str]]) -> ToolResult:
    sources = [Source(title=t, url=u, snippet=s) for t, u, s in results]
    lines = [f"[{i + 1}] {t}\n    {s}\n    {u}" for i, (t, u, s) in enumerate(results)]
    return ToolResult(content="\n".join(lines) or "No results.", sources=sources, data=results)


def _wikipedia(query: str, n: int) -> ToolResult:
    with httpx.Client(timeout=15, headers=_UA) as c:
        r = c.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": n})
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    results = []
    for h in hits:
        title = h["title"]
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        results.append((title, url, _strip(h.get("snippet", ""))))
    return _format(results)


@tool
def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Search the web (DuckDuckGo) and return the top results with snippets and URLs.

    Args:
        query: the search query.
        max_results: maximum number of results to return (default 5).
    """
    try:
        with httpx.Client(timeout=15, headers=_UA, follow_redirects=True) as c:
            r = c.post("https://html.duckduckgo.com/html/", data={"q": query})
        r.raise_for_status()
        results = _parse_ddg(r.text, max_results)
        if results:
            return _format(results)
        return _wikipedia(query, max_results)  # fallback
    except Exception as exc:
        try:
            return _wikipedia(query, max_results)
        except Exception:
            return ToolResult(content="", error=f"web_search unavailable: {exc}")


@tool
def wikipedia_search(query: str, max_results: int = 3) -> ToolResult:
    """Search Wikipedia and return matching article titles, snippets, and URLs.

    Args:
        query: the search query.
        max_results: maximum number of results to return (default 3).
    """
    try:
        return _wikipedia(query, max_results)
    except Exception as exc:
        return ToolResult(content="", error=f"wikipedia_search unavailable: {exc}")
