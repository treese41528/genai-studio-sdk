"""
``datacommons_lookup`` — ground a quantitative claim in Google Data Commons.

This is **DataGemma RIG reframed as a tool** rather than a fine-tune: any
tool-calling model can fetch a *real* statistic on demand instead of inventing
one. It resolves a place name to a Data Commons id, maps a free-text variable to
a statistical-variable id (via a small curated alias map — the paper's #2 failure
mode is bad NL→variable conversion), fetches the observation, and returns the
value with its source. Coverage is the dominant caveat: Data Commons has data for
only ~1/4 of statistical queries, so a miss returns an **honest "no observation"**
(as content, not an error, never a guess).

``datacommons-client`` is an optional dependency::

    pip install 'genai-studio-sdk[grounding]'        # + a free key from apikeys.datacommons.org

With ``nl=True`` the curated alias map is backed by Data Commons' NL statistical-
variable search, so *any* free-text variable ("carbon emissions", "number of
households") resolves — directly attacking the NL→variable failure mode. The DC
API specifics are isolated behind ``resolve_place`` / ``fetch_obs`` / ``var_search``
so the tool logic (aliasing, caching, honest no-data, ambiguity) is testable
without the library or a key — inject your own for tests or other backends.
"""

from __future__ import annotations

from typing import Callable

from genai_studio.agents import Source, ToolResult, tool

# Curated free-text -> Data Commons statistical-variable ids. Prefer passing an
# explicit DCID as ``variable``; this map just cuts the most common conversions.
_DEFAULT_ALIASES = {
    "population": "Count_Person",
    "median household income": "Median_Income_Household",
    "median income": "Median_Income_Person",
    "unemployment rate": "UnemploymentRate_Person",
    "median age": "Median_Age_Person",
    "life expectancy": "LifeExpectancy_Person",
    "poverty rate": "Count_Person_BelowPovertyLevelInThePast12Months",
}

_INSTALL_HINT = (
    "Data Commons is unavailable: install the extra "
    "(pip install 'genai-studio-sdk[grounding]') and pass a Data Commons API key "
    "(free from apikeys.datacommons.org)."
)


def make_datacommons_tool(api_key: str | None = None, *, dc_client=None,
                          resolve_place: Callable | None = None,
                          fetch_obs: Callable | None = None,
                          var_search: Callable | None = None,
                          aliases: dict | None = None, nl: bool = False,
                          name: str = "datacommons_lookup", cache: bool = True):
    """Build a ``datacommons_lookup`` grounding tool.

    Args:
        api_key: Data Commons API key (used to build the default client lazily).
        dc_client: a prebuilt ``datacommons_client`` instance (else built lazily).
        resolve_place: ``(name: str) -> list[str]`` of place DCIDs (overridable;
            defaults to a ``datacommons-client`` resolver). First = best match.
        fetch_obs: ``(var_dcid, place_dcid, date) -> dict | None`` returning
            ``{value, unit, date, source, source_url}`` or ``None`` for no data.
        var_search: ``(text: str) -> dcid | None`` resolving a free-text variable
            (overridable; defaults to the DC NL stat-var search when ``nl=True``).
        aliases: extra free-text -> DCID variable mappings (merged over defaults).
        nl: resolve variables NOT in the alias map via Data Commons' NL stat-var
            search (``datacommons.org/api/stats/stat-var-search``, backed by
            nl.datacommons.org) instead of assuming the text is a DCID. The alias
            map keeps precedence (curated accuracy for common stats; NL for the tail).
        name: the tool name the model sees.
        cache: memoize ``(variable, place, date)`` lookups in-process.
    """
    alias_map = {**_DEFAULT_ALIASES, **(aliases or {})}
    _cache: dict | None = {} if cache else None
    _lazy: dict = {"client": dc_client, "resolve": resolve_place, "fetch": fetch_obs,
                   "var_search": var_search}

    def _ensure_adapters():
        """Build the default datacommons-client adapters on first real use."""
        if _lazy["resolve"] is not None and _lazy["fetch"] is not None:
            return
        client = _lazy["client"]
        if client is None:
            from datacommons_client.client import DataCommonsClient  # optional dep
            client = DataCommonsClient(api_key=api_key)
            _lazy["client"] = client
        if _lazy["resolve"] is None:
            _lazy["resolve"] = lambda nm: _default_resolve(client, nm)
        if _lazy["fetch"] is None:
            _lazy["fetch"] = lambda v, p, d: _default_fetch(client, v, p, d)

    def _resolve_var(text: str) -> str:
        """Free-text variable -> DCID: alias map -> DCID passthrough -> NL search."""
        k = text.lower().strip()
        if k in alias_map:
            return alias_map[k]
        if " " not in text and ("_" in text or "/" in text):
            return text  # already looks like a DCID — don't second-guess it
        fn = _lazy["var_search"] or (
            (lambda t: _default_var_search(t, api_key)) if nl else None)
        if fn:
            try:
                hit = fn(text)
                if hit:
                    return hit
            except Exception:
                pass
        return text  # fall back: use the text as-is

    @tool(name=name,
          description="Look up an official statistic (e.g. population, median "
                      "income, unemployment rate) for a place from Google Data "
                      "Commons. Returns the real value with its source, or reports "
                      "honestly that no data is available — never a guess.")
    def datacommons_lookup(variable: str, place: str, date: str = "latest") -> ToolResult:
        """Fetch a real statistic for a place from Data Commons.

        Args:
            variable: the statistic — a known alias ('population', 'median household
                income') or an explicit Data Commons statistical-variable DCID.
            place: the place name ('California', 'France') or a place DCID.
            date: 'latest' (default) or a specific year/date.
        """
        ckey = (variable.lower().strip(), place.lower().strip(), date)
        if _cache is not None and ckey in _cache:
            return _cache[ckey]

        try:
            _ensure_adapters()
        except Exception:  # missing library OR missing/invalid API key -> degrade
            return ToolResult(content="", error=_INSTALL_HINT)

        var_dcid = _resolve_var(variable)
        try:
            place_dcids = list(_lazy["resolve"](place) or [])
        except Exception as exc:
            return ToolResult(content="", error=f"place resolution failed: {type(exc).__name__}: {exc}")

        if not place_dcids:
            return _store(_cache, ckey, ToolResult(
                content=f"No Data Commons place matches {place!r}."))

        place_dcid, ambiguous = place_dcids[0], len(place_dcids) > 1
        try:
            obs = _lazy["fetch"](var_dcid, place_dcid, date)
        except Exception as exc:
            return ToolResult(content="", error=f"observation fetch failed: {type(exc).__name__}: {exc}")

        if not obs:
            return _store(_cache, ckey, ToolResult(
                content=f"No Data Commons observation for {var_dcid!r} in {place!r} "
                        f"({place_dcid}). Treat this claim as UNVERIFIABLE, not false.",
                data={"variable": var_dcid, "place": place_dcid}))

        val, unit = obs.get("value"), obs.get("unit")
        obs_date, src = obs.get("date", date), obs.get("source")
        amb = f" (note: {place!r} was ambiguous; used {place_dcid})" if ambiguous else ""
        content = (f"{var_dcid} for {place} ({place_dcid}) = {val}"
                   f"{(' ' + unit) if unit else ''} as of {obs_date}{amb}. "
                   f"Source: {src or 'Data Commons'}.")
        result = ToolResult(
            content=content,
            data={"variable": var_dcid, "place": place_dcid, "value": val,
                  "unit": unit, "date": obs_date},
            sources=[Source(title=src or "Data Commons",
                            url=obs.get("source_url") or f"https://datacommons.org/browser/{place_dcid}",
                            metadata={"variable": var_dcid, "value": val})])
        return _store(_cache, ckey, result)

    return datacommons_lookup


def _store(cache, key, result):
    if cache is not None:
        cache[key] = result
    return result


def _default_var_search(text: str, api_key: str | None, *, limit: int = 15) -> str | None:
    """Free-text -> the most general statistical-variable DCID via Data Commons'
    NL/vector stat-var search (datacommons.org, backed by nl.datacommons.org).
    ``None`` on miss.

    Verified 2026-06-24: GET .../api/stats/stat-var-search?query=… ->
    ``{"statVars": [{"name", "dcid"}, …]}`` ranked by text match — which often
    surfaces hyper-specific variants first (e.g. "life expectancy" ->
    ``LifeExpectancy_Person_95OrMoreYears_Male``). So we re-rank to prefer curated
    (non ``dc/…``) vars and the MOST GENERAL one (fewest underscores). NL resolution
    is best-effort; the resolved DCID is surfaced in the tool result so a caller can
    catch a mis-resolution, and the curated alias map keeps precedence.
    """
    import httpx
    headers = {"X-API-Key": api_key} if api_key else {}
    r = httpx.get("https://datacommons.org/api/stats/stat-var-search",
                  params={"query": text, "limit": limit}, headers=headers, timeout=30)
    r.raise_for_status()
    svs = [s for s in (r.json().get("statVars") or []) if s.get("dcid")]
    if not svs:
        return None
    # text-rank is index; prefer named (non dc/…) then most general (fewest "_").
    svs.sort(key=lambda s: (s["dcid"].startswith("dc/"), s["dcid"].count("_")))
    return svs[0]["dcid"]


# ── default adapters over datacommons-client v2 ──────────────────────────────
# Live-verified against datacommons-client v2 (2026-06-24, api.datacommons.org).
# Inject resolve_place=/fetch_obs= to target another backend (e.g. the NL API).
def _default_resolve(client, name: str) -> list[str]:
    """Resolve a place name to DCIDs. If it already looks like a DCID, pass through.

    ``resolve.fetch_dcids_by_name(...).to_flat_dict()`` -> ``{name: [dcid, ...]}``
    (best match first, e.g. ``"California" -> ["geoId/06", ...]``).
    """
    if "/" in name:  # already a DCID (country/USA, geoId/06, wikidataId/Q99, …)
        return [name]
    resp = client.resolve.fetch_dcids_by_name(names=name)
    flat = resp.to_flat_dict() if hasattr(resp, "to_flat_dict") else {}
    dcids = flat.get(name, [])  # keyed by the exact query string; [] on a miss
    return [dcids] if isinstance(dcids, str) else list(dcids or [])


def _default_fetch(client, var_dcid: str, place_dcid: str, date: str) -> dict | None:
    """Fetch one observation. Returns {value, unit, date, source, source_url} or None.

    ``observation.fetch(...).to_observation_records()`` yields ``ObservationRecord``
    objects (pydantic) with ``value``/``date``/``unit``/``importName``/
    ``provenanceUrl``/``facetId``; LATEST returns one per facet, so pick the newest.
    """
    dc_date = "LATEST" if date == "latest" else date  # v2 selector, not 'latest'
    resp = client.observation.fetch(variable_dcids=[var_dcid],
                                    entity_dcids=[place_dcid], date=dc_date)
    records = resp.to_observation_records() if hasattr(resp, "to_observation_records") else []
    rows = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in records]
    rows = [r for r in rows if r.get("value") is not None]
    if not rows:
        return None
    rec = max(rows, key=lambda r: str(r.get("date") or ""))  # newest observation
    return {"value": rec.get("value"), "unit": rec.get("unit"),
            "date": rec.get("date", date), "source": rec.get("importName"),
            "source_url": rec.get("provenanceUrl")}
