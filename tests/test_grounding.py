"""Unit tests for the Data Commons grounding tool.

The Data Commons HTTP specifics are injected (``resolve_place`` / ``fetch_obs``)
so the *robust* logic — alias mapping, caching, honest no-data, ambiguity, and
graceful degradation when the optional library is missing — is tested with zero
network and no ``datacommons-client`` install.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult
from genai_studio.agents.tools import make_datacommons_tool


def _tool(**kw):
    # sensible default fakes; override per test
    kw.setdefault("resolve_place", lambda name: ["country/USA"])
    kw.setdefault("fetch_obs", lambda v, p, d: {"value": 331_000_000, "date": "2020",
                                                "source": "US Census"})
    return make_datacommons_tool(**kw)


def test_value_path_returns_number_and_source():
    res = _tool().run({"variable": "population", "place": "United States"})
    assert isinstance(res, ToolResult)
    assert res.error is None
    assert "331000000" in res.content
    assert res.data["value"] == 331_000_000
    assert len(res.sources) == 1 and res.sources[0].title == "US Census"


def test_alias_maps_freetext_to_dcid():
    seen = {}
    def fetch(v, p, d):
        seen["var"] = v
        return {"value": 1, "date": "2020"}
    _tool(fetch_obs=fetch).run({"variable": "population", "place": "USA"})
    assert seen["var"] == "Count_Person"  # 'population' -> Count_Person


def test_explicit_dcid_passes_through():
    seen = {}
    def fetch(v, p, d):
        seen["var"] = v
        return {"value": 1, "date": "2020"}
    _tool(fetch_obs=fetch).run({"variable": "Median_Age_Person", "place": "USA"})
    assert seen["var"] == "Median_Age_Person"


def test_no_data_is_honest_not_an_error():
    res = _tool(fetch_obs=lambda v, p, d: None).run(
        {"variable": "population", "place": "Atlantis"})
    assert res.error is None              # NOT an error
    assert "No Data Commons observation" in res.content
    assert "UNVERIFIABLE" in res.content  # explicitly not 'false'


def test_no_place_match():
    res = _tool(resolve_place=lambda name: []).run(
        {"variable": "population", "place": "Nowheresville"})
    assert res.error is None
    assert "No Data Commons place" in res.content


def test_ambiguous_place_is_flagged():
    res = _tool(resolve_place=lambda name: ["geoId/13", "country/GEO"]).run(
        {"variable": "population", "place": "Georgia"})
    assert "ambiguous" in res.content
    assert "geoId/13" in res.content  # used the first candidate


def test_cache_dedupes_calls():
    calls = {"n": 0}
    def fetch(v, p, d):
        calls["n"] += 1
        return {"value": 5, "date": "2020"}
    t = _tool(fetch_obs=fetch, cache=True)
    t.run({"variable": "population", "place": "USA"})
    t.run({"variable": "population", "place": "USA"})
    assert calls["n"] == 1  # second lookup served from cache


def test_missing_library_degrades_gracefully(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("datacommons_client"):
            raise ImportError("no datacommons_client")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # no client / resolve / fetch injected -> must try to import the lib -> fail soft
    res = make_datacommons_tool().run({"variable": "population", "place": "USA"})
    assert res.content == ""
    assert "install" in (res.error or "").lower()


def test_custom_alias_merge():
    seen = {}
    def fetch(v, p, d):
        seen["var"] = v
        return {"value": 1, "date": "2020"}
    t = _tool(fetch_obs=fetch, aliases={"widgets": "Count_Widget"})
    t.run({"variable": "widgets", "place": "USA"})
    assert seen["var"] == "Count_Widget"


# ── NL variable resolution (free-text -> stat-var DCID) ──────────────────────
def test_nl_resolves_unaliased_variable():
    seen = {}
    def vs(text):
        seen["q"] = text
        return "Annual_Emissions_CarbonDioxide"
    def fetch(v, p, d):
        seen["var"] = v
        return {"value": 42, "date": "2020"}
    _tool(fetch_obs=fetch, var_search=vs, nl=True).run(
        {"variable": "carbon dioxide emissions", "place": "USA"})
    assert seen["q"] == "carbon dioxide emissions"
    assert seen["var"] == "Annual_Emissions_CarbonDioxide"


def test_alias_takes_precedence_over_nl():
    called = {"vs": 0}
    def vs(text):
        called["vs"] += 1
        return "WRONG"
    def fetch(v, p, d):
        called["var"] = v
        return {"value": 1, "date": "2020"}
    _tool(fetch_obs=fetch, var_search=vs, nl=True).run(
        {"variable": "population", "place": "USA"})
    assert called["var"] == "Count_Person"   # alias wins
    assert called["vs"] == 0                  # NL search never called for an aliased var


def test_dcid_passthrough_skips_nl():
    called = {"vs": 0}
    def vs(text):
        called["vs"] += 1
        return "WRONG"
    def fetch(v, p, d):
        called["var"] = v
        return {"value": 1, "date": "2020"}
    _tool(fetch_obs=fetch, var_search=vs, nl=True).run(
        {"variable": "Count_Household", "place": "USA"})  # looks like a DCID
    assert called["var"] == "Count_Household"
    assert called["vs"] == 0                  # DCID-looking text not sent to NL search


def test_nl_disabled_passes_text_through():
    seen = {}
    def fetch(v, p, d):
        seen["var"] = v
        return {"value": 1, "date": "2020"}
    _tool(fetch_obs=fetch, nl=False).run(   # no var_search, nl off
        {"variable": "number of households", "place": "USA"})
    assert seen["var"] == "number of households"  # passed through unresolved
