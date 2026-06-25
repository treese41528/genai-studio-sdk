"""Live Data Commons integration test for the grounding tool.

Gated on ``DATACOMMONS_API_KEY`` (and the optional ``datacommons-client`` extra);
skipped otherwise so the default test run stays offline. Verifies the real v2
response shapes that the default adapters depend on.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("datacommons_client")
_KEY = os.getenv("DATACOMMONS_API_KEY")
pytestmark = pytest.mark.skipif(not _KEY, reason="DATACOMMONS_API_KEY not set")

from genai_studio.agents.tools import make_datacommons_tool


def test_population_lookup_returns_real_number():
    tool = make_datacommons_tool(api_key=_KEY)
    res = tool.run({"variable": "population", "place": "California"})
    assert res.error is None
    assert res.data and isinstance(res.data.get("value"), (int, float))
    assert res.data["value"] > 1_000_000          # California has >1M people
    assert res.data["place"] == "geoId/06"         # resolved to the state
    assert res.sources and res.sources[0].url      # provenance attached


def test_nonsense_place_degrades_gracefully():
    tool = make_datacommons_tool(api_key=_KEY)
    # a genuinely non-existent place (note: "Atlantis" is a real FL city!)
    res = tool.run({"variable": "population", "place": "Zzqxnowherevillexyzq"})
    assert res.error is None                        # graceful, never an exception
    assert "No Data Commons" in res.content         # honest "no place / no data"


def test_nl_resolves_an_unaliased_variable():
    # 'number of households' is NOT in the alias map -> resolved via NL stat-var
    # search, which (after re-ranking) should land on the general Count_Household.
    tool = make_datacommons_tool(api_key=_KEY, nl=True)
    res = tool.run({"variable": "number of households", "place": "California"})
    assert res.error is None
    assert res.data and isinstance(res.data.get("value"), (int, float))
    assert res.data["value"] > 1_000_000
    assert res.data["variable"].startswith("Count_Household")
