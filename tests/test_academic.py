"""Tests for arxiv_search + openalex_search (httpx stubbed; offline-safe)."""

from __future__ import annotations

import types

from genai_studio.agents.tools import academic
from genai_studio.agents.tools.academic import (
    _block, _openalex_abstract, arxiv_search, openalex_search,
)

ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Great Paper</title>
    <summary>We do great things.</summary>
    <published>2023-05-01T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1"/>
  </entry>
</feed>"""

OPENALEX_JSON = {"results": [{
    "title": "OA Paper", "publication_year": 2021, "cited_by_count": 7,
    "doi": "https://doi.org/10.1/x",
    "authorships": [{"author": {"display_name": "Jane Doe"}}],
    "abstract_inverted_index": {"An": [0], "abstract.": [1]},
    "primary_location": {"source": {"display_name": "J. Stuff"}},
}]}


class _Resp:
    def __init__(self, text="", json_obj=None):
        self.text, self._json, self.headers = text, json_obj, {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _Client:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get(self, *a, **k):
        return self._resp

    def post(self, *a, **k):
        return self._resp


class _BoomClient:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get(self, *a, **k):
        raise RuntimeError("no net")


def _patch(monkeypatch, resp):
    monkeypatch.setattr(academic, "httpx",
                        types.SimpleNamespace(Client=lambda *a, **k: _Client(resp)))


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_openalex_abstract_reconstruction():
    inv = {"Hello": [0], "world": [1], "again": [2, 4], "say": [3]}
    assert _openalex_abstract(inv) == "Hello world again say again"
    assert _openalex_abstract(None) == ""


def test_block_empty_is_no_results():
    assert "No results" in _block([]).content


# ── arxiv ────────────────────────────────────────────────────────────────────
def test_arxiv_search_parses_atom(monkeypatch):
    _patch(monkeypatch, _Resp(text=ARXIV_XML))
    res = arxiv_search("great", max_results=5)
    assert res.error is None and len(res.data) == 1
    p = res.data[0]
    assert p["title"] == "A Great Paper"
    assert p["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert p["url"] == "http://arxiv.org/pdf/1234.5678v1"     # pdf link preferred
    assert p["year"] == "2023"
    assert res.sources[0].title == "A Great Paper"


def test_arxiv_search_offline_is_error(monkeypatch):
    monkeypatch.setattr(academic, "httpx",
                        types.SimpleNamespace(Client=lambda *a, **k: _BoomClient()))
    res = arxiv_search("x")
    assert res.error and "unavailable" in res.error


# ── openalex ─────────────────────────────────────────────────────────────────
def test_openalex_search_parses(monkeypatch):
    _patch(monkeypatch, _Resp(json_obj=OPENALEX_JSON))
    res = openalex_search("oa", max_results=3)
    assert res.error is None
    p = res.data[0]
    assert p["title"] == "OA Paper" and p["year"] == 2021 and p["citations"] == 7
    assert p["doi"] == "10.1/x"                               # doi prefix stripped
    assert p["abstract"] == "An abstract."
    assert res.sources[0].metadata["citations"] == 7
