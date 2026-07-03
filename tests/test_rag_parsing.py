"""Regression tests for RAG list-endpoint parsing.

The gateway returns ``{"items": [...]}`` from ``/api/v1/knowledge/`` and
``/api/v1/files/``; earlier the list methods only checked ``data``/``knowledge``/
``files`` keys and silently returned ``[]``. These tests pin the ``items`` key
(plus backward-compat with the older shapes), with zero network.
"""

from __future__ import annotations

import pytest

from genai_studio import GenAIStudio


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


@pytest.fixture
def studio():
    # api_key bypasses env; validate_model=False avoids any network on init.
    return GenAIStudio(api_key="test-key", validate_model=False)


def _patch_get(monkeypatch, studio, payload):
    monkeypatch.setattr(studio, "_http_get", lambda path: _Resp(payload))


@pytest.mark.parametrize("payload", [
    {"items": [{"id": "k1", "name": "KB One"}]},          # current gateway shape
    {"data": [{"id": "k1", "name": "KB One"}]},           # older shape
    {"knowledge": [{"id": "k1", "name": "KB One"}]},      # alt shape
    [{"id": "k1", "name": "KB One"}],                     # bare list
])
def test_list_knowledge_bases_parses_all_shapes(monkeypatch, studio, payload):
    _patch_get(monkeypatch, studio, payload)
    kbs = studio.list_knowledge_bases()
    assert len(kbs) == 1
    assert kbs[0].id == "k1" and kbs[0].name == "KB One"


@pytest.mark.parametrize("payload", [
    {"items": [{"id": "f1", "filename": "a.pdf"}]},       # current gateway shape
    {"data": [{"id": "f1", "filename": "a.pdf"}]},        # older shape
    {"files": [{"id": "f1", "filename": "a.pdf"}]},       # alt shape
    [{"id": "f1", "filename": "a.pdf"}],                  # bare list
])
def test_list_files_parses_all_shapes(monkeypatch, studio, payload):
    _patch_get(monkeypatch, studio, payload)
    files = studio.list_files()
    assert len(files) == 1
    assert files[0].id == "f1" and files[0].filename == "a.pdf"


def test_empty_knowledge_list(monkeypatch, studio):
    _patch_get(monkeypatch, studio, {"items": []})
    assert studio.list_knowledge_bases() == []
