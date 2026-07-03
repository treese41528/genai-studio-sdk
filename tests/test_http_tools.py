"""Tests for http_get + fetch_json (httpx stubbed; offline-safe)."""

from __future__ import annotations

import json
import types

import httpx as _real_httpx

from genai_studio.agents.tools import http as httptool
from genai_studio.agents.tools.http import _host_ok, make_fetch_json, make_http_get


class _Stream:
    def __init__(self, body, headers):
        self._body, self.headers = body, headers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        for i in range(0, len(self._body), 8):
            yield self._body[i:i + 8]


class _Client:
    def __init__(self, body, headers):
        self._body, self.headers = body, headers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def stream(self, method, url):
        return _Stream(self._body, self.headers)


def _patch(monkeypatch, body, headers=None):
    monkeypatch.setattr(httptool, "httpx", types.SimpleNamespace(
        Client=lambda *a, **k: _Client(body, headers or {}), URL=_real_httpx.URL))


class _RedirectStream:
    def __init__(self, location):
        self.is_redirect, self.headers = True, {"location": location}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def raise_for_status(self):
        pass


class _RedirectClient:
    def __init__(self, location):
        self._loc = location

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def stream(self, method, url):
        return _RedirectStream(self._loc)


# ── allow-list logic ─────────────────────────────────────────────────────────
def test_host_ok():
    assert _host_ok("https://api.example.com/x", None)            # no list -> allowed
    assert _host_ok("https://api.example.com/x", ["example.com"]) # subdomain match
    assert _host_ok("https://example.com/x", ["example.com"])
    assert not _host_ok("https://evil.com/x", ["example.com"])


# ── http_get (block_private_ips off so fake public hosts skip real DNS) ──────
def test_http_get_returns_text(monkeypatch):
    _patch(monkeypatch, b"hello world", {"content-type": "text/plain"})
    res = make_http_get(block_private_ips=False)("https://x.org/page")
    assert res.error is None and res.content == "hello world"
    assert res.data["content_type"] == "text/plain"


def test_http_get_rejects_non_http_scheme():
    res = make_http_get()("file:///etc/passwd")                   # scheme check, no network
    assert res.error and "http_get failed" in res.error


def test_http_get_allowlist_blocks(monkeypatch):
    _patch(monkeypatch, b"x")
    res = make_http_get(allow_hosts=["good.org"], block_private_ips=False)("https://evil.org/x")
    assert res.error and "allow-list" in res.error


def test_http_get_size_cap(monkeypatch):
    _patch(monkeypatch, b"A" * 1000)
    res = make_http_get(max_bytes=100, block_private_ips=False)("https://x.org/big")
    assert res.error is None and len(res.content) == 100


# ── SSRF defenses: private-IP literals + redirect re-validation ──────────────
def test_blocks_loopback_ip_literal():
    res = make_http_get(block_private_ips=True)("http://127.0.0.1/x")   # IP literal, no DNS
    assert res.error and "private/internal" in res.error


def test_blocks_cloud_metadata_ip_literal():
    res = make_http_get(block_private_ips=True)("http://169.254.169.254/latest/meta-data/")
    assert res.error and "private/internal" in res.error


def test_redirect_to_blocked_host_is_refused(monkeypatch):
    # first hop is allow-listed, but it 302s to a non-allow-listed host -> refused.
    monkeypatch.setattr(httptool, "httpx", types.SimpleNamespace(
        Client=lambda *a, **k: _RedirectClient("http://evil.org/secret"), URL=_real_httpx.URL))
    res = make_http_get(allow_hosts=["good.org"], block_private_ips=False)("http://good.org/redir")
    assert res.error and "allow-list" in res.error           # the redirect TARGET was re-checked


# ── fetch_json ───────────────────────────────────────────────────────────────
def test_fetch_json_parses(monkeypatch):
    _patch(monkeypatch, json.dumps({"k": [1, 2, 3]}).encode())
    res = make_fetch_json(block_private_ips=False)("https://x.org/api")
    assert res.error is None and res.data == {"k": [1, 2, 3]}


def test_fetch_json_bad_json_is_error(monkeypatch):
    _patch(monkeypatch, b"not json at all")
    res = make_fetch_json(block_private_ips=False)("https://x.org/api")
    assert res.error and "fetch_json failed" in res.error
