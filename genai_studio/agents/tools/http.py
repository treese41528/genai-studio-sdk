"""
HTTP fetch tools (core — ``httpx``; no API keys).

- ``http_get``   — GET a URL and return decoded text (size-capped).
- ``fetch_json`` — GET a URL and return parsed JSON (compact preview in ``content``,
  the full object in ``ToolResult.data`` for ``python_exec`` to consume).

SAFETY: fetching an arbitrary, model-chosen URL is an SSRF / exfiltration surface
(internal metadata endpoints, file://-style tricks, huge bodies). These tools cap
the response size, time out, refuse non-http(s) schemes, and by default block
private/loopback/link-local targets (``block_private_ips``) — re-checked on every
redirect hop, since httpx's own redirect-following would otherwise skip the
allow-list. For stricter control build an allow-listed variant via
:func:`make_http_get` / :func:`make_fetch_json` or wrap the agent with a
:class:`Guard`. Treat any fetched content as untrusted: pair with the *sandboxed*
``python_exec`` downstream.
"""

from __future__ import annotations

import json as _json
import ipaddress
import socket
import urllib.parse

import httpx

from genai_studio.agents import ToolResult, tool

_UA = {"User-Agent": "genai-studio-sdk/1.0 (educational; +https://github.com/treese41528/genai-studio-sdk)"}
_DEFAULT_MAX_BYTES = 200_000
_MAX_HOPS = 6


def _host_ok(url: str, allow_hosts) -> bool:
    if not allow_hosts:
        return True
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(host == h.lower() or host.endswith("." + h.lower()) for h in allow_hosts)


def _is_blocked_ip(host: str) -> bool:
    """True if ``host`` is, or resolves to, a private / loopback / link-local /
    reserved address (blocks ``169.254.169.254``-style metadata SSRF)."""
    try:
        addrs = [ipaddress.ip_address(host)]                 # an IP literal — no DNS
    except ValueError:
        try:
            addrs = [ipaddress.ip_address(i[4][0]) for i in socket.getaddrinfo(host, None)]
        except Exception:
            return False                                     # unresolvable: let the real fetch fail
    return any(a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
               or a.is_multicast or a.is_unspecified for a in addrs)


def _read_capped(url: str, *, timeout: float, max_bytes: int, allow_hosts,
                 block_private_ips: bool) -> tuple[bytes, str]:
    """GET with a streamed byte cap; returns (body, content_type). Raises on error.

    Redirects are followed MANUALLY so the scheme + allow-list + private-IP checks
    re-run on EVERY hop — httpx's own ``follow_redirects`` would skip them on the
    redirect target, the classic allow-list-bypass SSRF.
    """
    with httpx.Client(timeout=timeout, headers=_UA, follow_redirects=False) as c:
        for _ in range(_MAX_HOPS):
            if not (url.startswith("http://") or url.startswith("https://")):
                raise ValueError("only http(s) URLs are allowed")
            host = urllib.parse.urlparse(url).hostname or ""
            if not _host_ok(url, allow_hosts):
                raise PermissionError(f"host not in allow-list: {host!r}")
            if block_private_ips and _is_blocked_ip(host):
                raise PermissionError(f"blocked private/internal address: {host!r}")
            with c.stream("GET", url) as r:
                if getattr(r, "is_redirect", False):
                    loc = r.headers.get("location")
                    if not loc:
                        r.raise_for_status()
                        return b"", r.headers.get("content-type", "")
                    url = str(httpx.URL(url).join(loc))      # re-checked at the top of the next hop
                    continue
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                chunks, total = [], 0
                for chunk in r.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        break
                return b"".join(chunks)[:max_bytes], ctype
        raise RuntimeError("too many redirects")


def make_http_get(*, allow_hosts=None, block_private_ips: bool = True,
                  max_bytes: int = _DEFAULT_MAX_BYTES, timeout: float = 20):
    """Build an ``http_get`` tool. ``allow_hosts`` restricts to a host list (a host
    matches itself or any subdomain); ``block_private_ips`` (default on) refuses
    private/loopback/link-local targets at every redirect hop — the safe default."""

    @tool(name="http_get",
          description="GET a URL and return the response body as text (size-capped). "
                      "Use for fetching pages/APIs that return text or HTML.")
    def http_get(url: str) -> ToolResult:
        """Fetch ``url`` (http/https) and return its decoded text body.

        Args:
            url: the http(s) URL to GET.
        """
        try:
            body, ctype = _read_capped(url, timeout=timeout, max_bytes=max_bytes,
                                       allow_hosts=allow_hosts, block_private_ips=block_private_ips)
        except Exception as exc:
            return ToolResult(content="", error=f"http_get failed: {exc}")
        text = body.decode("utf-8", errors="replace")
        return ToolResult(content=text, data={"url": url, "content_type": ctype, "bytes": len(body)})

    return http_get


def make_fetch_json(*, allow_hosts=None, block_private_ips: bool = True,
                    max_bytes: int = _DEFAULT_MAX_BYTES, timeout: float = 20):
    """Build a ``fetch_json`` tool (same safety options as :func:`make_http_get`)."""

    @tool(name="fetch_json",
          description="GET a URL and return parsed JSON. The full object is in the "
                      "result data; a compact preview is shown.")
    def fetch_json(url: str) -> ToolResult:
        """Fetch ``url`` and parse the body as JSON.

        Args:
            url: the http(s) URL returning a JSON body.
        """
        try:
            body, _ = _read_capped(url, timeout=timeout, max_bytes=max_bytes,
                                   allow_hosts=allow_hosts, block_private_ips=block_private_ips)
            obj = _json.loads(body.decode("utf-8", errors="replace"))
        except Exception as exc:
            return ToolResult(content="", error=f"fetch_json failed: {exc}")
        preview = _json.dumps(obj, indent=2)[:2000]
        return ToolResult(content=preview, data=obj)

    return fetch_json


# Ready-to-use, unrestricted variants (cap + timeout only). For untrusted use,
# prefer the allow-listed factory above or a host-allow-listing Guard.
http_get = make_http_get()
fetch_json = make_fetch_json()
