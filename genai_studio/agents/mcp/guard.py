"""``MCPGuard`` — a fail-closed ``before_tool`` gate for MCP tools.

Layered on top of the namespacing trick (which already forces every ``mcp__*`` call through approval):
this denies calls to any server not on the operator's allowlist, before the human is even prompted.
The per-tool definition hashes are captured at connect into a manifest — the rug-pull check they
enable fires when tools are re-listed (P3); in P1 the snapshot is immutable, so the manifest is the
provenance record. A guard that raises fails CLOSED (blocks) by the loop's contract.
"""

from __future__ import annotations

import hashlib
import json

from ..guard import ALLOW, Guard, deny
from .mapping import server_of


def tool_hash(spec) -> str:
    """Stable hash of a tool's (name, description, parameters) — the pin against silent drift."""
    payload = json.dumps({"n": spec.name, "d": spec.description, "p": spec.parameters},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MCPGuard(Guard):
    """Deny ``mcp__*`` calls to non-allowlisted servers OR to tools whose definition drifted since
    connect (rug-pull defense). Pass-through for non-MCP tools."""

    def __init__(self, *, allow_servers, manifest=None):
        self.allow_servers = set(allow_servers)
        self.manifest = dict(manifest or {})     # namespaced_name -> definition hash (captured at connect)
        self.drifted: set = set()                # names whose def changed/vanished on re-list (P3, via resync)

    def before_tool(self, call):
        name = getattr(call, "name", "") or ""
        if not name.startswith("mcp__"):
            return ALLOW                          # not ours; let other guards/approval decide
        if name in self.drifted:                  # definition changed since we pinned it -> rug-pull
            return deny(f"MCP tool {name!r} changed definition since connect (possible rug-pull); "
                        "reconnect to re-approve it")
        server = server_of(name)
        if server not in self.allow_servers:
            return deny(f"MCP server {server!r} is not on the allowlist")
        return ALLOW
