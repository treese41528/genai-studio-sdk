"""``MCPServerConfig`` + ``load_mcp_config`` — Claude-Code-style server declarations.

    {"mcpServers": {"<name>": {"command": ..., "args": [...], "env": {...},
                               "transport": "stdio", "trusted": false, "timeout": 30}}}

``${VAR}`` in ``env``/``headers`` expands from the process environment so secrets stay out of the
file. No config anywhere ⇒ ``[]`` ⇒ MCP disabled (the byte-identical default)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: tuple = ()
    env: dict = field(default_factory=dict)
    url: str | None = None                      # P2 (HTTP)
    headers: dict = field(default_factory=dict)  # P2
    trusted: bool = False
    timeout: float = 30.0

    def __post_init__(self):
        if "__" in self.name:                   # server name is a namespace segment (mcp__<name>__tool)
            raise ValueError(f"MCP server name {self.name!r} must not contain '__'")


def _one(name: str, d: dict) -> MCPServerConfig:
    env = {k: v for k, v in _expand(d.get("env", {})).items() if k != "GENAI_STUDIO_API_KEY"}
    return MCPServerConfig(
        name=name, transport=d.get("transport", "stdio"), command=d.get("command"),
        args=tuple(d.get("args", ())), env=env, url=d.get("url"),
        headers=_expand(d.get("headers", {})), trusted=bool(d.get("trusted", False)),
        timeout=float(d.get("timeout", 30.0)))


def load_mcp_config(source=None) -> list[MCPServerConfig]:
    """Load configs from a dict, a JSON file path, or the standard locations (kwarg →
    ``./.genai_studio/mcp.json`` → ``~/.genai_studio/mcp.json`` → ``$GENAI_STUDIO_MCP_CONFIG``).
    Returns ``[]`` when nothing is configured."""
    if isinstance(source, (list, tuple)) and all(isinstance(s, MCPServerConfig) for s in source):
        return list(source)
    data = None
    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, Path)) and Path(source).is_file():
        data = json.loads(Path(source).read_text("utf-8"))
    elif source is None:
        for p in (Path.cwd() / ".genai_studio" / "mcp.json",
                  Path.home() / ".genai_studio" / "mcp.json",
                  os.environ.get("GENAI_STUDIO_MCP_CONFIG")):
            if p and Path(p).is_file():
                data = json.loads(Path(p).read_text("utf-8"))
                break
    if not isinstance(data, dict):
        return []
    servers = data.get("mcpServers", data)
    return [_one(name, d) for name, d in servers.items() if isinstance(d, dict)]
