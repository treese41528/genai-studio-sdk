"""MCP (Model Context Protocol) client — connect to MCP servers and expose their tools, GATED.

External MCP tools are **untrusted**: they ride the harness's fail-closed seams rather than getting a
free pass. Every imported tool is namespaced ``mcp__<server>__<tool>``, which keeps it out of the
approval allowlists → ``assess()`` returns ``_ASK`` (re-prompts every call; a cached "always" grant
can't apply, so a rug-pulled definition can't be silently re-run), and an :class:`MCPGuard` adds a
server allowlist + a provenance banner. stdio servers are arbitrary local code, so spawning is
opt-in (``allow_stdio=True``) with the command shown for consent.

Off by default and byte-identical when unused: nothing here is imported unless a server is configured.
The official ``mcp`` SDK is a lazy import behind the ``[mcp]`` extra.

    from genai_studio.agents.mcp import mcp_tools
    tools, mgr = mcp_tools({"mcpServers": {"fs": {"command": "npx", "args": [...]}}}, allow_stdio=True)
    agent = Agent(client=..., tools=[*tools], guards=[mgr.guard, approval_guard])
    ...  # mgr.close() when done  (or `with mgr:`; or let assemble_agent(mcp=...) attach + Agent.close())
"""

from __future__ import annotations

from .config import MCPServerConfig, load_mcp_config
from .guard import MCPGuard

__all__ = ["mcp_tools", "MCPManager", "MCPServerConfig", "load_mcp_config", "MCPGuard"]


def mcp_tools(source, *, allow_stdio: bool = False, call_timeout: float = 30.0):
    """Connect every configured server, snapshot ``tools/list`` once, and return
    ``(tools, MCPManager)``. FAIL-OPEN on discovery — a dead/timing-out server contributes zero tools
    and logs, never crashing the build. The manager owns teardown (use it as a context manager or call
    ``.close()``). ``source`` is a dict, a JSON path, or None (standard locations)."""
    from .client import MCPManager
    return MCPManager.connect_all(load_mcp_config(source), allow_stdio=allow_stdio, call_timeout=call_timeout)
