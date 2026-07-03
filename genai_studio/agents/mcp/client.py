"""``MCPConnection`` (async-SDK → sync bridge) + ``MCPManager`` (lifecycle owner).

Each connection runs a dedicated daemon-thread event loop and opens the SDK ``ClientSession`` ONCE in
a single long-lived task that stays parked until ``close()`` — so the anyio context is entered and
exited in the *same* task (the cancel-scope rule). Tool calls are submitted onto that loop from the
caller thread via ``run_coroutine_threadsafe`` and block on the future (with a timeout — a hung server
must not wedge the caller). The tool wrappers are therefore plain sync callables that work under both
``Agent.run`` and ``arun`` (which dispatches sync tools via ``to_thread``).
"""

from __future__ import annotations

import asyncio
import logging
import threading

from ..tool import ToolResult
from .guard import MCPGuard, tool_hash
from .mapping import result_to_toolresult, to_tool

log = logging.getLogger("genai_studio.agents.mcp")


def _require_mcp():
    try:
        import mcp  # noqa: F401
    except ImportError as e:
        raise ImportError("MCP support needs the SDK: pip install 'genai-studio-sdk[mcp]'") from e


class MCPConnection:
    """One stdio MCP server, served by an owner-loop thread. Open/serve/close all in one task."""

    def __init__(self, config, *, call_timeout: float = 30.0):
        self.config = config
        self.call_timeout = call_timeout
        self._loop = None
        self._thread = None
        self._session = None
        self._stop = None
        self._ready = threading.Event()
        self._error = None

    def connect(self):
        self._thread = threading.Thread(target=self._serve_thread, name=f"mcp-{self.config.name}",
                                        daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self.call_timeout + 15):
            raise TimeoutError(f"MCP server {self.config.name!r} did not become ready")
        if self._error:
            raise self._error
        return self

    def _serve_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:                       # already recorded in _serve; guard the thread
            self._error = self._error or e
            self._ready.set()
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _serve(self):
        _require_mcp()
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self.config.command, args=list(self.config.args),
                                       env=self.config.env or None)
        try:
            async with AsyncExitStack() as stack:    # entered + exited in THIS task -> no cancel-scope error
                read, write = await stack.enter_async_context(stdio_client(params))
                self._session = await stack.enter_async_context(ClientSession(read, write))
                await self._session.initialize()
                self._ready.set()
                await self._stop.wait()              # park until close()
        except Exception as e:
            self._error = e
            self._ready.set()

    def _submit(self, coro, timeout):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def list_tools(self):
        return self._submit(self._session.list_tools(), self.call_timeout).tools

    def call(self, raw_name, arguments) -> ToolResult:
        try:
            res = self._submit(self._session.call_tool(raw_name, arguments or {}), self.call_timeout)
        except Exception as e:
            return ToolResult(content="", error=f"MCP call to {raw_name!r} failed: {type(e).__name__}: {e}")
        return result_to_toolresult(res)

    def close(self):
        if self._loop and self._loop.is_running() and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=10)


class MCPManager:
    """Owns every live connection + the wrapped tools + the ``MCPGuard``. Its ``close()`` tears down
    all servers (sync-safe — it owns its own threads). A context manager for teardown."""

    def __init__(self, connections, tools, guard):
        self._connections = connections
        self.tools = tools
        self.guard = guard

    @classmethod
    def connect_all(cls, configs, *, allow_stdio: bool = False, call_timeout: float = 30.0):
        conns, tools, manifest, allow = [], [], {}, set()
        for cfg in configs:
            if cfg.transport != "stdio":
                log.warning("MCP %r: transport %r unsupported in P1 (stdio only); skipped",
                            cfg.name, cfg.transport)
                continue
            if not allow_stdio:
                log.warning("MCP %r: stdio spawning is opt-in — pass allow_stdio=True to run %r; skipped",
                            cfg.name, cfg.command)
                continue
            try:
                conn = MCPConnection(cfg, call_timeout=min(call_timeout, cfg.timeout)).connect()
                mcp_tools_list = conn.list_tools()
            except Exception as e:                    # FAIL-OPEN on discovery: never crash the build
                log.warning("MCP %r failed to connect/list: %s", cfg.name, e)
                continue
            allow.add(cfg.name)
            conns.append(conn)
            for mt in mcp_tools_list:
                try:
                    t = to_tool(cfg.name, mt, conn.call)
                except ValueError as e:
                    log.warning("MCP %r: skipping tool %r (%s)", cfg.name, getattr(mt, "name", None), e)
                    continue
                tools.append(t)
                manifest[t.name] = tool_hash(t.spec)
        return tools, cls(conns, tools, MCPGuard(allow_servers=allow, manifest=manifest))

    def close(self):
        for c in self._connections:
            try:
                c.close()
            except Exception:
                pass
        self._connections = []

    aclose = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
