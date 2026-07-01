"""Dynamic + parallel sub-agent fan-out.

``parallel_agents`` runs one bounded sub-agent PER subtask CONCURRENTLY (asyncio) and returns their
results in order — the fan-out primitive (developer-facing, like the fixed ``supervisor`` but with a
*dynamic* work-list run in *parallel*). ``make_fanout_tool`` exposes it as a model-facing ``fan_out``
tool: the MODEL decides how many independent subtasks to spawn (up to a cap), and they run in
parallel.

**Rate-limit invariant:** every sub-agent shares the ONE parent ``client``, so the process-wide
``RateLimiter`` paces them — parallel fan-out never bursts the gateway. Keep worker tools READ-ONLY
so concurrent workers can't race on writes.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from .agent import Agent
from .tool import ToolResult, tool
from .trace import NullTracer


def parallel_agents(client, subtasks: Sequence[str], *, system: str = "", model=None, tools=(),
                    max_steps: int = 6, max_concurrency: int = 8, cancel=None) -> list:
    """Run one sub-agent per subtask CONCURRENTLY; return ``AgentResult``s in the SAME order as
    ``subtasks``. A sub-task that raises lands as an ``Exception`` in its slot (never propagates).
    Bounded by ``max_concurrency``; paced by the shared client's rate-limiter."""
    subs = [str(s) for s in subtasks]
    if not subs:
        return []

    async def _run():
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def _one(st):
            async with sem:
                child = Agent(client=client, model=model, system=system, tools=list(tools),
                              max_steps=max_steps, tracer=NullTracer())
                return await child.arun(st, cancel=cancel)

        return await asyncio.gather(*[_one(s) for s in subs], return_exceptions=True)

    return asyncio.run(_run())


def make_fanout_tool(client, *, system: str | None = None, model=None, worker_tools=(),
                     max_agents: int = 5, max_steps: int = 6, name: str = "fan_out"):
    """A model-facing ``fan_out(subtasks)`` tool: the MODEL chooses how many INDEPENDENT subtasks to
    spawn (up to ``max_agents``); each runs as its own sub-agent IN PARALLEL and the results come
    back together. Give workers READ-ONLY ``worker_tools`` so parallel workers can't race on writes."""
    sys = system or ("You are a focused worker. Complete the SINGLE subtask you are given and report "
                     "the result concisely. Do not ask questions; do the work.")

    @tool(name=name,
          description=(f"Run 2-{max_agents} INDEPENDENT subtasks IN PARALLEL, each handled by its own "
                       "sub-agent, and get all their results back at once. Use when a task cleanly "
                       "splits into independent pieces (research N topics, inspect N files/areas). "
                       "Give self-contained subtasks; do the final synthesis yourself."))
    def fan_out(subtasks: list[str]) -> ToolResult:
        subs = [s for s in (subtasks or []) if str(s).strip()][:max_agents]
        if not subs:
            return ToolResult(content="", error="pass a non-empty list of independent subtasks")
        results = parallel_agents(client, subs, system=sys, model=model, tools=worker_tools,
                                  max_steps=max_steps)
        parts = []
        for i, (st, r) in enumerate(zip(subs, results), 1):
            if isinstance(r, BaseException):
                parts.append(f"[{i}] {st[:70]} -> ERROR: {type(r).__name__}: {r}")
            else:
                parts.append(f"[{i}] {st[:70]}\n{(getattr(r, 'text', '') or '').strip()}")
        return ToolResult(content="\n\n".join(parts), data={"n": len(subs)})

    return fan_out
