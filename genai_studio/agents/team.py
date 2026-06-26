"""
``Team`` — a shared context that builds a whole agent tree *correctly by
construction*, so the easy way is the safe way.

Hand-wiring a multi-agent system means repeating four things on every agent and
getting all of them right: the same ``ModelClient`` (so the tree shares one
process-wide rate-limiter — the gateway silently drops bursts), one inner tracer
scoped per agent (so a sub-agent's "step 1" doesn't collide with the manager's),
the same guard policy, and the same house rules. Miss one and the failure is
quiet: a dropped request, a mis-attributed trace line, an un-policed tool.

A ``Team`` holds those four once and stamps them onto every agent it builds:

    team = Team(client, model="qwen2.5:72b",
                tracer=ConsoleTracer(),               # ONE inner, auto-scoped
                guards=[BudgetGuard(max_tool_calls=40)],  # tree-wide policy
                system_prefix=HOUSE_RULES)            # prepended to every agent

    researcher = team.agent("researcher", system="Find facts.", tools=[kb_search])
    writer     = team.agent("writer", system="Write the brief.")
    manager    = team.supervisor("Coordinate research then writing.",
                                 [researcher, writer])
    print(manager.run(question).text)

Every agent above shares ``client`` (→ one rate-limiter), carries its own
``ScopedTracer`` over the one ``ConsoleTracer`` (attributable, correctly-nested
output), enforces the same guards, and is prefixed with the same house rules —
none of it restated. ``supervisor`` / ``pipeline`` here additionally *enforce*
the shared client (they raise on a stranger), where the standalone
:func:`~genai_studio.agents.orchestrate.supervisor` can only warn.

**A Team shares the rate cap; it does not create one.** The limiter lives on the
client, so build the client with an explicit cap once
(``GenAIStudioClient(studio, rate_limiter=RateLimiter(20))`` or
``GENAI_STUDIO_RPM=20``); because the team shares that one client, the whole tree
is paced by it. If the client has *no* cap (no ``RateLimiter`` and
``GENAI_STUDIO_RPM`` unset), the tree shares a *no-op* limiter — i.e. no pacing —
so ``Team`` emits a one-time warning at construction. Sharing the client
guarantees the cap is *uniform*, not that one *exists*.

**Trace depth is single-level.** Each agent's events are attributed by *name*
(the anti-collision guarantee — ``[researcher] step 1`` never collides with
``[supervisor] step 1`` however deep the tree), and ``supervisor`` re-scopes its
direct workers one level deeper. Depth (the indentation) does *not* recurse into
a worker that is itself a composite (a ``team.supervisor`` used as a worker): its
grandchildren render at their parent's depth. Names stay correct; only the indent
flattens past one level — build trees one ``Team`` deep, or set depth by hand.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from .agent import Agent
from .orchestrate import pipeline as _pipeline, supervisor as _supervisor
from .trace import NullTracer, ScopedTracer

# Params Team sets itself on the agents it builds — passing them through a
# builder's **kwargs would collide, so we reject them with a Team-scoped message
# instead of a raw deep TypeError from Agent()/orchestrate.
_AGENT_RESERVED = ("client", "tracer")
_SUPERVISOR_RESERVED = ("client", "tracer", "cancel", "budget", "model")


@dataclass
class Team:
    """A shared build context for a multi-agent tree (see module docstring).

    Args:
        client: the ONE ``ModelClient`` every agent in the tree shares, so the
            whole tree shares one rate-limiter. Set the RPM cap on the client
            itself (``RateLimiter`` / ``GENAI_STUDIO_RPM``) — a Team shares the
            cap but does not create it, and warns if the client is uncapped.
        model: default model id for agents that don't override it.
        tracer: the ONE inner tracer; each agent gets its own :class:`ScopedTracer`
            over it so the run is attributable. ``None`` → silent (``NullTracer``).
        guards: tree-wide before/after-tool policy, prepended to every agent's
            own guards. Pass a single shared instance (e.g. one ``BudgetGuard``)
            to get a genuinely tree-wide counter.
        system_prefix: house rules prepended to every agent's system prompt
            (the AGENTS.md pattern — shared conventions stated once).
        cancel: a shared cancel token forwarded into supervisor/pipeline runs.
        budget: a default :class:`Budget` for supervisor/pipeline runs (each
            delegation/stage still runs under a fresh copy, per ``as_tool``).
    """

    client: Any
    model: str | None = None
    tracer: Any = None
    guards: Sequence = ()
    system_prefix: str | None = None
    cancel: Any = None
    budget: Any = None
    max_depth: int | None = None

    def __post_init__(self):
        if self.tracer is None:
            self.tracer = NullTracer()
        self.guards = tuple(self.guards)
        self._warn_if_unpaced()

    def _warn_if_unpaced(self) -> None:
        """Honesty about the headline guarantee: a shared client paces the tree
        only if that client actually has a cap. Reaching in read-only for ``_rl``
        is enough to tell; clients without the attribute (test doubles) are skipped.
        """
        rl = getattr(self.client, "_rl", None)
        if rl is None and not hasattr(self.client, "_rl"):
            return  # not a gateway client — the limiter concept doesn't apply
        if getattr(rl, "min_interval", 0) <= 0:    # _NullLimiter or RateLimiter(0)
            warnings.warn(
                "Team's client has no rate cap (no RateLimiter and GENAI_STUDIO_RPM "
                "unset), so the tree is UNPACED — the gateway silently drops bursts. "
                "Build the client with RateLimiter(20) or set GENAI_STUDIO_RPM=20.",
                stacklevel=3)

    # ── builders ─────────────────────────────────────────────────────────────
    def agent(self, name: str, *, system: str | None = None, tools: Sequence = (),
              model: str | None = None, depth: int = 0, guards: Sequence = (),
              **kwargs) -> Agent:
        """Build an :class:`Agent` wired to this team.

        Shares the team ``client``/``model``, gets a :class:`ScopedTracer`
        (``name`` at ``depth``) over the shared inner tracer, enforces
        ``team.guards`` then any per-agent ``guards``, and is prefixed with the
        team ``system_prefix``. Extra ``kwargs`` (``max_steps``, ``output_schema``,
        …) pass straight through to :class:`Agent`.

        A leaf agent enforces nothing on its own — the shared-client / rate-limit
        invariant is checked only when you compose with ``supervisor``/``pipeline``.
        """
        if not name:
            raise ValueError("Team.agent() requires a name (it scopes the trace).")
        _reject_reserved(kwargs, _AGENT_RESERVED, "Team.agent()")
        return Agent(
            client=self.client,
            model=model or self.model,
            name=name,
            system=self._system(system),
            tools=tuple(tools),
            tracer=ScopedTracer(self.tracer, name, depth),
            guards=(*self.guards, *guards),
            **kwargs,
        )

    def supervisor(self, system: str, workers: Sequence[Agent], *,
                   name: str = "supervisor", depth: int = 0,
                   extra_tools: Sequence = (), guards: Sequence = (),
                   max_depth: int | None = None, effort: str | None = None,
                   delegation_guide: bool = True, **kwargs) -> Agent:
        """Build a coordinator that delegates to ``workers`` (agents-as-tools).

        Like :func:`~genai_studio.agents.orchestrate.supervisor`, but *enforces*
        the shared client (raises on a worker built from a different one — the
        rate-limiter guarantee), re-scopes each direct worker one level deeper than
        the manager, and applies the team tracer/guards/prefix to the manager.
        Depth is single-level (see the module docstring): a worker that is itself a
        composite does not re-deepen its grandchildren. ``max_depth`` (falling back
        to the team's) caps delegation nesting; ``effort`` caps fan-out (see
        :func:`~genai_studio.agents.orchestrate.supervisor`).
        """
        workers = self._checked_workers(workers)
        _reject_reserved(kwargs, _SUPERVISOR_RESERVED, "Team.supervisor()")
        deeper = [self._rescope(w, depth + 1) for w in workers]
        # _checked_workers already guaranteed same client, so the mismatch
        # warnings.warn inside _supervisor can't fire on this path — the
        # load-bearing client check is the raise above, not that warning.
        return _supervisor(
            self.client, self._system(system), deeper,
            model=self.model, name=name, extra_tools=extra_tools,
            tracer=ScopedTracer(self.tracer, name, depth),
            cancel=self.cancel, budget=self.budget,
            max_depth=max_depth if max_depth is not None else self.max_depth,
            effort=effort, delegation_guide=delegation_guide,
            guards=(*self.guards, *guards), **kwargs,
        )

    def pipeline(self, stages: Sequence[Agent], *, depth: int = 0,
                 gate: Callable | None = None) -> Callable:
        """Compose ``stages`` into a fixed sequential workflow (enforces the
        shared client; re-scopes each stage at ``depth``). Returns
        ``run(prompt, *, budget=None)`` which defaults ``budget`` to the team's.
        """
        stages = self._checked_workers(stages)
        scoped = [self._rescope(s, depth) for s in stages]
        run = _pipeline(scoped, cancel=self.cancel, gate=gate)
        team_budget = self.budget

        def team_run(prompt, *, budget=None):
            return run(prompt, budget=budget if budget is not None else team_budget)

        team_run.stages = scoped
        return team_run

    # ── internals ────────────────────────────────────────────────────────────
    def _system(self, system: str | None) -> str | None:
        # idempotent: don't double-stamp the prefix onto a prompt that already
        # carries it (e.g. a prompt fed back through a builder).
        if self.system_prefix and system and system.startswith(self.system_prefix):
            return system
        parts = [p for p in (self.system_prefix, system) if p]
        return "\n\n".join(parts) if parts else None

    def _checked_workers(self, workers: Sequence[Agent]) -> list[Agent]:
        workers = list(workers)
        if not workers:
            raise ValueError("need at least one agent.")
        for w in workers:
            if not hasattr(w, "as_tool"):
                raise TypeError("Team compositions take Agents (each needs .as_tool()).")
            if getattr(w, "client", None) is not self.client:
                raise ValueError(
                    f"agent {getattr(w, 'name', '?')!r} was built from a different "
                    "ModelClient than this Team. Build every agent with this team's "
                    ".agent() so the whole tree shares one rate-limiter (the gateway "
                    "drops bursts).")
        return workers

    def _rescope(self, agent: Agent, depth: int) -> Agent:
        """Return a copy of ``agent`` scoped at ``depth`` over the team's inner
        tracer, named by the agent so it's attributed even if it was built outside
        ``team.agent()`` (any named, same-client agent). An unnamed agent with no
        existing scope can't be attributed and is returned unchanged."""
        tr = getattr(agent, "tracer", None)
        name = getattr(agent, "name", None) or (tr.agent if isinstance(tr, ScopedTracer) else None)
        if not name:
            return agent
        return replace(agent, tracer=ScopedTracer(self.tracer, name, depth))


def _reject_reserved(kwargs: dict, reserved, where: str) -> None:
    """Raise a Team-scoped error for a kwarg the team controls, instead of letting
    it surface as a raw ``got multiple values for ...`` TypeError from deep inside
    ``Agent()`` / ``orchestrate``."""
    clash = [k for k in reserved if k in kwargs]
    if clash:
        raise TypeError(
            f"{where} controls {', '.join(clash)} for the whole team — set "
            f"{'them' if len(clash) > 1 else 'it'} on Team(...), not per builder.")
