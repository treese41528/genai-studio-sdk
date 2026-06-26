"""
``supervisor`` / ``pipeline`` — the two minimal multi-agent topologies.

Thin factories over the core ``Agent`` (domain-agnostic, so they live in core and
export alongside ``Agent``): no new loop and no new control-flow primitive — they
just compose :meth:`Agent.as_tool` and the existing run loop.

- ``supervisor`` — ONE coordinator LLM that delegates to isolated worker agents
  exposed as tools (agents-as-tools). Workers always return their result to the
  manager, which keeps control. Use when routing is dynamic / LLM-decided.
- ``pipeline``   — a FIXED sequence of agent stages (code orchestration, no
  manager LLM): each stage runs on the previous stage's answer, with an optional
  validation gate that can abort. Use when the workflow is a known DAG.

Default to single-agent; reach for these only when a task genuinely decomposes
into bounded, independent sub-tasks. **Rate-limit invariant:** build every agent
in a team from the SAME ``ModelClient`` so the whole tree shares one process-wide
``RateLimiter`` — the gateway silently drops bursts. ``supervisor`` warns if a
worker's client differs from the manager's.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from dataclasses import replace
from typing import Callable, Sequence

from .agent import Agent, AgentResult, Budget
from .guard import BudgetGuard

DELEGATION_GUIDE = """\
You coordinate specialist sub-agents, each exposed to you as a tool. When you delegate:
- give ONE clear objective and state the exact output format you want back;
- delegate look-up / compute / research subtasks — keep the synthesis yourself;
- never re-delegate something a sub-agent already answered: reuse its result;
- match effort to the task — do not fan out many sub-agents for a simple question.
When you have enough to answer, STOP delegating and give the final answer."""

# Effort-scaling presets (Anthropic's lesson: vague delegation spawned ~50 subagents
# for a trivial query). Each maps a coarse task class to a fan-out ceiling (a
# BudgetGuard capping the MANAGER's own tool calls — its direct delegations — not
# the whole sub-tree), a delegation-depth cap, and a prompt hint, so "match effort
# to the task" is enforced, not just advised. Set effort on each supervisor for a
# multi-level tree, or use Team for one genuinely tree-wide shared guard.
_Effort = namedtuple("_Effort", "max_tool_calls max_depth hint")
EFFORT_PRESETS = {
    "simple": _Effort(
        4, 1, "EFFORT — simple: answer directly or use at most ONE sub-agent; do not fan out."),
    "comparison": _Effort(
        10, 2, "EFFORT — comparison: delegate a FEW focused look-ups (~2-4), then synthesize yourself."),
    "complex": _Effort(
        24, 2, "EFFORT — complex: decompose into bounded subtasks and reuse results; "
               "still avoid over-fanning-out."),
}


def effort_policy(level: str) -> _Effort:
    """Look up an effort preset (``simple`` / ``comparison`` / ``complex``)."""
    try:
        return EFFORT_PRESETS[level]
    except KeyError:
        raise ValueError(f"unknown effort {level!r}; choose from {sorted(EFFORT_PRESETS)}.")


def supervisor(client, system: str, workers: Sequence[Agent], *,
               model: str | None = None, name: str = "supervisor",
               extra_tools: Sequence = (), tracer=None, cancel=None,
               budget=None, max_depth: int | None = None, effort: str | None = None,
               delegation_guide: bool = True, **agent_kwargs) -> Agent:
    """Build a coordinator :class:`Agent` that delegates to ``workers`` (agents-as-tools).

    Each worker is exposed via :meth:`Agent.as_tool` (isolated context, returns a
    condensed result with its sources). Worker tool names are de-duplicated
    automatically. The manager keeps control and finishes by calling
    ``final_answer`` or replying in plain text.

    The manager's ``system`` prompt should *name each worker and what it's for* —
    worker tool descriptions are generic, so routing quality comes from the prompt
    (this is the prescriptive-delegation lesson; ``DELEGATION_GUIDE`` is appended).

    Args:
        client: the ModelClient — share it with every worker (one rate-limiter).
        system: the manager's system prompt (its role + how to use each worker).
        workers: the specialist agents to expose as tools (≥ 1; give them ``name``s).
        model: model id for the manager (defaults to the client's default).
        name: the manager agent's name.
        extra_tools: extra tools the manager may call directly (besides workers).
        tracer: tracer for the manager (workers keep their own).
        cancel: a shared cancel token forwarded into every worker run.
        budget: optional :class:`Budget`; each worker delegation runs under a fresh
            copy of it (per ``as_tool``) so a runaway worker degrades gracefully.
        max_depth: cap on delegation NESTING beneath the manager (forwarded into
            every worker's ``as_tool``) — the unbounded-recursion backstop.
        effort: ``"simple"`` / ``"comparison"`` / ``"complex"`` — a preset that
            caps this manager's own fan-out (a ``BudgetGuard`` on the manager's
            direct tool calls — NOT propagated into sub-supervisors), sets a
            sensible ``max_depth`` (unless given), and appends a prompt hint, so
            "match effort to the task" is enforced, not merely advised.
        delegation_guide: append the prescriptive ``DELEGATION_GUIDE`` to ``system``.
        **agent_kwargs: forwarded to the manager :class:`Agent` (e.g. ``max_steps``).
    """
    workers = list(workers)
    if not workers:
        raise ValueError("supervisor() needs at least one worker agent.")
    if not all(hasattr(w, "as_tool") for w in workers):
        raise TypeError("supervisor() workers must be Agents (each needs .as_tool()).")
    for w in workers:
        if getattr(w, "client", None) is not client:
            warnings.warn(
                "supervisor(): a worker uses a different ModelClient than the "
                "manager. Build every agent in the team from the SAME client so the "
                "whole tree shares one rate-limiter (the gateway drops bursts).",
                stacklevel=2)
            break

    guards = list(agent_kwargs.pop("guards", ()) or ())
    if effort:
        eff = effort_policy(effort)
        if max_depth is None:
            max_depth = eff.max_depth
        guards.append(BudgetGuard(max_tool_calls=eff.max_tool_calls))  # caps the manager's own fan-out
        system = f"{system}\n\n{eff.hint}"

    # Reserve the manager's own tool names so a worker can't shadow final_answer/
    # finish (which the loop treats as terminal) or an extra_tool.
    reserved = {"final_answer", "finish",
                *(getattr(t, "name", getattr(t, "__name__", None)) for t in extra_tools)}
    worker_tools = _unique_as_tools(workers, cancel=cancel, budget=budget,
                                    reserved=reserved, max_depth=max_depth)
    sys_prompt = system + ("\n\n" + DELEGATION_GUIDE if delegation_guide else "")
    if tracer is not None:
        agent_kwargs.setdefault("tracer", tracer)
    from .tools.general import final_answer  # lazy: keep agents-init light
    return Agent(client=client, model=model, name=name,
                 tools=[*worker_tools, *extra_tools, final_answer],
                 system=sys_prompt, guards=guards, **agent_kwargs)


def pipeline(stages: Sequence[Agent], *, cancel=None,
             gate: Callable[[AgentResult], bool] | None = None) -> Callable:
    """Compose ``stages`` into a fixed sequential workflow (pure code orchestration).

    Returns ``run(prompt, *, budget=None) -> AgentResult``: each stage runs on the
    previous stage's answer (result-only context — the token-cheap default), and
    the returned result carries the de-duplicated union of every stage's
    ``sources`` (the stage's own result object is not mutated). The pipeline stops
    early if a stage does not finish cleanly (``stopped != "final"``) or produces
    no non-whitespace text — it won't feed a failed/truncated/blank result into the
    next stage. An optional ``gate(result) -> bool`` runs after each clean stage;
    returning ``False`` aborts the pipeline there. Each stage runs under a fresh
    copy of any ``budget`` (independent counters), like ``as_tool``.

    Args:
        stages: the agents to run in order (≥ 1).
        cancel: a shared cancel token forwarded into every stage run.
        gate: optional per-stage validation; ``False`` stops the pipeline early.
    """
    stages = list(stages)
    if not stages:
        raise ValueError("pipeline() needs at least one stage.")

    def run(prompt, *, budget=None) -> AgentResult:
        text, last, collected = prompt, None, []
        for stage in stages:
            # fresh per-stage budget copy (independent counters), like as_tool.
            b = replace(budget) if isinstance(budget, Budget) else budget
            last = stage.run(text, cancel=cancel, budget=b)
            collected.extend(last.sources or [])
            text = last.text
            # stop on any non-clean stop (error/cancelled/budget/max_steps) or blank
            # output — don't feed a failed/truncated/empty result forward — or a
            # rejecting gate.
            if last.stopped != "final" or not (last.text or "").strip():
                break
            if gate is not None and not gate(last):
                break
        return replace(last, sources=_dedupe_sources(collected))

    run.stages = stages
    return run


def _unique_as_tools(workers, *, cancel, budget=None, reserved=None, max_depth=None):
    """Expose each worker as a tool with a name unique across workers AND the
    manager's reserved/extra tool names (so no worker shadows ``final_answer`` etc.)."""
    seen: set[str] = {n for n in (reserved or ()) if n}
    tools = []
    for i, w in enumerate(workers):
        base = w.name or f"worker{i + 1}"
        nm, k = base, 2
        while nm in seen:
            nm, k = f"{base}_{k}", k + 1
        seen.add(nm)
        tools.append(w.as_tool(name=nm, cancel=cancel, budget=budget, max_depth=max_depth))
    return tools


def _dedupe_sources(sources) -> list:
    """Ordered, de-duplicated union of sources (same key policy as the loop)."""
    seen, out = set(), []
    for s in sources or []:
        key = s.id or s.url or s.title or id(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
