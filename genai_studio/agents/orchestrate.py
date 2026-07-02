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

# The empirical decision knowledge (from this project's benchmark studies), so the ORCHESTRATOR
# model can route well itself rather than relying on a hard-coded router. Names the worker to pick
# by task type, the sampling that wins, and when tools/verification actually help. Appended to the
# manager prompt by ``supervisor(routing_guide=True)``; also usable standalone.
ROUTING_GUIDE = """\
Routing knowledge (measured on this gateway — use it to choose the worker, tools, and settings; if a
worker's name/description doesn't match a role below, fall back to the strongest general worker):

PICK THE MODEL/WORKER BY TASK TYPE
- Arithmetic / grade-school math: a fast general model (llama4 ~93%, qwen2.5:72b ~90% on GSM8K) — the
  cheapest that suffices.
- Facts / short factual answers / anything an unaided model often gets wrong: a strong general model
  (qwen2.5:72b) WITH grounding/retrieval tools — grounding clearly helps here.
- Hard multi-step reasoning / proofs: a reasoning model (e.g. deepseek-r1) at GREEDY decoding.
- Multi-hop questions: grounded retrieval + a strong general model.
- Best all-round default when unsure: qwen2.5:72b.

SAMPLING
- Reasoning models (deepseek-r1 / qwq / qwen3) are BEST at GREEDY (temperature 0) on agentic/tool-use
  tasks — do NOT raise their temperature for these.

TOOLS & VERIFICATION
- Exact math (arithmetic, algebra, calculus, matrices): COMPUTE with symbolic_math/matrix_op and CHECK
  with verify_math — never do the math in your head.
- To show a claim holds for ALL values: use prove (a sound solver) or lean_check (a proof kernel), and
  trust the certificate, not a guessed answer.
- Grounding/retrieval helps facts and multi-hop; it does NOT help a strong model on math it already
  does well — don't add it there.
- When an answer is CHEAP and INDEPENDENT to check (roots by substitution, a factorization by
  expanding, an inequality by a solver): sample several solutions, DISCARD any that fail the check
  (require it to be COMPLETE, not merely self-consistent), and take the majority of the survivors —
  this reliably turns "some sample is right" into the answer. When checking is as hard as solving
  (open-ended reasoning), skip this — verification adds nothing there."""

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
               delegation_guide: bool = True, routing_guide: bool = True, **agent_kwargs) -> Agent:
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
        routing_guide: append :data:`ROUTING_GUIDE` — the measured decision knowledge (which model/
            worker per task type, greedy sampling for reasoning models, when tools/verification help)
            — so the manager can route well itself instead of guessing. On by default.
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
    sys_prompt = system
    if delegation_guide:
        sys_prompt += "\n\n" + DELEGATION_GUIDE
    if routing_guide:
        sys_prompt += "\n\n" + ROUTING_GUIDE
    if tracer is not None:
        agent_kwargs.setdefault("tracer", tracer)
    from .tools.general import final_answer  # lazy: keep agents-init light
    return Agent(client=client, model=model, name=name,
                 tools=[*worker_tools, *extra_tools, final_answer],
                 system=sys_prompt, guards=guards, **agent_kwargs)


# Benchmark-optimal defaults (routing study): a strong all-round manager, a fast+reliable math model,
# a reasoning model run GREEDY, and a strong grounded research model. Override per gateway.
ROUTED_DEFAULTS = {"manager": "qwen2.5:72b", "math": "qwen2.5:72b",
                   "reasoning": "deepseek-r1:32b", "research": "qwen2.5:72b"}


def _math_worker(client, model, tracer):
    from .tools.general import calculator, final_answer
    from .tools.smt import prove, solve_constraints
    from .tools.symbolic import matrix_op, symbolic_math, verify_math
    system = ("You are the MATH specialist. Solve arithmetic, algebra, calculus, and linear algebra "
              "EXACTLY: COMPUTE with symbolic_math/matrix_op, CHECK every result with verify_math, and "
              "establish universal claims with prove. Never do math in your head. Return the exact answer.")
    return Agent(client=client, model=model, name="math_specialist", system=system, tracer=tracer,
                 tools=[symbolic_math, verify_math, matrix_op, prove, solve_constraints, calculator, final_answer])


def _reasoning_worker(client, model, tracer):
    from .tools.general import calculator, final_answer
    from .tools.symbolic import symbolic_math, verify_math
    system = ("You are the REASONING specialist for hard, multi-step problems. Think carefully step by "
              "step and verify intermediate results with the math tools when you can. Give the final answer.")
    return Agent(client=client, model=model, name="reasoning_specialist", system=system, tracer=tracer,
                 temperature=0.0,          # GREEDY — the routing study's best setting for reasoning models
                 tools=[symbolic_math, verify_math, calculator, final_answer])


def _research_worker(client, model, tracer):
    from .tools.general import final_answer
    from .tools.web import web_search, wikipedia_search
    tools = [web_search, wikipedia_search, final_answer]
    try:                                   # http / academic aren't re-exported; skip if unavailable
        from .tools.http import make_fetch_json, make_http_get
        tools = [make_http_get(), make_fetch_json(), *tools]
    except Exception:
        pass
    try:
        from .tools.academic import arxiv_search, openalex_search
        tools = [arxiv_search, openalex_search, *tools]
    except Exception:
        pass
    system = ("You are the RESEARCH specialist for facts and multi-hop questions. GROUND every claim "
              "with web_search / wikipedia / academic tools — never answer from memory — and report "
              "what you found with sources.")
    return Agent(client=client, model=model, name="research_specialist", system=system, tracer=tracer, tools=tools)


_ROUTED_BUILDERS = {"math": _math_worker, "reasoning": _reasoning_worker, "research": _research_worker}
_ROUTED_DESC = {"math": "math_specialist (exact arithmetic/algebra/calculus/linear-algebra + proofs)",
                "reasoning": "reasoning_specialist (hard multi-step reasoning, runs greedy)",
                "research": "research_specialist (facts + multi-hop, grounded retrieval)"}


def routed_team(client, *, manager_model: str | None = None, include=("math", "reasoning", "research"),
                models: dict | None = None, system: str | None = None, tracer=None,
                **supervisor_kwargs) -> Agent:
    """A :func:`supervisor` whose workers are PRE-WIRED to the benchmark-optimal model + sampling +
    tools for each role — so the manager's :data:`ROUTING_GUIDE` maps onto real specialists it can
    pick, end-to-end.

    Specialists (choose via ``include``): ``math`` (CAS/prove tools), ``reasoning`` (a reasoning model
    at greedy), ``research`` (grounded retrieval). Every worker AND the manager share ``client`` (the
    one-rate-limiter invariant); each worker overrides only its ``model``. Override any model via
    ``models={"reasoning": "qwq:latest", ...}`` or ``manager_model=``.

    Args:
        client: the ONE shared client for the whole team.
        manager_model: model for the coordinator (default: the all-round champion).
        include: which specialists to build.
        models: per-role model overrides merged over :data:`ROUTED_DEFAULTS`.
        system: manager system prompt (a routing-aware default is used if omitted); DELEGATION_GUIDE
            and ROUTING_GUIDE are appended by :func:`supervisor`.
    """
    m = {**ROUTED_DEFAULTS, **(models or {})}
    roles = [r for r in include if r in _ROUTED_BUILDERS]
    if not roles:
        raise ValueError(f"routed_team needs ≥1 known role from {sorted(_ROUTED_BUILDERS)}; got {include}.")
    workers = [_ROUTED_BUILDERS[r](client, m[r], tracer) for r in roles]
    if system is None:
        system = ("You are the coordinator. Your specialists (as tools): "
                  + "; ".join(_ROUTED_DESC[r] for r in roles)
                  + ". Route each subtask to the right specialist, then synthesize their results.")
    return supervisor(client, system, workers, model=manager_model or m["manager"],
                      tracer=tracer, **supervisor_kwargs)


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
