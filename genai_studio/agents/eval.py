"""
``eval`` — a tiny, JsonlTracer-backed evaluation harness.

One lucky pass tells you little; agents are stochastic. This harness runs each
case ``k`` times and reports *reliability*, layering the two grading levels from
the evals literature:

- **L1 — end-state grading** (``Case.check``): a deterministic function of the
  ``AgentResult`` (and, if it wants, the JSONL trace) — fast, cheap, run on every
  change. This is where most regressions are caught.
- **L2 — LLM-as-judge** (``judge=llm_judge(...)``): a model scores the answer
  against a rubric — a starting point you align to a domain expert's critiques.

and three reliability views over the ``k`` runs:

- **pass^k** — the case passes only if *all* k runs pass (the tau-bench
  reliability metric: would you trust it k times in a row?).
- **pass@k** — the case passes if *any* of k runs pass (the capability ceiling).
- **consistency** — the fraction of runs that agree on the majority answer
  (self-consistency; a cheap stand-in for semantic-entropy stability).

``JsonlTracer`` is the substrate: each run streams its trace to a file the grader
can assert on (tool was used, no retry storm, …). The harness owns the tracer's
lifetime so a big sweep never leaks file handles.

    from genai_studio.agents import Agent, JsonlTracer, NullTracer
    from genai_studio.agents.eval import Case, evaluate, contains, llm_judge

    cases = [Case("add", "What is 19 + 23? Use add, then state it.", check=contains("42"))]
    def factory(case, tracer):           # the harness hands you the per-run tracer
        return Agent(client=client, tools=[add], tracer=tracer,
                     system="Use the add tool, then state the number.")

    report = evaluate(factory, cases, k=5, judge=llm_judge(client))
    print(report.summary())
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .agent import AgentResult
from .trace import JsonlTracer, NullTracer

# A grader scores one run: (case, result, trace_path) -> (score in [0,1], detail).
# ``trace_path`` is None when no ``trace_dir`` was given.
Grader = Callable[["Case", AgentResult, "str | None"], "tuple[float, str]"]


# ════════════════════════════════════════════════════════════════════════════
# Cases + records
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Case:
    """One evaluation case. ``check`` is the L1 end-state grader (optional)."""
    id: str
    prompt: str
    check: Grader | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class RunRecord:
    run: int
    answer: str
    score: float
    passed: bool
    detail: str
    stopped: str
    tokens: int | None
    trace_path: str | None = None
    error: str | None = None


@dataclass
class CaseReport:
    case_id: str
    runs: list[RunRecord]
    _norm: Callable[[str], str] = field(default=lambda s: s, repr=False)

    @property
    def k(self) -> int:
        return len(self.runs)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.runs if r.passed)

    @property
    def pass_pow_k(self) -> bool:
        """Passes only if EVERY run passed (reliability)."""
        return bool(self.runs) and all(r.passed for r in self.runs)

    @property
    def pass_at_k(self) -> bool:
        """Passes if ANY run passed (capability ceiling)."""
        return any(r.passed for r in self.runs)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.k if self.k else 0.0

    def _safe_norm(self, s: str) -> str:
        """Normalize defensively: a user normalizer that raises on some answer
        must degrade to identity, not poison every report property / summary()."""
        try:
            return self._norm(s or "")
        except Exception:
            return s or ""

    def _answer_counts(self) -> Counter:
        return Counter(self._safe_norm(r.answer or "") for r in self.runs)

    @property
    def majority_answer(self) -> str | None:
        """A representative ORIGINAL answer from the most common (normalized) class."""
        counts = self._answer_counts()
        if not counts:
            return None
        winner, _ = counts.most_common(1)[0]
        for r in self.runs:                       # return the first un-normalized form
            if self._safe_norm(r.answer or "") == winner:
                return r.answer
        return None

    @property
    def consistency(self) -> float:
        """Fraction of runs agreeing on the majority (normalized) answer."""
        counts = self._answer_counts()
        if not counts:
            return 0.0
        return counts.most_common(1)[0][1] / self.k


@dataclass
class EvalReport:
    cases: list[CaseReport]

    def _mean(self, fn) -> float:
        return sum(fn(c) for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def pass_pow_k(self) -> float:
        return self._mean(lambda c: 1.0 if c.pass_pow_k else 0.0)

    @property
    def pass_at_k(self) -> float:
        return self._mean(lambda c: 1.0 if c.pass_at_k else 0.0)

    @property
    def pass_rate(self) -> float:
        return self._mean(lambda c: c.pass_rate)

    @property
    def consistency(self) -> float:
        return self._mean(lambda c: c.consistency)

    def summary(self) -> str:
        k = self.cases[0].k if self.cases else 0
        lines = [
            f"eval: {len(self.cases)} cases × k={k}",
            f"  pass^k     {self.pass_pow_k:6.1%}   (all {k} runs pass)",
            f"  pass@k     {self.pass_at_k:6.1%}   (any run passes)",
            f"  pass-rate  {self.pass_rate:6.1%}   (per-run mean)",
            f"  consistency{self.consistency:6.1%}   (majority-answer agreement)",
        ]
        for c in self.cases:
            mark = "✅" if c.pass_pow_k else ("🟡" if c.pass_at_k else "❌")
            lines.append(f"  {mark} {c.case_id:<24} {c.n_passed}/{c.k} pass  "
                         f"consistency={c.consistency:.0%}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Grader helpers (L1) + an LLM-as-judge (L2)
# ════════════════════════════════════════════════════════════════════════════

def contains(*needles: str, ci: bool = True) -> Grader:
    """L1: pass iff the final answer contains every ``needle`` (case-insensitive)."""
    def check(case, result, trace_path):
        text = result.text or ""
        hay = text.lower() if ci else text
        missing = [n for n in needles if (n.lower() if ci else n) not in hay]
        return (0.0, f"missing {missing}") if missing else (1.0, "ok")
    return check


def used_tool(name: str | None = None) -> Grader:
    """L1: pass iff a tool (optionally a specific one) was actually called.

    Reads ``result.steps`` (in-memory), so it works with or without a trace file.
    """
    def check(case, result, trace_path):
        for step in result.steps:
            for call in getattr(step, "tool_calls", None) or []:
                if name is None or getattr(call, "name", None) == name:
                    return (1.0, f"used {getattr(call, 'name', '?')}")
        return (0.0, f"tool {name or '<any>'} not used")
    return check


def all_of(*graders: Grader) -> Grader:
    """Combine graders with AND semantics: score is the MIN of the sub-scores, so
    the combined grader clears a threshold only when EVERY sub-grader does. (A mean
    would let a failed required grader be averaged away once ``threshold < 1.0`` —
    e.g. ``used_tool`` silently waived — so pass^k could call a case reliable while
    the mandatory tool was never used.)"""
    def check(case, result, trace_path):
        scores, details = [], []
        for g in graders:
            s, d = g(case, result, trace_path)
            scores.append(float(s))
            details.append(d)
        return (min(scores) if scores else 1.0), "; ".join(details)
    return check


def llm_judge(client, rubric: str | None = None, *, model: str | None = None,
              threshold: float = 1.0) -> Grader:
    """L2: an LLM scores the answer. Returns a :data:`Grader`.

    The judge is a STARTING POINT — align its rubric to a domain expert's
    critiques before trusting it. Defaults to a generic correctness rubric.
    Unparseable judge output scores 0.0 with the raw text in the detail (a
    judge failure must not silently pass a case).

    Caveat — judge vs. agent quality: a judge call that *errors* (an unreachable
    or flaky gateway) scores 0.0, which FAILS a run L1 may have passed and
    depresses pass^k. That's conservative (an answer you can't verify isn't
    certified), but it conflates infra flakiness with agent quality, so the error
    is surfaced on ``RunRecord.error`` (prefix ``judge-error:``) — isolate or
    retry the judge against a burst-dropping gateway rather than trusting a low
    pass^k blindly.
    """
    from .client import Message
    rubric = rubric or ("Did the answer correctly and clearly resolve the question?")

    def check(case, result, trace_path):
        prompt = (
            f"Question: {case.prompt}\nAnswer: {result.text}\n\n{rubric}\n"
            'Reply with ONLY JSON: {"pass": true/false, "reason": "<one line>"}'
        )
        try:
            resp = client.complete([Message.user(prompt)], model=model)
            text = resp.text or ""
            obj = _parse_json_obj(text)             # must be exactly ONE JSON object
            ok = _coerce_pass(obj.get("pass"))      # string "false" is a FAIL, not truthy
            return (1.0 if ok else 0.0, f"judge:{obj.get('reason', '?')}")
        except Exception as exc:                    # never let a judge error pass a case
            return (0.0, f"judge-error:{type(exc).__name__}")

    check._threshold = threshold                  # consulted by evaluate()
    return check


# ════════════════════════════════════════════════════════════════════════════
# The harness
# ════════════════════════════════════════════════════════════════════════════

def evaluate(agent_factory: Callable[[Case, Any], Any], cases: Sequence[Case], *,
             k: int = 5, judge: Grader | None = None, trace_dir: str | None = None,
             threshold: float = 1.0, normalize: Callable[[str], str] | None = None,
             on_run: Callable[[Case, RunRecord], None] | None = None,
             done_ids: set | None = None) -> EvalReport:
    """Run each case ``k`` times and report reliability (pass^k / pass@k / consistency).

    Args:
        agent_factory: ``(case, tracer) -> Agent``. The harness creates the per-run
            tracer (a :class:`JsonlTracer` when ``trace_dir`` is set, else a
            :class:`NullTracer`) and closes it after the run — *use the tracer it
            hands you* so the trace lands where the grader can read it.
        cases: the cases to evaluate (each may carry an L1 ``check``).
        k: independent runs per case (k≥1).
        judge: an optional tree-wide L2 grader (e.g. :func:`llm_judge`) applied to
            every run in addition to each case's ``check``.
        trace_dir: where to stream per-run JSONL traces (``{id}_r{run}.jsonl``).
        threshold: a run passes a grader when its score ≥ ``threshold``; a run
            passes overall when it clears EVERY grader (check and judge) AND the
            agent produced a non-blank answer.
        normalize: answer normalizer for the consistency/majority vote
            (default: lowercase + collapse whitespace + strip edge punctuation).
        on_run: called as ``on_run(case, run_record)`` after each run
            (checkpointing). A raising ``on_run`` is swallowed — a checkpoint sink
            must never abort a long sweep.
        done_ids: case ids to skip (already checkpointed in a prior, interrupted
            run) — crash-resumable sweeps, paired with ``on_run``.

    With no grader at all, a run "passes" if it merely finished with a non-blank
    answer — then ``consistency`` is the signal to watch, not pass^k.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    norm = normalize or _default_norm
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    reports: list[CaseReport] = []
    for case in cases:
        if done_ids and case.id in done_ids:
            continue                                    # resumability: already done
        runs: list[RunRecord] = []
        for r in range(k):
            trace_path = os.path.join(trace_dir, f"{case.id}_r{r}.jsonl") if trace_dir else None
            # mode='w': each {id}_r{run}.jsonl is owned by exactly one run, so a
            # re-run truncates instead of appending stale events.
            tracer = JsonlTracer(trace_path, mode="w") if trace_path else NullTracer()
            try:
                rec = _run_once(agent_factory, case, r, tracer, trace_path, judge, threshold)
            finally:
                if hasattr(tracer, "close"):
                    tracer.close()                      # always free the handle
            runs.append(rec)
            if on_run is not None:
                try:
                    on_run(case, rec)
                except Exception:                       # a checkpoint sink must not abort the sweep
                    pass
        reports.append(CaseReport(case.id, runs, _norm=norm))
    return EvalReport(reports)


def _run_once(agent_factory, case, r, tracer, trace_path, judge, threshold) -> RunRecord:
    try:
        agent = agent_factory(case, tracer)
        result = agent.run(case.prompt)
    except Exception as exc:                       # a crashed run is a failed run, not a crashed eval
        return RunRecord(run=r, answer="", score=0.0, passed=False,
                         detail=f"run-error:{type(exc).__name__}: {exc}",
                         stopped="error", tokens=None, trace_path=trace_path,
                         error=f"{type(exc).__name__}: {exc}")

    answer = result.text or ""
    graders: list[Grader] = []
    if case.check is not None:
        graders.append(case.check)
    if judge is not None:
        graders.append(judge)

    scores, details, passed = [], [], answer.strip() != ""
    for g in graders:
        thr = getattr(g, "_threshold", threshold)
        try:
            s, d = g(case, result, trace_path)
        except Exception as exc:                   # a grader bug fails the run, not the eval
            s, d = 0.0, f"grader-error:{type(exc).__name__}: {exc}"
        scores.append(float(s))
        details.append(d)
        passed = passed and (float(s) >= thr)
    score = sum(scores) / len(scores) if scores else (1.0 if passed else 0.0)
    # surface infra faults (grader raised, or llm_judge swallowed a gateway error)
    # so a run scored 0 by flakiness is distinguishable from a quality failure.
    infra = [d for d in details if "-error:" in d]
    return RunRecord(run=r, answer=answer, score=score, passed=bool(passed),
                     detail="; ".join(details) or ("finished" if passed else "blank/aborted"),
                     stopped=result.stopped,
                     tokens=getattr(result.usage, "total_tokens", None),
                     trace_path=trace_path, error="; ".join(infra) or None)


# ── internals ────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\.\,\!\?\:\;\"'`]+|[\s\.\,\!\?\:\;\"'`]+$")


def _default_norm(s: str) -> str:
    return _EDGE_PUNCT.sub("", _WS.sub(" ", (s or "").strip().lower()))


def _coerce_pass(v) -> bool:
    """A judge verdict is truthy ONLY for real affirmatives — the JSON string
    ``"false"``/``"no"`` is a non-empty string and would otherwise pass fail-open."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "pass", "passed")
    return bool(v)


def _parse_json_obj(text: str) -> dict:
    import json
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in judge output: {text[:80]!r}")
    return json.loads(text[start:end + 1])
