"""
Shared harness for replicating agent benchmarks with the genai_studio.agents SDK.

A `Task` bundles a prompt, an optional per-task workspace `setup`, and a
deterministic `grade` function. `run_suite` runs each task with a fresh agent,
captures a JSONL trace, grades the result, and prints an aggregate report.

This is the SDK's own "really test it" harness — faithful, offline-gradeable
slices of DSBench, DataSciBench, and ReAct (see the per-suite modules).
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from genai_studio import GenAIStudio
from genai_studio.agents import GenAIStudioClient, ReActClient

DEFAULT_MODEL = os.getenv("GENAI_STUDIO_MODEL", "qwen2.5:72b")
TRACE_DIR = os.path.join(os.path.dirname(__file__), "_traces")


def make_client(*, react: bool = False, model: str | None = None, **kw):
    """A GenAIStudioClient (native tools) or a ReActClient wrapping it."""
    studio = GenAIStudio(validate_model=False)
    base = GenAIStudioClient(studio, default_model=model or DEFAULT_MODEL, **kw)
    return ReActClient(base) if react else base


@dataclass
class Task:
    id: str
    prompt: str
    grade: Callable          # (result, workdir) -> (score: float in [0,1], detail: str)
    setup: Callable | None = None   # (workdir) -> None: write input files
    meta: dict = field(default_factory=dict)


@dataclass
class TaskOutcome:
    task_id: str
    score: float
    passed: bool
    detail: str
    stopped: str
    tokens: int | None
    elapsed: float
    error: str | None = None


def run_suite(name: str, tasks: list[Task], agent_factory: Callable, *,
              runs: int = 1, verbose: bool = True, quiet: bool = False,
              done_ids: set | None = None, on_outcome: Callable | None = None) -> dict:
    """Run a task suite. `agent_factory(task, workdir)` returns a fresh Agent.

    Returns a report dict with per-task outcomes and aggregate metrics
    (mean score, success rate = fraction with score == 1.0). ``quiet`` suppresses
    the per-task lines (for large sweeps).

    Resumability: tasks whose id is in ``done_ids`` are skipped (already
    checkpointed); ``on_outcome(outcome)`` is called after each completed task so
    the caller can persist it (per-question checkpoint).
    """
    os.makedirs(TRACE_DIR, exist_ok=True)
    outcomes: list[TaskOutcome] = []
    cwd0 = os.getcwd()

    print(f"\n{'=' * 70}\n{name}  —  tasks={len(tasks)}  runs={runs}\n{'=' * 70}")
    for task in tasks:
        if done_ids and task.id in done_ids:
            continue  # already completed in a prior (interrupted) run
        for r in range(runs):
            workdir = os.path.join(TRACE_DIR, f"{name}_{task.id}_r{r}")
            os.makedirs(workdir, exist_ok=True)
            if task.setup:
                task.setup(workdir)
            t0 = time.time()
            try:
                os.chdir(workdir)  # relative file I/O in agent code lands here
                agent = agent_factory(task, workdir)
                result = agent.run(task.prompt)
                os.chdir(cwd0)
                score, detail = task.grade(result, workdir)
                outcomes.append(TaskOutcome(
                    task_id=task.id, score=float(score), passed=(score >= 1.0),
                    detail=detail, stopped=result.stopped,
                    tokens=result.usage.total_tokens, elapsed=time.time() - t0))
            except Exception as exc:
                os.chdir(cwd0)
                outcomes.append(TaskOutcome(
                    task_id=task.id, score=0.0, passed=False,
                    detail="EXC", stopped="error", tokens=None,
                    elapsed=time.time() - t0, error=f"{type(exc).__name__}: {exc}"))
                if verbose:
                    traceback.print_exc()
            o = outcomes[-1]
            if on_outcome is not None:
                on_outcome(o)  # per-question checkpoint
            if not quiet:
                mark = "✅" if o.passed else ("🟡" if o.score > 0 else "❌")
                run_tag = f" r{r}" if runs > 1 else ""
                print(f"  {mark} {task.id}{run_tag:<4} score={o.score:.2f} "
                      f"[{o.stopped}] {o.elapsed:4.0f}s — {o.detail}"
                      + (f"  ERR {o.error}" if o.error else ""))

    mean = sum(o.score for o in outcomes) / len(outcomes) if outcomes else 0.0
    sr = sum(1 for o in outcomes if o.passed) / len(outcomes) if outcomes else 0.0
    print(f"{'-' * 70}\n{name}: mean_score={mean:.3f}  success_rate={sr:.3f}  "
          f"(n={len(outcomes)})\n")
    return {"name": name, "mean_score": mean, "success_rate": sr, "outcomes": outcomes}


def extract_final_text(result) -> str:
    """The model's final natural-language answer (last assistant text)."""
    if result.text and result.text.strip():
        return result.text
    for msg in reversed(result.messages):
        if getattr(msg, "role", None) == "assistant" and getattr(msg, "content", None):
            return msg.content
    return ""
