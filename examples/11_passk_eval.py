"""Reliability eval: pass^k, not one lucky pass.

Agents are stochastic, so a single green run proves little. The ``eval`` harness
runs each case k times and reports:
  - pass^k   — passes only if ALL k runs pass (the reliability bar a user feels)
  - pass@k   — passes if ANY run passes (the capability ceiling)
  - consistency — how often the k runs agree on the answer (self-consistency)

Two grading levels compose: L1 deterministic checks (``contains`` / ``used_tool``)
and an optional L2 ``llm_judge``. Each run streams a JSONL trace into ``trace_dir``.

Run: python examples/11_passk_eval.py
"""

from __future__ import annotations

from genai_studio.agents import Agent, tool
from genai_studio.agents.eval import (
    Case, all_of, contains, evaluate, llm_judge, used_tool,
)
from _common import make_client


@tool
def add(a: int, b: int) -> str:
    "Add two integers.\n\nArgs:\n    a: x.\n    b: y."
    return str(a + b)


CASES = [
    Case("add-42", "What is 19 + 23? Use the add tool, then state the number.",
         check=all_of(contains("42"), used_tool("add"))),
    Case("add-350", "What is 100 + 250? Use the add tool, then state the number.",
         check=all_of(contains("350"), used_tool("add"))),
]


def factory(case: Case, tracer):
    """The harness hands us the per-run tracer (a JsonlTracer when trace_dir is set).
    Use it so the trace lands where the graders can read it."""
    return Agent(client=client, tools=[add], tracer=tracer,
                 system="Use the add tool, then state the number.")


if __name__ == "__main__":
    client = make_client()
    report = evaluate(
        factory, CASES,
        k=3,                              # three independent runs per case
        # L2: a model double-checks each answer. NOTE: a flaky/unreachable judge
        # scores 0 and fails a run L1 passed (depressing pass^k) — the fault shows
        # up on RunRecord.error (prefix "judge-error:"); drop it to grade L1-only.
        judge=llm_judge(client),
        trace_dir="eval_traces",          # per-run JSONL traces (truncated each run)
    )
    print(report.summary())
