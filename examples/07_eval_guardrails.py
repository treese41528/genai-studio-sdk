"""Evaluation & guardrails.

Budgets cap a run; JsonlTracer captures the trace; then we evaluate:
- Level 1 (deterministic unit tests): assertions on the output AND the trace.
- Level 2 (LLM-as-judge stub): a model scores the answer against a rubric.

Per the evals literature, Level 1 catches most regressions cheaply; the judge
is a starting point you align to a domain expert's critiques.

Run: python examples/07_eval_guardrails.py
"""

from __future__ import annotations

import json

from genai_studio.agents import Agent, Budget, NullTracer, tool
from genai_studio.agents.trace import JsonlTracer
from _common import make_client


@tool
def add(a: int, b: int) -> str:
    "Add two integers.\n\nArgs:\n    a: x.\n    b: y."
    return str(a + b)


CASES = [
    {"q": "What is 19 + 23? Use the add tool, then state the number.", "must_contain": "42"},
    {"q": "What is 100 + 250? Use the add tool, then state the number.", "must_contain": "350"},
]


def level1(case, result, trace_path) -> None:
    """Deterministic assertions — fast, cheap, run on every change."""
    assert result.stopped in ("final", "max_steps"), f"did not finish: {result.stopped}"
    assert case["must_contain"] in result.text, f"missing {case['must_contain']!r}"
    events = [json.loads(line) for line in open(trace_path)]
    assert any(e["type"] == "ToolCallEvent" for e in events), "no tool was used"


def level2_judge(client, question, answer) -> dict:
    """LLM-as-judge STUB. Replace the rubric with critiques from a domain expert."""
    prompt = (
        f"Question: {question}\nAnswer: {answer}\n\n"
        "Did the answer correctly and clearly resolve the question? "
        'Reply with ONLY JSON: {"pass": true/false, "reason": "<one line>"}'
    )
    from genai_studio.agents import Message
    resp = client.complete([Message.user(prompt)])
    try:
        start = resp.text.find("{")
        return json.loads(resp.text[start:resp.text.rfind("}") + 1])
    except Exception:
        return {"pass": None, "reason": f"unparseable judge output: {resp.text[:80]}"}


if __name__ == "__main__":
    client = make_client()
    for i, case in enumerate(CASES):
        path = f"eval_run_{i}.jsonl"
        agent = Agent(client=client, tools=[add], tracer=JsonlTracer(path),
                      system="Use the add tool, then state the number.")
        result = agent.run(case["q"], budget=Budget(max_steps=4, max_tokens=20_000))
        try:
            level1(case, result, path)
            verdict = level2_judge(client, case["q"], result.text)
            print(f"case {i}: L1 PASS | judge={verdict.get('pass')} — {verdict.get('reason')}")
        except AssertionError as e:
            print(f"case {i}: L1 FAIL — {e}")
