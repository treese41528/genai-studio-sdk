"""MATH-500 eval — does grounding a non-frontier model with the math tools improve accuracy?

Runs each competition problem in two conditions and grades by sympy answer-equivalence:
  - **bare**      — the model reasons and boxes an answer (no tools). The baseline.
  - **grounded**  — the agent has the exact-math tools (symbolic_math / verify_math / matrix_op /
                    prove / solve_constraints) + calculator, and is told to COMPUTE and VERIFY with
                    them. Tests whether grounding reduces math hallucination.

The lift (grounded − bare) is the headline. Sequential, RPM-paced (one gateway process).

    export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
    python benchmarks/math_eval.py --model qwen2.5:72b --n 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from _bench import DEFAULT_MODEL, make_client                      # noqa: E402
from math_grade import extract_answer, is_equiv                    # noqa: E402

from genai_studio.agents import Agent, NullTracer                  # noqa: E402

_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
_CACHE = os.path.join(os.path.dirname(__file__), "_data", "benchmarks", "math500.jsonl")

BARE_SYSTEM = ("You are a careful mathematician. Solve the problem and put your FINAL answer in "
               "\\boxed{...}. Show brief reasoning, then the boxed answer.")
GROUNDED_SYSTEM = ("You are a careful mathematician with exact-math tools. COMPUTE and VERIFY with the "
                   "tools — never do arithmetic, algebra, or calculus in your head: use symbolic_math "
                   "to solve/simplify/integrate, matrix_op for linear algebra, verify_math to check an "
                   "equality, and prove for a claim that must hold for all values. Then put your FINAL "
                   "answer in \\boxed{...}.")


def _load(n: int, seed: int, stratify: str | None = None) -> list:
    if not os.path.exists(_CACHE):
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            open(_CACHE, "w").write(c.get(_URL).text)
    rows = [json.loads(ln) for ln in open(_CACHE) if ln.strip()]
    import random
    random.Random(seed).shuffle(rows)
    if not stratify:
        return rows[:n]
    from collections import defaultdict                        # balanced round-robin across classes
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(stratify, "?")].append(r)
    keys = sorted(buckets)
    out: list = []
    i = 0
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out[:n]


def _grounded_tools():
    from genai_studio.agents.tools import calculator, final_answer
    from genai_studio.agents.tools.symbolic import matrix_op, symbolic_math, verify_math
    from genai_studio.agents.tools.smt import prove, solve_constraints
    return [symbolic_math, verify_math, matrix_op, prove, solve_constraints, calculator, final_answer]


def _bare_tools():
    from genai_studio.agents.tools import final_answer
    return [final_answer]


def _run(problem: str, client, model, condition: str) -> str:
    tools = _grounded_tools() if condition == "grounded" else _bare_tools()
    system = GROUNDED_SYSTEM if condition == "grounded" else BARE_SYSTEM
    agent = Agent(client=client, model=model, tools=tools, system=system,
                  tracer=NullTracer(), max_steps=12)
    try:
        return agent.run(problem).text or ""
    except Exception as e:                                          # a crashed run counts as wrong
        return f"[error: {e}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="bare,grounded")
    ap.add_argument("--stratify", default=None, help="balance the sample by a field, e.g. 'subject'")
    args = ap.parse_args()

    client = make_client(model=args.model)
    model = args.model or DEFAULT_MODEL
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    problems = _load(args.n, args.seed, stratify=args.stratify)
    strat = f"  stratify={args.stratify}" if args.stratify else ""
    print(f"MATH-500 — model={model}  n={len(problems)}  conditions={conditions}{strat}\n")

    def _fresh():
        return {"correct": 0, "by_level": {}, "by_subject": {}}
    tally = {c: _fresh() for c in conditions}
    for i, r in enumerate(problems, 1):
        gold, level, subj = r["answer"], r.get("level", 0), r.get("subject", "?")
        line = f"[{i:>3}/{len(problems)}] L{level} {subj[:16]:16}"
        for c in conditions:
            out = _run(r["problem"], client, model, c)
            ok = is_equiv(out, gold)
            tally[c]["correct"] += ok
            for key, val in (("by_level", level), ("by_subject", subj)):
                cell = tally[c][key].setdefault(val, [0, 0])
                cell[0] += ok
                cell[1] += 1
            line += f"  {c}={'✓' if ok else '·'}({extract_answer(out)[:14]})"
        print(line, flush=True)

    n = len(problems)
    print("\n=== RESULTS ===")
    for c in conditions:
        print(f"  {c:9}: {tally[c]['correct']}/{n} = {tally[c]['correct']/n:.1%}" if n else c)
    if "bare" in tally and "grounded" in tally and n:
        print(f"  LIFT (grounded − bare): {(tally['grounded']['correct']-tally['bare']['correct'])/n:+.1%}")

    for dim, label in (("by_subject", "subject"), ("by_level", "level")):
        keys = sorted(set().union(*[set(tally[c][dim]) for c in conditions]))
        print(f"\n  by {label} ({' vs '.join(conditions)}):")
        for k in keys:
            cells = "  ".join(f"{c}={tally[c][dim].get(k,[0,0])[0]}/{tally[c][dim].get(k,[0,0])[1]}"
                              for c in conditions)
            print(f"    {str(k):22} {cells}")


if __name__ == "__main__":
    main()
