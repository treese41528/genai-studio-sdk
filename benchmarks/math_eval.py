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


def _load(n: int, seed: int) -> list:
    if not os.path.exists(_CACHE):
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            open(_CACHE, "w").write(c.get(_URL).text)
    rows = [json.loads(ln) for ln in open(_CACHE) if ln.strip()]
    import random
    random.Random(seed).shuffle(rows)
    return rows[:n]


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
    args = ap.parse_args()

    client = make_client(model=args.model)
    model = args.model or DEFAULT_MODEL
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    problems = _load(args.n, args.seed)
    print(f"MATH-500 — model={model}  n={len(problems)}  conditions={conditions}\n")

    tally = {c: {"correct": 0, "by_level": {}} for c in conditions}
    for i, r in enumerate(problems, 1):
        gold, level = r["answer"], r.get("level", 0)
        line = f"[{i:>3}/{len(problems)}] L{level} {r.get('subject','')[:14]:14}"
        for c in conditions:
            out = _run(r["problem"], client, model, c)
            ok = is_equiv(out, gold)
            tally[c]["correct"] += ok
            lv = tally[c]["by_level"].setdefault(level, [0, 0])
            lv[0] += ok
            lv[1] += 1
            line += f"  {c}={'✓' if ok else '·'}({extract_answer(out)[:16]})"
        print(line, flush=True)

    print("\n=== RESULTS ===")
    n = len(problems)
    for c in conditions:
        acc = tally[c]["correct"] / n if n else 0
        print(f"  {c:9}: {tally[c]['correct']}/{n} = {acc:.1%}")
    if "bare" in tally and "grounded" in tally:
        lift = (tally["grounded"]["correct"] - tally["bare"]["correct"]) / n
        print(f"  LIFT (grounded − bare): {lift:+.1%}")
    print("\n  by level (correct/total), grounded:" if "grounded" in tally else "")
    if "grounded" in tally:
        for lv in sorted(tally["grounded"]["by_level"]):
            a, b = tally["grounded"]["by_level"][lv]
            print(f"    L{lv}: {a}/{b}")


if __name__ == "__main__":
    main()
