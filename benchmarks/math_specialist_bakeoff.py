"""Per-role bake-off: which model is the best `math_specialist`?

Picks the specialist by DATA, not extrapolation from the general routing study. Each candidate model
gets the math_specialist system prompt + CAS tools and solves exact-math problems (4x4 determinants,
definite integrals, big exact arithmetic) that are hard in-head but trivial for the tools. We score
two things — because the concern with a cheap model (llama4) is that it ANSWERS FROM MEMORY instead
of grounding:
  - accuracy  — exact match to the sympy gold
  - tool-use  — did it actually CALL a CAS tool (not just answer)?
  - tokens    — cost

  export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
  python benchmarks/math_specialist_bakeoff.py --candidates qwen2.5:72b,llama4:latest,llama3.3:70b --n 24
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from math_grade import _last_boxed                              # noqa: E402

import sympy                                                    # noqa: E402
from genai_studio import GenAIStudio                            # noqa: E402
from genai_studio.agents import Agent, NullTracer               # noqa: E402
from genai_studio.agents.client import GenAIStudioClient        # noqa: E402
from genai_studio.agents.orchestrate import _math_worker        # noqa: E402  (the real specialist)

_X = sympy.Symbol("x")
_CAS = {"symbolic_math", "verify_math", "matrix_op", "prove", "solve_constraints", "calculator"}


def _client(model, timeout):
    return GenAIStudioClient(GenAIStudio(validate_model=False, timeout=timeout), default_model=model)


def _guard(fn, wall):
    box: dict = {}
    def run():
        try:
            box["r"] = fn()
        except Exception as e:
            box["r"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(wall)
    return box.get("r")


def _gen(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = ("det", "integral", "arith")[i % 3]
        if kind == "det":
            M = [[rng.randint(-5, 5) for _ in range(4)] for _ in range(4)]
            ans = sympy.Matrix(M).det()
            prob = f"Compute the determinant of the 4x4 matrix {M}."
        elif kind == "integral":
            P = sum(rng.randint(-4, 4) * _X**j for j in range(4))
            b = rng.randint(2, 5)
            ans = sympy.integrate(P, (_X, 0, b))
            prob = f"Compute the EXACT value of the definite integral of ({sympy.sstr(P)}) dx from x=0 to x={b}."
        else:
            a, e = rng.randint(21, 99), rng.randint(3, 4)
            ans = a**e
            prob = f"Compute the exact value of {a}^{e}."
        out.append({"problem": prob + " Put the final answer in \\boxed{}.", "ans": ans, "kind": kind})
    return out


def _grade(text, ans):
    if not isinstance(text, str):
        return False
    b = _last_boxed(text)
    if b is None:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        b = lines[-1] if lines else ""
    try:
        got = sympy.sympify(b.replace("^", "**").replace(",", "").strip("$ "))
        return bool(sympy.simplify(got - ans) == 0)
    except Exception:
        return False


def _used_cas(result):
    for m in getattr(result, "messages", []):
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "name", None) in _CAS:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="qwen2.5:72b,llama4:latest,llama3.3:70b")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--wall", type=float, default=120.0)
    args = ap.parse_args()

    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    probs = _gen(args.n, args.seed)
    print(f"MATH-SPECIALIST BAKE-OFF — candidates={cands} n={args.n}\n", flush=True)

    stat = {c: {"correct": 0, "tooled": 0, "tokens": 0, "n": 0} for c in cands}
    for i, pr in enumerate(probs, 1):
        line = f"[{i:>2}/{args.n}] {pr['kind']:8}"
        for c in cands:
            client = _client(c, args.timeout)

            def run():
                a = _math_worker(client, c, NullTracer())     # the REAL math_specialist (greedy math tools)
                return a.run(pr["problem"])
            res = _guard(run, args.wall)
            ok = tooled = False
            toks = 0
            if res is not None and not isinstance(res, BaseException):
                ok = _grade(getattr(res, "text", ""), pr["ans"])
                tooled = _used_cas(res)
                toks = getattr(getattr(res, "usage", None), "total_tokens", 0) or 0
            s = stat[c]
            s["n"] += 1
            s["correct"] += ok
            s["tooled"] += tooled
            s["tokens"] += toks
            line += f"  {c.split(':')[0][:8]:8}={'✓' if ok else '·'}{'T' if tooled else '-'}"
        print(line, flush=True)

    print("\n=== RESULTS (accuracy | tool-use rate | avg tokens) ===")
    for c in cands:
        s = stat[c]
        n = s["n"] or 1
        print(f"  {c:16}: acc {s['correct']}/{s['n']} = {s['correct']/n:5.0%} | "
              f"tool-use {s['tooled']/n:4.0%} | avg tokens {s['tokens']//n}")


if __name__ == "__main__":
    main()