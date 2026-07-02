"""Root-solving benchmark — where CHECK ≪ SOLVE, does PROVE-style verification-filtering win?

Solving a polynomial equation is hard (factor a quartic); CHECKING a proposed root is trivial
(substitute). This is the regime the literature says CAS verification should pay off — unlike
MATH-500 answer-matching, where checking ≈ re-solving. Three arms over k sampled solutions:

  bare@1          one greedy solve
  maj@k           majority-vote the solution SETS (self-consistency)
  prove_filtered  PROVE (arXiv:2410.12608) done properly: DISCARD every sample whose proposed roots
                  don't all satisfy the equation (deterministic sympy substitution — independent of
                  how the model solved), then majority-vote the SURVIVORS. This reshapes the vote
                  distribution instead of rubber-stamping the plurality.

Problems are generated with known integer roots (seeded), so the gold + the substitution check are
exact. Grading is set-equality of the roots (order-independent, sympy).

  export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
  python benchmarks/root_solve_eval.py --n 30 --k 8
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from math_grade import _last_boxed                              # noqa: E402

import sympy                                                    # noqa: E402
from genai_studio import GenAIStudio                            # noqa: E402
from genai_studio.agents import Agent, NullTracer               # noqa: E402
from genai_studio.agents.client import GenAIStudioClient        # noqa: E402
from genai_studio.agents.tools import final_answer              # noqa: E402

SYSTEM = ("You solve polynomial equations. Find ALL real solutions. Show brief work, then put your "
          "final answer as a comma-separated list of every solution in \\boxed{...}, e.g. "
          "\\boxed{-2, 1, 3}. Include every real root exactly once.")
_X = sympy.Symbol("x")


def _client(model, timeout):
    return GenAIStudioClient(GenAIStudio(validate_model=False, timeout=timeout), default_model=model)


def _guard(fn, wall):
    box: dict = {}
    def run():
        try:
            box["r"] = fn()
        except Exception as e:
            box["r"] = f"[error {type(e).__name__}]"
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(wall)
    return box.get("r", "")


def _gen(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        deg = rng.choice([2, 3, 3, 4, 4])                       # hard enough to slip, complete sets reachable
        roots = rng.sample(range(-6, 7), deg)                  # distinct integer roots
        P = sympy.expand(sympy.prod([_X - r for r in roots]))
        pretty = sympy.sstr(P).replace("**", "^").replace("*", "")
        out.append({"P": P, "gold": {sympy.Integer(r) for r in roots},
                    "problem": f"Find ALL real solutions to the equation  {pretty} = 0."})
    return out


def _solve(problem, client, model, temp, wall):
    def run():
        a = Agent(client=client, model=model, tools=[final_answer], system=SYSTEM,
                  tracer=NullTracer(), max_steps=4, temperature=temp)
        return a.run(problem).text or ""
    return _guard(run, wall)


def _extract(text):
    b = _last_boxed(text) or ""
    roots = []
    for p in re.split(r"[,;]", b):
        p = p.strip().strip("{}$ ").replace("x=", "").lstrip("=").strip()
        if not p:
            continue
        try:
            roots.append(sympy.nsimplify(sympy.sympify(p.replace("^", "**"))))
        except Exception:
            pass
    return roots


def _canon(r):
    try:
        return str(sympy.simplify(r))
    except Exception:
        return str(r)


def _key(roots):
    return tuple(sorted(_canon(r) for r in roots))


def _valid(P, roots, tol=1e-9):
    """True iff every proposed root actually satisfies P(x)=0 (the cheap independent check)."""
    if not roots:
        return False
    for r in roots:
        try:
            if abs(complex(P.subs(_X, r))) > tol:
                return False
        except Exception:
            return False
    return True


def _correct(roots, gold):
    return _key(roots) == _key(gold)


def _vote(samples):                                            # samples = list of root-lists
    from collections import Counter
    c = Counter(_key(s) for s in samples if s)
    if not c:
        return []
    best = c.most_common(1)[0][0]
    return [sympy.sympify(v) for v in best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--wall", type=float, default=100.0)
    args = ap.parse_args()

    client = _client(args.model, args.timeout)
    probs = _gen(args.n, args.seed)
    print(f"ROOT-SOLVE (check≪solve) — model={args.model} n={args.n} k={args.k} temp={args.temp}\n", flush=True)

    hit = {"bare": 0, "maj": 0, "filtered": 0, "passk": 0}
    filtered_frac = []
    for i, pr in enumerate(probs, 1):
        P, gold = pr["P"], pr["gold"]
        bare = _extract(_solve(pr["problem"], client, args.model, 0.0, args.wall))
        samples = [_extract(_solve(pr["problem"], client, args.model, args.temp, args.wall))
                   for _ in range(args.k)]
        survivors = [s for s in samples if _valid(P, s)]
        filtered_frac.append(1 - len(survivors) / max(1, len(samples)))
        picks = {"bare": bare, "maj": _vote(samples),
                 "filtered": _vote(survivors) if survivors else _vote(samples)}
        line = f"[{i:>3}/{args.n}] deg{sympy.degree(P)}"
        for arm in ("bare", "maj", "filtered"):
            ok = _correct(picks[arm], gold)
            hit[arm] += ok
            line += f"  {arm[:4]}={'✓' if ok else '·'}"
        hit["passk"] += any(_correct(s, gold) for s in samples)     # ceiling: any sample correct
        line += f"  (filt {len(samples)-len(survivors)}/{len(samples)})  gold={sorted(int(g) for g in gold)}"
        print(line, flush=True)

    n = args.n
    print("\n=== RESULTS ===")
    for arm in ("bare", "maj", "filtered"):
        print(f"  {arm:9}: {hit[arm]}/{n} = {hit[arm]/n:.1%}")
    print(f"  pass@{args.k} (ceiling): {hit['passk']}/{n} = {hit['passk']/n:.1%}")
    print(f"  self-consistency lift (maj − bare):     {(hit['maj']-hit['bare'])/n:+.1%}")
    print(f"  PROVE-filter lift    (filtered − maj):  {(hit['filtered']-hit['maj'])/n:+.1%}")
    print(f"  total lift           (filtered − bare): {(hit['filtered']-hit['bare'])/n:+.1%}")
    print(f"  avg samples filtered out: {sum(filtered_frac)/len(filtered_frac):.0%}")


if __name__ == "__main__":
    main()
