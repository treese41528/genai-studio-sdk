"""Factorization benchmark — the check≪solve payoff, generalized beyond root-solving.

Factoring a polynomial is hard; CHECKING a proposed factorization is trivial (expand and compare).
This is the same regime as root_solve_eval.py, so verified best-of-n should convert pass@k into
accuracy. Four arms over k sampled factorizations:

  bare@1     one greedy attempt
  maj@k      majority-vote the factorizations (self-consistency)
  verified   VERIFIED best-of-n (genai_studio.agents.verified_best_of): DISCARD every sample that
             does not expand back to P AND is not fully factored (deterministic sympy — independent
             of how the model factored), then vote among the survivors. This is the reusable
             check≪solve primitive; the checker is genai_studio.agents.factorization_check.
  pass@k     an upper bound — any of the k samples fully correct

Problems are products of distinct linear factors (seeded), so the gold + the expand-check are exact.
Grading: the answer expands to P AND is fully factored (sympy.factor is a fixed point).

  export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
  python benchmarks/factor_verify_eval.py --n 30 --k 8
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
from genai_studio.agents import Agent, NullTracer, verified_best_of   # noqa: E402
from genai_studio.agents.client import GenAIStudioClient        # noqa: E402
from genai_studio.agents.tools import final_answer              # noqa: E402
from genai_studio.agents.tools.symbolic import _parse           # noqa: E402

SYSTEM = ("You factor polynomials COMPLETELY over the integers. Show brief work, then put your final "
          "answer — the fully factored form as a product — in \\boxed{...}, e.g. \\boxed{(x-1)(x+2)(x+3)}.")
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


def _gen(n, seed, degrees):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        deg = degrees[i % len(degrees)]
        roots = rng.sample(range(-6, 7), deg)                  # distinct integer roots -> linear factors
        P = sympy.expand(sympy.prod([_X - r for r in roots]))
        pretty = sympy.sstr(P).replace("**", "^").replace("*", "")
        out.append({"P": P, "problem": f"Factor completely over the integers:  {pretty}"})
    return out


def _solve(problem, client, model, temp, wall):
    def run():
        a = Agent(client=client, model=model, tools=[final_answer], system=SYSTEM,
                  tracer=NullTracer(), max_steps=4, temperature=temp)
        return a.run(problem).text or ""
    return _guard(run, wall)


def _extract(text):
    """The boxed factorization as a normalized string (implicit-mult friendly, ^ -> **)."""
    b = (_last_boxed(text) or "").strip().strip("$ ")
    return b or None


def _valid(P, cand):
    """Sound VALIDITY: cand expands to P and is a genuine product (a correct factorization)."""
    if not cand:
        return False
    try:
        F = _parse(cand)
        return bool(sympy.expand(F - P) == 0 and (F.is_Mul or F.is_Pow) and sympy.expand(F) != F)
    except Exception:
        return False


def _complete(P, cand):
    """Sound COMPLETENESS: valid AND fully factored (sympy can't factor it further)."""
    if not _valid(P, cand):
        return False
    try:
        F = _parse(cand)
        return sympy.factor(F) == F
    except Exception:
        return False


def _key(cand):
    """Canonical vote key — sympy's factored form, so equivalent spellings collapse."""
    try:
        return sympy.sstr(sympy.factor(_parse(cand)))
    except Exception:
        return str(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--degrees", default="3,4", help="comma-separated factor counts to spread over")
    ap.add_argument("--wall", type=float, default=90.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("set GENAI_STUDIO_API_KEY (and GENAI_STUDIO_RPM=20)")

    degrees = [int(d) for d in args.degrees.split(",")]
    client = _client(args.model, args.timeout)
    probs = _gen(args.n, args.seed, degrees)
    arms = ("bare", "maj", "verified", "passk")
    c = dict.fromkeys(arms, 0)

    for i, pr in enumerate(probs, 1):
        P = pr["P"]
        bare = _extract(_solve(pr["problem"], client, args.model, 0.0, args.wall))
        samples = [_extract(_solve(pr["problem"], client, args.model, args.temp, args.wall))
                   for _ in range(args.k)]
        # verified best-of-n: filter by validity, prefer fully-factored, vote among survivors
        pick = verified_best_of(samples, check=lambda s: _valid(P, s),
                                complete=lambda s: _complete(P, s), key=_key)
        maj = verified_best_of(samples, check=lambda s: True, key=_key).answer   # plain majority
        got = {"bare": bare, "maj": maj, "verified": pick.answer}
        for arm in ("bare", "maj", "verified"):
            c[arm] += _complete(P, got[arm])
        c["passk"] += any(_complete(P, s) for s in samples)
        n = i
        print(f"[{i:3}/{args.n}] bare {c['bare']/n:4.0%} | maj {c['maj']/n:4.0%} | "
              f"VERIFIED {c['verified']/n:4.0%} | pass@k {c['passk']/n:4.0%}", flush=True)

    print("\n=== factorization: check≪solve ===")
    n = args.n
    for arm in arms:
        print(f"  {arm:9} {c[arm]/n:5.1%}")
    print(f"\nverified best-of-{args.k} lift over greedy: {(c['verified']-c['bare'])/n:+.1%}  "
          f"(pass@k ceiling {c['passk']/n:.0%})")


if __name__ == "__main__":
    main()
