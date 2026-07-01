"""Three-arm MATH-500: does CAS-VERIFIED self-consistency beat plain self-consistency and bare?

Tao's division of labor — LLMs GENERATE, formal tools VERIFY. This tests tools as a VERIFIER/FILTER
over self-consistency samples (PROVE, arXiv:2410.12608), NOT as an inline solver — the integration
style the literature says actually works, and which structurally avoids the tool-call spin/leak.

  A  bare@1        one greedy solution, box the answer (single-pass baseline)
  B  maj@k         k sampled solutions (temp>0), majority-vote the boxed answers (self-consistency,
                   Wang et al. arXiv:2203.11171) — the HONEST baseline arm C must beat
  C  cas_verified  same k samples; a CAS pass VERIFIES each distinct candidate answer (substitute
                   back / recompute with symbolic_math / verify_math / prove) and discards the ones it
                   refutes; majority-vote the survivors (falls back to plain maj if none survive)

Robust to gateway drop/hang: tight per-request timeout + a hard per-sample wall-clock guard, so a
stalled sample is scored as no-answer and never stalls the whole run.

  export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
  python benchmarks/math_selfconsist.py --model qwen2.5:72b --n 30 --k 8 --stratify subject
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from math_eval import _grounded_tools, _load                  # noqa: E402  (stratified loader + tools)
from math_grade import _normalize, extract_answer, is_equiv   # noqa: E402

from genai_studio import GenAIStudio                          # noqa: E402
from genai_studio.agents import Agent, NullTracer             # noqa: E402
from genai_studio.agents.client import GenAIStudioClient      # noqa: E402
from genai_studio.agents.tools import final_answer            # noqa: E402

SOLVE_SYSTEM = ("You are a careful mathematician. Solve the problem step by step and put your FINAL "
                "answer in \\boxed{...}.")
VERIFY_SYSTEM = ("You verify a PROPOSED answer to a math problem using exact-math tools. Use "
                 "symbolic_math / verify_math / matrix_op / prove to CHECK it: substitute the answer "
                 "back, recompute the key quantity, or verify the key equation/identity — do not just "
                 "re-solve from memory. Finish with a line 'VERDICT: VERIFIED' if the proposed answer "
                 "is correct, or 'VERDICT: REFUTED' if it is wrong.")


def _client(model: str, timeout: int) -> GenAIStudioClient:
    return GenAIStudioClient(GenAIStudio(validate_model=False, timeout=timeout), default_model=model)


def _guard(fn, wall: float):
    """Run fn() in a daemon thread; return '' if it doesn't finish within `wall` seconds (a dropped/
    hung gateway call self-terminates at the client timeout, so the orphan doesn't linger long)."""
    box: dict = {}

    def run():
        try:
            box["r"] = fn()
        except Exception as e:
            box["r"] = f"[error: {type(e).__name__}]"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(wall)
    return box.get("r", "")


def _solve(problem, client, model, temp, wall):
    def run():
        a = Agent(client=client, model=model, tools=[final_answer], system=SOLVE_SYSTEM,
                  tracer=NullTracer(), max_steps=4, temperature=temp)
        return a.run(problem).text or ""
    return _guard(run, wall)


def _verify(problem, answer, client, model, wall):
    """True/False/None — does the CAS uphold `answer`? (None = couldn't decide)."""
    def run():
        a = Agent(client=client, model=model, tools=_grounded_tools(), system=VERIFY_SYSTEM,
                  tracer=NullTracer(), max_steps=8, temperature=0.0)
        return a.run(f"Problem:\n{problem}\n\nProposed answer: {answer}\n\nVerify it.").text or ""
    out = _guard(run, wall).upper()
    iv, ir = out.rfind("VERIFIED"), out.rfind("REFUTED")
    if iv == -1 and ir == -1:
        return None
    return iv > ir                                            # whichever verdict appears last wins


def _cluster(samples):
    """Group answers by normalized key; return [(representative, votes)] sorted by votes desc."""
    groups: dict = {}
    for s in samples:
        ea = extract_answer(s) if s else ""
        if not ea:
            continue
        g = groups.setdefault(_normalize(ea) or ea, [ea, 0])
        g[1] += 1
    return sorted(groups.values(), key=lambda x: -x[1])


def _maj(samples):
    c = _cluster(samples)
    return c[0][0] if c else ""


def _cas_verified(problem, samples, client, model, wall, top=4):
    clusters = _cluster(samples)
    # verify the most-voted candidates (a singleton rarely wins the vote anyway)
    verified = [(ans, n) for ans, n in clusters[:top] if _verify(problem, ans, client, model, wall)]
    if verified:
        return max(verified, key=lambda x: x[1])[0]
    return clusters[0][0] if clusters else ""                # fallback: plain majority


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify", default="subject")
    ap.add_argument("--arms", default="bare,maj,cas")
    ap.add_argument("--timeout", type=int, default=45, help="per-request gateway timeout (s)")
    ap.add_argument("--wall", type=float, default=100.0, help="per-sample wall-clock guard (s)")
    args = ap.parse_args()

    client = _client(args.model, args.timeout)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    problems = _load(args.n, args.seed, stratify=args.stratify)
    print(f"MATH-500 self-consistency — model={args.model} n={len(problems)} k={args.k} "
          f"temp={args.temp} arms={arms}\n", flush=True)

    correct = {a: 0 for a in arms}
    for i, r in enumerate(problems, 1):
        prob, gold = r["problem"], r["answer"]
        samples = [_solve(prob, client, args.model, args.temp, args.wall)
                   for _ in range(args.k)] if ("maj" in arms or "cas" in arms) else []
        picks = {}
        if "bare" in arms:
            picks["bare"] = _solve(prob, client, args.model, 0.0, args.wall)
        if "maj" in arms:
            picks["maj"] = _maj(samples)
        if "cas" in arms:
            picks["cas"] = _cas_verified(prob, samples, client, args.model, args.wall)
        line = f"[{i:>3}/{len(problems)}] L{r.get('level','?')} {r.get('subject','')[:12]:12}"
        for a in arms:
            ok = is_equiv(picks[a], gold)
            correct[a] += ok
            line += f"  {a}={'✓' if ok else '·'}"
        line += f"  gold={gold[:16]}"
        print(line, flush=True)

    n = len(problems)
    print("\n=== RESULTS ===")
    for a in arms:
        print(f"  {a:12}: {correct[a]}/{n} = {correct[a]/n:.1%}" if n else a)
    if {"bare", "maj"} <= set(arms):
        print(f"  self-consistency lift (maj − bare): {(correct['maj']-correct['bare'])/n:+.1%}")
    if {"maj", "cas"} <= set(arms):
        print(f"  CAS-verify lift  (cas − maj):        {(correct['cas']-correct['maj'])/n:+.1%}")
    if {"bare", "cas"} <= set(arms):
        print(f"  total lift       (cas − bare):       {(correct['cas']-correct['bare'])/n:+.1%}")


if __name__ == "__main__":
    main()
