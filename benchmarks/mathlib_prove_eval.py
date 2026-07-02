"""End-to-end mathlib proving — can the agent get the Lean KERNEL to accept hard proofs?

For each target theorem the agent may call ``search_lemmas`` (find mathlib lemmas), ``lean_check``
(kernel-check a proof), and ``grade_proof``, iterating until the kernel accepts it. Grading is the
kernel itself: we RECORD every proof lean_check accepted and INDEPENDENTLY re-check the final one — the
model's prose is never trusted. A spread of difficulties is included so both wins and honest misses show.

  export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
  python benchmarks/mathlib_prove_eval.py --model qwen2.5:72b
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from genai_studio import GenAIStudio
from genai_studio.agents import Agent, NullTracer, ToolResult, tool
from genai_studio.agents.client import GenAIStudioClient
from genai_studio.agents.tools import final_answer
from genai_studio.agents.tools.lean import make_lean_check
from genai_studio.agents.tools.mathlib import make_search_lemmas, build_lemma_index, mathlib_project

# (id, natural-language goal, the exact Lean proposition to prove)
THEOREMS = [
    ("am_gm_2", "AM–GM for two reals (2ab ≤ a²+b²)",
     "(a b : ℝ) : 2 * (a * b) ≤ a ^ 2 + b ^ 2"),
    ("three_var", "a²+b²+c² ≥ ab+bc+ca for all reals",
     "(a b c : ℝ) : a ^ 2 + b ^ 2 + c ^ 2 ≥ a * b + b * c + c * a"),
    ("sqrt2_irrational", "√2 is irrational",
     ": Irrational (Real.sqrt 2)"),
    ("gauss_sum", "Gauss sum: 0+1+…+n = n(n+1)/2",
     "(n : ℕ) : (∑ i ∈ Finset.range (n + 1), (i : ℝ)) = n * (n + 1) / 2"),
    ("inf_primes", "there are arbitrarily large primes",
     "(N : ℕ) : ∃ p, N ≤ p ∧ Nat.Prime p"),
    ("cauchy_schwarz2", "2D Cauchy–Schwarz (as a real polynomial inequality)",
     "(a b c d : ℝ) : (a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)"),
]

SYSTEM = r"""You are a Lean 4 + mathlib proof engineer. Prove the given theorem so the KERNEL accepts it.

RULES:
- Write COMPLETE source: `import Mathlib` then the theorem. Call lean_check to verify.
- Your FIRST attempt MUST be a single powerful tactic — do NOT hand-write calc chains first:
    • real/polynomial inequalities:  `:= by nlinarith [sq_nonneg (a-b), sq_nonneg (b-c), sq_nonneg (a-c), sq_nonneg (a+b)]`
    • algebraic identities:  `:= by ring`      • numeric goals:  `:= by norm_num`
    • positivity:  `:= by positivity`          • linear ℕ/ℤ:  `:= by omega`
    • a known library fact: call search_lemmas, then `:= <lemma_name>` or `:= by exact?`
- THIS IS LEAN 4: calc steps use `:=` (e.g. `_ ≤ y := by nlinarith`), never `:`. Tactic blocks use `by`.
- If lean_check errors, READ it and try a DIFFERENT tactic/lemma. NEVER resubmit a proof that already failed.
- Call final_answer ONLY after lean_check reports PROOF CHECKED, with the verified source in a ```lean block.

A proof that the kernel accepts:
```lean
import Mathlib
theorem ex (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2 := by nlinarith [sq_nonneg (a - b)]
```"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=150.0)   # per lean_check (full mathlib import is slow)
    ap.add_argument("--only", default="", help="comma-separated theorem ids to run")
    args = ap.parse_args()
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("set GENAI_STUDIO_API_KEY (and GENAI_STUDIO_RPM=20)")
    proj = mathlib_project()
    if proj is None:
        sys.exit("no mathlib project (set GENAI_STUDIO_LEAN_PROJECT)")

    client = GenAIStudioClient(GenAIStudio(validate_model=False, timeout=180), default_model=args.model)
    base_lc = make_lean_check(project_dir=proj, timeout=args.timeout)
    index = build_lemma_index(proj)                           # ~180k decls, cached
    search = make_search_lemmas(index)
    todo = [t for t in THEOREMS if not args.only or t[0] in args.only.split(",")]

    results = []
    for tid, goal, prop in todo:
        accepted = []                                        # every source the KERNEL accepted this run

        @tool(name="lean_check", description=base_lc.spec.description)
        def lean_check(code: str) -> ToolResult:
            r = base_lc.run({"code": code})
            if r.data and r.data.get("ok"):
                accepted.append(code)
            return r

        agent = Agent(client=client, model=args.model, tools=[search, lean_check, final_answer],
                      system=SYSTEM, tracer=NullTracer(), max_steps=args.max_steps, temperature=0.0)
        task = (f"Prove this Lean theorem (name it `thm`):\n\n"
                f"    theorem thm {prop} := by sorry\n\nGoal in words: {goal}. Replace `sorry` with a real proof.")
        t0 = time.time()
        try:
            out = agent.run(task)
            text = out.text or ""
        except Exception as e:
            text = f"[error {type(e).__name__}: {e}]"
        dt = time.time() - t0

        # ground truth: INDEPENDENTLY re-check the last kernel-accepted source (don't trust the loop)
        proved, verified_src = False, ""
        if accepted:
            v = base_lc.run({"code": accepted[-1]})
            proved = bool(v.data and v.data.get("ok"))
            verified_src = accepted[-1]
        results.append((tid, goal, proved, dt, len(accepted), verified_src))
        print(f"[{tid:16}] {'PROVED ✓' if proved else 'not proved':11} "
              f"| {dt:5.0f}s | kernel-accepts={len(accepted)} | {goal}", flush=True)

    n = len(results)
    npr = sum(1 for r in results if r[2])
    print(f"\n=== mathlib end-to-end: {npr}/{n} proved (kernel-checked, independently re-verified) ===")
    for tid, goal, proved, dt, nacc, src in results:
        print(f"\n### {tid} — {'PROVED ✓' if proved else 'NOT proved'} ({dt:.0f}s)")
        if src:
            print("\n".join("    " + ln for ln in src.strip().splitlines()[:16]))


if __name__ == "__main__":
    main()
