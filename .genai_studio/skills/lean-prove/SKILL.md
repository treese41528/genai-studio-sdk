---
name: lean-prove
description: Prove a mathematical statement rigorously in Lean 4, checked by the kernel. Use when a claim must be PROVEN (not just numerically checked) — arithmetic facts, natural-number/integer identities, decidable logic. Writes Lean, runs lean_check, and repairs from the compiler's errors until the kernel accepts the proof.
allowed-tools: [lean_check]
---

# Proving in Lean 4 (kernel-checked)

Your job: turn an informal claim into a Lean 4 theorem and **prove it so `lean_check` reports
`PROOF CHECKED ✓`**. A proof is only real when the kernel accepts it — never say "proven" otherwise.

## The loop (do this every time)
1. **Formalize** the claim as a Lean 4 theorem, choosing the right types (`Nat`, `Int`, `Prop`).
2. **Write a proof** with `:= by <tactic>`.
3. **Call `lean_check`** with the complete source.
4. If it's not `✓`, **read the error** (it gives `line:col: error: …`), fix the statement or tactic,
   and call `lean_check` again. Iterate. Do not give up after one failure.
5. Report the final checked theorem, or — if it's outside what these tactics can reach — say so
   honestly (see *Scope* below), rather than faking a proof.

## Core tactics (this build has NO mathlib — use only these)
- **`decide`** — decidable propositions on *concrete, small* values: `2 + 2 = 4`, `Nat.Prime 7`,
  `3 < 5`, `¬ (2 = 3)`. Fails or hangs on large numbers or variables.
- **`omega`** — linear arithmetic over `Nat`/`Int`, including variables and hypotheses:
  `a + b = b + a`, `∀ n : Nat, n ≤ n + 1`, `2 * x + 1 ≠ 2 * y`. Your workhorse for arithmetic.
- **`rfl`** — things equal *by definition/computation*: `[1,2] ++ [3] = [1,2,3]`, `2 + 2 = 4`.
- **`simp`** — simplify with built-in lemmas; often `by simp` or `by simp; omega`.
- **`intro`, `constructor`, `cases`, `exact`, `apply`** — structural steps for `→`, `∧`, `∨`, `∀`.

## Patterns
```lean
theorem add_comm_nat (a b : Nat) : a + b = b + a := by omega
theorem two_two    : 2 + 2 = 4 := by decide
theorem le_succ    (n : Nat) : n ≤ n + 1 := by omega
theorem and_intro  (p q : Prop) (hp : p) (hq : q) : p ∧ q := by constructor <;> assumption
theorem seven_prime : Nat.Prime 7 := by decide
```
Combine when needed: `by intro h; omega` or `by simp [Nat.mul_comm]`.

## Scope (be honest)
Reachable here: concrete decidable facts, linear `Nat`/`Int` (in)equalities and identities, basic
propositional logic. **Out of reach without mathlib:** general real-number analysis, `ring`/`norm_num`
/`linarith`-style nonlinear algebra, induction-heavy number theory, calculus. If a claim needs those,
say it's outside this build's Lean fragment (mathlib required) instead of writing a proof that won't
check. For nonlinear *arithmetic over reals*, the `prove` tool (z3) may decide it without Lean.
