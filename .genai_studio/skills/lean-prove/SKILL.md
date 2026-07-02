---
name: lean-prove
description: Prove a mathematical statement rigorously in Lean 4 (with mathlib when available), checked by the kernel. Use whenever a claim must be PROVEN — not just numerically checked: real/polynomial inequalities, algebraic identities, number theory, analysis, induction. Find lemmas with search_lemmas, write Lean 4, run lean_check, and repair from the compiler's errors until the kernel accepts it.
allowed-tools: [search_lemmas, lean_check, grade_proof]
---

# Proving in Lean 4 + mathlib (kernel-checked)

Turn the claim into a Lean 4 theorem and prove it so `lean_check` reports **`PROOF CHECKED ✓`**. A proof
is real only when the kernel accepts it — never claim "proven" otherwise. Every snippet below is verified
to compile against Lean 4.31.0 + mathlib. (No mathlib? The `search_lemmas` tool will be absent — fall
back to the core tactics `decide`/`omega`/`rfl`/`simp`/`intro`/`exact`.)

## The loop
1. **State** the theorem with the right types (`ℝ`, `ℕ`, `ℤ`, `Prop`). Start the source with `import Mathlib`.
2. **Lead with ONE powerful tactic** (see the decision tree) — do NOT hand-write a `calc` chain first.
3. **`lean_check`** the complete source.
4. On error: **read it** (`line:col: error: …`), pick a *different* tactic or lemma — **never resubmit a
   proof that already failed** — and re-check.
5. Need a library fact? **`search_lemmas`** by concept, then cite it (`exact <name>` / `apply <name>` / `rw [<name>]`).
6. Report the checked theorem, or say honestly it's out of reach — don't fake it.

## Tactic decision tree (try in this order)
| Goal shape | First tactic |
|---|---|
| real/polynomial inequality (`≤`,`<`,`≥`) | `nlinarith [sq_nonneg (a-b), sq_nonneg (b-c), …]` — one `sq_nonneg` per squared difference |
| linear inequality from hypotheses | `linarith` |
| algebraic identity (`=`) in a ring | `ring` |
| identity depending on hypotheses `hᵢ` | `linear_combination c₁*h₁ + c₂*h₂` |
| concrete numerals (`3 < 5`, `¬ …`, primality) | `norm_num` |
| `0 ≤ e` / `0 < e` / `e ≠ 0` structurally | `positivity` |
| monotonicity (`a+c ≤ b+c`, `f a ≤ f b`) | `gcongr` |
| linear `ℕ`/`ℤ` (with vars, `%`, `/`) | `omega` |
| goal has fractions | `field_simp` **then** `ring`/`nlinarith` |
| a known library theorem exists | `search_lemmas` → `exact <name> <args>` |
| casts `↑` block a tactic (`ℕ→ℝ` …) | `push_cast` / `norm_cast` / `exact_mod_cast`, then the tactic |
| definitional / computational | `rfl`, `decide` (small only), `simp` |

## Lean 4 syntax — must-knows (these differ from Lean 3, the #1 source of errors)
- Lambda is `fun x => e` (or `λ x => e`). **`λ x, e` is a syntax error.**
- `calc` steps use **`:=`**, not `:`  →  `calc a ≤ b := h1` then `  _ < c := by nlinarith`.
- `induction n with | zero => … | succ k ih => …` (structured cases; not `induction n with k ih`).
- Subgoal bullets are **`·`**, not `{ }`. `refine` holes are **`?_`**, not `_`.
- `have h : T := …` / `obtain ⟨a, b⟩ := h` / `suffices h : T by …` — all need `:=` (or `by`).
- Finset sum notation is `∑ i ∈ s, f i` (with `∈`; the old `∑ i in s, …` is deprecated).
- `nlinarith`/`linarith` hints are a **square-bracket list**: `nlinarith [sq_nonneg (a-b)]` (not `{ }`).
- Prefer `push Not at h` over the deprecated `push_neg`; `exact?`/`apply?` replace `library_search`.
- Many order/division lemmas gained a `₀` suffix: `le_div_iff₀`, `div_le_iff₀`, `lt_div_iff₀`.

## The inequality workhorse: `nlinarith` + `sq_nonneg`
`nlinarith` adds/multiplies the hypotheses and hints you give it but **won't invent the key nonneg square**
— you supply it. For a goal in variables `a,b,c,…`, hint the squared differences that vanish at equality:
```lean
import Mathlib
theorem am_gm (a b : ℝ) : 2 * (a * b) ≤ a ^ 2 + b ^ 2 := by nlinarith [sq_nonneg (a - b)]

theorem three_var (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (c - a)]

theorem cauchy_schwarz (a b c d : ℝ) : (a*c + b*d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) := by
  nlinarith [sq_nonneg (a*d - b*c)]        -- hint = the Lagrange cross term
```

## Find & cite a library lemma (`search_lemmas`)
When the goal IS a known theorem, don't reprove it — find it and cite it. Search by concept or by the
shape of the statement; then `exact`/`apply`/`rw`.
```lean
import Mathlib
theorem sqrt2_irr : Irrational (Real.sqrt 2) := irrational_sqrt_two            -- search "irrational sqrt 2"
theorem big_prime (N : ℕ) : ∃ p, N ≤ p ∧ Nat.Prime p := Nat.exists_infinite_primes N   -- search "exists infinite primes"
```
Naming is compositional — read left→right: `add_comm : a+b=b+a`, `sq_nonneg : 0 ≤ a^2`,
`abs_le : |a| ≤ b ↔ -b ≤ a ∧ a ≤ b`, `mul_le_mul`, `Real.sq_sqrt : 0 ≤ a → sqrt a ^ 2 = a`.

## Induction over a Finset sum (peel the top term)
`Finset.sum_range_succ : ∑ i ∈ range (n+1), f i = (∑ i ∈ range n, f i) + f n`. Over `ℝ`, finish with
`push_cast; ring` (NOT `omega` — it can't do the nonlinear `k*(k+1)`):
```lean
import Mathlib
theorem gauss (n : ℕ) : (∑ i ∈ Finset.range (n + 1), (i : ℝ)) = n * (n + 1) / 2 := by
  induction n with
  | zero => simp
  | succ k ih => rw [Finset.sum_range_succ, ih]; push_cast; ring
```

## Structural proofs (logic)
```lean
import Mathlib
example (p q : Prop) (h : p ∧ q) : q ∧ p := by obtain ⟨hp, hq⟩ := h; exact ⟨hq, hp⟩
example : ∃ n : ℕ, n > 3 := ⟨4, by norm_num⟩
example (f g : ℕ → ℕ) (h : ∀ x, f x = g x) : f = g := by funext x; exact h x
example (p : Prop) (h : ¬¬p) : p := by by_contra hp; exact h hp
```

## Repairing from errors
| Lean says | Do |
|---|---|
| `unsolved goals` | goal isn't closed — add a finishing tactic (`ring`/`nlinarith`/`simp`); after `field_simp` append `ring` |
| `type mismatch` / cast `↑` | insert `push_cast`/`norm_cast`/`exact_mod_cast` before the tactic |
| `unknown identifier/constant` | wrong/renamed lemma — `search_lemmas` (or `exact?`) for the real name; mind the `₀`/`notMem` renames |
| `unexpected token ':'` in calc | Lean 3 syntax — change the step's `:` to `:=` |
| `ring failed` | identity needs a hypothesis → `linear_combination`; or goal has division → `field_simp` first |
| `no goals` | a trailing tactic ran after the proof closed — remove it (don't append `ring` when `field_simp` already closed it) |

## Scope (be honest)
`nlinarith`/`ring`/`norm_num`/`positivity`/`omega`/`linarith` + `search_lemmas`-and-cite cover a lot of
competition and undergraduate math. Deep results needing a long bespoke argument (hard olympiad
inequalities, nontrivial analysis/algebra) may be out of reach in a few steps — prove what you can and
state plainly what remains, rather than emitting a `sorry` (NOT a proof) or claiming success. For
nonlinear arithmetic over the reals, the `prove` tool (z3) may decide it without Lean.
