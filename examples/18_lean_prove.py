"""Kernel-checked proving in Lean 4 (model-writes / kernel-checks).

An agent given the ``lean_check`` tool writes a Lean 4 theorem + proof, runs it through the Lean
kernel, and REPAIRS from the compiler's errors until the kernel accepts it — the ordinary agent loop
IS the proof-repair loop. The result is a *machine-checked* proof, not a plausible-looking one; a
wrong proof (2+2=5) or an admitted gap (``sorry``) is rejected.

Needs the Lean 4 toolchain (``lean`` on PATH, or ~/.elan/bin/lean). Core tactics (decide / omega /
rfl) work without mathlib. Run: python examples/18_lean_prove.py
"""

from __future__ import annotations

from genai_studio.agents import Agent, ConsoleTracer
from genai_studio.agents.tools import final_answer
from genai_studio.agents.tools.lean import lean_available, make_lean_check
from _common import make_client

SYSTEM = (
    "You prove theorems in Lean 4, verified by the kernel. For each claim: formalize it as a Lean 4 "
    "theorem (pick the right types — Nat/Int/Prop), write a proof with `:= by <tactic>`, and call "
    "lean_check. If it is not accepted, READ the error, fix it, and call lean_check again — iterate. "
    "Use only core tactics (decide for concrete decidable facts; omega for linear Nat/Int arithmetic "
    "with variables; rfl for definitional equalities; simp). Report the final CHECKED theorem, or say "
    "honestly if it needs mathlib. Never claim proven without a lean_check ✓."
)

CLAIMS = [
    "For all natural numbers a and b, a + b = b + a.",
    "2 + 2 = 4.",
    "For every natural number n, n <= n + 1.",
]

if __name__ == "__main__":
    if lean_available() is None:
        raise SystemExit("Lean 4 not installed — see leanprover-community.github.io/get_started.html")
    client = make_client()
    agent = Agent(client=client, tools=[make_lean_check(), final_answer], system=SYSTEM,
                  tracer=ConsoleTracer(), max_steps=10)
    for claim in CLAIMS:
        print(f"\n{'='*70}\nCLAIM: {claim}\n{'='*70}")
        print(agent.run(claim).text)
