"""``lean_check`` — verify a Lean 4 proof with the Lean kernel (model-writes / kernel-checks).

The model writes a Lean 4 theorem + proof; ``lean_check`` runs the Lean compiler and returns
SUCCESS (the kernel accepted it — a *machine-checked* proof) or the compiler ERRORS so the model can
repair and retry. This grounds PROOF the way sympy/z3 ground computation: the model cannot fake a
proof the kernel rejects, and the normal agent loop IS the repair loop (call → read errors → fix →
recall). ``sorry`` (an admitted gap) is rejected as incomplete.

Needs the Lean 4 toolchain on ``PATH`` (or at ``~/.elan/bin/lean``). Core tactics (``decide``,
``omega``, ``rfl``, ``simp``, ``constructor``, ``exact``) work WITHOUT mathlib; ``norm_num``/``ring``/
``linarith`` and real competition math need mathlib (a heavy, separate install)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from genai_studio.agents import ToolResult, tool


def lean_available(lean: str = "lean") -> str | None:
    """Path to the Lean executable, or None if not installed."""
    exe = shutil.which(lean)
    if exe:
        return exe
    fallback = os.path.expanduser("~/.elan/bin/lean")
    return fallback if os.path.exists(fallback) else None


def make_lean_check(*, lean: str = "lean", timeout: float = 40):
    """Return the ``lean_check`` tool (runs the Lean kernel on model-written proofs)."""
    exe = lean_available(lean)

    @tool
    def lean_check(code: str) -> ToolResult:
        """Check a Lean 4 proof with the Lean kernel. Write a COMPLETE theorem with its proof, e.g.
        `theorem t : 2 + 2 = 4 := by decide`. Returns SUCCESS if the kernel accepts it (a real,
        machine-checked proof), or the compiler errors so you can fix and retry. Core tactics work
        (decide, omega, rfl, simp, constructor, exact); norm_num/ring/linarith need mathlib.

        Args:
            code: complete Lean 4 source — the theorem statement and its proof.
        """
        if exe is None:
            return ToolResult(content="", error="Lean 4 not installed (put `lean` on PATH; "
                              "install via https://leanprover-community.github.io/get_started.html)")
        d = tempfile.mkdtemp()
        path = os.path.join(d, "Proof.lean")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([exe, path], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return ToolResult(content="", error=f"lean timed out after {timeout}s")
        except Exception as e:
            return ToolResult(content="", error=f"could not run lean: {e}")
        out = (r.stdout + r.stderr).strip()
        low = out.lower()
        if r.returncode == 0 and "error" not in low and "sorry" not in low:
            return ToolResult(content="PROOF CHECKED ✓ — the Lean kernel accepted this proof.",
                              data={"ok": True})
        if "sorry" in low:
            return ToolResult(content=f"INCOMPLETE — uses `sorry` (an admitted gap), not a real proof.\n{out[:1500]}",
                              data={"ok": False})
        return ToolResult(content=f"NOT proven — Lean reported:\n{out[:2000]}", data={"ok": False})

    return lean_check


def make_grade_proof(*, lean: str = "lean", timeout: float = 40):
    """Return ``grade_proof`` — grade a submitted proof CERTIFICATE against a claim. The claim (a Lean
    proposition) and the proof (a term or ``by`` block) are given SEPARATELY, assembled into a theorem,
    and kernel-checked. This is the *check≪solve* proof arm: verifying a candidate proof is far cheaper
    than producing one, so propose several and keep the one the kernel accepts."""
    checker = make_lean_check(lean=lean, timeout=timeout)

    @tool
    def grade_proof(claim: str, proof: str, imports: str = "") -> ToolResult:
        """Grade a Lean 4 proof CERTIFICATE: assemble ``theorem _ : <claim> := <proof>`` and check it
        with the Lean kernel. SOUND — the kernel cannot be fooled. Verifying a proof is cheaper than
        finding one, so you can propose several candidate proofs and keep the accepted one.

        Args:
            claim: the proposition, e.g. "2 + 2 = 4", "∀ n : Nat, n + 0 = n".
            proof: the proof — a term or ``by`` block, e.g. "by decide", "by omega", "fun n => rfl".
            imports: optional import lines (e.g. "import Mathlib") when the proof needs them.
        """
        head = (imports.strip() + "\n") if imports.strip() else ""
        code = f"{head}theorem grade_thm : {claim} := {proof.strip()}\n"
        return checker.run({"code": code})

    return grade_proof
