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
