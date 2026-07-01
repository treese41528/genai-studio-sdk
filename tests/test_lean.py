"""lean_check — kernel-checked Lean 4 proof verification (gated on the Lean toolchain)."""

from __future__ import annotations

import pytest

import genai_studio.agents.tools.lean as L
from genai_studio.agents.tools.lean import lean_available, make_lean_check

_needs_lean = pytest.mark.skipif(lean_available() is None, reason="Lean 4 toolchain not installed")


@_needs_lean
def test_checks_correct_decide_proof():
    r = make_lean_check().run({"code": "theorem t : 2 + 2 = 4 := by decide"})
    assert r.data["ok"] is True and "CHECKED" in r.content


@_needs_lean
def test_checks_omega_proof_with_variables():
    r = make_lean_check().run({"code": "theorem c (a b : Nat) : a + b = b + a := by omega"})
    assert r.data["ok"] is True


@_needs_lean
def test_rejects_false_theorem():
    r = make_lean_check().run({"code": "theorem bad : 2 + 2 = 5 := by decide"})
    assert r.data["ok"] is False and "NOT proven" in r.content


@_needs_lean
def test_rejects_sorry_as_incomplete():
    r = make_lean_check().run({"code": "theorem gap : 1 = 1 := by sorry"})
    assert r.data["ok"] is False and "INCOMPLETE" in r.content


def test_clear_error_when_lean_absent(monkeypatch):
    monkeypatch.setattr(L, "lean_available", lambda lean="lean": None)
    r = make_lean_check().run({"code": "theorem t : True := trivial"})
    assert r.error and "not installed" in r.error
