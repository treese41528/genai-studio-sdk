"""The optional Lean+mathlib proof track — offline (scanner/search) + mathlib-gated live tests."""

from __future__ import annotations

import pytest

from genai_studio.agents.tools.mathlib import (LemmaDecl, make_search_lemmas, mathlib_project,
                                               mathlib_tools, scan_file)

_SAMPLE = """
/-- The sum is commutative. -/
theorem add_comm' (a b : Nat) : a + b = b + a := by omega

@[simp] lemma foo_bar (x : Nat) : x + 0 = x := by simp

protected theorem baz : 1 = 1 := rfl
"""


# ── offline: the declaration scanner ─────────────────────────────────────────
def test_scan_file_extracts_name_signature_doc():
    decls = scan_file(_SAMPLE, "Test.Module")
    by = {d.name: d for d in decls}
    assert {"add_comm'", "foo_bar", "baz"} <= set(by)         # incl. @[simp] and `protected`
    ac = by["add_comm'"]
    assert "a + b = b + a" in ac.signature and "commutative" in ac.doc and ac.module == "Test.Module"
    assert ":=" not in ac.signature                            # proof body excluded


# ── offline: hybrid retrieval (keyword) ──────────────────────────────────────
def _idx():
    return [LemmaDecl("gcd_dvd", "gcd a b ∣ a", "M", "gcd divides"),
            LemmaDecl("add_comm", "a + b = b + a", "M", "commutativity of addition"),
            LemmaDecl("mul_comm", "a * b = b * a", "M", "commutativity of multiplication")]


def test_search_lemmas_keyword_ranks():
    tool = make_search_lemmas(_idx())
    assert "gcd_dvd" in tool.run({"query": "gcd divides", "k": 1}).data["names"]
    assert tool.run({"query": "commutativity addition", "k": 1}).data["names"] == ["add_comm"]
    assert tool.run({"query": "zzz nomatch qqq"}).data is None or \
        not tool.run({"query": "zzz nomatch qqq"}).data                    # no hit -> no names


def test_search_lemmas_index_is_lazy():
    calls = {"n": 0}
    def load():
        calls["n"] += 1
        return _idx()
    tool = make_search_lemmas(load)
    assert calls["n"] == 0                                     # not built at construction
    tool.run({"query": "gcd"})
    tool.run({"query": "comm"})
    assert calls["n"] == 1                                     # built once, on first search


# ── mathlib-gated live tests (run only where a mathlib project exists) ────────
_needs_mathlib = pytest.mark.skipif(mathlib_project() is None, reason="no mathlib project")


@_needs_mathlib
def test_mathlib_backed_lean_check_proves_ring():
    from genai_studio.agents.tools.lean import make_lean_check
    lc = make_lean_check(project_dir=mathlib_project())
    r = lc.run({"code": "import Mathlib.Tactic\n"
                        "example (a b : ℝ) : (a+b)^2 = a^2 + 2*a*b + b^2 := by ring"})
    assert r.data["ok"] is True


@_needs_mathlib
def test_mathlib_tools_bundle_and_real_search():
    names = {t.name for t in mathlib_tools()}
    assert {"lean_check", "grade_proof", "search_lemmas"} <= names
    sl = next(t for t in mathlib_tools() if t.name == "search_lemmas")
    assert any("gcd" in n for n in sl.run({"query": "gcd dvd", "k": 5}).data["names"])
