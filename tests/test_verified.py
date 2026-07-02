"""The check≪solve primitive — verified_best_of (dependency-free) + the sound checkers."""

from __future__ import annotations

import pytest

from genai_studio.agents.verified import (VerifiedPick, factorization_check, inequality_check,
                                          verified_best_of)


# ── the primitive (pure, no sympy/z3) ────────────────────────────────────────
def test_filters_then_votes_among_survivors():
    cands = ["good", "bad", "good", "good"]            # 3 valid, 1 not
    pick = verified_best_of(cands, check=lambda c: c == "good")
    assert isinstance(pick, VerifiedPick)
    assert pick.answer == "good" and pick.verified and pick.n_verified == 3 and pick.method == "verified-vote"


def test_fallback_vote_when_none_verify_is_honest():
    pick = verified_best_of(["w1", "w2", "w2"], check=lambda c: False)
    assert pick.answer == "w2" and pick.verified is False and pick.method == "fallback-vote"


def test_empty_candidates():
    pick = verified_best_of([None, None], check=lambda c: True)
    assert pick.answer is None and pick.n_candidates == 0 and pick.method == "empty"


def test_completeness_is_preferred_over_validity():
    # validity accepts A and B; completeness accepts only B -> pick B
    pick = verified_best_of(["A", "B", "A"], check=lambda c: c in ("A", "B"), complete=lambda c: c == "B")
    assert pick.answer == "B" and pick.verified and pick.n_verified == 1


def test_completeness_falls_back_to_validity():
    # nothing is complete -> fall back to the validity filter, still verified
    pick = verified_best_of(["A", "A", "C"], check=lambda c: c == "A", complete=lambda c: False)
    assert pick.answer == "A" and pick.verified and pick.method == "validity-vote"


def test_a_checker_that_raises_counts_as_reject():
    def boom(c):
        raise ValueError("nope")
    assert verified_best_of(["x"], check=boom).verified is False


# ── the sound checkers (need sympy / z3) ─────────────────────────────────────
def test_inequality_checker():
    pytest.importorskip("z3")
    pytest.importorskip("sympy")
    assert inequality_check("x**2 + y**2 >= 2*x*y") is True     # true for all reals -> z3 proves it
    assert inequality_check("x**2 >= x") is False               # false (x=1/2)


def test_factorization_checker():
    pytest.importorskip("sympy")
    assert factorization_check("x**3 - 8", "(x-2)*(x**2+2*x+4)") is True
    assert factorization_check("x**2 - 1", "(x+1)*(x-2)") is False    # wrong product
    assert factorization_check("x**2 - 1", "x**2 - 1") is False       # equal but NOT factored


def test_verified_best_of_over_real_factorizations():
    pytest.importorskip("sympy")
    cands = ["(x+2)*(x+3)", "(x+1)*(x+6)", "(x+2)*(x+3)", "x**2+5*x+6", "(x+2)*(x+3)"]
    pick = verified_best_of(cands, check=lambda c: factorization_check("x**2+5*x+6", c))
    assert pick.answer == "(x+2)*(x+3)" and pick.verified and pick.n_verified == 3
