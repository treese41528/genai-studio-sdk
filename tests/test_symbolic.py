"""Exact-math grounding tools — verify_math / symbolic_math / matrix_op (needs sympy = [math])."""

from __future__ import annotations

import pytest

pytest.importorskip("sympy")

from genai_studio.agents.tools.symbolic import matrix_op, symbolic_math, verify_math


# ── verify_math (the flagship) ───────────────────────────────────────────────
def test_verify_true_identity():
    r = verify_math.run({"claim": "x**2 - 1 == (x-1)*(x+1)"})
    assert r.data["verdict"] is True and "PASS" in r.content


def test_verify_float_trap_exact():
    # the classic 0.1+0.2 trap: exact CAS says these ARE equal
    assert verify_math.run({"claim": "1/10 + 2/10 == 3/10"}).data["verdict"] is True


def test_verify_false_with_counterexample():
    r = verify_math.run({"claim": "(x+1)**2 == x**2 + 1"})
    assert r.data["verdict"] is False and "FAIL" in r.content


def test_verify_numeric_equality():
    assert verify_math.run({"claim": "sqrt(2)*sqrt(2) == 2"}).data["verdict"] is True


def test_verify_inequality():
    assert verify_math.run({"claim": "2**10 > 1000"}).data["verdict"] is True
    assert verify_math.run({"claim": "1 > 2"}).data["verdict"] is False


def test_verify_bad_claim():
    assert verify_math.run({"claim": "just some text"}).error


# ── symbolic_math ────────────────────────────────────────────────────────────
def test_solve_all_roots():
    r = symbolic_math.run({"operation": "solve", "expression": "x**2 = 2"})
    assert "sqrt(2)" in r.content and "-sqrt(2)" in r.content       # BOTH roots, not one


def test_integrate_exact_definite():
    r = symbolic_math.run({"operation": "integrate", "expression": "sin(x)",
                           "lower": "0", "upper": "pi"})
    assert r.content.strip().endswith("2")                          # exactly 2, not 1.9999


def test_diff_and_factor():
    assert "cos(x)" in symbolic_math.run({"operation": "diff", "expression": "sin(x)"}).content
    assert "(x - 1)*(x + 1)" in symbolic_math.run({"operation": "factor", "expression": "x**2-1"}).content


def test_evaluate_exact_rational():
    assert "3/10" in symbolic_math.run({"operation": "evaluate", "expression": "1/10 + 2/10"}).content


def test_symbolic_bad_op():
    assert symbolic_math.run({"operation": "teleport", "expression": "x"}).error


# ── matrix_op ────────────────────────────────────────────────────────────────
def test_matrix_det_and_inv():
    assert "-2" in matrix_op.run({"matrix": [[1, 2], [3, 4]], "operation": "det"}).content
    assert matrix_op.run({"matrix": [[1, 2], [3, 4]], "operation": "inv"}).error is None


def test_matrix_singular_stays_zero():
    # a singular matrix: det is EXACTLY 0 (float numpy would report ~1e-16 and "invert" it)
    det = matrix_op.run({"matrix": [[1, 2], [2, 4]], "operation": "det"})
    assert det.content.strip().endswith("0")
    assert "singular" in matrix_op.run({"matrix": [[1, 2], [2, 4]], "operation": "inv"}).error


def test_matrix_solve():
    r = matrix_op.run({"matrix": [[2, 0], [0, 4]], "operation": "solve", "vector": [4, 8]})
    assert "2" in r.content                                          # x = [2, 2]
