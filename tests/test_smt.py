"""Sound theorem proving over arithmetic — prove / solve_constraints (z3, the [smt] extra),
and verify_math's z3 integration (it now PROVES claims it used to return UNKNOWN for)."""

from __future__ import annotations

import pytest

pytest.importorskip("z3")
pytest.importorskip("sympy")

from genai_studio.agents.tools.smt import prove, solve_constraints
from genai_studio.agents.tools.symbolic import verify_math


# ── prove ────────────────────────────────────────────────────────────────────
def test_prove_amgm_inequality():
    # x^2 + y^2 >= 2xy holds for ALL reals — a genuine proof obligation
    r = prove.run({"claim": "x**2 + y**2 >= 2*x*y"})
    assert r.data["verdict"] == "proven" and "PROVEN" in r.content


def test_prove_algebraic_identity():
    assert prove.run({"claim": "(a+b)**2 == a**2 + 2*a*b + b**2"}).data["verdict"] == "proven"


def test_prove_disproven_with_counterexample():
    r = prove.run({"claim": "x**2 >= x"})                    # false over reals (x = 1/2)
    assert r.data["verdict"] == "disproven" and "counterexample" in r.content


def test_prove_with_assumption():
    # x^2 >= x IS true once we assume x >= 1
    assert prove.run({"claim": "x**2 >= x", "assume": "x >= 1"}).data["verdict"] == "proven"


def test_prove_unsupported_transcendental():
    assert prove.run({"claim": "sin(x) <= 1"}).error         # outside z3's arithmetic fragment


# ── solve_constraints ────────────────────────────────────────────────────────
def test_solve_constraints_sat():
    r = solve_constraints.run({"constraints": ["x + y == 10", "x - y == 2"]})
    assert r.data["result"] == "sat" and "x=" in r.content


def test_solve_constraints_unsat():
    assert solve_constraints.run({"constraints": ["x > 5", "x < 3"]}).data["result"] == "unsat"


def test_solve_constraints_integers():
    r = solve_constraints.run({"constraints": ["2*x + 3*y == 12", "x > 0", "y > 0"], "domain": "int"})
    assert r.data["result"] == "sat"


# ── verify_math now leans on z3 for real proofs ──────────────────────────────
def test_verify_math_proves_via_smt():
    # previously UNKNOWN (sympy couldn't decide the sign) — now PROVEN by z3
    r = verify_math.run({"claim": "x**2 + y**2 >= 2*x*y"})
    assert r.data["verdict"] is True and "SMT" in r.content


def test_verify_math_numeric_is_a_check_not_a_proof():
    # sin(x) == x: sympy inconclusive, z3 can't (transcendental) -> numeric heuristic disproves it
    r = verify_math.run({"claim": "sin(x) == x"})
    assert r.data["verdict"] is False                        # counterexample found numerically
