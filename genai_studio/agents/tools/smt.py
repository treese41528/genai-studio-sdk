"""``prove`` / ``solve_constraints`` — SOUND theorem proving over arithmetic (z3, the `[smt]` extra).

`verify_math`/`symbolic_math` GROUND COMPUTATION; they do not *prove*. This adds a sound decision
procedure for the fragment where one exists: **polynomial (in)equalities and logic over the reals or
integers**. `prove` establishes a universally-quantified claim by checking its negation is UNSAT
(→ PROVEN for all values), or returns a concrete COUNTEREXAMPLE; `solve_constraints` finds a witness
or proves infeasibility. z3 is a *sound* procedure — a PROVEN result is a real proof, not a sample
check.

HONEST LIMITS: this is not a general theorem prover. Unbounded induction (∀n with n in an exponent /
factorial), transcendental functions, analysis/topology, and novel theorems fall OUTSIDE z3's
decidable fragment → it returns UNKNOWN (never a false claim). Machine-checked proofs of arbitrary
theorems need an interactive prover (Lean/Isabelle) + a proof-writing model — see the design doc.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

from ..datascience._guard import _require
from .symbolic import _parse, _split_relation


class _Unsupported(Exception):
    """A construct outside z3's arithmetic fragment (symbolic exponent, transcendental, …)."""


def _to_z3(sp, z3, e, domain: str, env: dict):
    """Translate a sympy expression to a z3 arithmetic expression (polynomials over R/Z)."""
    if e.is_Symbol:
        n = str(e)
        if n not in env:
            env[n] = z3.Int(n) if domain == "int" else z3.Real(n)
        return env[n]
    if e.is_Integer:
        return z3.IntVal(int(e)) if domain == "int" else z3.RealVal(int(e))
    if e.is_Rational:
        return z3.Q(int(e.p), int(e.q))
    if e.is_Float:
        return z3.RealVal(str(e))
    if e.is_Add:
        r = None
        for a in e.args:
            t = _to_z3(sp, z3, a, domain, env)
            r = t if r is None else r + t
        return r
    if e.is_Mul:
        r = None
        for a in e.args:
            t = _to_z3(sp, z3, a, domain, env)
            r = t if r is None else r * t
        return r
    if e.is_Pow:
        base, ex = e.as_base_exp()
        if ex.is_Integer and int(ex) >= 1:
            b = _to_z3(sp, z3, base, domain, env)
            r = b
            for _ in range(int(ex) - 1):
                r = r * b
            return r
        raise _Unsupported(f"exponent {ex} (only non-negative integer powers)")
    raise _Unsupported(f"{type(e).__name__}: {e}")


_REL = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}


def _relation_to_z3(sp, z3, text: str, domain: str, env: dict):
    """Parse "lhs OP rhs" and build the z3 boolean constraint."""
    rel = _split_relation(text)
    if rel is None:
        raise _Unsupported(f"not a relation: {text!r}")
    lhs, op, rhs = rel
    return _REL[op](_to_z3(sp, z3, _parse(lhs), domain, env),
                    _to_z3(sp, z3, _parse(rhs), domain, env))


def z3_decide(claim: str, domain: str = "real", assume: str | None = None):
    """Decide a universally-quantified (in)equality with z3. Returns
    ``("proven"|"disproven"|"unknown"|"unsupported", counterexample|None)``. Free vars are ∀-quantified
    over ``domain``; ``assume`` is a comma-separated list of hypotheses. Used by ``prove`` and
    (opportunistically) by ``verify_math``. Never raises — a non-arithmetic construct → "unsupported"."""
    try:
        z3 = _require("z3", "smt")
    except Exception:
        return "unsupported", None                          # z3 not installed
    sp = _require("sympy", "math")
    env: dict = {}
    try:
        goal = _relation_to_z3(sp, z3, claim, domain, env)
        hyps = [_relation_to_z3(sp, z3, h, domain, env) for h in (assume or "").split(",") if h.strip()]
    except _Unsupported:
        return "unsupported", None
    except Exception:
        return "unsupported", None
    s = z3.Solver()
    s.set("timeout", 5000)
    s.add(z3.Not(goal), *hyps)                              # a counterexample = hyps ∧ ¬goal
    res = s.check()
    if res == z3.unsat:
        return "proven", None
    if res == z3.sat:
        m = s.model()
        ce = ", ".join(f"{d.name()}={m[d]}" for d in m.decls())
        return "disproven", ce or "(trivial)"
    return "unknown", None


@tool
def prove(claim: str, domain: str = "real", assume: str | None = None) -> ToolResult:
    """PROVE a universally-quantified (in)equality with a SOUND solver (z3). Establishes the claim
    holds for ALL values (of the free variables) satisfying the assumptions, or returns a concrete
    COUNTEREXAMPLE, or UNKNOWN when the claim is outside z3's decidable fragment (e.g. unbounded
    induction). A PROVEN result is a real proof, not a sample check.

    Args:
        claim: an (in)equality, e.g. "x**2 + y**2 >= 2*x*y", "(a+b)**2 == a**2 + 2*a*b + b**2".
        domain: "real" or "int" — the domain of the free variables.
        assume: comma-separated hypotheses, e.g. "x>0, y>0".
    """
    verdict, ce = z3_decide(claim, domain, assume)
    if verdict == "proven":
        return ToolResult(content=f"PROVEN (holds for all {domain} values): {claim}",
                          data={"verdict": "proven"})
    if verdict == "disproven":
        return ToolResult(content=f"DISPROVEN: {claim}\ncounterexample: {ce}",
                          data={"verdict": "disproven", "counterexample": ce})
    if verdict == "unsupported":
        return ToolResult(content="", error="claim is outside z3's arithmetic fragment (or [smt] not "
                          "installed): only polynomial (in)equalities over reals/integers; no "
                          "transcendentals, factorials, or unbounded induction.")
    return ToolResult(content=f"UNKNOWN: z3 could not decide {claim} within the time limit "
                      "(likely nonlinear/unbounded — outside its decidable fragment).",
                      data={"verdict": "unknown"})


@tool
def solve_constraints(constraints: list[str], domain: str = "real") -> ToolResult:
    """Find values satisfying a set of constraints (SMT), or PROVE none exist. Returns SAT + a
    witness assignment, UNSAT (proven infeasible), or UNKNOWN. Use for logic/number puzzles,
    feasibility, and "is there an x with …".

    Args:
        constraints: (in)equalities that must ALL hold, e.g. ["x + y == 10", "x - y == 2", "x > 0"].
        domain: "real" or "int".
    """
    try:
        z3 = _require("z3", "smt")
    except Exception as e:
        return ToolResult(content="", error=str(e))
    sp = _require("sympy", "math")
    env: dict = {}
    try:
        cons = [_relation_to_z3(sp, z3, c, domain, env) for c in constraints if str(c).strip()]
    except _Unsupported as e:
        return ToolResult(content="", error=f"unsupported constraint: {e}")
    if not cons:
        return ToolResult(content="", error="pass a non-empty list of constraints")
    s = z3.Solver()
    s.set("timeout", 5000)
    s.add(*cons)
    res = s.check()
    if res == z3.sat:
        m = s.model()
        sol = ", ".join(f"{d.name()}={m[d]}" for d in m.decls())
        return ToolResult(content=f"SAT — a solution: {sol}", data={"result": "sat", "model": sol})
    if res == z3.unsat:
        return ToolResult(content="UNSAT — no assignment satisfies all constraints (proven infeasible).",
                          data={"result": "unsat"})
    return ToolResult(content="UNKNOWN — z3 could not decide within the time limit.",
                      data={"result": "unknown"})
