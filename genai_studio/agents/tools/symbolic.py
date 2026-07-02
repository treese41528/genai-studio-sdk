"""Exact-math grounding tools — `verify_math`, `symbolic_math`, `matrix_op` (the `[math]` extra).

Non-frontier models slip on arithmetic, algebra, calculus, and linear algebra. These tools make the
model **compute and verify with a real CAS (sympy) instead of reasoning numbers in its head**:
- `verify_math` — the flagship: decide a claimed equality/inequality TRUE/FALSE (self-verification).
- `symbolic_math` — exact solve / simplify / factor / expand / diff / integrate / limit / series /
  evaluate (exact rationals + radicals; `∫sin` over 0..π is exactly 2, `1/10+2/10` is `3/10`).
- `matrix_op` — exact linear algebra over rationals (a singular matrix's det stays exactly 0).

Safe: expressions are parsed with sympy's parser (no `exec`), so unlike `python_exec` these are fine
to expose on any input, and they're pure computation (read-only, no state). sympy is imported lazily
(a one-line install hint on a missing `[math]` install)."""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

from ..datascience._guard import _require


def _sp():
    return _require("sympy", "math")


def _parse(expr: str):
    """Parse a math expression: implicit multiplication (2x → 2*x), `^` as power, `n!` factorial.
    Uses ``implicit_multiplication`` (NOT the ``_application`` variant, whose ``split_symbols`` mangles
    subscripted names — ``a1`` → ``a*1`` = ``a``, ``x1 + x2`` → ``3*x``). So ``x1``, ``a2`` stay single
    symbols; write ``x*y`` for a product of two distinct variables."""
    sp = _sp()
    from sympy.parsing.sympy_parser import (convert_xor, factorial_notation,
                                            implicit_multiplication, standard_transformations)
    tr = standard_transformations + (implicit_multiplication, convert_xor, factorial_notation)
    return sp.parsing.sympy_parser.parse_expr(expr, transformations=tr)


_RELS = [("==", "eq"), ("!=", "ne"), ("<=", "le"), (">=", "ge"), ("<", "lt"), (">", "gt")]


def _split_relation(claim: str):
    for tok, op in _RELS:
        if tok in claim:
            i = claim.index(tok)
            return claim[:i].strip(), op, claim[i + len(tok):].strip()
    if "=" in claim:                                    # a lone '=' means equality
        i = claim.index("=")
        return claim[:i].strip(), "eq", claim[i + 1:].strip()
    return None


def _zero_state(sp, expr):
    """True/False/None — is `expr` symbolically zero (None = undecided)."""
    try:
        s = sp.simplify(expr)
        if s == 0:
            return True
        z = s.is_zero
        return z if z is not None else None
    except Exception:
        return None


def _numeric_zero(sp, expr, tol):
    """Numeric fallback: sample free symbols, is |expr| < tol everywhere? -> (verdict, witness)."""
    syms = sorted(expr.free_symbols, key=str)
    for trial in ([{}] if not syms else [{s: sp.Integer(v) for s, v in zip(syms, vals)}
                                         for vals in ((2, 3, 5, 7), (1, 4, 6, 9), (3, 5, 8, 2))]):
        try:
            val = complex(sp.N(expr.subs(trial)))
        except Exception:
            return None, None
        if abs(val) > max(tol, 1e-9):
            return False, {str(k): str(v) for k, v in trial.items()}
    return True, None


def _z3_verify(claim: str):
    """Try a SOUND z3 decision (over the reals). Returns a proven/disproven ToolResult, or None when
    z3 is unavailable / the claim is outside its arithmetic fragment / it's inconclusive."""
    try:
        from .smt import z3_decide
    except Exception:
        return None
    verdict, ce = z3_decide(claim, "real")
    if verdict == "proven":
        return ToolResult(content=f"PASS (TRUE): {claim}  [proven for all reals by SMT]",
                          data={"verdict": True, "proof": "smt"})
    if verdict == "disproven":
        return ToolResult(content=f"FAIL (FALSE): {claim}  [SMT counterexample: {ce}]",
                          data={"verdict": False, "counterexample": ce})
    return None


@tool
def verify_math(claim: str, assume: str | None = None, tol: float = 1e-12) -> ToolResult:
    """Decide whether a math CLAIM is TRUE or FALSE with an exact CAS — use this to CHECK any
    equality, identity, or inequality BEFORE you rely on it. Symbolic-first (proves it for all
    values); falls back to arbitrary-precision numeric only when symbolic proof is inconclusive.
    Echoes the parsed claim so a mis-read is visible; gives a counterexample when FALSE.

    Args:
        claim: e.g. "sqrt(2)*sqrt(2) == 2", "x**2-1 == (x-1)*(x+1)", "1/10 + 2/10 == 3/10",
            "factorial(n) > 2**n". Use "==" for equality (not "="); "^" means power.
        assume: (currently informational) assumptions like "x>0", "n integer".
        tol: numeric tolerance for the fallback when symbolic proof is inconclusive.
    """
    sp = _sp()
    rel = _split_relation(claim)
    if rel is None:
        return ToolResult(content="", error="claim must contain a relation: ==, !=, <, >, <=, >=")
    lhs_s, op, rhs_s = rel
    try:
        L, R = _parse(lhs_s), _parse(rhs_s)
    except Exception as e:
        return ToolResult(content="", error=f"could not parse claim: {e}")
    diff = L - R
    _sym = {"eq": "==", "ne": "!=", "le": "<=", "ge": ">=", "lt": "<", "gt": ">"}[op]
    parsed = f"{sp.sstr(L)} {_sym} {sp.sstr(R)}"

    def _out(verdict, note="", extra=None):
        head = "PASS (TRUE)" if verdict else "FAIL (FALSE)"
        body = f"{head}: {parsed}" + (f"  [{note}]" if note else "") + (f"\n{extra}" if extra else "")
        return ToolResult(content=body, data={"verdict": bool(verdict), "parsed": parsed})

    if op in ("eq", "ne"):
        z = _zero_state(sp, diff)
        if z is not None:                                    # symbolic proof (exact)
            return _out(z if op == "eq" else not z, "proven symbolically",
                        extra=None if z else f"difference simplifies to {sp.sstr(sp.simplify(diff))}")
        v = _z3_verify(claim)                                # sound SMT proof, if in z3's fragment
        if v is not None:
            return v
        z, witness = _numeric_zero(sp, diff, tol)            # heuristic — explicitly NOT a proof
        if z is None:
            return ToolResult(content=f"UNKNOWN: could not decide {parsed} (try prove())",
                              data={"verdict": None})
        return _out(z if op == "eq" else not z,
                    "checked numerically — NOT a proof" if z else "counterexample " + str(witness),
                    extra=None if z else f"difference simplifies to {sp.sstr(sp.simplify(diff))}")
    # inequalities: sign of (L - R)
    d = sp.simplify(diff)
    decide = {"lt": d.is_negative, "gt": d.is_positive, "le": d.is_nonpositive, "ge": d.is_nonnegative}[op]
    if decide is not None:                                    # symbolic proof
        return _out(decide, "proven symbolically", extra=f"L - R simplifies to {sp.sstr(d)}")
    v = _z3_verify(claim)                                     # sound SMT proof (over the reals)
    if v is not None:
        return v
    return ToolResult(content=f"UNKNOWN: cannot decide {parsed} symbolically or via SMT "
                      "(for an integer-only claim try prove(domain='int'))", data={"verdict": None})


@tool
def symbolic_math(operation: str, expression: str, symbol: str = "x", point: str | None = None,
                  lower: str | None = None, upper: str | None = None, order: int = 1,
                  precision: int | None = None) -> ToolResult:
    """Exact symbolic algebra & calculus — use for anything you'd otherwise work out by hand.
    The framework picks the right sympy backend for you; results are EXACT (rationals + radicals),
    never a float artifact.

    Args:
        operation: one of solve | simplify | factor | expand | diff | integrate | limit | series |
            evaluate.
        expression: the expression (e.g. "x**2 - 2", "sin(x)"). For solve you may write "x**2 = 2".
        symbol: the variable to act on (default x).
        point: limit point / series center.
        lower: lower bound for a definite integral.
        upper: upper bound for a definite integral.
        order: derivative order / series order.
        precision: if set, also return an N-significant-digit decimal.
    """
    sp = _sp()
    op = operation.strip().lower()
    valid = {"solve", "simplify", "factor", "expand", "diff", "integrate", "limit", "series", "evaluate"}
    if op not in valid:
        return ToolResult(content="", error=f"unknown operation {operation!r}; choose from {sorted(valid)}")
    try:
        x = sp.Symbol(symbol)
        if op == "solve":
            if "=" in expression and "==" not in expression:
                l, r = expression.split("=", 1)
                eq = sp.Eq(_parse(l), _parse(r))
            else:
                eq = _parse(expression)
            sols = sp.solve(eq, x, dict=False)
            result = sols
        else:
            e = _parse(expression)
            if op == "simplify":
                result = sp.simplify(e)
            elif op == "factor":
                result = sp.factor(e)
            elif op == "expand":
                result = sp.expand(e)
            elif op == "diff":
                result = sp.diff(e, x, order)
            elif op == "integrate":
                result = sp.integrate(e, (x, _parse(lower), _parse(upper))) if (lower and upper) \
                    else sp.integrate(e, x)
            elif op == "limit":
                result = sp.limit(e, x, _parse(point) if point else 0)
            elif op == "series":
                result = sp.series(e, x, _parse(point) if point else 0, order)
            else:  # evaluate
                result = sp.nsimplify(e) if e.free_symbols == set() else sp.simplify(e)
    except Exception as ex:
        return ToolResult(content="", error=f"{op} failed: {type(ex).__name__}: {ex}")
    text = sp.sstr(result)
    dec = ""
    if precision:
        try:
            dec = f"\n≈ {sp.N(result if not isinstance(result, list) else result[0], precision)}"
        except Exception:
            pass
    return ToolResult(content=f"{op}: {text}{dec}", data={"result": text})


@tool
def matrix_op(matrix: list[list[float]], operation: str, vector: list[float] | None = None) -> ToolResult:
    """Exact linear algebra over RATIONALS — a singular matrix's determinant stays exactly 0 (float
    numpy reports ~1e-16 and wrongly "inverts" it). Use for any 2x2+ matrix work.

    Args:
        matrix: a list of equal-length numeric rows, e.g. [[1, 2], [3, 4]].
        operation: det | inv | rref | rank | eigenvals | eigenvects | solve | nullspace | transpose.
        vector: right-hand side b (a list) for operation="solve" (Ax = b).
    """
    sp = _sp()
    op = operation.strip().lower()
    valid = {"det", "inv", "rref", "rank", "eigenvals", "eigenvects", "solve", "nullspace", "transpose"}
    if op not in valid:
        return ToolResult(content="", error=f"unknown operation {operation!r}; choose from {sorted(valid)}")
    try:
        M = sp.Matrix([[sp.Rational(str(v)) for v in row] for row in matrix])   # exact entries
    except Exception as e:
        return ToolResult(content="", error=f"bad matrix: {e}")
    try:
        if op == "det":
            r = M.det()
        elif op == "inv":
            if M.det() == 0:
                return ToolResult(content="", error="matrix is singular (determinant is exactly 0) — no inverse")
            r = M.inv()
        elif op == "rref":
            r = M.rref()[0]
        elif op == "rank":
            r = M.rank()
        elif op == "eigenvals":
            r = M.eigenvals()
        elif op == "eigenvects":
            r = M.eigenvects()
        elif op == "nullspace":
            r = M.nullspace()
        elif op == "transpose":
            r = M.T
        else:  # solve Ax = b
            if vector is None:
                return ToolResult(content="", error="operation 'solve' needs a vector b")
            b = sp.Matrix([sp.Rational(str(v)) for v in vector])
            r = M.solve(b)
    except Exception as ex:
        return ToolResult(content="", error=f"{op} failed: {type(ex).__name__}: {ex}")
    return ToolResult(content=f"{op}: {sp.sstr(r)}", data={"result": sp.sstr(r)})
