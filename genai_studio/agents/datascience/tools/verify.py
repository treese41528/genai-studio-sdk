"""``verify_stat`` — ground a claimed statistic by RE-COMPUTING it over the live data namespace.

The data-science analog of ``verify_math``, and the check≪solve idea applied to analysis: checking a
number is far cheaper than producing the analysis, so the agent should VERIFY every figure it reports
against the actual data before concluding — instead of confabulating a plausible value (the failure mode
the reliability eval caught: tools can INCREASE hallucination when the model narrates from memory). The
tool evaluates a pandas/numpy expression over the SAME namespace as ``python_exec``/``load_dataset`` and
confirms it matches the claim.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

_SAFE_BUILTINS = {"len": len, "sum": sum, "abs": abs, "min": min, "max": max, "round": round}


def make_verify_stat(namespace: dict | None = None):
    """Return the ``verify_stat`` tool bound to a shared data ``namespace`` (so it sees the same ``df``
    as python_exec/load_dataset)."""
    namespace = namespace if namespace is not None else {}

    @tool
    def verify_stat(expression: str, expected: float, tol: float = 1e-6, relative: bool = False) -> ToolResult:
        """VERIFY a claimed numeric result by re-computing it over the current data. Evaluates a
        pandas/numpy ``expression`` against the live namespace (the same ``df`` as python_exec) and
        confirms it ~ ``expected``. Use this to CHECK every statistic you report BEFORE concluding —
        never trust a number you did not re-compute.

        Args:
            expression: a Python expression over the namespace that yields ONE number, e.g.
                "df['age'].mean()", "len(df)", "(df['y'] > 0).mean()", "df['a'].corr(df['b'])".
            expected: the value you claim it equals.
            tol: tolerance — absolute by default (or a fraction of ``expected`` if ``relative``).
            relative: compare ``|actual-expected| <= tol*|expected|`` instead of absolute.
        """
        env = dict(namespace)
        try:
            import numpy as np
            env.setdefault("np", np)
        except Exception:
            pass
        try:
            import pandas as pd
            env.setdefault("pd", pd)
        except Exception:
            pass
        try:
            val = eval(expression, {"__builtins__": _SAFE_BUILTINS}, env)   # noqa: S307 — analyst's own ns
        except Exception as e:
            return ToolResult(content="", error=f"could not evaluate {expression!r}: {e} "
                              "(is the data loaded? name it as in python_exec)")
        try:
            actual = float(val)
        except (TypeError, ValueError):
            return ToolResult(content="", error=f"{expression!r} is not a single number "
                              f"(got {type(val).__name__}) — reduce it to one value")
        bound = tol * abs(expected) if relative else tol
        ok = abs(actual - expected) <= bound
        head = "VERIFIED ✓" if ok else "MISMATCH ✗"
        return ToolResult(content=f"{head}: {expression} = {actual:.6g}  (claimed {expected:.6g}, "
                          f"tol {'rel ' if relative else ''}{tol:g})",
                          data={"verdict": ok, "actual": actual, "expected": expected})

    return verify_stat
