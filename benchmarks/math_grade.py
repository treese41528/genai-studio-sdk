"""MATH answer extraction + equivalence grading (sympy-backed).

Grading competition-math answers is finicky: the gold is a LaTeX expression and the model's answer is
free text. ``extract_answer`` pulls the final answer (preferring ``\\boxed{…}``); ``is_equiv``
normalizes both to a sympy-parseable form and checks symbolic/numeric equality, falling back to a
normalized string match. Grading noise is small and — crucially — IDENTICAL across conditions, so the
measured baseline-vs-grounded *lift* is valid even where an individual grade is arguable.
"""

from __future__ import annotations

import re


def _last_boxed(s: str) -> str | None:
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(s) and s[i] != "{":
        i += 1
    depth, start = 0, i
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
    return None


def extract_answer(text: str) -> str:
    """Pull the final answer from model output: \\boxed{…}, then 'answer is …', then last line."""
    if not text:
        return ""
    b = _last_boxed(text)
    if b is not None:
        return b.strip()
    m = re.findall(r"(?:final answer|the answer)\s*(?:is|:|=)?\s*\$?([^\n$]+?)\$?\s*\.?\s*$",
                   text, re.I | re.M)
    if m:
        return m[-1].strip().rstrip(".")
    line = [ln for ln in text.strip().splitlines() if ln.strip()]
    return line[-1].strip() if line else ""


def _normalize(s: str) -> str:
    """LaTeX -> a compact sympy-parseable string (best-effort)."""
    s = s.strip().strip("$").strip()
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("\\ ", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\cdot", "*"),
                 ("\\times", "*"), ("\\pi", "pi"), ("^{\\circ}", ""), ("^\\circ", ""),
                 ("\\%", ""), ("%", ""), ("\\$", ""), (",\\!", ""), ("\\displaystyle", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\(?:text|mbox|mathrm|operatorname)\{[^{}]*\}", "", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\w)", r"((\1)/(\2))", s)     # \frac{14}3
    s = re.sub(r"\\frac(\w)\{([^{}]*)\}", r"((\1)/(\2))", s)     # \frac1{2}
    s = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", s)             # \frac14
    s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}", r"((\2))**(1/(\1))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\w)", r"sqrt(\1)", s)
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("\\", "")
    return s.strip().rstrip(".").strip()


def is_equiv(model_output: str, gold: str) -> bool:
    """True if the model's extracted answer equals the gold (symbolically, numerically, or as
    normalized strings)."""
    a, b = extract_answer(model_output), (gold or "").strip()
    na, nb = _normalize(a), _normalize(b)
    if na == nb and na != "":
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                                standard_transformations)
        tr = standard_transformations + (implicit_multiplication_application,)
        ea = sympy.parsing.sympy_parser.parse_expr(na, transformations=tr)
        eb = sympy.parsing.sympy_parser.parse_expr(nb, transformations=tr)
        d = sympy.simplify(ea - eb)
        if d == 0 or getattr(d, "is_zero", False):
            return True
        if abs(float(ea) - float(eb)) < 1e-6:
            return True
    except Exception:
        pass
    return na.replace(" ", "") == nb.replace(" ", "") and na != ""
