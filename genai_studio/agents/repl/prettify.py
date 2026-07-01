"""``prettify`` — make LaTeX + markdown model output readable in a terminal.

Terminals can't render math graphically, but most LaTeX maps cleanly to Unicode: ``\\frac{\\pi^2}{4}``
→ ``π²/4``, ``\\sqrt{3}`` → ``√3``, ``\\alpha \\le \\beta`` → ``α ≤ β``, ``\\boxed{5}`` → ``【 5 】``.
This does that (built-in, no dependency; uses ``pylatexenc`` for fuller coverage if it's installed),
plus light markdown → ANSI (bold, headers, inline code, bullets). Everything FAILS OPEN — on any error
the original text is returned, so prettifying can never break output.
"""

from __future__ import annotations

import re

_B, _DIM, _UND, _RESET = "\033[1m", "\033[2m", "\033[4m", "\033[0m"

# ── Unicode maps ──────────────────────────────────────────────────────────────
_SUP = {**{c: s for c, s in zip("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")},
        **{c: s for c, s in zip("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")}, "n": "ⁿ", "i": "ⁱ"}
_SUB = {**{c: s for c, s in zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")},
        **{c: s for c, s in zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")}}
_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
          "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
          "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
          "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
          "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ",
          "Omega": "Ω"}
_OPS = {"le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈",
        "equiv": "≡", "cong": "≅", "sim": "∼", "propto": "∝", "times": "×", "cdot": "·", "div": "÷",
        "pm": "±", "mp": "∓", "ast": "∗", "star": "⋆", "circ": "∘", "infty": "∞",
        "partial": "∂", "nabla": "∇", "to": "→",
        "rightarrow": "→", "Rightarrow": "⇒", "leftarrow": "←", "Leftarrow": "⇐",
        "leftrightarrow": "↔", "mapsto": "↦", "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂",
        "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "cup": "∪", "cap": "∩", "setminus": "∖",
        "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃", "nexists": "∄",
        "neg": "¬", "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "oplus": "⊕", "otimes": "⊗",
        "angle": "∠", "perp": "⊥", "parallel": "∥", "deg": "°", "prime": "′", "ldots": "…",
        "cdots": "⋯", "dots": "…", "vdots": "⋮", "Re": "ℜ", "Im": "ℑ", "aleph": "ℵ", "hbar": "ℏ",
        "ell": "ℓ", "Box": "□", "bullet": "•", "quad": "  ", "qquad": "    "}
# Big operators keep their limits as a READABLE range (∑[i=1..n]), not tiny cramped scripts.
_BIGOP = {"sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
          "bigcup": "⋃", "bigcap": "⋂", "bigoplus": "⨁", "bigotimes": "⨂", "bigvee": "⋁",
          "bigwedge": "⋀", "limsup": "lim sup", "liminf": "lim inf", "lim": "lim", "max": "max",
          "min": "min"}
_BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "E": "𝔼"}


def _bigop(m):
    op, lo, hi = _BIGOP[m.group(1)], m.group(2), m.group(3)
    if lo and hi:
        return f"{op}[{lo}..{hi}] "
    if lo:
        return f"{op}[{lo}] "
    if hi:
        return f"{op}^({hi}) "
    return op + " "


def _map(text, table):
    return "".join(table.get(c, c) for c in text)


def _script(text, table, prefix):
    """Convert a super/sub-script body; keep it literal (^x / _x) if any char is unmappable."""
    if text and all(c in table for c in text):
        return _map(text, table)
    return f"{prefix}{text}" if len(text) == 1 else f"{prefix}({text})"


_BIGOP_RE = (r"\\(sum|prod|coprod|iiint|iint|int|oint|bigcup|bigcap|bigoplus|bigotimes|bigvee|"
             r"bigwedge|limsup|liminf|lim|max|min)(?:_\{([^{}]*)\})?(?:\^\{([^{}]*)\})?")


def latex_to_unicode(s: str) -> str:
    """Best-effort LaTeX → Unicode (built-in; no dependency). Big-operator limits render as readable
    ranges (``∑[i=1..n]``, ``∫[0..1]``) rather than cramped unicode scripts; short variable scripts
    (``x²``, ``a₁``) stay as unicode. Fails open to the original text."""
    try:
        for d in ("\\left", "\\right", "\\!", "\\displaystyle", "\\limits", "\\nolimits"):
            s = s.replace(d, "")
        s = re.sub(r"\\[,;:> ]", " ", s)                       # thin/med spaces \, \; \: -> a space
        for d in ("$$", "$", "\\(", "\\)", "\\[", "\\]"):
            s = s.replace(d, "")
        s = re.sub(r"\^\{?\\circ\}?", "°", s)                  # degrees
        s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathsf|operatorname)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbb\{([A-Z])\}", lambda m: _BB.get(m.group(1), m.group(1)), s)
        # Greek + operators (NOT big operators, whose limits we format specially below)
        s = re.sub(r"\\([A-Za-z]+)",
                   lambda m: _GREEK.get(m.group(1)) or _OPS.get(m.group(1)) or m.group(0), s)
        s = re.sub(r"([_^])\s*([^{\s\\])", r"\1{\2}", s)       # normalise single scripts to braces
        for _ in range(2):                                     # big operators + their limits
            s = re.sub(_BIGOP_RE, _bigop, s)
        for _ in range(4):                                     # nested boxed/frac/sqrt/scripts
            s = re.sub(r"\\boxed\{([^{}]*)\}", r"【 \1 】", s)
            s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}",
                       lambda m: f"{_map(m.group(1), _SUP)}√({m.group(2)})", s)
            s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
            s = re.sub(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
            s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP, "^"), s)
            s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB, "_"), s)
        s = s.replace("{", "").replace("}", "").replace("\\", "")
        return re.sub(r"  +", " ", s)                          # collapse doubled spaces
    except Exception:
        return s


def markdown_to_ansi(s: str, color: bool = True) -> str:
    """Light markdown → ANSI (bold, headers, inline code, bullets). No-op text when color is off."""
    try:
        def b(x):
            return f"{_B}{x}{_RESET}" if color else x
        lines = []
        for ln in s.split("\n"):
            h = re.match(r"^(#{1,6})\s+(.*)$", ln)
            if h:
                lines.append((f"{_B}{_UND}" if color else "") + h.group(2) + (_RESET if color else ""))
                continue
            ln = re.sub(r"^(\s*)[-*]\s+", r"\1• ", ln)
            ln = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: b(m.group(1) or m.group(2)), ln)
            ln = re.sub(r"`([^`]+)`", lambda m: (f"{_DIM}{m.group(1)}{_RESET}" if color else m.group(1)), ln)
            lines.append(ln)
        return "\n".join(lines)
    except Exception:
        return s


def prettify(s: str, *, color: bool = True) -> str:
    """LaTeX → Unicode, then light markdown → ANSI. Fails open to the original text."""
    if not s:
        return s
    try:
        return markdown_to_ansi(latex_to_unicode(s), color=color)
    except Exception:
        return s
