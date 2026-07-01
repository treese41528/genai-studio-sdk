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
        "pm": "±", "mp": "∓", "ast": "∗", "star": "⋆", "circ": "∘", "infty": "∞", "sum": "∑",
        "prod": "∏", "int": "∫", "oint": "∮", "partial": "∂", "nabla": "∇", "to": "→",
        "rightarrow": "→", "Rightarrow": "⇒", "leftarrow": "←", "Leftarrow": "⇐",
        "leftrightarrow": "↔", "mapsto": "↦", "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂",
        "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "cup": "∪", "cap": "∩", "setminus": "∖",
        "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃", "nexists": "∄",
        "neg": "¬", "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "oplus": "⊕", "otimes": "⊗",
        "angle": "∠", "perp": "⊥", "parallel": "∥", "deg": "°", "prime": "′", "ldots": "…",
        "cdots": "⋯", "dots": "…", "vdots": "⋮", "Re": "ℜ", "Im": "ℑ", "aleph": "ℵ", "hbar": "ℏ",
        "ell": "ℓ", "Box": "□", "bullet": "•", "quad": "  ", "qquad": "    "}
_BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "E": "𝔼"}


def _map(text, table):
    return "".join(table.get(c, c) for c in text)


def _script(text, table, prefix):
    """Convert a super/sub-script body; keep it literal (^x / _x) if any char is unmappable."""
    if text and all(c in table for c in text):
        return _map(text, table)
    return f"{prefix}{text}" if len(text) == 1 else f"{prefix}({text})"


def latex_to_unicode(s: str) -> str:
    """Best-effort LaTeX → Unicode (built-in; no dependency)."""
    try:
        return _pylatexenc(s)
    except Exception:
        pass
    try:
        for d in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\displaystyle", "$$", "$",
                  "\\(", "\\)", "\\[", "\\]", "\\ "):
            s = s.replace(d, " " if d in ("\\ ", "\\quad") else "")
        s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbb\{([A-Z])\}", lambda m: _BB.get(m.group(1), m.group(1)), s)
        for _ in range(3):                                     # a few passes for nested braces
            s = re.sub(r"\\boxed\{([^{}]*)\}", r"【 \1 】", s)
            s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}", lambda m: f"{_map(m.group(1), _SUP)}√({m.group(2)})", s)
            s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
            s = re.sub(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
            s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP, "^"), s)
            s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB, "_"), s)
        s = re.sub(r"\^(\w)", lambda m: _script(m.group(1), _SUP, "^"), s)
        s = re.sub(r"_(\w)", lambda m: _script(m.group(1), _SUB, "_"), s)
        s = re.sub(r"\\([A-Za-z]+)",
                   lambda m: _GREEK.get(m.group(1)) or _OPS.get(m.group(1)) or m.group(0), s)
        return s.replace("{", "").replace("}", "").replace("\\", "")
    except Exception:
        return s


def _pylatexenc(s: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text          # optional, fuller coverage
    # only run it on math spans so prose/markdown is untouched
    def conv(m):
        return LatexNodes2Text().latex_to_text(m.group(1))
    out = re.sub(r"\$([^$]+)\$", conv, s)
    out = re.sub(r"\\\[(.+?)\\\]", conv, out, flags=re.S)
    out = re.sub(r"\\\((.+?)\\\)", conv, out)
    return out


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
