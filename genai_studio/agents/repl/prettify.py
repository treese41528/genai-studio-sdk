"""``prettify`` — make LaTeX + markdown model output readable in a terminal.

Terminals can't typeset math, but most LaTeX maps cleanly to Unicode. This converts, in order of
cleverness: symbols (Greek, operators, relations, arrows, blackboard/accents), fractions, roots,
super/sub-scripts, big-operator limits as ranges (``∑[i=1..n]``), binomials, accents (``\\vec v`` →
``v⃗``), and multi-line **matrices** (``\\begin{bmatrix}`` → a bracketed grid) and **cases**. Plus
light markdown → ANSI (headers, bold, code, bullets). Everything FAILS OPEN — on any error the
original text is returned, so prettifying can never break output. Built-in, no dependency.
"""

from __future__ import annotations

import re

_B, _DIM, _UND, _RESET = "\033[1m", "\033[2m", "\033[4m", "\033[0m"

# ── Unicode maps ──────────────────────────────────────────────────────────────
_SUP = {**{c: s for c, s in zip("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")},
        **{c: s for c, s in zip("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")}}
_SUB = {**{c: s for c, s in zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")},
        **{c: s for c, s in zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")}}
_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
          "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
          "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "varpi": "ϖ",
          "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ",
          "phi": "φ", "varphi": "ϕ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ",
          "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
          "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "nabla": "∇", "partial": "∂"}
_OPS = {"le": "≤", "leq": "≤", "leqslant": "≤", "ge": "≥", "geq": "≥", "geqslant": "≥", "ne": "≠",
        "neq": "≠", "approx": "≈", "approxeq": "≊", "equiv": "≡", "cong": "≅", "simeq": "≃",
        "sim": "∼", "asymp": "≍", "propto": "∝", "doteq": "≐", "ll": "≪", "gg": "≫", "prec": "≺",
        "succ": "≻", "preceq": "⪯", "succeq": "⪰", "times": "×", "cdot": "·", "div": "÷", "ast": "∗",
        "star": "⋆", "circ": "∘", "bullet": "•", "pm": "±", "mp": "∓", "oplus": "⊕", "ominus": "⊖",
        "otimes": "⊗", "odot": "⊙", "infty": "∞", "aleph": "ℵ", "hbar": "ℏ", "ell": "ℓ", "wp": "℘",
        "Re": "ℜ", "Im": "ℑ", "to": "→", "gets": "←", "rightarrow": "→", "longrightarrow": "⟶",
        "Rightarrow": "⇒", "Longrightarrow": "⟹", "implies": "⟹", "leftarrow": "←",
        "Leftarrow": "⇐", "leftrightarrow": "↔", "Leftrightarrow": "⇔", "iff": "⟺", "mapsto": "↦",
        "uparrow": "↑", "downarrow": "↓", "nearrow": "↗", "searrow": "↘", "hookrightarrow": "↪",
        "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂", "subseteq": "⊆", "subsetneq": "⊊",
        "supset": "⊃", "supseteq": "⊇", "sqsubseteq": "⊑", "cup": "∪", "cap": "∩", "uplus": "⊎",
        "sqcup": "⊔", "sqcap": "⊓", "setminus": "∖", "emptyset": "∅", "varnothing": "∅",
        "forall": "∀", "exists": "∃", "nexists": "∄", "neg": "¬", "lnot": "¬", "land": "∧",
        "wedge": "∧", "lor": "∨", "vee": "∨", "top": "⊤", "bot": "⊥", "vdash": "⊢", "models": "⊨",
        "therefore": "∴", "because": "∵", "angle": "∠", "measuredangle": "∡", "perp": "⊥",
        "parallel": "∥", "nparallel": "∦", "mid": "∣", "triangle": "△", "square": "□",
        "blacksquare": "∎", "diamond": "⋄", "deg": "°", "prime": "′", "dagger": "†", "ddagger": "‡",
        "ldots": "…", "cdots": "⋯", "dots": "…", "vdots": "⋮", "ddots": "⋱", "cong ": "≅",
        "surd": "√", "checkmark": "✓", "clubsuit": "♣", "diamondsuit": "♦", "heartsuit": "♥",
        "spadesuit": "♠", "flat": "♭", "sharp": "♯", "natural": "♮", "quad": "  ", "qquad": "    ",
        "colon": ":", "cdotp": "·", "lVert": "‖", "rVert": "‖", "vert": "|", "Vert": "‖",
        "backslash": "\\", "langle": "⟨", "rangle": "⟩", "lvert": "|", "rvert": "|", "lceil": "⌈",
        "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋", "imath": "ı", "jmath": "ȷ", "intercal": "⊺",
        "nmid": "∤", "smallsetminus": "∖", "leftrightarrows": "⇄", "rightrightarrows": "⇉",
        "twoheadrightarrow": "↠", "longmapsto": "⟼", "hookleftarrow": "↩", "Vdash": "⊩"}
_BIGOP = {"sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
          "bigcup": "⋃", "bigcap": "⋂", "bigoplus": "⨁", "bigotimes": "⨂", "bigvee": "⋁",
          "bigwedge": "⋀", "bigsqcup": "⨆", "limsup": "limsup", "liminf": "liminf", "lim": "lim",
          "max": "max", "min": "min", "sup": "sup", "inf": "inf", "argmax": "argmax", "argmin": "argmin"}
_BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "E": "𝔼", "F": "𝔽", "H": "ℍ",
       "D": "𝔻", "A": "𝔸", "K": "𝕂"}
_CAL = {"L": "ℒ", "F": "ℱ", "H": "ℋ", "N": "𝒩", "O": "𝒪", "P": "𝒫", "B": "ℬ", "E": "ℰ", "R": "ℛ",
        "M": "ℳ", "A": "𝒜", "C": "𝒞", "D": "𝒟", "G": "𝒢", "S": "𝒮", "T": "𝒯", "X": "𝒳"}
_ACCENT = {"hat": "̂", "widehat": "̂", "bar": "̄", "overline": "̄",
           "vec": "⃗", "tilde": "̃", "widetilde": "̃", "dot": "̇",
           "ddot": "̈", "check": "̌", "acute": "́", "grave": "̀",
           "breve": "̆", "mathring": "̊"}
_BRACKETS = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "Bmatrix": ("❴", "❵"),
             "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"), "matrix": (" ", " ")}


def _map(text, table):
    return "".join(table.get(c, c) for c in text)


def _script(text, table, prefix):
    """Convert a super/sub-script body; keep it literal (^x / _x) if any char is unmappable."""
    if text and all(c in table for c in text):
        return _map(text, table)
    return f"{prefix}{text}" if len(text) == 1 else f"{prefix}({text})"


def _bigop(m):
    op, lo, hi = _BIGOP[m.group(1)], m.group(2), m.group(3)
    if lo and hi:
        return f"{op}[{lo}..{hi}]"
    if lo:
        return f"{op}[{lo}]"
    if hi:
        return f"{op}^({hi})"
    return op


def _accent(name, text):
    return "".join(ch + _ACCENT[name] for ch in text) if text else ""


def _matrix(env, body):
    rows = [r for r in re.split(r"\\\\", body) if r.strip()]
    grid = [[latex_to_unicode(c.strip()) for c in row.split("&")] for row in rows]
    ncol = max((len(r) for r in grid), default=0)
    width = [max((len(g[c]) for g in grid if c < len(g)), default=0) for c in range(ncol)]
    lb, rb = _BRACKETS.get(env, ("[", "]"))
    out = []
    for g in grid:
        cells = "  ".join((g[c] if c < len(g) else "").ljust(width[c]) for c in range(ncol))
        out.append(f"{lb} {cells} {rb}")
    return "\n" + "\n".join(out) + "\n"


def _cases(body):
    rows = [r for r in re.split(r"\\\\", body) if r.strip()]
    cells = [[latex_to_unicode(p.strip()) for p in row.split("&")] for row in rows]
    w0 = max((len(c[0]) for c in cells if c), default=0)
    n = len(cells)
    out = []
    for i, c in enumerate(cells):
        brace = "❴" if n == 1 else ("⎧" if i == 0 else "⎩" if i == n - 1 else "⎨")
        expr = (c[0] if c else "").ljust(w0)
        cond = ("   if " + c[1]) if len(c) > 1 and c[1] else ""
        out.append(f" {brace} {expr}{cond}")
    return "\n" + "\n".join(out) + "\n"


_BIGOP_RE = (r"\\(sum|prod|coprod|iiint|iint|int|oint|bigcup|bigcap|bigoplus|bigotimes|bigsqcup|"
             r"bigvee|bigwedge|limsup|liminf|lim|max|min|sup|inf)(?:_\{([^{}]*)\})?(?:\^\{([^{}]*)\})?")


def latex_to_unicode(s: str) -> str:
    """Best-effort LaTeX → Unicode (built-in; no dependency). Fails open to the original text."""
    try:
        # 1. environments that need multi-line layout — do first, render cells recursively
        s = re.sub(r"\\begin\{(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix)\}(.*?)\\end\{\1\}",
                   lambda m: _matrix(m.group(1), m.group(2)), s, flags=re.S)
        s = re.sub(r"\\begin\{(?:cases|dcases)\}(.*?)\\end\{(?:cases|dcases)\}",
                   lambda m: _cases(m.group(1)), s, flags=re.S)
        s = re.sub(r"\\begin\{(?:align\*?|aligned|gather\*?|equation\*?)\}(.*?)\\end\{[^}]*\}",
                   lambda m: m.group(1), s, flags=re.S)
        # 2. delimiters + spacing
        s = re.sub(r"\\(?:bigg?|Bigg?|biggg?)[lrm]?", "", s)   # sizing delimiters \bigl \Bigr …
        for d in ("\\left", "\\right", "\\!", "\\displaystyle", "\\limits", "\\nolimits", "\\,"):
            s = s.replace(d, " " if d == "\\," else "")
        s = re.sub(r"\\[;:> ]", " ", s)
        s = re.sub(r"\\begingroup|\\endgroup", "", s)
        for d in ("$$", "$", "\\(", "\\)", "\\[", "\\]", "\\|"):
            s = s.replace(d, "‖" if d == "\\|" else "")
        s = re.sub(r"\\\\\s*", "\n", s)                        # row break outside envs -> newline
        s = re.sub(r"\^\{?\\circ\}?", "°", s)                  # degrees
        s = re.sub(r"\^\s*\{?\s*\\?(?:top|intercal)\s*\}?", "ᵀ", s)   # transpose superscript -> ᵀ
        s = re.sub(r"\^\s*\{?\s*\\?dagger\s*\}?", "†", s)      # adjoint/dagger superscript
        s = re.sub(r"\^\{?T\}?(?![\w])", "ᵀ", s)              # A^T transpose
        s = re.sub(r"\\x(right|left)arrow\s*\{([^{}]*)\}",     # labelled arrows
                   lambda m: (f" →[{m.group(2)}] " if m.group(1) == "right" else f" ←[{m.group(2)}] "), s)
        s = re.sub(r"\\xrightarrow", "→", s)
        s = re.sub(r"\\xleftarrow", "←", s)
        # 3. text / styled spans + named number sets
        s = re.sub(r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|operatorname)\*?\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbb\{([A-Z])\}", lambda m: _BB.get(m.group(1), m.group(1)), s)
        s = re.sub(r"\\mathcal\{([A-Z])\}", lambda m: _CAL.get(m.group(1), m.group(1)), s)
        # 5. binomials, mod, escapes
        s = re.sub(r"\\[dt]?binom\{([^{}]*)\}\{([^{}]*)\}", r"C(\1, \2)", s)
        s = re.sub(r"\\pmod\{([^{}]*)\}", r"(mod \1)", s)
        s = re.sub(r"\\bmod\b", "mod", s)
        s = re.sub(r"\\([%$#&_{}])", r"\1", s)
        # 6. Greek + operators (NOT big operators, handled below)
        s = re.sub(r"\\([A-Za-z]+)",
                   lambda m: _GREEK.get(m.group(1)) or _OPS.get(m.group(1)) or m.group(0), s)
        # accents AFTER symbol conversion, so \hat{\theta} -> θ̂ (not the literal word)
        s = re.sub(r"\\(hat|widehat|bar|overline|vec|tilde|widetilde|dot|ddot|check|acute|grave|breve|mathring)\{([^{}]*)\}",
                   lambda m: _accent(m.group(1), m.group(2)), s)
        s = re.sub(r"\\(hat|bar|vec|tilde|dot|check)\s+(\S)", lambda m: _accent(m.group(1), m.group(2)), s)
        s = re.sub(r"([_^])\s*([^{\s\\])", r"\1{\2}", s)       # normalise single scripts to braces
        for _ in range(2):
            s = re.sub(_BIGOP_RE, _bigop, s)
        # 7. boxed / frac / sqrt / remaining scripts (nested passes)
        for _ in range(5):
            s = re.sub(r"\\boxed\{([^{}]*)\}", r"【 \1 】", s)
            s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}",
                       lambda m: f"{_map(m.group(1), _SUP)}√({m.group(2)})", s)
            s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
            s = re.sub(r"\\(?:[dtc])?frac\s*\{([^{}]*)\}\{([^{}]*)\}",   # frac/dfrac/tfrac/cfrac
                       lambda m: f"{_par(m.group(1))}/{_pardenom(m.group(2))}", s)
            s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP, "^"), s)
            s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB, "_"), s)
        s = s.replace("{", "").replace("}", "").replace("\\", "")
        # readability spacing — source LaTeX omits spaces the eye wants (n!∑ -> n! ∑, n≥1 -> n ≥ 1).
        s = re.sub(r"([⌊⌈⟨])\s+", r"\1", s)                         # tighten inside floor/ceil/angle
        s = re.sub(r"\s+([⌋⌉⟩])", r"\1", s)
        s = re.sub(r"(?<=\S)([∑∏∫∬∭∮⋃⋂⨁⨂⋁⋀])", r" \1", s)          # space before a big operator
        s = re.sub(r"(\])(?=[^\s\].,;:)\]])", r"\1 ", s)             # space after a [lo..hi] range
        stash: list = []                                            # protect range interiors (keep i=1 tight)
        s = re.sub(r"\[[^\[\]]*\.\.[^\[\]]*\]",
                   lambda m: (stash.append(m.group(0)), f"\x00{len(stash)-1}\x00")[1], s)
        s = re.sub(r"\s*([=≠≤≥≈≡≅→←↔↦⟹⟸⟺±×÷∈∉])\s*", r" \1 ", s)   # pad binary relations/ops
        s = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)
        s = re.sub(r" +([,.;:)])", r"\1", s)                        # no space before punctuation
        s = re.sub(r"(?m)^ (?=[^\s([{❴⎧⎨⎩|‖])", "", s)             # drop spurious leading space
        # collapse doubled spaces on ordinary lines; a matrix/cases ROW starts with its bracket, so
        # leaving those lines untouched preserves column alignment while tidying inline math.
        return "\n".join(ln if re.match(r"^\s*[\[(❴⎧⎨⎩‖]", ln) else re.sub(r"  +", " ", ln)
                         for ln in s.split("\n"))
    except Exception:
        return s


def _par(x: str) -> str:
    """Parenthesise a fraction part only if it has a TOP-LEVEL operator (so `1`, `k!`, `(-1)ᵏ`,
    `n(n+1)` stay bare, but `1 + x`, `p-1`, `-b ± √…` get wrapped)."""
    x = (x or "").strip()
    depth = 0
    for i, ch in enumerate(x):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and (ch in "+±∓" or (ch == "-" and i > 0 and x[i - 1] not in "+-±∓*/(=,^_ ")):
            return f"({x})"
    return x


def _pardenom(x: str) -> str:
    """Like ``_par`` but also wraps an implicit product denominator (`2a`, `2π`) so `x/2a` can't be
    misread as `x/2·a`."""
    x = x.strip()
    if _par(x) != x:
        return f"({x})"
    return f"({x})" if re.match(r"^-?\d+\s*[A-Za-zπ(]", x) else x


def markdown_to_ansi(s: str, color: bool = True) -> str:
    """Light markdown → ANSI (headers, bold, inline code, bullets). Plain text when color is off."""
    try:
        def b(x):
            return f"{_B}{x}{_RESET}" if color else x
        lines = []
        for ln in s.split("\n"):
            h = re.match(r"^(#{1,6})\s+(.*)$", ln)
            if h:
                lines.append((f"{_B}{_UND}" if color else "") + h.group(2) + (_RESET if color else ""))
                continue
            ln = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", ln)
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
