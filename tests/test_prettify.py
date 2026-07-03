"""Terminal math rendering — LaTeX→Unicode + markdown, and the renderer/command wiring."""

from __future__ import annotations

import types

from genai_studio.agents.repl.commands import ReplContext, _pretty, build_registry
from genai_studio.agents.repl.prettify import latex_to_unicode, markdown_to_ansi, prettify
from genai_studio.agents.repl.render import StreamRenderer


def test_latex_common_math():
    assert latex_to_unicode(r"$\boxed{\frac{\pi^2}{4}}$") == "【 π²/4 】"
    assert latex_to_unicode(r"$x^2 + \sqrt{3}$") == "x² + √(3)"
    assert latex_to_unicode(r"$\alpha \le \beta \ne \gamma$") == "α ≤ β ≠ γ"
    assert latex_to_unicode(r"$x \in \mathbb{R}$") == "x ∈ ℝ"
    assert latex_to_unicode(r"$a_{ij} \times 10^{-2}$") == "aᵢⱼ × 10⁻²"


def test_snake_case_identifiers_survive_prettify():
    # regression: '_' inside an identifier must NOT be read as a math subscript
    # (mcp__filesystem__list_allowed_directories -> mcpfilesystemlistₐllowed_directories)
    for ident in ("mcp__filesystem__list_allowed_directories", "read_file", "get_file_info",
                  "a_long_snake_case_name", "list_directory_with_sizes"):
        assert latex_to_unicode(ident) == ident
    # but genuine single-variable math subscripts still convert
    assert latex_to_unicode(r"$a_1 + x_i$") == "a₁ + xᵢ"


def test_latex_big_operators_as_readable_ranges():
    # limits render as ranges, not cramped subscripts; thin-space \, becomes a real space
    assert latex_to_unicode(r"$\sum_{i=1}^{n} i$") == "∑[i=1..n] i"
    assert latex_to_unicode(r"$\int_0^1 x\,dx$") == "∫[0..1] x dx"
    assert latex_to_unicode(r"$\prod_{k=1}^{n} k$") == "∏[k=1..n] k"
    assert latex_to_unicode(r"$\lim_{x \to 0} f$") == "lim[x → 0] f"
    assert latex_to_unicode(r"$\int_{-\infty}^{\infty} e^{-x^2}\,dx$") == "∫[-∞..∞] e^(-x²) dx"


def test_latex_accents_binom_setops():
    assert latex_to_unicode(r"$\vec{v} + \hat{x}$") == "v⃗ + x̂"
    assert latex_to_unicode(r"$\bar{X}$") == "X̄"
    assert latex_to_unicode(r"$\binom{n}{k}$") == "C(n, k)"
    assert latex_to_unicode(r"$A \cup B \subseteq \mathbb{R}$") == "A ∪ B ⊆ ℝ"
    assert latex_to_unicode(r"$x \equiv 3 \pmod{7}$") == "x ≡ 3 (mod 7)"


def test_latex_transpose_floor_angle_cfrac():
    assert latex_to_unicode(r"$\vec{x}^{\top}$") == "x⃗ᵀ"                    # transpose superscript
    assert latex_to_unicode(r"$\lfloor x \rfloor$") == "⌊x⌋"                 # floor, tightened
    assert latex_to_unicode(r"$\langle u,v \rangle$") == "⟨u,v⟩"            # angle brackets
    assert latex_to_unicode(r"$\cfrac{1}{1 + \cfrac{1}{x}}$") == "1/(1 + 1/x)"  # nested continued fraction
    assert latex_to_unicode(r"$\bigl(a\bigr)$") == "(a)"                    # sizing delimiters stripped


def test_latex_matrix_grid():
    out = latex_to_unicode(r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}")
    assert "[ 1  2 ]" in out and "[ 3  4 ]" in out


def test_latex_cases_block():
    out = latex_to_unicode(r"\begin{cases} a & x>0 \\ b & x<0 \end{cases}")
    assert "⎧ a" in out and "⎩ b" in out and "if x>0" in out


def test_markdown_plain_when_no_color():
    out = markdown_to_ansi("# Title\n**bold** `code`\n- one", color=False)
    assert "Title" in out and "bold" in out and "\033[" not in out and "• one" in out


def test_markdown_bold_adds_ansi_with_color():
    assert "\033[1m" in markdown_to_ansi("**hi**", color=True)


def test_prettify_fails_open_on_garbage():
    # never raises; returns a string even for pathological input
    assert isinstance(prettify(r"\frac{unbalanced", color=False), str)
    assert prettify("", color=False) == ""


def test_renderer_pretty_flag():
    on = StreamRenderer(color=False, pretty=True)
    off = StreamRenderer(color=False, pretty=False)
    assert on._pretty(r"$x^2$") == "x²"
    assert off._pretty(r"$x^2$") == r"$x^2$"           # raw when off


def test_renderer_unwraps_json_answer_envelope():
    import io
    buf = io.StringIO()
    r = StreamRenderer(color=False, pretty=False, stream=buf)
    r._seg = ['{"answer": "Access is limited to /tmp."}']    # model wrapped its prose in JSON
    r._flush_text()
    out = buf.getvalue()
    assert "Access is limited to /tmp." in out and '"answer"' not in out


def test_renderer_unwraps_NESTED_json_envelope():
    from genai_studio.agents.repl.render import _unwrap_answer
    # the leaked shape from qwen: {"id":…, "response":{"result":"…"}} must extract the inner text
    assert _unwrap_answer('{"id":"text-b8dde5","response":{"result":"(x-2)*(x**2+2*x+4)"}}') == "(x-2)*(x**2+2*x+4)"
    assert _unwrap_answer('{"a":1,"b":2,"c":3}') == '{"a":1,"b":2,"c":3}'   # no answer field -> unchanged
    assert _unwrap_answer("Just plain text.") == "Just plain text."         # non-JSON untouched


def test_pretty_command_toggles(tmp_path):
    ctx = ReplContext(agent=None, tools=[], approval_config=None, recorder=None, client=None,
                      cfg=types.SimpleNamespace(), cwd=tmp_path, registry=build_registry())
    assert ctx.pretty is True
    _pretty(ctx, "")
    assert ctx.pretty is False
    _pretty(ctx, "on")
    assert ctx.pretty is True
    assert build_registry().get("pretty") is not None
