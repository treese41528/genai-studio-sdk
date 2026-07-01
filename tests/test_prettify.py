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


def test_latex_big_operators_as_readable_ranges():
    # limits render as ranges, not cramped subscripts; thin-space \, becomes a real space
    assert latex_to_unicode(r"$\sum_{i=1}^{n} i$") == "∑[i=1..n] i"
    assert latex_to_unicode(r"$\int_0^1 x\,dx$") == "∫[0..1] x dx"
    assert latex_to_unicode(r"$\prod_{k=1}^{n} k$") == "∏[k=1..n] k"
    assert latex_to_unicode(r"$\lim_{x \to 0} f$") == "lim[x → 0] f"
    assert latex_to_unicode(r"$\int_{-\infty}^{\infty} e^{-x^2}\,dx$") == "∫[-∞..∞] e^(-x²) dx"


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


def test_pretty_command_toggles(tmp_path):
    ctx = ReplContext(agent=None, tools=[], approval_config=None, recorder=None, client=None,
                      cfg=types.SimpleNamespace(), cwd=tmp_path, registry=build_registry())
    assert ctx.pretty is True
    _pretty(ctx, "")
    assert ctx.pretty is False
    _pretty(ctx, "on")
    assert ctx.pretty is True
    assert build_registry().get("pretty") is not None
