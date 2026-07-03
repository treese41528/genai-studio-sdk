"""assemble_system — the single always-on system-prompt block combiner (priority order + budget)."""

from __future__ import annotations

from genai_studio.agents.compose import assemble_system


def test_orders_blocks_and_keeps_base():
    out = assemble_system("BASE", "MEM", "RECALL", "SKILLS", "DEFER")
    assert out.startswith("BASE")
    # priority order preserved
    assert out.index("MEM") < out.index("RECALL") < out.index("SKILLS") < out.index("DEFER")


def test_skips_empty_blocks():
    out = assemble_system("BASE", "", "  ", "SKILLS")
    assert out == "BASE\n\nSKILLS"


def test_budget_drops_lowest_priority_first():
    # base + first block fit; the second would overflow -> it (and nothing after) is included.
    out = assemble_system("BASE", "A" * 30, "B" * 30, "C" * 30, budget_chars=40)
    assert "AAA" in out and "BBB" not in out and "CCC" not in out


def test_base_always_kept_even_over_budget():
    assert assemble_system("X" * 100, "Y" * 100, budget_chars=10) == "X" * 100


def test_no_budget_includes_everything():
    out = assemble_system("BASE", "A" * 5000, "B" * 5000, budget_chars=0)
    assert "AAAA" in out and "BBBB" in out


def test_build_system_prompt_still_works():
    # the REPL helper now delegates to assemble_system; back-compat preserved
    from genai_studio.agents.repl.memory import build_system_prompt
    assert build_system_prompt("BASE", "") == "BASE"
    out = build_system_prompt("BASE", "PROJECT RULES")
    assert out.startswith("BASE") and "PROJECT RULES" in out


def test_frontmatter_codec_reexported_from_custom():
    # skills.py imports the codec from agents.frontmatter; custom.py re-exports for back-compat
    from genai_studio.agents.frontmatter import parse_frontmatter as pf_core
    from genai_studio.agents.repl.custom import parse_frontmatter as pf_repl
    assert pf_core is pf_repl
    meta, body = pf_core("---\ndescription: hi\n---\nBODY")
    assert meta.get("description") == "hi" and body == "BODY"
