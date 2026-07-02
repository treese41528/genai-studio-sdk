"""The REPL splash banner + capability summary."""

from __future__ import annotations

import types
from pathlib import Path

from genai_studio import __version__
from genai_studio.agents.repl.cli import _banner, _capabilities, _slash_completer
from genai_studio.agents.repl.commands import build_registry


def _tools(*names):
    return [types.SimpleNamespace(name=n) for n in names]


def test_capabilities_summary():
    caps = _capabilities(_tools("grep", "web_search", "symbolic_math", "prove", "lean_check",
                                "python_exec", "load_dataset", "mcp__fs__read", "final_answer"))
    for expect in ("code", "web", "exact math + proofs", "Lean", "data science", "MCP"):
        assert expect in caps
    assert _capabilities(_tools("final_answer")) == "—"


def test_splash_is_informative_and_plain_without_tty():
    cfg = types.SimpleNamespace(model="qwen2.5:72b", profile="general")
    config = types.SimpleNamespace(mode=types.SimpleNamespace(value="suggest"),
                                   sandbox=types.SimpleNamespace(value="workspace-write"))
    tools = _tools("grep", "symbolic_math", "prove", "web_search", "python_exec", "final_answer")
    out = _banner(cfg, tools, config, Path("/tmp"), 2, [Path("CLAUDE.md")],
                  n_skills=3, preset="balanced", temperature=0.0)
    assert f"v{__version__}" in out and "GenAI Studio" in out          # header + version
    assert "qwen2.5:72b" in out and "balanced" in out and "greedy" in out
    assert "suggest / workspace-write" in out and "memory: CLAUDE.md" in out
    assert "6 tools" in out and "3 skills" in out and "2 custom" in out
    assert "/help" in out and "/quit" in out
    assert "\033[" not in out                                          # no ANSI when stdout isn't a tty


def test_slash_completer_cycles_matches():
    comp = _slash_completer(build_registry())
    matches, state = [], 0
    while (m := comp("/p", state)) is not None:                        # cycle all "/p…" commands
        matches.append(m)
        state += 1
    assert "/plan" in matches and "/pretty" in matches
    assert all(m.startswith("/p") for m in matches)
    assert comp("hello", 0) is None                                   # non-slash: no completion
    assert comp("/help", 0) == "/help" and comp("/help", 5) is None   # exhausts past the matches
