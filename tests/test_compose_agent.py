"""assemble_agent / wire_capabilities — the single composition path (REPL == headless)."""

from __future__ import annotations

from genai_studio.agents import assemble_agent, wire_capabilities
from genai_studio.agents.tool import tool

from conftest import ScriptedClient


@tool
def base_t(x: str) -> str:
    """A base tool.

    Args:
        x: input.
    """
    return x


def _skill(root, name, desc, body="do the thing"):
    d = root / ".genai_studio" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\ndescription: {desc}\n---\n{body}", encoding="utf-8")


def test_wire_capabilities_adds_skills_memory_defer(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")   # isolate user dir
    _skill(tmp_path, "greeter", "Greet the user")
    tools, blocks, tool_search, n_skills = wire_capabilities(
        [base_t], cwd=tmp_path, memory_dir=tmp_path / "mem", skills=True, memory=True, defer=True)
    names = {t.name for t in tools}
    assert {"base_t", "use_skill", "write_memory", "recall_memory"} <= names
    assert n_skills == 1 and tool_search is not None
    assert any("Skills" in b for b in blocks)                            # skills catalog block present


def test_assemble_agent_ready_with_deferral(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    _skill(tmp_path, "greeter", "Greet the user")
    agent = assemble_agent(ScriptedClient([]), profile="general", cwd=tmp_path, model="m",
                           memory_dir=tmp_path / "mem", defer=True)
    names = {t.name for t in agent.tools}
    assert {"use_skill", "write_memory", "recall_memory"} <= names       # meta-tools in the tool list
    assert agent._registry.get("search_tools") is not None               # meta-tool added by deferral setup
    assert "Skills" in agent.system                                      # catalog injected into system
    assert agent.tool_search is not None and len(agent.guards) >= 1      # deferral + approval guard wired
    # the meta-tools stay EAGER under deferral (always visible, not hidden behind search)
    for n in ("use_skill", "write_memory", "recall_memory"):
        assert not agent._registry.is_deferred(n)
