"""Offline tests for project-memory loading + /init."""

from __future__ import annotations

import types

from genai_studio.agents.repl.commands import ReplContext, _init, build_registry
from genai_studio.agents.repl.memory import build_system_prompt, find_memory_files, load_project_memory


def test_ancestor_order_root_before_leaf(tmp_path):
    (tmp_path / "AGENTS.md").write_text("root")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    (sub / "AGENTS.md").write_text("leaf")
    names = [str(p) for p in find_memory_files(sub)]
    assert names.index(str(tmp_path / "AGENTS.md")) < names.index(str(sub / "AGENTS.md"))


def test_load_and_build_system_prompt(tmp_path):
    (tmp_path / "AGENTS.md").write_text("PROJECT RULES")
    text, files = load_project_memory(tmp_path)
    assert "PROJECT RULES" in text and any(p.name == "AGENTS.md" for p in files)
    assert build_system_prompt("BASE", text).startswith("BASE") and "PROJECT RULES" in build_system_prompt("BASE", text)
    assert build_system_prompt("BASE", "") == "BASE"      # empty memory => no change


def test_init_writes_and_reloads(tmp_path):
    agent = types.SimpleNamespace(model="m", system="OLD")
    ctx = ReplContext(agent=agent, tools=[], approval_config=None, recorder=None, client=None,
                      cfg=types.SimpleNamespace(sessions_dir=tmp_path, allow_shell_expansion=False),
                      cwd=tmp_path, registry=build_registry(), base_system="BASE")
    _init(ctx, "")
    assert (tmp_path / "AGENTS.md").exists()               # vendor-neutral
    assert not (tmp_path / "CLAUDE.md").exists()           # not another tool's file
    assert agent.system.startswith("BASE") and "Project overview" in agent.system
