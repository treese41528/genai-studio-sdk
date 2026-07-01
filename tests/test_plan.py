"""Planning: the update_plan working-memory tool + the /plan read-only toggle."""

from __future__ import annotations

import types

from genai_studio.agents.approval import ApprovalConfig, ApprovalMode, SandboxPolicy
from genai_studio.agents.repl.commands import ReplContext, _plan, build_registry
from genai_studio.agents.tools._workspace import WorkspaceConfig
from genai_studio.agents.tools.plan import make_plan_tool


def test_update_plan_renders_and_counts():
    up = make_plan_tool()
    res = up.run({"steps": ["[x] read the code", "[~] write the fix", "[ ] run tests"]})
    assert res.error is None
    assert "1/3 complete" in res.content and "now: write the fix" in res.content
    assert "[ ] run tests" in res.content


def test_update_plan_persists_and_updates():
    up = make_plan_tool()
    up.run({"steps": ["[ ] a", "[ ] b"]})
    res = up.run({"steps": ["[x] a", "[~] b"]})
    assert "1/2 complete" in res.content and "now: b" in res.content


def test_update_plan_empty_errors():
    assert make_plan_tool().run({"steps": []}).error


def test_plan_command_toggles_readonly(tmp_path):
    cfg = ApprovalConfig(workspace=WorkspaceConfig(root=tmp_path), mode=ApprovalMode.suggest,
                         sandbox=SandboxPolicy.workspace_write)
    ctx = ReplContext(agent=None, tools=[], approval_config=cfg, recorder=None, client=None,
                      cfg=types.SimpleNamespace(), cwd=tmp_path, registry=build_registry(), base_system="")
    _plan(ctx, "")
    assert cfg.sandbox == SandboxPolicy.read_only               # entered plan mode
    _plan(ctx, "")
    assert cfg.sandbox == SandboxPolicy.workspace_write         # restored on exit


def test_plan_command_is_registered():
    assert build_registry().get("plan") is not None
