"""The new REPL slash commands (offline-testable: cost/retry/undo/preset/export/reload/mcp/doctor)."""

from __future__ import annotations

import types

from genai_studio.agents.repl.commands import (ReplContext, _cost, _doctor, _export, _mcp, _preset,
                                               _reload, _retry, _undo, _verify, build_registry)


def _ctx(tmp_path, **kw):
    agent = types.SimpleNamespace(model="qwen2.5:72b", system="", temperature=None, sampling={},
                                  tools=[], _registry=None, tool_search=None, guards=[])

    class _Client:
        def complete(self, *a, **k):
            raise RuntimeError("offline")

    ctx = ReplContext(agent=agent, tools=[types.SimpleNamespace(name="grep")], approval_config=None,
                      recorder=None, client=_Client(), cfg=types.SimpleNamespace(memory_dir=None),
                      cwd=tmp_path, registry=build_registry())
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def test_all_new_commands_registered():
    names = set(build_registry().names())
    assert {"cost", "retry", "undo", "preset", "profile", "export", "reload", "mcp",
            "verify", "doctor"} <= names


def test_undo_truncates_history(tmp_path):
    ctx = _ctx(tmp_path, history=[1, 2, 3, 4], turn_marks=[0, 2])
    _undo(ctx, "")
    assert ctx.history == [1, 2] and ctx.turn_marks == [0]
    _undo(ctx, "")
    assert ctx.history == [] and ctx.turn_marks == []
    _undo(ctx, "")                                       # nothing to undo — must not raise


def test_retry_returns_last_prompt_and_drops_prior(tmp_path):
    ctx = _ctx(tmp_path, last_prompt="do X", history=[1, 2], turn_marks=[0])
    r = _retry(ctx, "")
    assert r.prompt == "do X" and ctx.history == []      # replaced the previous attempt


def test_cost_reports_tokens(tmp_path, capsys):
    ctx = _ctx(tmp_path, total_usage={"prompt": 100, "completion": 50, "total": 150}, turn_marks=[0])
    _cost(ctx, "")
    assert "150" in capsys.readouterr().out


def test_preset_switches_model_and_sampling(tmp_path):
    ctx = _ctx(tmp_path)
    _preset(ctx, "careful")
    assert "deepseek" in ctx.agent.model and ctx.agent.temperature == 0.0   # careful = greedy reasoner
    _preset(ctx, "fast")
    assert "llama4" in ctx.agent.model


def test_export_writes_markdown(tmp_path):
    ctx = _ctx(tmp_path, history=[types.SimpleNamespace(role="user", content="hi", tool_calls=None),
                                  types.SimpleNamespace(role="assistant", content="hello", tool_calls=None)])
    _export(ctx, str(tmp_path / "out.md"))
    txt = (tmp_path / "out.md").read_text()
    assert "hi" in txt and "hello" in txt and "user" in txt


def test_mcp_reload_doctor_run_offline(tmp_path, capsys):
    ctx = _ctx(tmp_path)
    _mcp(ctx, "")
    _reload(ctx, "")
    _doctor(ctx, "")                                     # gateway ping raises -> handled, no crash
    out = capsys.readouterr().out
    assert "MCP" in out and "reloaded" in out and "environment" in out and "gateway" in out


def test_verify_without_claim_is_usage(tmp_path, capsys):
    _verify(_ctx(tmp_path, history=[]), "")              # no claim, no answer -> usage, no gateway call
    assert "usage" in capsys.readouterr().out
