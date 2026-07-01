"""Offline tests for the REPL slash registry, custom commands, and session persistence."""

from __future__ import annotations

import types

from genai_studio.agents.client import Message, ToolCall
from genai_studio.agents.repl import session as S
from genai_studio.agents.repl.custom import expand_template, parse_frontmatter, load_custom_commands
from genai_studio.agents.repl.commands import (
    ReplContext, SlashRegistry, build_registry, load_custom_into,
)


# ── custom-command template expansion ─────────────────────────────────────────
def test_expand_arguments_and_positional():
    assert expand_template("hi $ARGUMENTS end", "a b c", ".") == "hi a b c end"
    assert expand_template("[$1][$2][$3]", "x y", ".") == "[x][y][]"


def test_expand_file_inline(tmp_path):
    (tmp_path / "notes.txt").write_text("FILEBODY")
    out = expand_template("see @notes.txt now", "", tmp_path)
    assert "FILEBODY" in out and "```notes.txt" in out


def test_arg_cannot_smuggle_shell(tmp_path):
    # author shell directives expand BEFORE user args; a !`cmd` inside an argument must stay inert
    # (the directive survives LITERALLY -> it was never executed)
    out = expand_template("task: $ARGUMENTS", "!`echo PWNED`", tmp_path, allow_shell=True)
    assert "!`echo PWNED`" in out               # directive intact == not executed


def test_shell_gated_off_by_default(tmp_path):
    out = expand_template("x !`echo HI`", "", tmp_path, allow_shell=False)
    assert "disabled" in out                    # command was NOT run (left as a gated literal)


def test_parse_frontmatter():
    meta, body = parse_frontmatter("---\ndescription: do a thing\nargument-hint: pr-number\n---\nBODY $1")
    assert meta["description"] == "do a thing" and meta["argument-hint"] == "pr-number" and body == "BODY $1"
    meta2, body2 = parse_frontmatter("no frontmatter here")
    assert meta2 == {} and body2 == "no frontmatter here"


def test_load_custom_commands(tmp_path):
    d = tmp_path / ".claude" / "commands"
    d.mkdir(parents=True)
    (d / "greet.md").write_text("---\ndescription: greet\n---\nSay hello to $ARGUMENTS")
    cmds = load_custom_commands(tmp_path)
    assert "greet" in cmds and cmds["greet"].description == "greet" and cmds["greet"].source == "project"


# ── Message serialization fidelity (tool_calls + tool_call_id) ─────────────────
def test_message_roundtrip_with_tool_pair():
    assistant = Message(role="assistant", content="ok",
                        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "x"})])
    tool = Message(role="tool", content="data", tool_call_id="c1", name="read_file")
    for m in (assistant, tool):
        back = S.deserialize_message(S.serialize_message(m))
        assert back.to_openai() == m.to_openai()


# ── session recorder + resume + compact marker ────────────────────────────────
def test_session_record_and_load(tmp_path):
    rec = S.SessionRecorder(tmp_path, model="m", cwd=tmp_path)
    msgs = [Message.user("hello"),
            Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": "1"})]),
            Message(role="tool", content="1", tool_call_id="c1", name="echo"),
            Message.assistant("done")]
    rec.write_messages(msgs, types.SimpleNamespace(stopped="final"))
    rec.close()
    loaded = S.load_history(rec.path)
    assert [m.role for m in loaded] == ["user", "assistant", "tool", "assistant"]
    assert loaded[1].tool_calls[0].id == "c1" and loaded[2].tool_call_id == "c1"


def test_load_history_honors_compact(tmp_path):
    rec = S.SessionRecorder(tmp_path, model="m", cwd=tmp_path)
    rec.write_messages([Message.user("a"), Message.assistant("b")], None)
    rec.write_marker("compact", summary="SUMMARY")
    rec.write_messages([Message.user("c")], None)
    rec.close()
    loaded = S.load_history(rec.path)
    assert loaded[0].content == "SUMMARY" and loaded[-1].content == "c" and len(loaded) == 2


def test_list_sessions(tmp_path):
    S.SessionRecorder(tmp_path, model="m1", cwd=tmp_path).close()
    infos = S.list_sessions(tmp_path)
    assert len(infos) == 1 and infos[0].model == "m1"


# ── slash registry ────────────────────────────────────────────────────────────
def _ctx(tmp_path):
    reg = build_registry()
    agent = types.SimpleNamespace(model="m1")
    ctx = ReplContext(agent=agent, tools=[types.SimpleNamespace(name="read_file")],
                      approval_config=None, recorder=None, client=None,
                      cfg=types.SimpleNamespace(sessions_dir=tmp_path, allow_shell_expansion=False),
                      cwd=tmp_path, registry=reg, history=[Message.user("x")])
    return ctx, reg


def test_registry_clear_and_model(tmp_path):
    ctx, reg = _ctx(tmp_path)
    assert reg.dispatch("/clear", ctx).prompt is None and ctx.history == []
    reg.dispatch("/model gpt-oss:120b", ctx)
    assert ctx.agent.model == "gpt-oss:120b"
    assert reg.dispatch("/help", ctx).is_exit is False
    assert reg.dispatch("/quit", ctx).is_exit is True


def test_registry_custom_command_returns_prompt(tmp_path):
    d = tmp_path / ".claude" / "commands"
    d.mkdir(parents=True)
    (d / "say.md").write_text("Please say: $ARGUMENTS")
    ctx, reg = _ctx(tmp_path)
    load_custom_into(reg, tmp_path)
    res = reg.dispatch("/say hello world", ctx)
    assert res.prompt == "Please say: hello world"
