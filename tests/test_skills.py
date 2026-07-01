"""P0 skills — load/precedence, catalog, in-context use_skill dispatch (zero network)."""

from __future__ import annotations

from genai_studio.agents import tool
from genai_studio.agents.skills import (Skill, build_skill_tools, load_skills,
                                        render_skills_catalog)

from conftest import ScriptedClient, calls_tool, says

_called = []


@tool
def _echo(x: str) -> str:
    """Echo text.

    Args:
        x: the text.
    """
    return x


@tool
def _secret(x: str) -> str:
    """A privileged op that records when it runs.

    Args:
        x: the input.
    """
    _called.append(x)
    return f"SECRET:{x}"


def _write_skill(root, name, frontmatter, body):
    d = root / ".genai_studio" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return d


def test_load_and_catalog(tmp_path):
    _write_skill(tmp_path, "greeter", "description: Greet the user warmly\nwhen_to_use: on hello",
                 "Say hello to $ARGUMENTS.")
    skills = load_skills(tmp_path)
    assert "greeter" in skills and skills["greeter"].description == "Greet the user warmly"
    cat = render_skills_catalog(skills)
    assert "greeter: Greet the user warmly" in cat and "use when: on hello" in cat
    assert "use_skill" in cat


def test_use_skill_in_context_expands_body(tmp_path):
    _write_skill(tmp_path, "echoer", "description: Echo", "Repeat this back: $ARGUMENTS")
    bundle = build_skill_tools(tmp_path)
    res = bundle.tool.run({"name": "echoer", "task": "hello world"})
    assert res.error is None and res.content == "Repeat this back: hello world"


def test_unknown_skill_fails_closed(tmp_path):
    _write_skill(tmp_path, "a", "description: A", "body")
    bundle = build_skill_tools(tmp_path)
    res = bundle.tool.run({"name": "nope", "task": ""})
    assert res.error and "Unknown skill" in res.error and "a" in res.error


def test_no_skills_returns_empty_bundle(tmp_path):
    bundle = build_skill_tools(tmp_path)
    assert bundle.tool is None and bundle.catalog == "" and bundle.skills == {}


def test_isolated_flag_parsed(tmp_path):
    _write_skill(tmp_path, "scoped", "description: Scoped\nallowed-tools: [read_file]", "Do the thing.")
    skills = load_skills(tmp_path)
    assert skills["scoped"].isolated is True
    assert skills["scoped"].allowed_tools == ("read_file",)


def test_descriptionless_skipped(tmp_path, recwarn):
    _write_skill(tmp_path, "blank", "model: x", "body with no description")
    skills = load_skills(tmp_path)
    assert "blank" not in skills


def test_isolated_skill_runs_scoped_child(tmp_path):
    _write_skill(tmp_path, "scoped", "description: Scoped skill\nallowed-tools: [_echo]",
                 "Use only your tools.")
    client = ScriptedClient([calls_tool("_echo", {"x": "hi"}), says("done")])
    bundle = build_skill_tools(tmp_path, base_tools=[_echo, _secret], client=client, default_model="m")
    res = bundle.tool.run({"name": "scoped", "task": "go"})
    assert res.error is None and "done" in res.content          # child ran + returned its narrow answer


def test_isolated_skill_filter_blocks_disallowed_tool(tmp_path):
    _called.clear()
    _write_skill(tmp_path, "scoped", "description: Scoped\nallowed-tools: [_echo]", "Do it.")
    # the child tries a NON-allowed tool -> ToolFilterGuard denies -> the function never runs
    client = ScriptedClient([calls_tool("_secret", {"x": "z"}), says("could not, done")])
    bundle = build_skill_tools(tmp_path, base_tools=[_echo, _secret], client=client, default_model="m")
    res = bundle.tool.run({"name": "scoped", "task": "go"})
    assert res.error is None and _called == []                  # _secret was blocked, never executed


def test_isolated_skill_no_client_falls_back_in_context(tmp_path):
    _write_skill(tmp_path, "scoped", "description: Scoped\nallowed-tools: [_echo]", "Instructions here.")
    res = build_skill_tools(tmp_path, base_tools=[_echo], client=None).tool.run(
        {"name": "scoped", "task": "go"})
    assert "Instructions here." in res.content and "in-context" in res.content


def test_project_overrides_user(tmp_path, monkeypatch):
    # user skill, then a project skill of the same name -> project wins
    fake_home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    _write_skill(fake_home, "dup", "description: USER version", "user body")
    project = tmp_path / "proj"
    _write_skill(project, "dup", "description: PROJECT version", "project body")
    skills = load_skills(project)
    assert skills["dup"].description == "PROJECT version" and skills["dup"].source == "project"
