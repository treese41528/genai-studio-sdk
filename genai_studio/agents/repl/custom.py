"""File-based custom slash commands — ``.claude/commands/*.md`` (project + ``~/.claude``).

A command file's BODY is the prompt sent to the agent. Optional YAML frontmatter carries
``description``/``argument-hint``/``allowed-tools``/``model``. The frontmatter + template-
expansion codec now lives in ``genai_studio.agents.frontmatter`` (shared with skills); it is
re-exported here for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Codec moved to the agents-core level so skills.py can reuse it without a core→repl import.
from ..frontmatter import _aslist, _parse_yaml_flat, expand_template, parse_frontmatter  # noqa: F401


@dataclass
class CustomCommand:
    name: str
    description: str | None
    argument_hint: str | None
    allowed_tools: list | None
    model: str | None
    body: str
    source: str


def _command_dirs(cwd: Path):
    return [(Path.home() / ".claude" / "commands", "user"),
            (Path(cwd) / ".claude" / "commands", "project")]


def load_custom_commands(cwd) -> dict:
    """Load custom commands; project dir overrides user dir on name collision."""
    cmds: dict = {}
    for d, source in _command_dirs(Path(cwd)):       # user first, project second -> project wins
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.md")):
            try:
                text = p.read_text("utf-8")
            except OSError:
                continue
            meta, body = parse_frontmatter(text)
            cmds[p.stem] = CustomCommand(
                name=p.stem, description=meta.get("description"),
                argument_hint=meta.get("argument-hint"), allowed_tools=_aslist(meta.get("allowed-tools")),
                model=meta.get("model"), body=body, source=source)
    return cmds
# expand_template / _expand_files / _expand_shell now live in genai_studio.agents.frontmatter
# (re-exported above for backward compatibility).
