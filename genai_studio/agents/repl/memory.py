"""Project memory — auto-load ``CLAUDE.md`` / ``AGENTS.md`` and inject into the system
prompt; ``/init`` writes a starter ``CLAUDE.md``.

Walks ancestor dirs root→cwd so the most specific (cwd) file is most salient, plus a user
global ``~/.claude/CLAUDE.md``. The combined text is appended to the agent's system prompt
ONCE (the REPL strips ``system`` from threaded history, so it never double-injects).
"""

from __future__ import annotations

from pathlib import Path

_NAMES = ("CLAUDE.md", "AGENTS.md")

_TEMPLATE = """# Project overview

Describe what this project does and how it is structured.

## Build & test
- (commands to build, run, and test)

## Conventions
- (code style, naming, patterns to follow)

## Do / don't
- (anything the agent should always or never do)
"""


def find_memory_files(start) -> list:
    """Memory files from the user global down to ``start`` (root→cwd order)."""
    start = Path(start).resolve()
    out: list = []
    user = Path.home() / ".claude" / "CLAUDE.md"
    if user.is_file():
        out.append(user)
    for d in reversed([start, *start.parents]):          # filesystem root -> cwd
        for name in _NAMES:
            p = d / name
            if p.is_file():
                out.append(p)
    seen, deduped = set(), []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    return deduped


def load_project_memory(start):
    """Return ``(combined_text, files_used)``."""
    files = find_memory_files(start)
    blocks = []
    for p in files:
        try:
            blocks.append(f"## {p}\n{p.read_text('utf-8').strip()}")
        except OSError:
            continue
    return "\n\n".join(blocks), files


def build_system_prompt(base: str, project_memory: str) -> str:
    """Append the project-memory block via the shared ``assemble_system`` combiner (single
    injection point shared with the skills/recall-memory blocks)."""
    from ..compose import assemble_system
    block = ("# Project memory (CLAUDE.md / AGENTS.md)\n" + project_memory.strip()
             if (project_memory or "").strip() else "")
    return assemble_system(base, block)


def init_claude_md(cwd) -> Path:
    p = Path(cwd) / "CLAUDE.md"
    if not p.exists():
        p.write_text(_TEMPLATE, encoding="utf-8")
    return p
