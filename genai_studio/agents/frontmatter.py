"""YAML-frontmatter + template-expansion codec — shared by file-defined REPL custom commands
(``repl/custom.py``) and headless **skills** (``skills.py``).

Lives at the agents-core level (NOT under ``repl/``) so ``skills.py`` can reuse it without a
core→repl dependency. ``expand_template`` runs AUTHOR-controlled directives FIRST (``@path``
inlines a file; `` !`cmd` `` inlines shell output, gated by ``allow_shell``), THEN user
``$ARGUMENTS``/``$1..`` as INERT text — so a user argument can never smuggle a ``!`` shell
directive. Frontmatter parsing is yaml-or-flat and optional-dependency-tolerant.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


def parse_frontmatter(text: str):
    """Split ``---\\n<yaml>\\n---\\n<body>`` into ``(meta: dict, body: str)``; no frontmatter
    → ``({}, text)``."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[text.find("\n") + 1:end]
            body = text[end + 4:].lstrip("\n")
            return _parse_yaml_flat(fm), body
    return {}, text


def _parse_yaml_flat(fm: str) -> dict:
    """Parse frontmatter via PyYAML when present, else a flat ``key: value`` fallback (so the
    codec works without the optional yaml dependency)."""
    try:
        import yaml
        d = yaml.safe_load(fm)
        return d if isinstance(d, dict) else {}
    except Exception:
        meta: dict = {}
        for line in fm.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta


def _aslist(v):
    """Coerce a frontmatter value to a list (``None`` → ``None``; CSV string → list)."""
    if v is None:
        return None
    if isinstance(v, list):
        return v
    return [x.strip() for x in str(v).split(",") if x.strip()]


def expand_template(body: str, args: str, cwd, *, allow_shell: bool = False) -> str:
    """Expand a command/skill body: author ``@path`` + `` !`cmd` `` FIRST, then user args as
    inert text (ordering prevents arg-smuggled shell directives)."""
    out = _expand_files(body, Path(cwd))                          # 1. author @path
    out = _expand_shell(out, Path(cwd), allow_shell)             # 1. author !`cmd`
    out = out.replace("$ARGUMENTS", args or "")                  # 2. user args (inert)
    try:
        argv = shlex.split(args or "")
    except ValueError:
        argv = (args or "").split()

    def _pos(m):
        i = int(m.group(1))
        return argv[i - 1] if 1 <= i <= len(argv) else ""

    return re.sub(r"\$(\d+)", _pos, out)


def _expand_files(body: str, cwd: Path) -> str:
    def repl(m):
        path = m.group(1)
        try:
            content = (cwd / path).read_text("utf-8")
        except OSError:
            return m.group(0)                                    # leave literal if not a file
        return f"\n```{path}\n{content}\n```\n"

    return re.sub(r"@([\w./~-][^\s`]*)", repl, body)


def _expand_shell(body: str, cwd: Path, allow_shell: bool) -> str:
    def repl(m):
        cmd = m.group(1)
        if not allow_shell:
            return f"`(shell expansion disabled: !{cmd})`"
        try:
            return subprocess.run(["bash", "-lc", cmd], cwd=str(cwd), capture_output=True,
                                  text=True, timeout=15).stdout.strip()
        except Exception as e:
            return f"(shell error: {e})"

    return re.sub(r"!`([^`]+)`", repl, body)
