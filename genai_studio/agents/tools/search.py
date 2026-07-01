"""``grep`` / ``glob`` — workspace-confined codebase search (the "understand the repo" tools).

A coding agent that can only ``read_file`` a KNOWN path is half-blind; these let it EXPLORE:
``grep`` searches file contents by regex (ripgrep fast-path — which also respects ``.gitignore`` —
with a pure-Python fallback so it always works), ``glob`` finds files by name pattern. Both are bound
to one :class:`WorkspaceConfig` and confined to it (results never escape the workspace); both skip
noise dirs (``.git``, ``__pycache__``, ``.venv``, ``node_modules``, …) and cap output so a match on a
huge tree can't blow the context window.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from genai_studio.agents import ToolResult, tool

from ._workspace import PathEscape, WorkspaceConfig, _under, resolve_in_workspace

_NOISE = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache",
          ".mypy_cache", ".ruff_cache", ".genai_studio", ".tox", "dist", "build"}


def _is_noise(p: Path, root: Path) -> bool:
    try:
        parts = p.relative_to(root).parts
    except ValueError:
        return True
    return any(part in _NOISE or part.endswith(".egg-info") for part in parts)


def make_search_tools(ws: WorkspaceConfig) -> list:
    """Return ``[grep, glob]`` bound to ``ws``."""

    @tool
    def grep(pattern: str, path: str = ".", glob: str | None = None,
             ignore_case: bool = False, max_matches: int = 100) -> ToolResult:
        """Search file CONTENTS for a regular-expression pattern (like ripgrep). Use this to find
        where a symbol/string is defined or used across the codebase.

        Args:
            pattern: A regular expression to search for.
            path: Directory or file to search under, relative to the workspace root (default: all).
            glob: Optional filename glob to restrict the search, e.g. "*.py" or "**/*.md".
            ignore_case: Case-insensitive match.
            max_matches: Cap on the number of matching lines returned.
        """
        try:
            root = resolve_in_workspace(ws, path, for_write=False)
        except PathEscape as e:
            return ToolResult(content="", error=str(e))
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return ToolResult(content="", error=f"invalid regex: {e}")
        cap = max(1, min(max_matches, 1000))
        hits = _ripgrep(root, pattern, glob, ignore_case, cap) if shutil.which("rg") \
            else _py_grep(root, rx, glob, cap, ws)
        if not hits:
            return ToolResult(content=f"no matches for {pattern!r}"
                              + (f" in {glob}" if glob else ""))
        lines = [f"{rel}:{ln}: {text}" for rel, ln, text in hits[:cap]]
        more = "\n... (more matches; narrow the pattern or set a path/glob)" if len(hits) > cap else ""
        return ToolResult(content="\n".join(lines) + more,
                          data={"matches": len(hits), "files": len({h[0] for h in hits})})

    @tool
    def glob(pattern: str, path: str = ".") -> ToolResult:
        """Find files by NAME pattern (like a glob), newest-first. Use this to locate files before
        reading them.

        Args:
            pattern: A glob such as "*.py", "src/**/*.ts", or "**/test_*.py".
            path: Directory to search under, relative to the workspace root.
        """
        try:
            root = resolve_in_workspace(ws, path, for_write=False)
        except PathEscape as e:
            return ToolResult(content="", error=str(e))
        try:
            found = [p for p in root.glob(pattern)
                     if p.is_file() and _under(p.resolve(), ws.root) and not _is_noise(p, ws.root)]
        except (ValueError, OSError) as e:
            return ToolResult(content="", error=f"bad glob {pattern!r}: {e}")
        if not found:
            return ToolResult(content=f"no files match {pattern!r}")
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        rels = [str(p.relative_to(ws.root)) for p in found[:200]]
        more = "\n... (more; narrow the pattern)" if len(found) > 200 else ""
        return ToolResult(content="\n".join(rels) + more, data={"files": len(found)})

    return [grep, glob]


def _ripgrep(root: Path, pattern: str, glob_pat, ignore_case: bool, cap: int) -> list:
    """Fast path: ripgrep (respects .gitignore). Returns [(relpath, lineno, line)]."""
    cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "--max-count", str(cap)]
    if ignore_case:
        cmd.append("--ignore-case")
    if glob_pat:
        cmd += ["--glob", glob_pat]
    cmd += ["--", pattern, "."]
    try:
        out = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    hits = []
    for line in out.splitlines():
        parts = line.split(":", 2)                       # relpath:lineno:content
        if len(parts) == 3 and parts[1].isdigit():
            rel = os.path.normpath(os.path.join(os.path.relpath(root, root), parts[0]))
            hits.append((parts[0], int(parts[1]), parts[2].strip()[:300]))
        if len(hits) >= cap:
            break
    return hits


def _py_grep(root: Path, rx: re.Pattern, glob_pat, cap: int, ws: WorkspaceConfig) -> list:
    """Fallback: walk + regex (no ripgrep). Skips noise dirs + non-UTF-8 files."""
    hits: list = []
    files = (p for p in root.glob(glob_pat)) if glob_pat else _walk_files(root)
    for p in files:
        if not p.is_file() or _is_noise(p, ws.root) or not _under(p.resolve(), ws.root):
            continue
        try:
            text = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue                                     # binary / unreadable -> skip
        rel = str(p.relative_to(ws.root))
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append((rel, i, line.strip()[:300]))
                if len(hits) >= cap:
                    return hits
    return hits


def _walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _NOISE and not d.endswith(".egg-info")]
        for name in filenames:
            yield Path(dirpath) / name
