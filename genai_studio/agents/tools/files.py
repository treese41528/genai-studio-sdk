"""``read_file`` / ``write_file`` / ``edit_file`` — workspace-confined coding tools.

Built by a factory bound to one :class:`WorkspaceConfig` (the confinement floor lives
in ``resolve_in_workspace``). Writes are atomic (``mkstemp`` in the target dir +
``os.replace``) so a crash never leaves a half-written file. ``edit_file`` requires the
``old`` string to occur EXACTLY ONCE (mirrors the harness Edit), so the model must add
context rather than blindly replacing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from genai_studio.agents import ToolResult, tool

from ._workspace import PathEscape, WorkspaceConfig, resolve_in_workspace


def make_file_tools(ws: WorkspaceConfig) -> list:
    """Return ``[read_file, write_file, edit_file]`` bound to ``ws``."""

    @tool
    def read_file(path: str, max_bytes: int = 100_000) -> ToolResult:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: File path, relative to the workspace root (or absolute inside it).
            max_bytes: Maximum number of bytes to read; the rest is truncated.
        """
        try:
            p = resolve_in_workspace(ws, path, for_write=False)
        except PathEscape as e:
            return ToolResult(content="", error=str(e))
        if not p.is_file():
            return ToolResult(content="", error=f"not a file: {path}")
        try:
            size = p.stat().st_size
            data = p.read_bytes()[: max(0, max_bytes)]
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(content="", error=f"{path} is not UTF-8 text")
        except OSError as e:
            return ToolResult(content="", error=f"cannot read {path}: {e}")
        suffix = f"\n... (truncated at {len(data)} of {size} bytes)" if size > len(data) else ""
        return ToolResult(content=text + suffix, data={"bytes": len(data), "path": str(p)})

    @tool
    def write_file(path: str, content: str) -> ToolResult:
        """Create or overwrite a text file in the workspace (atomic write).

        Args:
            path: File path to write — must be inside the workspace and not under .git.
            content: The full new file contents.
        """
        try:
            p = resolve_in_workspace(ws, path, for_write=True)
        except PathEscape as e:
            return ToolResult(content="", error=str(e))
        existed = p.exists()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(p, content)
        except OSError as e:
            return ToolResult(content="", error=f"cannot write {path}: {e}")
        verb = "overwrote" if existed else "created"
        return ToolResult(content=f"{verb} {path} ({len(content.encode('utf-8'))} bytes)")

    @tool
    def edit_file(path: str, old: str, new: str) -> ToolResult:
        """Replace an exact string in a file. The ``old`` string must occur exactly once.

        Args:
            path: File to edit (inside the workspace).
            old: Exact existing text to replace; must be UNIQUE in the file.
            new: Replacement text.
        """
        try:
            p = resolve_in_workspace(ws, path, for_write=True)
        except PathEscape as e:
            return ToolResult(content="", error=str(e))
        if not p.is_file():
            return ToolResult(content="", error=f"not a file: {path}")
        try:
            text = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError) as e:
            return ToolResult(content="", error=f"cannot read {path}: {e}")
        n = text.count(old)
        if n == 0:
            return ToolResult(content="", error="old string not found in file")
        if n > 1:
            return ToolResult(content="", error=f"old string is not unique ({n} matches); add surrounding context")
        try:
            _atomic_write(p, text.replace(old, new, 1))
        except OSError as e:
            return ToolResult(content="", error=f"cannot write {path}: {e}")
        return ToolResult(content=f"edited {path} (1 replacement)")

    return [read_file, write_file, edit_file]


def _atomic_write(p: Path, content: str) -> None:
    """Write ``content`` to ``p`` atomically (temp in the same dir + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
