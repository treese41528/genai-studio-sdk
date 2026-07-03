"""``apply_patch`` — multi-hunk ATOMIC file edits via SEARCH/REPLACE blocks.

More efficient and safer than several ``edit_file`` calls: the model sends several search/replace
blocks in ONE call; each SEARCH must occur EXACTLY ONCE in the current file, and either ALL blocks
apply or NONE do (no half-edited file). Format (aider-style):

    <<<<<<< SEARCH
    exact existing text
    =======
    replacement text
    >>>>>>> REPLACE
"""

from __future__ import annotations

import re

from genai_studio.agents import ToolResult, tool

from ._workspace import PathEscape, WorkspaceConfig, resolve_in_workspace
from .files import _atomic_write

_BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)


def make_patch_tool(ws: WorkspaceConfig):
    """Return the ``apply_patch`` tool bound to ``ws``."""

    @tool
    def apply_patch(path: str, patch: str) -> ToolResult:
        """Apply several edits to one file AT ONCE, atomically, using SEARCH/REPLACE blocks. Each
        block replaces its SEARCH text (which must occur EXACTLY ONCE) with its REPLACE text; either
        all blocks apply or none do. Prefer this over multiple edit_file calls for multi-part edits.

        Args:
            path: File to edit (inside the workspace).
            patch: One or more blocks of the form:
                <<<<<<< SEARCH
                <exact existing text>
                =======
                <replacement text>
                >>>>>>> REPLACE
        """
        try:
            p = resolve_in_workspace(ws, path, for_write=True)
        except PathEscape as e:
            return ToolResult(content="", error=str(e))
        if not p.is_file():
            return ToolResult(content="", error=f"not a file: {path}")
        blocks = _BLOCK.findall(patch)
        if not blocks:
            return ToolResult(content="", error="no SEARCH/REPLACE blocks found; use the documented "
                              "<<<<<<< SEARCH / ======= / >>>>>>> REPLACE format")
        try:
            text = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError) as e:
            return ToolResult(content="", error=f"cannot read {path}: {e}")
        new_text = text
        for i, (search, replace) in enumerate(blocks, 1):        # validate+apply; write only at end
            n = new_text.count(search)
            if n == 0:
                return ToolResult(content="", error=f"block {i}: SEARCH text not found — no change made")
            if n > 1:
                return ToolResult(content="", error=f"block {i}: SEARCH text is not unique ({n} matches); "
                                  "add surrounding context — no change made")
            new_text = new_text.replace(search, replace, 1)
        try:
            _atomic_write(p, new_text)
        except OSError as e:
            return ToolResult(content="", error=f"cannot write {path}: {e}")
        return ToolResult(content=f"applied {len(blocks)} edit(s) to {path}")

    return apply_patch
