"""``WorkspaceConfig`` + the path-confinement floor for the coding tools.

ONE ``WorkspaceConfig`` is built per session and threaded through the file/shell tool
factories AND the approval engine, so the tool-level safety floor and the approval
guard agree on exactly which paths count as "inside the workspace". The confinement is
enforced at the TOOL level (here) so it holds even if a guard is misconfigured or
absent — the guard is policy/UX on top, not the only line of defence.

Honest limits: ``resolve_in_workspace`` resolves symlinks at check time
(``os.path.realpath``), which blocks the common symlink-escape, but a symlink swapped
between the check and the open is a residual TOCTOU race (same class of risk OS jails
exist to close — we have none here). For untrusted input at scale, run in a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class PathEscape(ValueError):
    """Raised when a path resolves outside the workspace's allowed roots."""


@dataclass
class WorkspaceConfig:
    """The session's filesystem boundary.

    ``root`` is the agent's working directory; ``extra_writable`` opt-in extra dirs.
    ``read_only_subpaths`` are carved read-only even inside a writable root (so the
    agent can't rewrite ``.git`` or its own session log).
    """

    root: Path
    extra_writable: tuple[Path, ...] = ()
    read_only_subpaths: tuple[str, ...] = (".git", ".genai_studio")
    max_read_bytes: int = 100_000

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.extra_writable = tuple(Path(p).resolve() for p in self.extra_writable)

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.root, *self.extra_writable)


def _under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:                      # different drives / mixed abs+rel
        return False


def resolve_in_workspace(ws: WorkspaceConfig, path: str, *, for_write: bool) -> Path:
    """Resolve ``path`` (relative to ``ws.root``) to a real path and REQUIRE it inside
    an allowed root; resolve symlinks first (anti-escape). For writes, also refuse paths
    directly under a carved read-only subpath (``.git``/``.genai_studio``). Raises
    :class:`PathEscape` on violation."""
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (ws.root / raw)
    real = Path(os.path.realpath(candidate))
    roots = ws.roots                         # read and write share the same roots here
    home_root = next((r for r in roots if _under(real, r)), None)
    if home_root is None:
        raise PathEscape(f"path {path!r} resolves outside the workspace ({real})")
    if for_write:
        rel_parts = Path(os.path.relpath(real, home_root)).parts
        for carved in ws.read_only_subpaths:
            cparts = Path(carved).parts
            if rel_parts[: len(cparts)] == cparts:
                raise PathEscape(f"path {path!r} is under read-only {carved!r}")
    return real
