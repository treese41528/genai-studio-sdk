"""REPL configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _root() -> Path:
    """The single ``genai_studio`` user root — skills, memory, and sessions all live under it
    (project ``./.genai_studio/`` overrides this user root for skills + memory)."""
    return Path.home() / ".genai_studio"


def _default_sessions_dir() -> Path:
    return _root() / "sessions"


def _default_memory_dir() -> Path:
    return _root() / "memory"


@dataclass
class ReplConfig:
    model: str
    profile: str = "general"
    max_steps: int = 25
    stream: bool = True
    sessions_dir: Path = field(default_factory=_default_sessions_dir)
    memory_dir: Path = field(default_factory=_default_memory_dir)   # recall-memory user store
    allow_shell_expansion: bool = False         # gates !`cmd` in custom commands (Phase 3)
    compact_token_threshold: int = 0            # 0 => derive from model window (Phase 3)
