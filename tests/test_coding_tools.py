"""Tests for the workspace-confined coding tools (read/write/edit/run_shell)."""

from __future__ import annotations

import os

import pytest

from genai_studio.agents.tools._workspace import WorkspaceConfig
from genai_studio.agents.tools.files import make_file_tools


@pytest.fixture
def ws(tmp_path):
    return WorkspaceConfig(root=tmp_path)


@pytest.fixture
def ftools(ws):
    rf, wf, ef = make_file_tools(ws)
    return {"read": rf, "write": wf, "edit": ef}


def test_write_read_roundtrip(ftools, tmp_path):
    assert ftools["write"].run({"path": "a/b.txt", "content": "hello"}).error is None
    assert (tmp_path / "a" / "b.txt").read_text() == "hello"
    r = ftools["read"].run({"path": "a/b.txt"})
    assert r.error is None and "hello" in r.content


def test_edit_unique_and_errors(ftools):
    ftools["write"].run({"path": "f.txt", "content": "one two two"})
    assert ftools["edit"].run({"path": "f.txt", "old": "one", "new": "ONE"}).error is None
    assert "ONE two two" in ftools["read"].run({"path": "f.txt"}).content
    assert "not unique" in (ftools["edit"].run({"path": "f.txt", "old": "two", "new": "X"}).error or "")
    assert "not found" in (ftools["edit"].run({"path": "f.txt", "old": "zzz", "new": "X"}).error or "")


def test_path_escape_blocked(ftools, tmp_path):
    assert "outside" in (ftools["read"].run({"path": "../../etc/passwd"}).error or "")
    assert "outside" in (ftools["write"].run({"path": "/etc/passwd", "content": "x"}).error or "")


def test_git_carved_read_only(ftools):
    assert "read-only" in (ftools["write"].run({"path": ".git/config", "content": "x"}).error or "")


def test_symlink_escape_blocked(ftools, ws, tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.write_text("secret")
    (tmp_path / "link").symlink_to(outside)        # symlink inside workspace -> outside
    assert "outside" in (ftools["read"].run({"path": "link"}).error or "")


# ── run_shell (Unix-gated, mirrors test_sandbox.py) ───────────────────────────
@pytest.mark.skipif(not hasattr(os, "fork"), reason="run_shell needs Unix")
def test_run_shell_basic_and_cwd(ws, tmp_path):
    from genai_studio.agents.tools.shell import make_run_shell
    sh = make_run_shell(ws)
    assert "hi" in sh.run({"command": "echo hi"}).content
    assert str(tmp_path.resolve()) in sh.run({"command": "pwd"}).content


@pytest.mark.skipif(not hasattr(os, "fork"), reason="run_shell needs Unix")
def test_run_shell_wall_clock(ws):
    from genai_studio.agents.tools.shell import make_run_shell
    sh = make_run_shell(ws)
    r = sh.run({"command": "sleep 30", "timeout": 1})
    assert r.error and "wall-clock" in r.error
