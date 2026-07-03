"""apply_patch (multi-hunk atomic edits) + background exec (run_background / check_job)."""

from __future__ import annotations

import os
import time

import pytest

from genai_studio.agents.tools._workspace import WorkspaceConfig
from genai_studio.agents.tools.patch import make_patch_tool

_PATCH = """<<<<<<< SEARCH
alpha
=======
ALPHA
>>>>>>> REPLACE
<<<<<<< SEARCH
gamma
=======
GAMMA
>>>>>>> REPLACE"""


def _ws(tmp_path):
    return WorkspaceConfig(root=tmp_path)


# ── apply_patch ──────────────────────────────────────────────────────────────
def test_apply_patch_multiple_blocks(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    res = make_patch_tool(_ws(tmp_path)).run({"path": "x.txt", "patch": _PATCH})
    assert res.error is None and "2 edit" in res.content
    assert f.read_text() == "ALPHA\nbeta\nGAMMA\n"


def test_apply_patch_atomic_on_failure(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("alpha\ngamma\n")                     # 'delta' below won't be found
    bad = _PATCH + "\n<<<<<<< SEARCH\ndelta\n=======\nD\n>>>>>>> REPLACE"
    res = make_patch_tool(_ws(tmp_path)).run({"path": "x.txt", "patch": bad})
    assert res.error and "not found" in res.error
    assert f.read_text() == "alpha\ngamma\n"           # NOTHING changed (atomic)


def test_apply_patch_non_unique(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("dup\ndup\n")
    res = make_patch_tool(_ws(tmp_path)).run(
        {"path": "x.txt", "patch": "<<<<<<< SEARCH\ndup\n=======\nX\n>>>>>>> REPLACE"})
    assert res.error and "not unique" in res.error and f.read_text() == "dup\ndup\n"


def test_apply_patch_no_blocks(tmp_path):
    (tmp_path / "x.txt").write_text("hi\n")
    assert make_patch_tool(_ws(tmp_path)).run({"path": "x.txt", "patch": "nonsense"}).error


# ── background exec ──────────────────────────────────────────────────────────
@pytest.mark.skipif(os.name != "posix", reason="background jobs are Unix-only")
def test_run_background_and_check_job(tmp_path):
    from genai_studio.agents.tools.background import make_background_tools
    run_bg, check = make_background_tools(_ws(tmp_path))
    started = run_bg.run({"command": "echo hello-bg; sleep 0.05"})
    jid = started.data["job"]
    assert jid and "started" in started.content
    for _ in range(40):                                 # poll until the output appears
        out = check.run({"job_id": jid})
        if "hello-bg" in out.content:
            break
        time.sleep(0.05)
    assert "hello-bg" in out.content
    assert check.run({"job_id": "nope"}).error          # unknown job -> error
