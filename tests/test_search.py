"""Codebase-search tools: grep + glob (ripgrep path + forced Python fallback, workspace-confined)."""

from __future__ import annotations

import genai_studio.agents.tools.search as S
from genai_studio.agents.tools._workspace import WorkspaceConfig
from genai_studio.agents.tools.search import make_search_tools


def _ws(tmp_path):
    return WorkspaceConfig(root=tmp_path)


def test_grep_finds_matches_across_files(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return BAR\n")
    (tmp_path / "b.py").write_text("x = 1\nfoo()\n")
    grep, _ = make_search_tools(_ws(tmp_path))
    res = grep.run({"pattern": r"foo"})
    assert res.error is None and "a.py:1" in res.content and "b.py:2" in res.content


def test_grep_glob_filter(tmp_path):
    (tmp_path / "a.py").write_text("TARGET\n")
    (tmp_path / "a.txt").write_text("TARGET\n")
    grep, _ = make_search_tools(_ws(tmp_path))
    res = grep.run({"pattern": "TARGET", "glob": "*.py"})
    assert "a.py" in res.content and "a.txt" not in res.content


def test_grep_python_fallback_when_no_ripgrep(tmp_path, monkeypatch):
    monkeypatch.setattr(S.shutil, "which", lambda name: None)     # force the pure-Python path
    (tmp_path / "a.py").write_text("NEEDLE here\n")
    grep, _ = make_search_tools(_ws(tmp_path))
    assert "a.py:1" in grep.run({"pattern": "NEEDLE"}).content


def test_grep_skips_noise_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(S.shutil, "which", lambda name: None)     # test the fallback's noise skip
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("SECRET\n")
    (tmp_path / "a.py").write_text("SECRET\n")
    grep, _ = make_search_tools(_ws(tmp_path))
    res = grep.run({"pattern": "SECRET"})
    assert "a.py" in res.content and ".git" not in res.content


def test_grep_no_match_and_bad_regex(tmp_path):
    (tmp_path / "a.py").write_text("hello\n")
    grep, _ = make_search_tools(_ws(tmp_path))
    assert "no matches" in grep.run({"pattern": "zzzznone"}).content
    assert "invalid regex" in (grep.run({"pattern": "["}).error or "")


def test_grep_confined(tmp_path):
    grep, _ = make_search_tools(_ws(tmp_path))
    assert "outside the workspace" in (grep.run({"pattern": "x", "path": "/etc"}).error or "")


def test_glob_finds_files_recursive(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("")
    (tmp_path / "y.py").write_text("")
    _, glob = make_search_tools(_ws(tmp_path))
    res = glob.run({"pattern": "**/*.py"})
    out = res.content.replace("\\", "/")
    assert "y.py" in out and "src/x.py" in out


def test_glob_confined_and_empty(tmp_path):
    _, glob = make_search_tools(_ws(tmp_path))
    assert "outside the workspace" in (glob.run({"pattern": "*", "path": "../.."}).error or "")
    assert "no files match" in glob.run({"pattern": "*.nonexistent"}).content
