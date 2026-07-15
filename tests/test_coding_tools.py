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


# ── read_file: PDF text extraction ────────────────────────────────────────────
def _write_pdf(path, pages):
    """Write a minimal, valid multi-page PDF (Helvetica text) whose pages carry the
    given strings — hand-built so no PDF-authoring dependency is needed, and verified
    to round-trip through pypdf's ``extract_text``. A page given as ``None`` is left
    blank (no content stream) to simulate a scanned/image page."""
    parts, page_nums, content_nums, next_num = {}, [], [], 3
    for _ in pages:
        page_nums.append(next_num); next_num += 1
        content_nums.append(next_num); next_num += 1
    font_num = next_num
    parts[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{p} 0 R" for p in page_nums)
    parts[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [ {kids} ] >>"
    for i, text in enumerate(pages):
        pn, cn = page_nums[i], content_nums[i]
        parts[pn] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
                     f"/Resources << /Font << /F1 {font_num} 0 R >> >> /Contents {cn} 0 R >>")
        stream = "" if text is None else f"BT /F1 14 Tf 20 150 Td ({text}) Tj ET"
        parts[cn] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    parts[font_num] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out, offsets = b"%PDF-1.4\n", {}
    for num in range(1, font_num + 1):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{parts[num]}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    out += f"xref\n0 {font_num + 1}\n0000000000 65535 f \n".encode()
    for num in range(1, font_num + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {font_num + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    with open(path, "wb") as f:
        f.write(out)


def test_read_file_extracts_pdf_text(ftools, tmp_path):
    pytest.importorskip("pypdf")
    _write_pdf(tmp_path / "doc.pdf", ["Hello from page one", "Second page here"])
    r = ftools["read"].run({"path": "doc.pdf"})
    assert r.error is None, r.error
    assert "Hello from page one" in r.content and "Second page here" in r.content
    assert "page 1 of 2" in r.content
    assert r.data and r.data.get("pdf") is True and r.data.get("pages") == 2


def test_read_file_detects_pdf_by_magic(ftools, tmp_path):
    # PDF content behind a .txt name is still extracted, not decoded as UTF-8
    pytest.importorskip("pypdf")
    _write_pdf(tmp_path / "mislabeled.txt", ["Magic-sniffed text"])
    r = ftools["read"].run({"path": "mislabeled.txt"})
    assert r.error is None and "Magic-sniffed text" in r.content


def test_read_file_scanned_pdf_reports_no_text(ftools, tmp_path):
    pytest.importorskip("pypdf")
    _write_pdf(tmp_path / "scan.pdf", [None])        # image-only page: no text operators
    r = ftools["read"].run({"path": "scan.pdf"})
    assert r.error and "no extractable text" in r.error


def test_read_file_truncates_large_pdf(ftools, tmp_path):
    pytest.importorskip("pypdf")
    _write_pdf(tmp_path / "big.pdf", [f"Page {i} content padding here" for i in range(20)])
    r = ftools["read"].run({"path": "big.pdf", "max_bytes": 120})
    assert r.error is None and "truncated" in r.content
    assert len(r.content) < 300


def test_text_files_unchanged_by_pdf_path(ftools):
    ftools["write"].run({"path": "notes.md", "content": "# Title\nplain text"})
    r = ftools["read"].run({"path": "notes.md"})
    assert r.error is None and r.content == "# Title\nplain text"
    assert r.data.get("pdf") is None                 # text path untouched, no pdf marker


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
