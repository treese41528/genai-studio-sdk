"""Tests for r_exec (uses real Rscript when present; always tests the error paths)."""

from __future__ import annotations

import shutil

import pytest

from genai_studio.agents.datascience.tools.r_exec import make_r_exec

_HAS_R = shutil.which("Rscript") is not None
_needs_r = pytest.mark.skipif(not _HAS_R, reason="Rscript not installed")


def test_missing_binary_is_a_clean_error():
    res = make_r_exec(rscript="definitely-not-a-real-rscript-xyz")("cat(1)")
    assert res.error and "not found" in res.error


@_needs_r
def test_runs_real_r_and_captures_output():
    res = make_r_exec()("cat(1 + 1)")
    assert res.error is None and "2" in res.content


@_needs_r
def test_nonzero_exit_surfaces_error():
    res = make_r_exec()('stop("boom")')
    assert res.error and "exited" in res.error
    assert "boom" in res.content                       # stderr captured into content


@_needs_r
def test_timeout_is_a_clean_error():
    res = make_r_exec(timeout=1)("Sys.sleep(5)")
    assert res.error and "timed out" in res.error
