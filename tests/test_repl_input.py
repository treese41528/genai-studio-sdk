"""REPL input handling: multi-line paste coalescing + backslash continuation.

``_read_input`` is exercised with injected reader/pending fakes (no tty needed);
the bracketed-paste path is readline-level and shows up here simply as a single
returned string that already contains newlines.
"""

from __future__ import annotations

import pytest

from genai_studio.agents.repl.cli import _read_input


def _reader(lines):
    it = iter(lines)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _input


def test_single_line_passthrough():
    got = _read_input(_input=_reader(["hello"]), _pending=lambda t=0: False)
    assert got == "hello"


def test_paste_burst_coalesces_into_one_submission():
    lines = ["def f():", "    return 1", "print(f())"]
    pending = iter([True, True, False])          # two more lines buffered, then quiet
    got = _read_input(_input=_reader(lines), _pending=lambda t=0: next(pending))
    assert got == "def f():\n    return 1\nprint(f())"


def test_bracketed_paste_returns_multiline_as_is():
    got = _read_input(_input=_reader(["a\nb\nc"]), _pending=lambda t=0: False)
    assert got == "a\nb\nc"


def test_backslash_continuation():
    got = _read_input(_input=_reader(["first \\", "second \\", "third"]),
                      _pending=lambda t=0: False)
    assert got == "first\nsecond\nthird"


def test_backslash_not_applied_to_pasted_block():
    # a pasted block whose last line ends in "\" must not prompt for more
    lines = ["x = 1", "y = 2 \\"]
    pending = iter([True, False])
    got = _read_input(_input=_reader(lines), _pending=lambda t=0: next(pending))
    assert got == "x = 1\ny = 2 \\"


def test_eof_on_continuation_ends_submission():
    got = _read_input(_input=_reader(["only \\"]), _pending=lambda t=0: False)
    assert got == "only"


def test_eof_on_first_read_propagates():
    with pytest.raises(EOFError):
        _read_input(_input=_reader([]), _pending=lambda t=0: False)
