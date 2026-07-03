"""MATH-500 answer extraction + equivalence grading (benchmarks/math_grade.py)."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("sympy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from math_grade import extract_answer, is_equiv


def test_extract_boxed_and_nested():
    assert extract_answer("so the answer is \\boxed{42}.") == "42"
    assert extract_answer("thus \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_answer_is_phrase():
    assert extract_answer("Reasoning...\nThe answer is 17").strip() == "17"


def test_equiv_fraction_and_decimal():
    assert is_equiv("\\boxed{0.5}", "\\frac{1}{2}")
    assert is_equiv("\\boxed{\\frac{14}{3}}", "\\frac{14}3")


def test_equiv_symbolic_and_sqrt():
    assert is_equiv("\\boxed{x^2+1}", "1 + x^2")
    assert is_equiv("\\boxed{2\\sqrt{3}}", "2\\sqrt{3}")


def test_not_equivalent():
    assert not is_equiv("\\boxed{7}", "8")
    assert not is_equiv("no answer here", "5")
