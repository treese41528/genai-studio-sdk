"""Shipped tools: finish/final_answer termination, calculator, DS tools."""

from __future__ import annotations

import sys

import pytest

from genai_studio.agents import Agent, NullTracer, ModelResponse, ToolCall, Usage
from conftest import ScriptedClient, calls_tool, says


def test_general_tools_import_is_light():
    # Fresh interpreter: general tools (incl. web) must not pull the scientific stack.
    import subprocess

    code = ("import sys; import genai_studio.agents.tools;"
            "assert 'pandas' not in sys.modules, 'general tools leaked pandas';"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_final_answer_tool_terminates(add_tool):
    from genai_studio.agents.tools import final_answer
    c = ScriptedClient([calls_tool("final_answer", {"answer": "the result is 7"})])
    res = Agent(client=c, tools=[final_answer, add_tool], tracer=NullTracer()).run("go")
    assert res.text == "the result is 7" and res.stopped == "final"
    assert len(res.steps) == 1  # ended on the finish call, no extra round-trip


def test_phantom_finish_is_honored():
    # The model "calls" finish without it being registered — still ends cleanly.
    c = ScriptedClient([calls_tool("finish", {"answer": "done"})])
    res = Agent(client=c, tools=[], tracer=NullTracer()).run("go")
    assert res.text == "done" and res.stopped == "final"


def test_finish_answer_extraction_variants():
    from genai_studio.agents.agent import _finish_answer
    assert _finish_answer({"answer": "a"}) == "a"
    assert _finish_answer({"value": 42}) == "42"
    assert _finish_answer({"text": "t"}) == "t"
    assert _finish_answer({"foo": "bar"}) == "bar"  # first scalar
    assert _finish_answer({}) == ""


def test_calculator():
    from genai_studio.agents.tools import calculator
    assert abs(float(calculator(expression="70 * 1.02 ** 8").content) - 82.0159) < 0.01
    assert calculator(expression="comb(10, 7) / 2 ** 10").content == "0.1171875"
    # unsafe input is rejected, not executed
    assert calculator(expression="__import__('os').system('echo hi')").error
    assert calculator(expression="open('x')").error


pd = pytest.importorskip("pandas")  # DS tools need the extra


def test_describe_and_hypothesis_test(tmp_path):
    import numpy as np

    from genai_studio.agents.datascience.tools import describe_data, hypothesis_test
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["a"] * 30 + ["b"] * 30,
                       "x": np.r_[rng.normal(0, 1, 30), rng.normal(2, 1, 30)]})
    df["y"] = 2 * df["x"] + rng.normal(0, 0.5, 60)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)

    dd = describe_data(file=str(p))
    assert "shape: 60 rows" in dd.content and "strongest correlations" in dd.content

    ht = hypothesis_test(file=str(p), test="ttest_ind", columns=["x"], group_by="g")
    assert ht.data["reject_h0"] and ht.data["p_value"] < 0.05  # groups differ

    pe = hypothesis_test(file=str(p), test="pearson", columns=["x", "y"])
    assert pe.data["statistic"] > 0.9 and pe.data["reject_h0"]

    bad = hypothesis_test(file=str(p), test="nope", columns=["x"])
    assert bad.error
