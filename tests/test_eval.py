"""Tests for the pass^k evaluation harness."""

from __future__ import annotations

import json

import pytest

from genai_studio.agents import Agent, tool
from genai_studio.agents.eval import (
    Case, CaseReport, RunRecord, all_of, contains, evaluate, llm_judge, used_tool,
)

from conftest import ScriptedClient, calls_tool, says


@tool
def add(a: int, b: int) -> str:
    "Add.\n\nArgs:\n  a: x.\n  b: y."
    return str(a + b)


def _scripted_factory(scripts, tools=()):
    """A factory drawing a fresh independent ScriptedClient for each run."""
    state = {"i": 0}

    def factory(case, tracer):
        sc = scripts[state["i"]]
        state["i"] += 1
        return Agent(client=ScriptedClient(sc), tools=tools, tracer=tracer)

    return factory


# ── grader helpers ───────────────────────────────────────────────────────────
def test_contains_pass_and_fail():
    res = type("R", (), {"text": "The answer is 42."})()
    assert contains("42")(None, res, None) == (1.0, "ok")
    s, _ = contains("99")(None, res, None)
    assert s == 0.0


def test_contains_is_case_insensitive_by_default():
    res = type("R", (), {"text": "PARIS"})()
    assert contains("paris")(None, res, None)[0] == 1.0
    assert contains("paris", ci=False)(None, res, None)[0] == 0.0


def test_used_tool_reads_steps():
    call = type("C", (), {"name": "add"})()
    step = type("S", (), {"tool_calls": [call]})()
    res = type("R", (), {"steps": [step], "text": "x"})()
    assert used_tool("add")(None, res, None)[0] == 1.0
    assert used_tool("search")(None, res, None)[0] == 0.0
    assert used_tool()(None, res, None)[0] == 1.0          # any tool


def test_all_of_is_and_via_min():
    g = all_of(contains("a"), contains("b"))
    res = type("R", (), {"text": "a only"})()
    score, _ = g(None, res, None)
    assert score == 0.0                                     # min(1.0, 0.0): one failed -> AND fails


def test_all_of_keeps_and_semantics_below_threshold():
    # the regression the review caught: a required sub-grader must not be waived
    # at threshold < 1.0. used_tool fails (no tool) -> the case must FAIL at 0.5.
    rep = evaluate(_scripted_factory([[says("42, no tool used")]]),
                   [Case("c", "q", check=all_of(contains("42"), used_tool("add")))],
                   k=1, threshold=0.5)
    assert rep.cases[0].runs[0].passed is False


# ── reliability metrics ──────────────────────────────────────────────────────
def test_passk_metrics_and_consistency():
    scripts = [
        [calls_tool("add", {"a": 19, "b": 23}), says("The answer is 42.")],
        [calls_tool("add", {"a": 19, "b": 23}), says("The answer is 42.")],
        [says("The answer is 41.")],                        # wrong, no tool
    ]
    rep = evaluate(_scripted_factory(scripts, tools=[add]),
                   [Case("add", "19+23?", check=all_of(contains("42"), used_tool("add")))],
                   k=3)
    c = rep.cases[0]
    assert c.pass_rate == pytest.approx(2 / 3)
    assert c.pass_pow_k is False and c.pass_at_k is True
    assert c.consistency == pytest.approx(2 / 3)
    assert c.majority_answer == "The answer is 42."
    # report-level aggregates (single case)
    assert rep.pass_pow_k == 0.0 and rep.pass_at_k == 1.0


def test_pass_pow_k_true_when_all_pass():
    scripts = [[says("42")], [says("forty-two = 42")]]
    rep = evaluate(_scripted_factory(scripts), [Case("c", "q", check=contains("42"))], k=2)
    assert rep.cases[0].pass_pow_k is True and rep.pass_pow_k == 1.0


def test_consistency_full_agreement():
    scripts = [[says("Paris.")], [says("paris")], [says(" PARIS ")]]
    rep = evaluate(_scripted_factory(scripts), [Case("c", "capital?")], k=3)
    assert rep.cases[0].consistency == pytest.approx(1.0)   # normalized to one class


# ── judge (L2) ───────────────────────────────────────────────────────────────
def test_llm_judge_passes_and_combines_with_check():
    # agent answers; judge client returns a passing verdict
    answer_scripts = [[says("42 is the answer")]]
    judge_client = ScriptedClient([says('{"pass": true, "reason": "correct"}')])
    rep = evaluate(_scripted_factory(answer_scripts),
                   [Case("c", "q", check=contains("42"))],
                   k=1, judge=llm_judge(judge_client))
    r = rep.cases[0].runs[0]
    assert r.passed is True and "judge:correct" in r.detail


def test_llm_judge_can_fail_a_run_the_check_passed():
    answer_scripts = [[says("42 is the answer")]]
    judge_client = ScriptedClient([says('{"pass": false, "reason": "unclear"}')])
    rep = evaluate(_scripted_factory(answer_scripts),
                   [Case("c", "q", check=contains("42"))],
                   k=1, judge=llm_judge(judge_client))
    assert rep.cases[0].runs[0].passed is False             # judge gate vetoes


def test_unparseable_judge_scores_zero_not_crash():
    judge_client = ScriptedClient([says("totally not json")])
    rep = evaluate(_scripted_factory([[says("hello")]]),
                   [Case("c", "q")], k=1, judge=llm_judge(judge_client))
    r = rep.cases[0].runs[0]
    assert r.passed is False and "judge-error" in r.detail


# ── robustness / ethos ───────────────────────────────────────────────────────
def test_run_error_is_a_failed_run_not_a_crashed_eval():
    def boom_factory(case, tracer):
        class Boom:
            def run(self, prompt):
                raise RuntimeError("kaboom")
        return Boom()

    rep = evaluate(boom_factory, [Case("c", "q", check=contains("x"))], k=2)
    c = rep.cases[0]
    assert c.n_passed == 0 and all(r.error and "kaboom" in r.error for r in c.runs)
    assert c.pass_pow_k is False


def test_grader_error_fails_run_not_eval():
    def bad_grader(case, result, trace_path):
        raise ValueError("grader bug")
    rep = evaluate(_scripted_factory([[says("hi")]]), [Case("c", "q", check=bad_grader)], k=1)
    r = rep.cases[0].runs[0]
    assert r.passed is False and "grader-error" in r.detail


def test_no_grader_passes_on_liveness_only():
    rep = evaluate(_scripted_factory([[says("some answer")]]), [Case("c", "q")], k=1)
    assert rep.cases[0].runs[0].passed is True              # finished + non-blank
    rep2 = evaluate(_scripted_factory([[says("   ")]]), [Case("c", "q")], k=1)
    assert rep2.cases[0].runs[0].passed is False            # blank answer fails liveness


def test_trace_dir_writes_jsonl(tmp_path):
    rep = evaluate(_scripted_factory([[calls_tool("add", {"a": 1, "b": 2}), says("3")]], tools=[add]),
                   [Case("addcase", "1+2?", check=contains("3"))],
                   k=1, trace_dir=str(tmp_path))
    p = tmp_path / "addcase_r0.jsonl"
    assert p.exists()
    types = {json.loads(line)["type"] for line in p.read_text().splitlines()}
    assert "ToolCallEvent" in types and "AgentEnd" in types
    assert rep.cases[0].runs[0].trace_path == str(p)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        evaluate(_scripted_factory([]), [Case("c", "q")], k=0)


def test_threshold_gates_a_continuous_grader():
    # threshold's legitimate use: a single continuous grader (e.g. an F1 score)
    def f1_like(case, result, trace_path):
        return 0.7, "f1=0.7"
    passing = evaluate(_scripted_factory([[says("x")]]),
                       [Case("c", "q", check=f1_like)], k=1, threshold=0.6)
    failing = evaluate(_scripted_factory([[says("x")]]),
                       [Case("c", "q", check=f1_like)], k=1, threshold=0.8)
    assert passing.cases[0].runs[0].passed is True
    assert failing.cases[0].runs[0].passed is False


# ── review fixes: never-crash holes, fail-open judge, trace/resume ───────────
def test_judge_string_false_is_a_fail_not_truthy():
    judge_client = ScriptedClient([says('{"pass": "false", "reason": "wrong"}')])
    rep = evaluate(_scripted_factory([[says("42")]]),
                   [Case("c", "q", check=contains("42"))], k=1, judge=llm_judge(judge_client))
    assert rep.cases[0].runs[0].passed is False             # "false" string must not pass


def test_on_run_error_does_not_abort_eval():
    def boom(case, rec):
        raise RuntimeError("checkpoint sink down")
    rep = evaluate(_scripted_factory([[says("x")], [says("x")]]),
                   [Case("c", "q")], k=2, on_run=boom)
    assert rep.cases[0].k == 2                               # sweep completed despite on_run raising


def test_raising_normalize_degrades_not_poisons_report():
    def bad_norm(s):
        raise ValueError("normalizer blew up")
    rep = evaluate(_scripted_factory([[says("a")], [says("a")]]),
                   [Case("c", "q")], k=2, normalize=bad_norm)
    # accessing report properties must not raise; degrades to identity
    assert rep.consistency == pytest.approx(1.0)
    assert "cases" in rep.summary()


def test_grader_and_judge_errors_surface_on_record_error():
    def bad_grader(case, result, trace_path):
        raise ValueError("grader bug")
    rep = evaluate(_scripted_factory([[says("hi")]]), [Case("c", "q", check=bad_grader)], k=1)
    assert "grader-error" in (rep.cases[0].runs[0].error or "")
    judge_client = ScriptedClient([says("not json")])       # judge raises internally -> judge-error
    rep2 = evaluate(_scripted_factory([[says("hi")]]),
                    [Case("c", "q")], k=1, judge=llm_judge(judge_client))
    assert "judge-error" in (rep2.cases[0].runs[0].error or "")


def test_trace_dir_rerun_truncates_not_appends(tmp_path):
    case = Case("addcase", "1+2?", check=contains("3"))
    for _ in range(2):                                       # run twice into the same dir
        evaluate(_scripted_factory([[calls_tool("add", {"a": 1, "b": 2}), says("3")]], tools=[add]),
                 [case], k=1, trace_dir=str(tmp_path))
    lines = (tmp_path / "addcase_r0.jsonl").read_text().splitlines()
    assert sum(1 for ln in lines if '"AgentEnd"' in ln) == 1   # one run's worth, not accumulated


def test_done_ids_skips_completed_cases():
    rep = evaluate(_scripted_factory([[says("x")]]),
                   [Case("a", "q"), Case("b", "q")], k=1, done_ids={"a"})
    assert [c.case_id for c in rep.cases] == ["b"]           # 'a' skipped
