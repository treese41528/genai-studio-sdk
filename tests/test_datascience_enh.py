"""DS 2.0 enhancements: the `datascience` profile, make_datascience_tools, sandboxed data_analyst."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("pandas")   # the DS tool modules import the scientific stack at module load

from genai_studio.agents.datascience import data_analyst
from genai_studio.agents.datascience.tools import make_datascience_tools, make_verify_stat
from genai_studio.agents.profiles import build_tools


def test_verify_stat_recomputes_a_claimed_number():
    import pandas as pd
    ns = {"df": pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 0, 1, 0]})}
    vs = make_verify_stat(ns)
    assert vs.run({"expression": "df['x'].mean()", "expected": 2.5}).data["verdict"] is True
    assert vs.run({"expression": "(df['y'] > 0).mean()", "expected": 0.5}).data["verdict"] is True
    mismatch = vs.run({"expression": "df['x'].mean()", "expected": 9.9})
    assert mismatch.data["verdict"] is False and mismatch.data["actual"] == 2.5
    assert vs.run({"expression": "df", "expected": 1}).error            # not a single number
    assert vs.run({"expression": "df['nope'].sum()", "expected": 0}).error   # bad expression -> error, not crash


def test_make_datascience_tools_full_set():
    names = [t.name for t in make_datascience_tools({})]
    assert names[0] == "python_exec"
    for t in ("load_dataset", "load_table", "describe_data", "fit_model", "hypothesis_test", "plot"):
        assert t in names


def test_datascience_profile_is_analysis_first():
    tools, _guard, _cfg = build_tools("datascience", workspace_root=".", prompt_fn=lambda *a: "allow")
    names = {t.name for t in tools}
    # full DS stack + math grounding + read/search/web, but NOT codebase-write tools
    for want in ("python_exec", "describe_data", "fit_model", "hypothesis_test", "plot",
                 "verify_math", "read_file", "grep", "web_search"):
        assert want in names, f"missing {want}"
    for banned in ("write_file", "apply_patch", "run_shell"):
        assert banned not in names, f"datascience profile should not include {banned}"


class _C:
    supports_native_tools = True
    def complete(self, *a, **k): ...


def test_data_analyst_default_and_sandboxed_build():
    a = data_analyst(_C(), model="m")
    names = {t.name for t in a.tools}
    assert {"python_exec", "describe_data", "verify_math", "final_answer"} <= names
    if sys.platform != "win32":                      # sandbox is Unix-only; just confirm it wires
        s = data_analyst(_C(), model="m", sandboxed=True)
        assert any(t.name == "python_exec" for t in s.tools)


def test_stats_critic_panel_lenses_and_review_option():
    # P1.2 — adversarial statistical review
    from genai_studio.agents.datascience import stats_panel_tool
    from genai_studio.agents.panel import STATS_LENSES
    assert stats_panel_tool(_C(), model="m").name == "review_analysis"
    for lens in ("assumptions", "power", "multiplicity", "leakage", "confounding"):
        assert lens in STATS_LENSES
    assert "review_analysis" in {t.name for t in data_analyst(_C(), model="m", review=True).tools}


def test_data_analyst_remember_and_sql_options(tmp_path):
    # P1.3 memory + P2.3 sql_query
    a = data_analyst(_C(), model="m", remember=True, cwd=".", memory_dir=str(tmp_path), database=":memory:")
    assert {"write_memory", "recall_memory", "sql_query"} <= {t.name for t in a.tools}


def test_data_science_team_decomposes_to_specialists():
    # P2.1 — a supervisor delegating to explorer/modeler/statistician
    from genai_studio.agents.datascience import data_science_team
    team = data_science_team(_C(), model="m")
    assert {"explorer", "modeler", "statistician", "final_answer"} <= {t.name for t in team.tools}
