"""Data-science extra: import-guard discipline + tool behaviour."""

from __future__ import annotations

import sys

import pytest

from genai_studio.agents import NullTracer
from conftest import ScriptedClient, calls_tool, says


def test_core_import_does_not_pull_pandas():
    # Must run in a FRESH interpreter: this pytest process loads pandas for the
    # other tests, so an in-process sys.modules check would be meaningless.
    import subprocess

    code = (
        "import sys;"
        "import genai_studio.agents;"
        "import genai_studio.agents.datascience;"
        "import genai_studio.agents.datascience.tools;"
        "assert 'pandas' not in sys.modules, 'core/tool-spec import leaked pandas';"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


pd = pytest.importorskip("pandas")  # the rest needs the [datascience] extra


def test_python_exec_stateful_and_errors():
    from genai_studio.agents.datascience.tools import make_python_exec

    px = make_python_exec()
    px("x = 40")
    r = px("x + 2")
    assert r.data == 42 and "42" in r.content
    err = px("1/0")
    assert err.error and not err.ok and "ZeroDivision" in err.content


def test_load_dataset_and_shared_namespace():
    from genai_studio.agents.datascience.tools import make_load_dataset, make_python_exec

    ns: dict = {}
    ld = make_load_dataset(ns)
    px = make_python_exec(ns)
    out = ld("iris")
    assert out.data.shape == (150, 5)
    assert "150" in px("df.shape").content  # python_exec sees the loaded df

    bad = ld("does_not_exist")
    assert bad.error and "Available" in bad.error


def test_data_analyst_assembles_and_runs():
    from genai_studio.agents.datascience import data_analyst

    c = ScriptedClient([
        calls_tool("load_dataset", {"name": "iris"}),
        calls_tool("python_exec", {"code": "print(df['target'].nunique())"}),
        says("Iris has 3 species."),
    ])
    agent = data_analyst(c, tracer=NullTracer())
    names = {t.spec.name for t in agent._registry._tools.values()}
    assert {"python_exec", "load_dataset", "fit_model", "plot"} <= names
    res = agent.run("analyze iris")
    assert res.stopped == "final" and "3 species" in res.text


def test_plot_headless(tmp_path):
    import os

    from genai_studio.agents.datascience.tools import plot

    r = plot("plt.plot([1, 2, 3], [1, 4, 9]); plt.title('demo')")
    assert r.error is None and os.path.exists(r.data)
    os.remove(r.data)
