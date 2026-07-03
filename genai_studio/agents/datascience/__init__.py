"""
Data-science extra — heavy scientific stack + pre-assembled agents.

Lives OUTSIDE the light core: ``import genai_studio.agents`` never imports this.
Install with ``pip install 'genai-studio-sdk[datascience]'``; the tools load
pandas/sklearn/etc. only when they actually run.

    from genai_studio import GenAIStudio
    from genai_studio.agents import GenAIStudioClient
    from genai_studio.agents.datascience import data_analyst

    agent = data_analyst(GenAIStudioClient(GenAIStudio(), default_model="qwen2.5:72b"))
    print(agent.run("Load iris and say which features best separate the species.").text)
"""

from __future__ import annotations

from genai_studio.agents import Agent

from .prompts import DATA_ANALYST_SYSTEM


def data_analyst(client, *, model: str = "qwen2.5:72b", sandboxed: bool = False,
                 system: str | None = None, remember: bool = False, database: str | None = None,
                 review: bool = False, cwd=".", memory_dir=None, **kw) -> Agent:
    """Pre-assembled data-science agent: python_exec + load_dataset + load_table
    (your own CSV/Parquet/Excel/JSON, sharing one namespace) + describe_data + fit_model +
    hypothesis_test + plot + verify_stat, plus exact-math grounding (verify_math/symbolic_math/matrix_op).
    A thin factory over the core ``Agent``.

    Options (all default OFF — byte-identical when unused):
        sandboxed: run python_exec in the hardened subprocess sandbox (Unix-only; own persistent state,
            so it does NOT share the load_dataset namespace).
        system: override the default DATA_ANALYST_SYSTEM (used to build specialist workers).
        remember: add write_memory/recall_memory over a per-project store (schema, findings persist).
        database: a SQLite path/URI — adds a read-only ``sql_query`` tool.
        review: add ``review_analysis`` — an adversarial statistical critic panel for conclusions.
    """
    # Imported lazily so importing this package stays light.
    from genai_studio.agents.tools import calculator, final_answer

    from ..tools.symbolic import matrix_op, symbolic_math, verify_math
    from .tools import make_datascience_tools

    tools = [*make_datascience_tools({}, sandboxed=sandboxed),
             calculator, verify_math, symbolic_math, matrix_op]
    if database:
        from .tools.io_tools import make_sql_query
        tools.append(make_sql_query(database))
    if remember:
        from pathlib import Path
        from ..memory import make_memory_tools, open_store
        mdir = memory_dir or (Path.home() / ".genai_studio" / "memory")
        tools += make_memory_tools(open_store(cwd, mdir), studio=getattr(client, "studio", None))
    if review:
        from .analysis import stats_panel_tool
        tools.append(stats_panel_tool(client, model=model))
    tools.append(final_answer)
    return Agent(client=client, model=model, tools=tools, system=system or DATA_ANALYST_SYSTEM, **kw)


def __getattr__(name):                    # lazy re-export (keeps package import light)
    if name in ("data_science_team", "stats_panel_tool"):
        from . import analysis
        return getattr(analysis, name)
    raise AttributeError(name)


__all__ = ["data_analyst", "DATA_ANALYST_SYSTEM", "data_science_team", "stats_panel_tool"]
