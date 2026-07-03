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


def data_analyst(client, *, model: str = "qwen2.5:72b", sandboxed: bool = False, **kw) -> Agent:
    """Pre-assembled data-science agent: python_exec + load_dataset + load_table
    (your own CSV/Parquet/Excel/JSON, sharing one namespace) + describe_data + fit_model +
    hypothesis_test + plot, plus exact-math grounding (verify_math/symbolic_math/matrix_op). A thin
    factory over the core ``Agent``. For SQL or R, add ``make_sql_query(db)`` / ``make_r_exec()``.

    ``sandboxed=True`` runs python_exec in the hardened subprocess sandbox (Unix-only; keeps its own
    persistent state, so it does not share the load_dataset namespace).
    """
    # Imported lazily so importing this package stays light.
    from genai_studio.agents.tools import calculator, final_answer

    from ..tools.symbolic import matrix_op, symbolic_math, verify_math
    from .tools import make_datascience_tools

    tools = [*make_datascience_tools({}, sandboxed=sandboxed),
             calculator, verify_math, symbolic_math, matrix_op, final_answer]
    return Agent(client=client, model=model, tools=tools, system=DATA_ANALYST_SYSTEM, **kw)


__all__ = ["data_analyst", "DATA_ANALYST_SYSTEM"]
