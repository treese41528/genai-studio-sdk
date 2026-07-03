"""Data-science verification + orchestration (2.0 framework, applied to DS).

- ``stats_panel_tool`` (P1.2): an adversarial statistical review — before a conclusion is reported, an
  independent panel stress-tests it through statistical lenses (assumptions, power, multiplicity,
  leakage, confounding). Reuses ``panel.critic_panel``; the lenses live in ``panel.STATS_LENSES``.
- ``data_science_team`` (P2): a supervisor that DECOMPOSES a task across specialist analysts (explorer /
  modeler / statistician) — the eval found multi-hop analysis needs decomposition, not more tools. One
  shared client (one rate limiter). Reuses ``orchestrate.supervisor`` + ``data_analyst`` workers.
"""

from __future__ import annotations


def stats_panel_tool(client, *, model: str | None = None, n: int | None = None):
    """A ``review_analysis`` tool: an independent panel of statistical critics tries to REFUTE a
    data-science conclusion (one critic per statistical lens). The model passes the claim; it gets back
    whether the conclusion survives. ``n``/lenses are fixed here so the model can't weaken its own check."""
    from ..panel import STATS_LENSES, panel_tool
    return panel_tool(client, models=[model] if model else None, lenses=STATS_LENSES,
                      n=n or len(STATS_LENSES), name="review_analysis",
                      description="Have an independent panel of STATISTICAL critics stress-test a "
                      "data-science conclusion BEFORE you report it — through the lenses of assumptions, "
                      "sample size/power, multiple comparisons, data leakage, and confounding/causality. "
                      "Pass the conclusion (with its numbers) as `claim`; you get back whether it survives.")


_EXPLORER = ("You are a data EXPLORER. Profile the dataset — shape, dtypes, missingness, distributions, "
             "and relationships — and report findings + data-quality issues. Verify every number with the "
             "tools. Do NOT fit models or run hypothesis tests; that's for the other specialists.")
_MODELER = ("You are a MODELER. Build and HONESTLY evaluate a predictive model on HELD-OUT data with the "
            "right metric versus a trivial baseline; guard against leakage and imbalance; report what "
            "drives the prediction (associational, not causal). Never report training-set performance.")
_STATISTICIAN = ("You are a STATISTICIAN. Answer with the CORRECT test, CHECK its assumptions, report the "
                 "effect size alongside p, and correct for multiple comparisons. Say 'underpowered / "
                 "can't tell' rather than forcing a conclusion.")
_MANAGER = ("You coordinate a data-science team on ONE task. Decompose it and DELEGATE — do not compute "
            "yourself: `explorer` profiles the data, `modeler` builds/evaluates predictive models, "
            "`statistician` runs hypothesis tests. Give each worker a self-contained subtask (name the "
            "dataset). Sanity-check what each returns, synthesize, and give ONE clear final report with the "
            "numbers and honest caveats. If the data can't answer the question, say so.")


def data_science_team(client, *, model: str = "qwen2.5:72b", manager_model: str | None = None,
                      sandboxed: bool = False, **kw):
    """A DS supervisor delegating to three specialist analysts (explorer / modeler / statistician), all on
    the SAME client. Use for a multi-part analysis that benefits from decomposition. Returns an ``Agent``
    you ``.run(task)``."""
    from ..orchestrate import supervisor
    from . import data_analyst

    def _worker(name, system):
        return data_analyst(client, model=model, system=system, sandboxed=sandboxed, name=name)

    workers = [_worker("explorer", _EXPLORER), _worker("modeler", _MODELER),
               _worker("statistician", _STATISTICIAN)]
    return supervisor(client, system=_MANAGER, workers=workers, model=manager_model or model, **kw)
