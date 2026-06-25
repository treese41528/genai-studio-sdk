"""``fit_model`` — fit a simple model on a bundled dataset and return a summary.

A thin convenience; for anything richer the model can compute inside ``python_exec``.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

from .._guard import _require
from .datasets import _LOADERS


@tool
def fit_model(dataset: str, target: str | None = None, kind: str = "ols") -> ToolResult:
    """Fit a simple model on a bundled dataset and return a text summary.

    Args:
        dataset: bundled dataset name (e.g. 'diabetes', 'breast_cancer').
        target: target column; defaults to the dataset's standard target.
        kind: 'ols' (linear), 'logit' (logistic), or 'rf' (random forest).
    """
    if dataset not in _LOADERS:
        return ToolResult(content="", error=f"Unknown dataset {dataset!r}. "
                          f"Available: {', '.join(_LOADERS)}.")
    sk = _require("sklearn.datasets")
    _require("pandas")
    bunch = getattr(sk, _LOADERS[dataset])(as_frame=True)
    df = bunch.frame
    ycol = target or getattr(bunch, "target_names", None) and "target" or "target"
    if ycol not in df.columns:
        ycol = "target"
    X = df.drop(columns=[ycol])
    y = df[ycol]

    try:
        if kind == "rf":
            ensemble = _require("sklearn.ensemble")
            metrics = _require("sklearn.metrics")
            model = ensemble.RandomForestRegressor(n_estimators=100, random_state=0)
            model.fit(X, y)
            r2 = metrics.r2_score(y, model.predict(X))
            imp = sorted(zip(X.columns, model.feature_importances_),
                         key=lambda t: -t[1])[:10]
            lines = "\n".join(f"  {n}: {v:.4f}" for n, v in imp)
            return ToolResult(content=f"RandomForest in-sample R²={r2:.4f}\n"
                              f"feature importances:\n{lines}", data=model)
        sm = _require("statsmodels.api")
        Xc = sm.add_constant(X)
        if kind == "logit":
            res = sm.Logit(y, Xc).fit(disp=0)
        else:
            res = sm.OLS(y, Xc).fit()
        return ToolResult(content=str(res.summary()), data=res)
    except Exception as exc:
        return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")
