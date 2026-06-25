"""
DataSciBench-mini — a faithful slice of DataSciBench's Task–Function–Code (TFC)
evaluation (Zhang et al., 2025).

Faithful to the paper's methodology: the agent must WRITE & EXECUTE code that
produces named OUTPUT FILES; grading runs deterministic checker functions (the
"C" in TFC) against the executed outputs using the paper's rule types (exact
count ==, error ≤ threshold, score ≥ threshold, file-exists). We report the
paper's two headline metrics:
  • Completion Rate (CR) = passed checkers / total checkers  (partial credit)
  • Success Rate  (SR)   = fraction of tasks where ALL checkers pass

Reference (DataSciBench Table 2, coarse SR%): GPT-4o ≈ 66%, Deepseek-Coder-33B
≈ 56%, Qwen2.5-7B ≈ 44%.

Run:  python benchmarks/datascibench_mini.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from _bench import Task, make_client, run_suite

DSB_SYSTEM = (
    "You are a data scientist. Use the python_exec tool to write and run Python "
    "(pandas, scikit-learn, matplotlib) that produces the requested OUTPUT FILES "
    "in the current working directory. You MUST create every named output file "
    "exactly as specified. For plots, set a non-interactive backend first: "
    "`import matplotlib; matplotlib.use('Agg')`. After saving the files, briefly "
    "confirm what you wrote."
)


# ── checker helpers (the TFC "C") ────────────────────────────────────────────
def _exists(workdir, name) -> bool:
    p = os.path.join(workdir, name)
    return os.path.exists(p) and os.path.getsize(p) > 0


def _numeric_cells(workdir, name) -> list[float]:
    p = os.path.join(workdir, name)
    if not os.path.exists(p):
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    out = []
    for v in df.to_numpy().ravel():
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _has_value_in(workdir, name, lo, hi) -> bool:
    return any(lo <= v <= hi for v in _numeric_cells(workdir, name))


def grade_checkers(checkers: list):
    """checkers: list of (label, fn(workdir)->bool). Score = fraction passed."""
    def _g(result, workdir):
        results = [(label, bool(fn(workdir))) for label, fn in checkers]
        passed = sum(1 for _, ok in results if ok)
        detail = " ".join(f"{lab}={'✓' if ok else '✗'}" for lab, ok in results)
        return passed / len(results), detail
    return _g


# ── tasks ────────────────────────────────────────────────────────────────────
def _t_cleaning() -> Task:
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "Quantity": rng.integers(1, 20, n).astype(float),
        "Revenue": rng.normal(200, 50, n).round(2),
    })
    df.loc[rng.choice(n, 8, replace=False), "Revenue"] = np.nan  # missing
    df.loc[rng.choice(n, 4, replace=False), "Revenue"] = 99999.0  # outliers

    def setup(wd):
        df.to_csv(os.path.join(wd, "sales.csv"), index=False)

    def chk_no_nulls(wd):
        cells = _numeric_cells(wd, "cleaned.csv")
        return _exists(wd, "cleaned.csv") and not any(np.isnan(v) for v in cells)

    def chk_rows(wd):
        p = os.path.join(wd, "cleaned.csv")
        return os.path.exists(p) and len(pd.read_csv(p)) == n

    def chk_normalized(wd):
        p = os.path.join(wd, "cleaned.csv")
        if not os.path.exists(p):
            return False
        d = pd.read_csv(p)
        for col in d.select_dtypes("number").columns:
            s = d[col]
            if abs(s.mean()) < 0.15 and abs(s.std() - 1) < 0.25:
                return True  # at least one column was z-score normalized
        return False

    prompt = (
        "Read ./sales.csv (columns TransactionID, Quantity, Revenue). Then: "
        "(1) impute missing Revenue values with the column median; "
        "(2) clip Quantity and Revenue to their 1st and 99th percentiles to remove "
        "outliers; (3) z-score normalize the Quantity and Revenue columns. "
        "Save the fully cleaned dataset (all 100 rows) to a file named cleaned.csv.")
    return Task(id="cleaning", prompt=prompt, setup=setup,
                grade=grade_checkers([("file+no_nulls", chk_no_nulls),
                                      ("rows==100", chk_rows),
                                      ("normalized", chk_normalized)]))


def _t_stats() -> Task:
    from sklearn.datasets import load_iris
    b = load_iris(as_frame=True)
    df = b.frame.drop(columns=["target"])
    gt_means = df.mean().to_dict()

    def setup(wd):
        df.to_csv(os.path.join(wd, "iris.csv"), index=False)

    def chk_means(wd):
        cells = _numeric_cells(wd, "stats.csv")
        # each GT mean must appear among the reported numbers (±0.05)
        return all(any(abs(c - m) <= 0.05 for c in cells) for m in gt_means.values())

    prompt = ("Read ./iris.csv (4 numeric flower-measurement columns). Compute the "
              "mean of each column and save the results to a file named stats.csv "
              "(include the mean value for every numeric column).")
    return Task(id="stats", prompt=prompt, setup=setup,
                grade=grade_checkers([("stats.csv exists", lambda wd: _exists(wd, "stats.csv")),
                                      ("means≈GT", chk_means)]))


def _t_classification() -> Task:
    from sklearn.datasets import load_wine
    b = load_wine(as_frame=True)

    def setup(wd):
        b.frame.to_csv(os.path.join(wd, "wine.csv"), index=False)

    prompt = ("Read ./wine.csv (features + a 'target' class column). Split into train "
              "and test sets with test_size=0.25, random_state=42. Train a "
              "RandomForestClassifier, evaluate it on the test set, and save the test "
              "accuracy to a file named accuracy.csv.")
    return Task(id="classification", prompt=prompt, setup=setup,
                grade=grade_checkers([
                    ("accuracy.csv exists", lambda wd: _exists(wd, "accuracy.csv")),
                    ("accuracy≥0.85", lambda wd: _has_value_in(wd, "accuracy.csv", 0.85, 1.0))]))


def _t_regression() -> Task:
    from sklearn.datasets import load_diabetes
    b = load_diabetes(as_frame=True)

    def setup(wd):
        b.frame.to_csv(os.path.join(wd, "diabetes.csv"), index=False)

    prompt = ("Read ./diabetes.csv (features + numeric 'target'). Split with "
              "test_size=0.25, random_state=42, fit a LinearRegression, and save the "
              "test-set R-squared and MSE to a file named regression_metrics.csv.")
    return Task(id="regression", prompt=prompt, setup=setup,
                grade=grade_checkers([
                    ("metrics file exists", lambda wd: _exists(wd, "regression_metrics.csv")),
                    ("R2≥0.35", lambda wd: _has_value_in(wd, "regression_metrics.csv", 0.35, 1.0))]))


def _t_clustering() -> Task:
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.0)
    df = pd.DataFrame(X, columns=["x", "y"])

    def setup(wd):
        df.to_csv(os.path.join(wd, "blobs.csv"), index=False)

    prompt = ("Read ./blobs.csv (columns x, y). Run KMeans with n_clusters=3, "
              "random_state=42. Compute the silhouette score of the clustering and "
              "save it to a file named silhouette.csv.")
    return Task(id="clustering", prompt=prompt, setup=setup,
                grade=grade_checkers([
                    ("silhouette.csv exists", lambda wd: _exists(wd, "silhouette.csv")),
                    ("silhouette≥0.5", lambda wd: _has_value_in(wd, "silhouette.csv", 0.5, 1.0))]))


def _t_visualization() -> Task:
    rng = np.random.default_rng(2)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    df = pd.DataFrame({"Month": months, "Sales": rng.integers(100, 900, len(months))})

    def setup(wd):
        df.to_csv(os.path.join(wd, "monthly.csv"), index=False)

    prompt = ("Read ./monthly.csv (columns Month, Sales). Create a bar chart of Sales "
              "by Month and save it as a PNG file named sales_plot.png. Remember to set "
              "matplotlib to the 'Agg' backend before plotting.")
    return Task(id="visualization", prompt=prompt, setup=setup,
                grade=grade_checkers([
                    ("sales_plot.png exists", lambda wd: _exists(wd, "sales_plot.png"))]))


def build_tasks() -> list[Task]:
    return [_t_cleaning(), _t_stats(), _t_classification(),
            _t_regression(), _t_clustering(), _t_visualization()]


def make_agent(task, workdir, react: bool = False):
    from genai_studio.agents import Agent
    from genai_studio.agents.datascience.tools import make_python_exec
    from genai_studio.agents.trace import JsonlTracer
    return Agent(client=make_client(react=react), tools=[make_python_exec()],
                 system=DSB_SYSTEM,
                 tracer=JsonlTracer(os.path.join(workdir, "trace.jsonl")),
                 max_steps=8)


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_suite("DataSciBench-mini", build_tasks()[:limit], make_agent)
