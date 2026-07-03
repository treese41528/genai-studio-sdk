"""``hypothesis_test`` — classical statistical tests on CSV columns (scipy.stats).

Classical inference: t-tests, correlation tests, chi-square
independence, and one-way ANOVA, each reporting the statistic, p-value, and a
reject/fail-to-reject conclusion at the chosen significance level.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

from .._guard import _require

_TESTS = ("ttest_ind", "ttest_1samp", "pearson", "spearman", "chi2", "anova")


@tool
def hypothesis_test(
    file: str,
    test: str,
    columns: list[str],
    group_by: str | None = None,
    popmean: float | None = None,
    alpha: float = 0.05,
) -> ToolResult:
    """Run a statistical hypothesis test on a CSV and report statistic, p-value,
    and conclusion.

    Args:
        file: path to a CSV file.
        test: one of 'ttest_ind' (two-sample t), 'ttest_1samp' (one-sample t vs
            popmean), 'pearson'/'spearman' (correlation), 'chi2' (independence of
            two categorical columns), 'anova' (one-way ANOVA across groups).
        columns: the column(s) used. ttest_1samp/anova: [value_col]; ttest_ind:
            [value_col] with group_by, or [col_a, col_b]; pearson/spearman/chi2:
            [col_a, col_b].
        group_by: grouping column (for ttest_ind / anova by group).
        popmean: population mean for ttest_1samp.
        alpha: significance level (default 0.05).
    """
    pd = _require("pandas")
    stats = _require("scipy.stats")
    if test not in _TESTS:
        return ToolResult(content="", error=f"unknown test {test!r}; choose from {_TESTS}")
    try:
        df = pd.read_csv(file)
        stat, p, extra = _run(stats, df, test, columns, group_by, popmean)
    except Exception as exc:
        return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")

    reject = p < alpha
    concl = (f"p={p:.4g} < alpha={alpha}: reject H0 (statistically significant)."
             if reject else
             f"p={p:.4g} >= alpha={alpha}: fail to reject H0 (not significant).")
    content = f"{test}: statistic={stat:.4g}, p-value={p:.4g}\n{extra}\n{concl}"
    return ToolResult(content=content.strip(),
                      data={"test": test, "statistic": stat, "p_value": p, "reject_h0": reject})


def _run(stats, df, test, columns, group_by, popmean):
    pd = _require("pandas")
    if test == "ttest_1samp":
        if popmean is None:
            raise ValueError("ttest_1samp requires popmean")
        s = df[columns[0]].dropna()
        r = stats.ttest_1samp(s, popmean)
        return r.statistic, r.pvalue, f"n={len(s)}, mean={s.mean():.4g}, popmean={popmean}"
    if test == "ttest_ind":
        a, b = _two_samples(df, columns, group_by)
        r = stats.ttest_ind(a, b, equal_var=False)
        return r.statistic, r.pvalue, f"group means: {a.mean():.4g} vs {b.mean():.4g}"
    if test in ("pearson", "spearman"):
        x = df[columns[0]].dropna()
        y = df[columns[1]].dropna()
        n = min(len(x), len(y))
        fn = stats.pearsonr if test == "pearson" else stats.spearmanr
        r = fn(x.iloc[:n], y.iloc[:n])
        return r[0], r[1], f"r={r[0]:+.4g} between {columns[0]} and {columns[1]}"
    if test == "chi2":
        table = pd.crosstab(df[columns[0]], df[columns[1]])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        return chi2, p, f"dof={dof}, contingency {table.shape}"
    if test == "anova":
        groups = [g.dropna().values for _, g in df.groupby(group_by)[columns[0]]]
        r = stats.f_oneway(*groups)
        return r.statistic, r.pvalue, f"{len(groups)} groups on {columns[0]}"
    raise ValueError(f"unsupported test {test!r}")


def _two_samples(df, columns, group_by):
    if group_by:
        levels = df[group_by].dropna().unique()
        if len(levels) != 2:
            raise ValueError(f"ttest_ind by group needs exactly 2 groups, got {len(levels)}")
        a = df[df[group_by] == levels[0]][columns[0]].dropna()
        b = df[df[group_by] == levels[1]][columns[0]].dropna()
        return a, b
    return df[columns[0]].dropna(), df[columns[1]].dropna()
