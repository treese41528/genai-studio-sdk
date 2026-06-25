"""``describe_data`` — one-call exploratory data analysis of a CSV."""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

from .._format import MAX_CHARS, _clip
from .._guard import _require


@tool
def describe_data(file: str) -> ToolResult:
    """Load a CSV and summarise it: shape, dtypes, missing-value counts, summary
    statistics, and the strongest pairwise correlations.

    Args:
        file: path to a CSV file.
    """
    pd = _require("pandas")
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return ToolResult(content="", error=f"could not read {file!r}: {exc}")

    parts = [
        f"shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"columns: {list(df.columns)}",
        f"dtypes:\n{df.dtypes.to_string()}",
        f"missing values per column:\n{df.isna().sum().to_string()}",
        f"summary statistics:\n{df.describe(include='all').to_string()}",
    ]

    num = df.select_dtypes("number")
    if num.shape[1] >= 2:
        corr = num.corr()
        pairs = []
        cols = list(num.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((abs(corr.iloc[i, j]), cols[i], cols[j], corr.iloc[i, j]))
        pairs.sort(reverse=True)
        top = "\n".join(f"  {a} ~ {b}: r={r:+.3f}" for _, a, b, r in pairs[:8])
        parts.append("strongest correlations:\n" + top)

    return ToolResult(content=_clip("\n\n".join(parts), MAX_CHARS), data=df)
