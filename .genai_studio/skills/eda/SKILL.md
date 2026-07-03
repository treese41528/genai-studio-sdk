---
name: eda
description: Exploratory data analysis done rigorously — profile a dataset before modeling or drawing conclusions. Use when asked to explore, summarize, understand, or "look at" a dataset. Loads the data, profiles shape/types/missingness/distributions/relationships with the tools (never by eye), and reports only what the data supports.
---

# Exploratory data analysis (EDA)

Goal: understand a dataset before any claim or model. **Compute every number with a tool** (`python_exec`
/ `describe_data`) — never estimate. Report only what you actually observed.

## The workflow
1. **Load** with `load_dataset` (bundled) or `load_table` (your CSV/Parquet/Excel/JSON). It persists as
   `df` for `python_exec`. Confirm it loaded: shape, first rows.
2. **Profile the structure** — `describe_data` (or `python_exec`): row/col counts, dtypes, and
   **missingness per column** (`df.isna().mean()`). State the unit of observation (what one row IS).
3. **Univariate** — for numerics: mean/median/std/min/max + skew and obvious outliers (IQR); for
   categoricals: value counts + cardinality. Flag anything degenerate (constant columns, 100%-unique IDs,
   near-constant, high-cardinality categoricals).
4. **Relationships** — correlations among numerics (`df.corr(numeric_only=True)`); for a target, how
   features relate to it (grouped means, a `plot`). Note direction and rough strength, not causation.
5. **Data-quality flags** — missingness, duplicates (`df.duplicated().sum()`), impossible values
   (negative ages, dates in the future), class imbalance if there's a label, and possible leakage
   (a feature that trivially encodes the target).
6. **Report** — a short, ordered summary: what the data is, its size/quality, the notable
   distributions/relationships, and the caveats. Give the numbers you computed.

## Rigor rules
- If a column is the intended **target**, say so and report its balance/distribution first.
- Distinguish **correlation from causation** explicitly; EDA suggests hypotheses, it doesn't confirm them.
- Don't impute or drop silently — if you clean, say exactly what and why (and prefer to just *report*
  the issue in EDA).
- If the data can't answer the question (wrong columns, too few rows, all-missing), **say so** rather
  than inventing a finding.

## Example moves
```python
df.shape, df.dtypes                       # structure
df.isna().mean().sort_values(ascending=False).head()   # missingness
df.describe(include="all")                # summary
df.corr(numeric_only=True)                # numeric relationships
df.duplicated().sum()                     # dupes
```
Then `plot` a distribution or a relationship worth showing, and summarize in plain language.
