"""Shared value summariser: turn a Python result into a compact string the model
can read, without dumping a whole DataFrame/array (token blow-up)."""

from __future__ import annotations

MAX_CHARS = 2000


def summarize(value, max_chars: int = MAX_CHARS) -> str:
    """A compact, model-facing summary of ``value``.

    DataFrames -> shape + dtypes + ``head()``; Series -> head; ndarrays ->
    shape/dtype + a small preview; everything else -> truncated ``repr``.
    """
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return _clip(
                f"DataFrame shape={value.shape}\n"
                f"columns: {list(value.columns)}\n"
                f"dtypes:\n{value.dtypes.to_string()}\n"
                f"head:\n{value.head().to_string()}",
                max_chars,
            )
        if isinstance(value, pd.Series):
            return _clip(
                f"Series name={value.name!r} len={len(value)} dtype={value.dtype}\n"
                f"{value.head(10).to_string()}",
                max_chars,
            )
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _clip(
                f"ndarray shape={value.shape} dtype={value.dtype}\n"
                f"{np.array2string(value, threshold=50, edgeitems=3)}",
                max_chars,
            )
    except ImportError:
        pass

    return _clip(repr(value), max_chars)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
