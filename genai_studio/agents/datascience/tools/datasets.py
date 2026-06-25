"""``load_dataset`` — load a bundled toy dataset (from scikit-learn).

Uses scikit-learn's built-in toy sets (numpy arrays inside the wheel, tens of KB),
so nothing is vendored or downloaded. When bound to a shared namespace (as the
``data_analyst`` factory does), the loaded DataFrame is also injected as ``df`` and
under its name, so ``python_exec`` can use it directly in the next step.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

from .._format import summarize
from .._guard import _require

_LOADERS = {
    "iris": "load_iris",
    "wine": "load_wine",
    "diabetes": "load_diabetes",
    "breast_cancer": "load_breast_cancer",
}


def make_load_dataset(namespace: dict | None = None):
    """Build a ``load_dataset`` tool, optionally sharing a namespace with python_exec."""

    @tool(name="load_dataset",
          description="Load a bundled toy dataset (iris, wine, diabetes, breast_cancer) "
                      "and return a short summary.")
    def load_dataset(name: str) -> ToolResult:
        """Load a bundled toy dataset and summarise it (shape, columns, dtypes).

        Args:
            name: one of 'iris', 'wine', 'diabetes', 'breast_cancer'.
        """
        if name not in _LOADERS:
            return ToolResult(
                content="",
                error=f"Unknown dataset {name!r}. Available: {', '.join(_LOADERS)}.",
            )
        sk = _require("sklearn.datasets")
        _require("pandas")  # ensure pandas present for as_frame=True
        bunch = getattr(sk, _LOADERS[name])(as_frame=True)
        df = bunch.frame
        if namespace is not None:
            namespace[name] = df
            namespace["df"] = df
        hint = f"\n(available in python_exec as `df` and `{name}`)" if namespace is not None else ""
        return ToolResult(content=f"Loaded '{name}':\n{summarize(df)}{hint}", data=df)

    return load_dataset


load_dataset = make_load_dataset()
