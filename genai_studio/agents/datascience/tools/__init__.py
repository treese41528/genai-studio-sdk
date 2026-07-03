"""Shipped data-science tools (require the ``[datascience]`` extra to *run*).

Importing this module is light — heavy deps load only when a tool executes.
"""

from .datasets import load_dataset, make_load_dataset
from .describe import describe_data
from .modeling import fit_model
from .plotting import plot
from .python_exec import make_python_exec, python_exec
from .sandbox import make_sandboxed_python_exec
from .stats import hypothesis_test


def make_datascience_tools(namespace: dict | None = None, *, sandboxed: bool = False) -> list:
    """The full data-science tool set over ONE shared ``namespace`` (a DataFrame persists across
    calls): python_exec + load_dataset + load_table + describe_data + fit_model + hypothesis_test + plot.
    Shared by ``data_analyst`` and the ``datascience`` REPL profile so they stay in lock-step.

    ``sandboxed=True`` swaps python_exec for the hardened subprocess variant (Unix-only; it keeps its
    OWN persistent state, so it does NOT share ``namespace`` with load_dataset — a documented tradeoff)."""
    namespace = namespace if namespace is not None else {}
    from .io_tools import make_load_table
    from .verify import make_verify_stat
    py = make_sandboxed_python_exec() if sandboxed else make_python_exec(namespace)
    return [py, make_load_dataset(namespace), make_load_table(namespace),
            describe_data, fit_model, hypothesis_test, plot, make_verify_stat(namespace)]


from .verify import make_verify_stat   # noqa: E402

__all__ = [
    "python_exec", "make_python_exec", "make_sandboxed_python_exec", "make_datascience_tools",
    "make_verify_stat", "load_dataset", "make_load_dataset",
    "describe_data", "fit_model", "hypothesis_test", "plot",
]
