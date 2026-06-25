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

__all__ = [
    "python_exec", "make_python_exec", "make_sandboxed_python_exec",
    "load_dataset", "make_load_dataset",
    "describe_data", "fit_model", "hypothesis_test", "plot",
]
