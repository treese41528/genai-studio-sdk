"""Standalone sandbox worker (stdlib ONLY) for ``make_sandboxed_python_exec``.

Invoked as ``python _sandbox_worker.py <input.pkl> <output.pkl>`` in a fresh,
resource-limited subprocess. Resource caps (RLIMIT_AS / RLIMIT_CPU) are applied by
the PARENT via ``subprocess`` ``preexec_fn`` *before* this interpreter starts, so
this worker only restores state, runs the code, and ships results back via a temp
file (never via stdout, which user ``print()``s are captured away from).

Deliberately imports NO genai_studio code — keeps the sandbox light and avoids
re-triggering SDK imports inside the jailed process.
"""

import ast
import contextlib
import io
import pickle
import sys
import traceback
import types


def _block_network():
    """Best-effort Python-level network block — SOFT, not an OS guarantee. Bypassable
    via os.system/subprocess, _socket, importlib.reload(socket), or ctypes."""
    import socket

    def _denied(*a, **k):
        raise OSError("network access is disabled in the sandbox")

    for attr in ("socket", "create_connection", "socketpair", "create_server",
                 "fromfd", "getaddrinfo"):
        if hasattr(socket, attr):
            setattr(socket, attr, _denied)


def _clip(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def _summarize(value, max_chars):
    """A compact, model-facing summary of the trailing-expression value.

    Mirrors ``genai_studio.agents.datascience._format.summarize`` — keep roughly in
    sync (this worker is stdlib-only and cannot import it).
    """
    if value is None:
        return ""
    try:
        import pandas as pd  # only if the sandboxed code already pulled it in
        if isinstance(value, pd.DataFrame):
            return _clip(f"DataFrame shape={value.shape}\ncolumns: {list(value.columns)}\n"
                         f"dtypes:\n{value.dtypes.to_string()}\n"
                         f"head:\n{value.head().to_string()}", max_chars)
        if isinstance(value, pd.Series):
            return _clip(f"Series name={value.name!r} len={len(value)} dtype={value.dtype}\n"
                         f"{value.head(10).to_string()}", max_chars)
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return _clip(f"ndarray shape={value.shape} dtype={value.dtype}\n"
                         f"{np.array2string(value, threshold=50, edgeitems=3)}", max_chars)
    except Exception:
        pass
    return _clip(repr(value), max_chars)


def _pickle_subset(ns):
    """Pickle picklable, non-dunder, non-module vars. Returns (bytes, dropped names)."""
    keep, dropped = {}, []
    for k, v in ns.items():
        if k.startswith("__") or isinstance(v, types.ModuleType):
            continue
        try:
            pickle.dumps(v)
            keep[k] = v
        except Exception:
            dropped.append(k)
    try:
        return pickle.dumps(keep), dropped
    except Exception:
        return b"", sorted(k for k in ns if not k.startswith("__"))


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        req = pickle.load(f)

    max_chars = req.get("max_chars", 2000)
    if not req.get("allow_network", False):
        _block_network()

    ns = {}
    if req.get("state"):
        try:
            ns = pickle.loads(req["state"])
        except Exception:
            ns = {}

    stdout = io.StringIO()
    result_value, err = None, None
    try:
        tree = ast.parse(req["code"], mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            if tree.body:
                exec(compile(tree, "<sandbox>", "exec"), ns)
            if last_expr is not None:
                result_value = eval(compile(last_expr, "<sandbox>", "eval"), ns)
    except Exception:
        err = traceback.format_exc()

    state_bytes, dropped = _pickle_subset(ns)
    result = {"out": stdout.getvalue(), "err": err,
              "summary": _summarize(result_value, max_chars),
              "state": state_bytes, "dropped": dropped}
    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
