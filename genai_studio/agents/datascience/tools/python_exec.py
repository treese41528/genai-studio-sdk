"""
``python_exec`` — execute Python in a persistent namespace.

╔══════════════════════════════════════════════════════════════════════════╗
║  UNSAFE — RUNS ARBITRARY CODE IN THIS PROCESS.                            ║
║  Trusted, in-process inputs ONLY. A model (or a prompt-injected one) can  ║
║  run ``__import__('os').system(...)``, read files, open sockets, etc.     ║
║  The moment an agent processes shared, user-supplied, or web-fetched      ║
║  data, replace this with the hardened subprocess/rlimit drop-in (see the  ║
║  "Production hardening" note below). This very transparency is the        ║
║  explicit-sandbox lesson.                                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

State persists across calls within one agent run, so a DataFrame created in one
step is available in the next. Execution output maps to a ``ToolResult``:
- ``content`` : captured stdout + a compact summary of the last expression's value
- ``data``    : the raw last-expression value (a DataFrame/array/figure for UIs)
- ``error``   : the full traceback on failure (fed back so the model self-corrects)

Production hardening: use the implemented drop-in ``make_sandboxed_python_exec``
(``tools/sandbox.py``) — same ``code: str -> ToolResult`` signature, but each call
runs in a resource-limited subprocess (``RLIMIT_AS``/``RLIMIT_CPU`` + a wall-clock
SIGKILL + a soft network block; picklable state shipped across calls). That bounds
runaway memory/CPU/time and casual network use but is NOT an OS jail — for
untrusted input at scale, run the whole agent inside a container/nsjail.
"""

from __future__ import annotations

import ast
import contextlib
import io
import traceback

from genai_studio.agents import ToolResult, tool

from .._format import MAX_CHARS, summarize


def make_python_exec(namespace: dict | None = None):
    """Build a fresh ``python_exec`` tool bound to its own (or a shared) namespace."""
    ns: dict = namespace if namespace is not None else {}

    @tool(name="python_exec",
          description="Execute Python code in a persistent namespace. Variables "
                      "(e.g. DataFrames) persist across calls. Use print() to show output.")
    def python_exec(code: str) -> ToolResult:
        """Run Python ``code``. State persists between calls within one run.

        Args:
            code: Python source to execute. The value of a trailing expression
                is summarised and returned (notebook-style).
        """
        stdout = io.StringIO()
        result_value = None
        try:
            tree = ast.parse(code, mode="exec")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                if tree.body:
                    exec(compile(tree, "<python_exec>", "exec"), ns)
                if last_expr is not None:
                    result_value = eval(  # noqa: S307 - intentional, see UNSAFE banner
                        compile(last_expr, "<python_exec>", "eval"), ns)
        except Exception:
            tb = traceback.format_exc()
            out = stdout.getvalue()
            content = (out + "\n" if out.strip() else "") + "Error:\n" + tb
            return ToolResult(content=content[:MAX_CHARS], error=tb)

        out = stdout.getvalue()
        parts = []
        if out.strip():
            parts.append(out.rstrip())
        if result_value is not None:
            parts.append(summarize(result_value))
        content = "\n".join(parts) if parts else "(no output)"
        return ToolResult(content=content[:MAX_CHARS], data=result_value)

    return python_exec


# Module-level convenience tool (its own shared namespace). For per-agent
# isolation, use make_python_exec() — the data_analyst factory does.
python_exec = make_python_exec()
