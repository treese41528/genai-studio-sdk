"""
``r_exec`` — run R code via ``Rscript`` for R-first data-science users.

Many statisticians live in R, so this is the R counterpart to ``python_exec``: it
writes the code to a temp ``.R`` file and runs it under ``Rscript`` in a FRESH
process each call (no persistent state between calls — pass data via files or a
single self-contained script). Requires R on the PATH; a missing R degrades to a
clear ``ToolResult.error`` rather than crashing the run.

SAFETY: like ``python_exec``, this executes arbitrary code. The in-process variant
has no OS isolation — for untrusted input run it behind a :class:`Guard`/approval,
or wrap ``Rscript`` with the same rlimit/subprocess hardening as
``make_sandboxed_python_exec``. ``timeout`` is the one always-on backstop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from genai_studio.agents import ToolResult, tool

from .._format import MAX_CHARS


def make_r_exec(*, rscript: str = "Rscript", timeout: float = 30):
    """Build an ``r_exec`` tool that runs R code via ``Rscript`` (``timeout`` seconds)."""

    @tool(name="r_exec",
          description="Execute R code via Rscript and return its output. Each call is a "
                      "fresh R process (no state persists). Use print()/cat() to show output.")
    def r_exec(code: str) -> ToolResult:
        """Run R ``code`` with Rscript.

        Args:
            code: R source to execute (use print()/cat() to display values).
        """
        exe = shutil.which(rscript)
        if not exe:
            return ToolResult(
                content="", error=f"{rscript!r} not found on PATH — install R to use r_exec.")
        fd, path = tempfile.mkstemp(suffix=".R")
        try:
            os.write(fd, code.encode("utf-8"))
            os.close(fd)
            proc = subprocess.run([exe, "--vanilla", path], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(content="", error=f"r_exec timed out after {timeout}s")
        except Exception as exc:
            return ToolResult(content="", error=f"r_exec failed: {exc}")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        out = proc.stdout or ""
        if (proc.stderr or "").strip():
            out += "\n[stderr]\n" + proc.stderr
        out = out.strip()[:MAX_CHARS]
        if proc.returncode != 0:
            return ToolResult(content=out, error=f"R exited with code {proc.returncode}")
        return ToolResult(content=out or "(no output)", data={"returncode": proc.returncode})

    return r_exec


r_exec = make_r_exec()
