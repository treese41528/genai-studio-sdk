"""``run_shell`` — a workspace-confined shell tool.

Reuses the process-isolation principle of ``make_sandboxed_python_exec``: each command
runs in its own session/process group (``os.setsid``) with ``RLIMIT_AS``/``RLIMIT_CPU``
set pre-exec and a wall-clock timeout that SIGKILLs the WHOLE group (so forked
descendants can't outlive it). ``cwd`` is pinned to the workspace root and the env is
scrubbed to a minimal allow-list.

HONEST LIMITS (a poor-man's confinement, NOT an OS jail): there is no Landlock/seccomp/
bwrap here, so an APPROVED command can still read/write anything the user can — the real
control is the approval engine + the read-only command allow-list (networked / writing
commands are not auto-approved and always prompt). ``allow_network=False`` is advisory
(no enforced network namespace under WSL2/non-root). **Unix only.**
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass

from genai_studio.agents import ToolResult, tool

from ._workspace import WorkspaceConfig

_MAX_OUT = 50_000
_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER")


@dataclass
class ShellSandboxConfig:
    mem_mb: int = 1024
    cpu_s: int = 30
    allow_network: bool = False     # advisory only (see module note)
    max_procs: int | None = None


def make_run_shell(ws: WorkspaceConfig, *, sandbox: ShellSandboxConfig | None = None):
    """Return a ``run_shell`` tool bound to ``ws``. Raises on non-Unix platforms."""
    if not hasattr(os, "fork"):
        raise RuntimeError("run_shell requires Unix (os.setsid + resource); omit it on this platform.")
    import resource
    cfg = sandbox or ShellSandboxConfig()

    def _child_setup() -> None:        # post-fork, pre-exec, in the child
        os.setsid()                    # own session/group -> killable as a unit
        limits = [(resource.RLIMIT_AS, cfg.mem_mb * 1024 * 1024), (resource.RLIMIT_CPU, cfg.cpu_s)]
        if cfg.max_procs is not None:
            limits.append((resource.RLIMIT_NPROC, cfg.max_procs))
        for what, lim in limits:
            try:
                resource.setrlimit(what, (lim, lim))
            except (ValueError, OSError):
                pass

    @tool
    def run_shell(command: str, timeout: float = 30) -> ToolResult:
        """Run a shell command in the workspace directory and return its combined output.

        Args:
            command: The shell command to run (executed via ``bash -lc``).
            timeout: Wall-clock seconds before the command (and its whole process group) is killed.
        """
        if not shutil.which("bash"):
            return ToolResult(content="", error="bash not found on PATH")
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["GENAI_STUDIO_SHELL"] = "1"
        try:
            proc = subprocess.Popen(["bash", "-lc", command], cwd=str(ws.root), env=env,
                                    preexec_fn=_child_setup, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as e:
            return ToolResult(content="", error=f"could not start shell: {e}")
        try:
            out, _ = proc.communicate(timeout=timeout)
            _killpg(proc.pid)                          # reap any forked group survivors
        except subprocess.TimeoutExpired:
            _killpg(proc.pid)
            try:
                out, _ = proc.communicate(timeout=2)
            except Exception:
                proc.wait()
                out = ""
            return ToolResult(content=(out or "")[:_MAX_OUT],
                              error=f"command exceeded the {timeout}s wall-clock limit (killed)")
        out = (out or "")[:_MAX_OUT]
        if proc.returncode != 0:
            return ToolResult(content=out, error=f"command exited with status {proc.returncode}")
        return ToolResult(content=out if out.strip() else "(no output)")

    return run_shell


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
