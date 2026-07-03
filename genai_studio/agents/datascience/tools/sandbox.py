"""
``make_sandboxed_python_exec`` — a hardened, OPT-IN drop-in for ``python_exec``.

Each ``code`` call runs in a SEPARATE subprocess made its own **session/process
group** (``os.setsid``) so the whole subtree can be reaped. In the child, before
exec, ``preexec_fn`` sets ``RLIMIT_AS`` (memory) + ``RLIMIT_CPU`` (and optionally
``RLIMIT_NPROC`` / ``RLIMIT_FSIZE``). The parent enforces a wall-clock timeout and
SIGKILLs the **entire group** (so forked grandchildren can't outlive it). A soft
Python-level network block is applied, and picklable namespace state is persisted
across calls (unpicklable vars are dropped, with a note). Same
``code: str -> ToolResult`` signature as ``make_python_exec``; limits/timeouts come
back as ``ToolResult.error`` (the loop never crashes).

WHAT IT BOUNDS: per-call address space (memory), CPU-seconds, wall-clock time, and
— if you set them — process count and per-file write size. Forked descendants are
reaped on every call (via the group kill), so an orphan can't linger.

WHAT IT DOES **NOT** do (it is a poor-man's sandbox, NOT an OS jail):
  * The network block is *Python-level and soft*: trivially bypassable via
    ``os.system``/``subprocess`` (curl), ``_socket``, ``importlib.reload(socket)``,
    or ``ctypes`` raw syscalls. Treat ``allow_network=False`` as "discourages casual
    network use," not a guarantee.
  * No filesystem isolation: the code can READ and OVERWRITE any file the host user
    can. Total disk use isn't bounded unless you set ``max_file_mb`` (and even that
    is per-file, not total).
  * No ``seccomp``/``Landlock`` — it does not stop determined syscall-level abuse.
  * ``RLIMIT_AS`` caps *virtual* address space, not RSS: numpy/pandas reserve far
    more AS than they use, so the pandas stack needs ≈2 GiB just to import (hence
    the ``mem_mb`` default). It bounds gross over-allocation, not precise RSS.
  * ``ToolResult.data`` is always ``None`` (the value lives in a dead subprocess);
    only the text summary crosses back. And unlike ``make_python_exec(namespace)``,
    this runs out-of-process and CANNOT share a live namespace with other tools
    (e.g. ``load_dataset``'s ``df`` injection is invisible) — load data *inside* the
    sandboxed code instead.
For genuinely untrusted input at scale, run the whole agent in a container/nsjail.
**Unix only** (needs ``os.fork`` + ``resource`` + ``preexec_fn``); raises on *call*
(not import) on other platforms — use ``make_python_exec`` there.
"""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import tempfile

from genai_studio.agents import ToolResult, tool

from .._format import MAX_CHARS

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_worker.py")


def make_sandboxed_python_exec(*, mem_mb: int = 2048, cpu_s: int = 10, wall_s: int = 15,
                               allow_network: bool = False, max_procs: int | None = None,
                               max_file_mb: int | None = None, approve_fn=None,
                               name: str = "python_exec"):
    """Build a resource-limited, subprocess-isolated ``python_exec`` tool.

    Args:
        mem_mb: address-space cap per call, MiB (``RLIMIT_AS`` — *virtual*, not RSS).
            The default (2048) is the realistic floor for importing numpy/pandas;
            values below ~1.5 GiB will fail to ``import pandas``.
        cpu_s: CPU-seconds cap per call (``RLIMIT_CPU``).
        wall_s: wall-clock cap per call (parent SIGKILLs the whole group). Keep > ``cpu_s``.
        allow_network: if False (default), apply the SOFT network block (see module note).
        max_procs: if set, ``RLIMIT_NPROC`` cap (per-user — set with care; the group
            kill already reaps forked descendants on each call).
        max_file_mb: if set, ``RLIMIT_FSIZE`` per-file write cap, MiB.
        approve_fn: optional ``(code: str) -> bool | str`` policy gate run BEFORE exec;
            return ``False`` or a NON-EMPTY reason string to reject (``True``/``None``/
            empty allow).
        name: the tool name the model sees.
    """
    if not hasattr(os, "fork"):  # preexec_fn + setrlimit + setsid need Unix
        raise RuntimeError(
            "make_sandboxed_python_exec requires Unix (os.fork + resource + setsid). "
            "Use make_python_exec on this platform, or run inside a Linux container.")
    import resource

    def _child_setup():  # runs post-fork, pre-exec, in the child
        os.setsid()  # own session/process group -> killable as a unit
        limits = [(resource.RLIMIT_AS, mem_mb * 1024 * 1024), (resource.RLIMIT_CPU, cpu_s)]
        if max_procs is not None:
            limits.append((resource.RLIMIT_NPROC, max_procs))
        if max_file_mb is not None:
            limits.append((resource.RLIMIT_FSIZE, max_file_mb * 1024 * 1024))
        for what, lim in limits:
            try:
                resource.setrlimit(what, (lim, lim))
            except (ValueError, OSError):  # pragma: no cover - platform/perm dependent
                pass

    state = {"bytes": b""}  # persisted picklable namespace across calls

    @tool(name=name,
          description="Execute Python code in a RESOURCE-LIMITED subprocess (memory/"
                      "CPU/time caps; network soft-blocked). Variables persist across "
                      "calls; the trailing expression is summarised as text (the object "
                      "itself is not returned). Use print() to show output.")
    def python_exec(code: str) -> ToolResult:
        """Run Python ``code`` in a sandboxed subprocess; state persists across calls.

        Args:
            code: Python source to execute. The trailing expression is summarised.
        """
        if approve_fn is not None:
            verdict = approve_fn(code)
            if verdict is False or (isinstance(verdict, str) and verdict.strip()):
                reason = verdict if isinstance(verdict, str) else "rejected by policy"
                return ToolResult(content="", error=f"code not run: {reason}")

        req = {"code": code, "state": state["bytes"],
               "allow_network": allow_network, "max_chars": MAX_CHARS}
        in_fd, in_path = tempfile.mkstemp(suffix=".in.pkl")
        out_fd, out_path = tempfile.mkstemp(suffix=".out.pkl")  # mkstemp -> unpredictable, O_EXCL
        os.write(in_fd, pickle.dumps(req))
        os.close(in_fd)
        os.close(out_fd)
        try:
            proc = subprocess.Popen([sys.executable, _WORKER, in_path, out_path],
                                    preexec_fn=_child_setup,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pgid, errb, timed_out = proc.pid, b"", False
            try:
                _, errb = proc.communicate(timeout=wall_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _killpg(pgid)  # SIGKILL the whole group -> worker + any forked descendants
                try:
                    _, errb = proc.communicate(timeout=2)  # drain pipes + reap the killed worker
                except Exception:
                    proc.wait()
            else:
                _killpg(pgid)  # reap any forked group survivors (orphan / fork bomb)
            if timed_out:
                return ToolResult(content="", error=(
                    f"execution exceeded the {wall_s}s wall-clock limit (killed)."))
            try:
                with open(out_path, "rb") as f:
                    res = pickle.load(f)
            except Exception:
                return ToolResult(content="", error=_kill_reason(proc.returncode, errb))
        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        state["bytes"] = res.get("state") or b""
        if res.get("err"):
            out = res.get("out", "")
            content = (out + "\n" if out.strip() else "") + "Error:\n" + res["err"]
            return ToolResult(content=content[:MAX_CHARS], error=res["err"])

        parts = []
        if res.get("out", "").strip():
            parts.append(res["out"].rstrip())
        if res.get("summary"):
            parts.append(res["summary"])
        if res.get("dropped"):
            d = res["dropped"]
            parts.append(f"(note: {len(d)} variable(s) not retained across calls "
                         f"[unpicklable]: {', '.join(d[:5])})")
        content = "\n".join(parts) if parts else "(no output)"
        return ToolResult(content=content[:MAX_CHARS])

    return python_exec


def _killpg(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _kill_reason(rc, errb) -> str:
    """A clear message for a child that produced no result (died on a signal/limit)."""
    if rc is not None and rc < 0:
        sig = -rc
        if sig == signal.SIGXCPU:
            why = "the CPU limit (SIGXCPU)"
        elif sig == signal.SIGXFSZ:
            why = "the file-size limit (SIGXFSZ)"
        elif sig == signal.SIGKILL:
            why = "SIGKILL (memory limit / wall-clock / OOM killer)"
        else:
            try:
                why = f"signal {signal.Signals(sig).name}"
            except ValueError:  # pragma: no cover
                why = f"signal {sig}"
        msg = f"sandbox terminated the code: {why}."
    else:
        msg = f"sandbox produced no result (exit code {rc})."
    tail = (errb or b"")[-300:].decode("utf-8", "replace").strip()
    return f"{msg} {tail}".strip()
