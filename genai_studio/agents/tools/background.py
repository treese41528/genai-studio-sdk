"""``run_background`` / ``check_job`` — start a long-running shell command and poll it.

For processes that outlive one tool call (dev servers, builds, long test runs): ``run_background``
starts the command detached (workspace cwd, its own process group, output captured to a temp file)
and returns a job id; ``check_job`` reports whether it's still running plus the recent output (and
can terminate it). A session-local job registry. Unix-oriented, like ``run_shell``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile

from genai_studio.agents import ToolResult, tool

from ._workspace import WorkspaceConfig


def make_background_tools(ws: WorkspaceConfig) -> list:
    """Return ``[run_background, check_job]`` bound to ``ws`` (share one job registry)."""
    if os.name != "posix":                                    # matches make_run_shell's posix-only stance
        raise RuntimeError("background jobs are Unix-only on this build")
    jobs: dict = {}
    counter = {"n": 0}

    @tool
    def run_background(command: str) -> ToolResult:
        """Start a shell command in the BACKGROUND (for long-running processes: a dev server, a
        build, a long test run). Returns a job id; poll it with check_job(id). Runs with the
        workspace as its working directory.

        Args:
            command: The shell command to run in the background.
        """
        counter["n"] += 1
        jid = f"job{counter['n']}"
        fd, out_path = tempfile.mkstemp(prefix=f"{jid}-", suffix=".log")
        os.close(fd)
        try:
            f = open(out_path, "wb")
            proc = subprocess.Popen(["bash", "-lc", command], cwd=str(ws.root), stdout=f,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        except Exception as e:
            return ToolResult(content="", error=f"could not start: {e}")
        jobs[jid] = {"proc": proc, "out": out_path, "cmd": command}
        return ToolResult(content=f"started {jid} (pid {proc.pid}): {command}\n"
                          f"poll it with check_job('{jid}')", data={"job": jid, "pid": proc.pid})

    @tool
    def check_job(job_id: str, tail_lines: int = 40, kill: bool = False) -> ToolResult:
        """Check a background job's status and recent output, or terminate it.

        Args:
            job_id: The id returned by run_background.
            tail_lines: How many trailing output lines to show.
            kill: If true, terminate the job (SIGTERM its process group).
        """
        job = jobs.get(job_id)
        if job is None:
            return ToolResult(content="", error=f"unknown job {job_id!r}; running: {sorted(jobs) or '(none)'}")
        proc = job["proc"]
        if kill and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
        code = proc.poll()
        status = "running" if code is None else f"exited ({code})"
        try:
            out = open(job["out"], "r", errors="replace").read()
        except OSError:
            out = ""
        tail = "\n".join(out.splitlines()[-max(1, tail_lines):])
        return ToolResult(content=f"{job_id}: {status}  ({job['cmd']})\n--- output (last {tail_lines} lines) ---\n{tail}",
                          data={"status": status})

    return [run_background, check_job]
