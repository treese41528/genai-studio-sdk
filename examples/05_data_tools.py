"""Data tools + a sandbox-safety lesson.

The data_analyst agent drives python_exec (a persistent namespace) and
load_dataset to analyze a bundled dataset.

⚠️  SANDBOX SAFETY: python_exec runs code IN THIS PROCESS. It is for
trusted-input only. The bottom of this script demonstrates *why* — printing
the live namespace and showing that arbitrary code (including os access) runs
with your privileges. For untrusted/web data, use the hardened subprocess
drop-in described in python_exec.py.

Run: python examples/05_data_tools.py   (needs: pip install '.[datascience]')
"""

from __future__ import annotations

from genai_studio.agents import ConsoleTracer
from genai_studio.agents.datascience import data_analyst
from genai_studio.agents.datascience.tools import make_python_exec
from _common import make_client


if __name__ == "__main__":
    agent = data_analyst(make_client(), model=None, tracer=ConsoleTracer())
    # model=None -> falls back to the client's default_model
    result = agent.run(
        "Load the iris dataset and tell me which two features best separate the species. "
        "Show the evidence."
    )
    print("\n=== FINDINGS ===\n", result.text)

    # ── The safety lesson ────────────────────────────────────────────────
    print("\n=== Why python_exec is UNSAFE (trusted input only) ===")
    px = make_python_exec()
    px("import os")
    r = px("os.getcwd()")  # arbitrary host access runs with YOUR privileges
    print("model-run code can read the host:", r.content)
    print("State persists across calls in a live namespace — convenient, but a")
    print("prompt-injected model could run anything. Harden before untrusted data.")
