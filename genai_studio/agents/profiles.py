"""``build_tools`` — assemble a tool set + its approval guard for a profile.

One call returns ``(tools, approval_guard, config)``: a deduped tool list, the
``before_tool`` guard enforcing approval, and the mutable :class:`ApprovalConfig` (so
``/approvals`` can edit it live). Threads ONE shared ``namespace`` through the data
tools (so a DataFrame persists across calls, like ``data_analyst``) and ONE
``WorkspaceConfig`` through the coding tools and the approval engine. Datascience tools
and platform-specific shell are imported lazily and skipped if unavailable.
"""

from __future__ import annotations

from pathlib import Path

from .approval import ApprovalConfig, ApprovalMode, SandboxPolicy, approval_guard
from .tools._workspace import WorkspaceConfig


def build_tools(profile: str = "general", *, workspace_root=None,
                mode: ApprovalMode = ApprovalMode.suggest,
                sandbox: SandboxPolicy = SandboxPolicy.workspace_write,
                prompt_fn=None, include_datascience: bool = True):
    """Return ``(tools, approval_guard, config)`` for ``profile`` in
    {``research``, ``coding``, ``general``}."""
    from .tools.general import calculator, final_answer, finish
    from .tools.web import web_search, wikipedia_search
    from .tools.files import make_file_tools

    ws = WorkspaceConfig(root=Path(workspace_root or Path.cwd()))
    config = ApprovalConfig(workspace=ws, mode=mode, sandbox=sandbox, prompt_fn=prompt_fn)
    namespace: dict = {}

    research = [web_search, wikipedia_search, calculator, final_answer, finish]
    try:                                            # http/academic aren't re-exported
        from .tools.http import make_fetch_json, make_http_get
        research += [make_http_get(), make_fetch_json()]
    except Exception:
        pass
    try:
        from .tools.academic import arxiv_search, openalex_search
        research += [arxiv_search, openalex_search]
    except Exception:
        pass

    file_tools = make_file_tools(ws)                # [read_file, write_file, edit_file]
    read_file = file_tools[0]
    coding = list(file_tools)
    try:
        from .tools.shell import make_run_shell
        coding.append(make_run_shell(ws))
    except RuntimeError:                            # non-Unix
        pass

    data: list = []
    if include_datascience:
        try:
            from .datascience.tools.datasets import make_load_dataset
            from .datascience.tools.io_tools import make_load_table
            from .datascience.tools.python_exec import make_python_exec
            data = [make_python_exec(namespace), make_load_table(namespace), make_load_dataset(namespace)]
        except Exception:
            pass

    if profile == "research":
        tools = research + [read_file]
    elif profile == "coding":
        tools = research + coding + data
    else:                                           # general
        tools = research + coding + data

    seen: set = set()
    deduped = []
    for t in tools:
        if t.name not in seen:
            seen.add(t.name)
            deduped.append(t)
    return deduped, approval_guard(config), config
