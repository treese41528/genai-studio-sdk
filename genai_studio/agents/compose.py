"""Composition helpers — the single assembly point for the always-on system-prompt blocks.

``assemble_system`` joins the base system prompt with the always-on disclosure blocks (project
memory, recalled memory, skills catalog, deferred-tool catalog — each pre-rendered WITH its own
header) in priority order under ONE byte budget. Routing every block through one function means
the REPL's "strip system from threaded history, inject once per turn" invariant covers them all
uniformly, and the combined floor stays bounded (lower-priority blocks at the end are dropped
first if the budget is exceeded; ``base`` is always kept).

``assemble_agent`` (added once skills/memory/deferred land) will be the single wiring entry point
so the REPL and headless callers configure an Agent identically.
"""

from __future__ import annotations


def assemble_system(base: str, *blocks: str, budget_chars: int = 8000) -> str:
    """Join ``base`` with always-on ``blocks`` (each self-headered) in the given priority order,
    under ``budget_chars``. A block is included only if it fits the remaining budget; once one
    doesn't fit, it and every lower-priority block after it are dropped (so the highest-priority
    blocks are always the ones kept). ``base`` is always included even if it alone exceeds the
    budget. ``budget_chars=0`` disables the cap."""
    out = (base or "").rstrip()
    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue
        if budget_chars and len(out) + len(b) + 2 > budget_chars:
            break                                  # drop this + all lower-priority blocks
        out += "\n\n" + b
    return out


def wire_capabilities(tools, *, cwd, client=None, model=None, memory_dir=None,
                      skills=True, memory=True, defer=False, shared_guards=(), studio=None):
    """Augment a base tool list with skills + recall-memory + (optional) deferred-tool search.

    Returns ``(tools, blocks, tool_search, n_skills)``: the augmented tool list, the always-on
    system blocks in PRIORITY order (recalled-memory index, then skills catalog), a ``ToolSearch``
    (or ``None``), and the skill count. The REPL (``cli.py``) and the headless ``assemble_agent``
    both wire capabilities through THIS function so they can never drift.
    """
    from pathlib import Path
    tools = list(tools)
    blocks, n_skills, tool_search = [], 0, None
    meta_eager = []                                 # meta-tools that must stay always-visible under deferral
    if memory:
        from .memory import make_memory_tools, memory_index_text, open_store
        store = open_store(cwd, memory_dir or (Path.home() / ".genai_studio" / "memory"))
        tools += make_memory_tools(store, studio=studio)   # studio -> embedding rerank (fail-open to keyword)
        meta_eager += ["write_memory", "recall_memory"]
        blocks.append(memory_index_text(store))     # recalled memory ranks ABOVE the skills catalog
    if skills:
        from .skills import build_skill_tools
        bundle = build_skill_tools(cwd, base_tools=tools, client=client, default_model=model,
                                   shared_guards=shared_guards)
        if bundle.tool is not None:
            tools.append(bundle.tool)
            meta_eager.append("use_skill")
        blocks.append(bundle.catalog)
        n_skills = len(bundle.skills)
    if defer:                                       # defer the bulk tools, but keep meta-tools eager
        from .tool_search import ToolSearch, embedding_rank
        rank = embedding_rank(studio) if studio is not None else None   # embed-ranked if studio given
        tool_search = ToolSearch(deferred=("*",), eager=tuple(meta_eager), rank=rank)
    return tools, blocks, tool_search, n_skills


def assemble_agent(client, profile="general", cwd=None, *, model=None, skills=True, memory=True,
                   defer=False, mode="auto", sandbox="workspace-write", prompt_fn=None,
                   system_base="", memory_dir=None, max_steps=10, guards=(), studio=None, **agent_kw):
    """Single composition entry point — build profile tools + skills + recall-memory + optional
    deferred-tool search, assemble the system prompt, and return a ready headless ``Agent``.

    The same builders the REPL uses, in one call, so a headless caller configures an Agent
    identically. Returns a plain ``Agent`` (adds no new state); wire manually for custom cases.
    """
    from pathlib import Path

    from .agent import Agent
    from .approval import ApprovalMode, SandboxPolicy
    from .profiles import build_tools
    cwd = Path(cwd) if cwd else Path.cwd()
    base_tools, approval_guard, _cfg = build_tools(
        profile, workspace_root=cwd, mode=ApprovalMode(mode), sandbox=SandboxPolicy(sandbox),
        prompt_fn=prompt_fn)
    tools, blocks, tool_search, _ = wire_capabilities(
        base_tools, cwd=cwd, client=client, model=model, memory_dir=memory_dir,
        skills=skills, memory=memory, defer=defer, shared_guards=[approval_guard], studio=studio)
    return Agent(client=client, tools=tools, system=assemble_system(system_base, *blocks),
                 model=model, guards=[approval_guard, *guards], tool_search=tool_search,
                 max_steps=max_steps, **agent_kw)
