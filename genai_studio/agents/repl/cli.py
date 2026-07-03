"""``run_repl`` — the interactive agent loop behind ``genai-studio agent``."""

from __future__ import annotations

import sys
from pathlib import Path

from ..agent import Agent, Cancel
from ..approval import ApprovalMode, SandboxPolicy
from ..client import GenAIStudioClient
from ..compose import assemble_system, wire_capabilities
from ..presets import resolve_preset
from ..profiles import build_tools
from ..trace import NullTracer
from .commands import ReplContext, build_registry, load_custom_into
from .commands import _resume as resume_cmd
from .config import ReplConfig
from .interrupt import turn_interrupt
from .memory import load_project_memory
from .render import StreamRenderer
from .session import SessionRecorder

try:                                     # enables arrow-key editing + in-session history
    import readline  # noqa: F401
except ImportError:                      # not available on some platforms (e.g. plain Windows)
    pass


def _slash_completer(registry):
    """A readline completer over slash commands — matches the typed ``/prefix`` (read live from the
    registry, so custom commands are included). Repeated Tab cycles through the matches."""
    def completer(text, state):
        if not text.startswith("/"):
            return None
        matches = sorted(f"/{n}" for n in registry.names() if f"/{n}".startswith(text))
        return matches[state] if state < len(matches) else None
    return completer


def _install_slash_completion(registry):
    """Bind Tab to CYCLE through slash-command completions (like the ``/`` menu in Claude Code):
    Tab next, Shift-Tab previous, and list the options when ambiguous. Fails open if unavailable."""
    try:
        import readline
    except ImportError:
        return
    readline.set_completer(_slash_completer(registry))
    readline.set_completer_delims(" \t\n")            # keep "/model" as one token
    for binding in ("tab: menu-complete",             # cycle forward through matches
                    '"\\e[Z": menu-complete-backward',  # Shift-Tab: cycle backward
                    "set show-all-if-ambiguous on",   # first Tab also lists the options
                    "set menu-complete-display-prefix on"):
        try:
            readline.parse_and_bind(binding)
        except Exception:
            pass

BASE_SYSTEM = (
    "You are an interactive coding and research assistant running in a terminal. Use the "
    "available tools to read files, search the codebase (grep/glob), run shell commands, search "
    "the web, and compute. Work step by step and prefer reading before writing; on a multi-step "
    "task, lay out a plan with update_plan and keep it current. For any non-trivial arithmetic, "
    "algebra, calculus, or matrix work, COMPUTE with symbolic_math/matrix_op and CHECK results with "
    "verify_math — never do math in your head; to PROVE a claim holds for all values use prove (a "
    "sound solver) or lean_check (a proof kernel). For a Lean 4 / mathlib proof, FIRST load the "
    "lean-prove skill (use_skill) for the syntax, tactic tree, and lemma-citation rules. When an "
    "answer is cheap to check (e.g. a root "
    "by substitution), verify it before trusting it. If a tool returns UNKNOWN or says a claim is "
    "OUTSIDE its fragment, do NOT retry the same call — reason that step yourself and move on. When "
    "you have the answer, call final_answer (or finish). Your reply to the user is ALWAYS plain "
    "prose/markdown — NEVER a JSON object or a dump of a tool call (no wrapping in {…}, no "
    "'response'/'result'/'output' fields). Keep replies brief for simple questions — but "
    "when you PROVE, verify, solve, or derive something, a bare verdict ('this has been proven') is NOT "
    "enough: you MUST explain WHY it is true. Give a clear, STEP-BY-STEP write-up a student could "
    "follow — the key algebraic steps or the decisive identity/lemma, in order, each with a word on why "
    "it holds, ending in the conclusion — presented as readable math, NOT as a recap of your tool "
    "calls. (E.g. for a²+b²≥2ab: state it, note a²+b²−2ab = (a−b)², observe a square is ≥ 0, conclude.)"
)


def _make_prompt_fn():
    def prompt_fn(call, preview) -> str:
        sys.stdout.write(f"\n  ⚠ approve {preview}\n  [A]llow  al[w]ays  [d]eny > ")
        sys.stdout.flush()
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "deny"
        if ans in ("w", "always"):
            return "always"
        if ans in ("d", "n", "no", "deny"):
            return "deny"
        return "allow"                       # enter / a / y default to allow-once
    return prompt_fn


def _capabilities(tools) -> str:
    """A short capability summary from the wired tools (nicer than dumping every tool name)."""
    n = {getattr(t, "name", "") for t in tools}
    caps = []
    if n & {"grep", "glob", "read_file", "edit_file", "run_shell", "apply_patch"}:
        caps.append("code")
    if "web_search" in n:
        caps.append("web")
    if n & {"symbolic_math", "prove", "verify_math", "matrix_op"}:
        caps.append("exact math + proofs")
    if "lean_check" in n:
        caps.append("Lean")
    if n & {"python_exec", "load_dataset", "fit_model", "load_table"}:
        caps.append("data science")
    if any(name.startswith("mcp__") for name in n):
        caps.append("MCP")
    return " · ".join(caps) or "—"


def _banner(cfg, tools, config, cwd, n_custom, mem_files, n_skills=0, preset=None, temperature=None) -> str:
    """A welcoming, informative splash: framed header + version/tagline, aligned session info, a
    capability summary, and next-step hints."""
    from genai_studio import __version__  # single-sourced: bumping __version__ updates the splash too.
    # (When a NEW tool CATEGORY is added, extend _capabilities() so the "capable" line reflects it.)

    color = sys.stdout.isatty()

    def c(s, code):
        return f"\033[{code}m{s}\033[0m" if color and s else s

    CY, DIM, BOLD, TITLE = "36", "2", "1", "1;36"
    W, inner = 58, 56
    ver = f"v{__version__}"
    tag = "A tool-using agent over the Purdue GenAI Studio gateway"
    bar = c("│", CY)
    header = [
        c("╭" + "─" * W + "╮", CY),
        f"{bar} " + c("GenAI Studio · Agent".ljust(inner - len(ver)), TITLE) + c(ver, DIM) + f" {bar}",
        f"{bar} " + c(tag.ljust(inner), DIM) + f" {bar}",
        c("╰" + "─" * W + "╯", CY),
    ]

    def row(label, value):
        return "  " + c(label.ljust(10), DIM) + value

    model = str(cfg.model) + (f"  [{preset}]" if preset else "") + (c("  · greedy", DIM) if temperature == 0.0 else "")
    ctx = []
    if mem_files:
        ctx.append("memory: " + ", ".join(p.name for p in mem_files))
    if n_skills:
        ctx.append(f"{n_skills} skills (/skills)")
    ctx.append(f"{len(tools)} tools (/tools)")
    hint = ("  " + c("Type a task, or ", DIM) + c("/help", BOLD) + c(" for commands", DIM)
            + (c(f"  ·  {n_custom} custom", DIM) if n_custom else "") + c("  ·  /quit to exit", DIM))
    info = [
        "",
        row("model", c(model, BOLD)),
        row("profile", str(cfg.profile)),
        row("approvals", f"{config.mode.value} / {config.sandbox.value}" if config else "—"),
        row("cwd", str(cwd)),
        row("context", "   ·   ".join(ctx)),
        row("capable", c(_capabilities(tools), CY)),
        "",
        hint,
    ]
    return "\n".join(header + info)


def run_repl(ai, args, *, tools=None, approval_guard=None, approval_config=None) -> int:
    # Benchmark-informed preset picks the model + sampling (greedy for reasoning models); an
    # explicit --model overrides the model but keeps the preset's sampling.
    model, temperature, sampling = resolve_preset(getattr(args, "preset", None),
                                                  getattr(args, "model", None))
    cfg = ReplConfig(model=model, profile=getattr(args, "profile", None) or "general",
                     max_steps=getattr(args, "max_steps", None) or 25,
                     stream=not getattr(args, "no_stream", False))
    client = GenAIStudioClient(ai, default_model=model)
    cwd = Path.cwd()

    if tools is None:
        mode = ApprovalMode(getattr(args, "approval", None) or "suggest")
        sandbox = SandboxPolicy(getattr(args, "sandbox", None) or "workspace-write")
        tools, approval_guard, approval_config = build_tools(
            cfg.profile, workspace_root=cwd, mode=mode, sandbox=sandbox, prompt_fn=_make_prompt_fn())

    base_system = BASE_SYSTEM if not getattr(args, "system", None) else args.system + "\n\n" + BASE_SYSTEM
    mem_text, mem_files = load_project_memory(cwd)

    # Optional MCP servers (external tools, GATED). --mcp <config> or an auto-loaded
    # ./.genai_studio/mcp.json; stdio spawning requires --allow-stdio. FAIL-OPEN: a bad/unreachable
    # server logs and contributes no tools. Tools are appended before wire_capabilities so they flow
    # through the same registry/dedup/defer path; the guard is prepended below.
    mcp_mgr = None
    mcp_source = getattr(args, "mcp", None)
    if mcp_source is not None or (cwd / ".genai_studio" / "mcp.json").is_file():
        try:
            from ..mcp import mcp_tools
            mcp_list, mcp_mgr = mcp_tools(mcp_source, allow_stdio=getattr(args, "allow_stdio", False))
            tools = [*tools, *mcp_list]
            if mcp_list:
                print(f"MCP: connected {len(mcp_list)} tool(s) — {', '.join(t.name for t in mcp_list[:6])}"
                      + (" …" if len(mcp_list) > 6 else ""))
            elif not getattr(args, "allow_stdio", False):
                print("MCP: servers configured but stdio spawning is off — pass --allow-stdio to connect.")
        except Exception as e:
            print(f"(MCP setup skipped: {e})")

    # Skills (P0 in-context tier): discover .genai_studio/skills, add the use_skill meta-tool,
    # and inject the always-on catalog alongside project memory via the single assemble_system point.
    # Skills + recall-memory via the SAME shared wiring path as the headless assemble_agent.
    tools, cap_blocks, tool_search, n_skills = wire_capabilities(
        tools, cwd=cwd, client=client, model=model, memory_dir=cfg.memory_dir,
        skills=True, memory=True, defer=False, shared_guards=[approval_guard], studio=ai)

    # Dynamic parallel fan-out: workers get the READ-ONLY tools (safe concurrent explore/research;
    # the shared client's rate-limiter paces them). Excludes the meta-tools to avoid nesting.
    from ..approval import READ_ONLY_TOOLS
    from ..fanout import make_fanout_tool
    _meta = {"use_skill", "search_tools", "recall_memory", "update_plan"}
    worker_tools = [t for t in tools if t.name in READ_ONLY_TOOLS and t.name not in _meta]
    tools = [*tools, make_fanout_tool(client, model=model, worker_tools=worker_tools, max_agents=5)]

    mem_block = ("# Project memory (AGENTS.md)\n" + mem_text.strip()
                 if (mem_text or "").strip() else "")
    # priority order: base -> AGENTS.md -> recalled facts -> skills catalog
    system = assemble_system(base_system, mem_block, *cap_blocks)
    guards = [mcp_mgr.guard, approval_guard] if mcp_mgr else [approval_guard]   # MCP allowlist first
    agent = Agent(client=client, tools=tools, system=system, tracer=NullTracer(),
                  guards=guards, model=model, max_steps=cfg.max_steps,
                  tool_search=tool_search, temperature=temperature, sampling=sampling)
    if mcp_mgr is not None:
        agent._closeables.append(mcp_mgr)                # agent.close() tears the MCP servers down

    recorder = SessionRecorder(cfg.sessions_dir, model=model, cwd=cwd)
    registry = build_registry()
    ctx = ReplContext(agent=agent, tools=tools, approval_config=approval_config, recorder=recorder,
                      client=client, cfg=cfg, cwd=cwd, registry=registry, base_system=base_system)
    ctx.pretty = True                                # LaTeX→Unicode + markdown rendering (/pretty toggles)
    n_custom = load_custom_into(registry, cwd)
    _install_slash_completion(registry)              # Tab cycles through /commands (custom included)

    if getattr(args, "resume", None):
        resume_cmd(ctx, "" if args.resume == "__pick__" else args.resume)

    if getattr(args, "prompt", None):                # one-shot mode
        _turn(ctx, args.prompt, args.prompt)
        agent.close()                                # tear down MCP servers, if any
        recorder.close()
        return 0

    print(_banner(cfg, tools, approval_config, cwd, n_custom, mem_files, n_skills,
                  preset=getattr(args, "preset", None), temperature=temperature))
    try:
        while True:
            try:
                line = input("\n› ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("  (Ctrl-C — type /quit or Ctrl-D to exit)")
                continue
            if not line:
                continue
            if line in ("exit", "quit", "q", ":q"):
                break
            if line.startswith("/"):
                res = registry.dispatch(line, ctx)
                if res.is_exit:
                    break
                if res.prompt is None:
                    continue
                _turn(ctx, res.prompt, line)
                continue
            _turn(ctx, line, line)
    finally:
        agent.close()                                # tear down MCP servers, if any
        recorder.close()
    print("Goodbye.")
    return 0


def _run_turn(agent, text, history, recorder=None, raw=None, pretty=True) -> list:
    n0 = len(history)
    if recorder is not None and raw is not None:
        recorder.write_input(raw)
    renderer = StreamRenderer(pretty=pretty)
    renderer.start()
    tok = Cancel()
    gen = agent.stream(text, memory=history, cancel=tok)
    try:
        with turn_interrupt(tok):
            for ev in gen:
                renderer.handle(ev)
    except KeyboardInterrupt:
        try:
            gen.close()
        except Exception:
            pass
        renderer.abort()
        return history, None                 # discard the forcibly-aborted turn
    res = renderer.result
    if res is None:
        return history, None
    new_history = [m for m in res.messages if getattr(m, "role", None) != "system"]
    if recorder is not None:
        recorder.write_messages(new_history[n0:], res)
    return new_history, res


def _turn(ctx, text, raw=None) -> list:
    """One turn + per-session bookkeeping (undo marks, last prompt, cumulative token usage)."""
    ctx.turn_marks = getattr(ctx, "turn_marks", [])
    ctx.turn_marks.append(len(ctx.history))
    ctx.last_prompt = text
    new_history, res = _run_turn(ctx.agent, text, ctx.history, ctx.recorder, raw, ctx.pretty)
    if res is None:
        if ctx.turn_marks:
            ctx.turn_marks.pop()             # aborted/empty turn — drop its mark
    else:
        u = getattr(res, "usage", None)
        if u is not None:
            t = getattr(ctx, "total_usage", None) or {"prompt": 0, "completion": 0, "total": 0}
            t["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            t["completion"] += getattr(u, "completion_tokens", 0) or 0
            t["total"] += getattr(u, "total_tokens", 0) or 0
            ctx.total_usage = t
    ctx.history = new_history
    return new_history
