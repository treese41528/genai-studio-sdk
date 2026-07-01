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

BASE_SYSTEM = (
    "You are an interactive coding and research assistant running in a terminal. Use the "
    "available tools to read files, search the codebase (grep/glob), run shell commands, search "
    "the web, and compute. Work step by step and prefer reading before writing; on a multi-step "
    "task, lay out a plan with update_plan and keep it current. For any non-trivial arithmetic, "
    "algebra, calculus, or matrix work, COMPUTE with symbolic_math/matrix_op and CHECK results with "
    "verify_math — never do math in your head; to PROVE a claim holds for all values, use prove (a "
    "sound solver). When you have the answer, call final_answer (or finish) with a concise result. "
    "Keep replies brief and to the point."
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


def _banner(cfg, tools, config, cwd, n_custom, mem_files, n_skills=0, preset=None, temperature=None) -> str:
    custom = f"  (+{n_custom} custom cmds)" if n_custom else ""
    skills = f"  ({n_skills} skills — /skills)" if n_skills else ""
    approvals = f"{config.mode.value}/{config.sandbox.value}" if config else "?"
    mem = ("  memory: " + ", ".join(p.name for p in mem_files)) if mem_files else ""
    model = f"model={cfg.model}" + (f" [{preset}]" if preset else "") + (" greedy" if temperature == 0.0 else "")
    return (f"genai-studio agent  ·  {model}  profile={cfg.profile}  approvals={approvals}\n"
            f"cwd={cwd}{mem}{skills}\ntools: {', '.join(t.name for t in tools)}\n/help for commands{custom}\n")


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

    mem_block = ("# Project memory (CLAUDE.md / AGENTS.md)\n" + mem_text.strip()
                 if (mem_text or "").strip() else "")
    # priority order: base -> CLAUDE.md -> recalled facts -> skills catalog
    system = assemble_system(base_system, mem_block, *cap_blocks)
    agent = Agent(client=client, tools=tools, system=system, tracer=NullTracer(),
                  guards=[approval_guard], model=model, max_steps=cfg.max_steps,
                  tool_search=tool_search, temperature=temperature, sampling=sampling)

    recorder = SessionRecorder(cfg.sessions_dir, model=model, cwd=cwd)
    registry = build_registry()
    ctx = ReplContext(agent=agent, tools=tools, approval_config=approval_config, recorder=recorder,
                      client=client, cfg=cfg, cwd=cwd, registry=registry, base_system=base_system)
    ctx.pretty = True                                # LaTeX→Unicode + markdown rendering (/pretty toggles)
    n_custom = load_custom_into(registry, cwd)

    if getattr(args, "resume", None):
        resume_cmd(ctx, "" if args.resume == "__pick__" else args.resume)

    if getattr(args, "prompt", None):                # one-shot mode
        ctx.history = _run_turn(agent, args.prompt, ctx.history, recorder, args.prompt, ctx.pretty)
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
                ctx.history = _run_turn(agent, res.prompt, ctx.history, recorder, line, ctx.pretty)
                continue
            ctx.history = _run_turn(agent, line, ctx.history, recorder, line, ctx.pretty)
    finally:
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
        return history                       # discard the forcibly-aborted turn
    res = renderer.result
    if res is None:
        return history
    new_history = [m for m in res.messages if getattr(m, "role", None) != "system"]
    if recorder is not None:
        recorder.write_messages(new_history[n0:], res)
    return new_history
