"""Slash-command registry + built-ins for the agent REPL.

Metadata-driven (name/description/handler), so ``/help`` and completion are generic.
Handlers take ``(ctx, arg) -> CommandResult``; ``CommandResult.prompt is None`` means the
command was handled in-process (no model turn), otherwise ``prompt`` is sent to the agent
(this is how file-based custom commands expand to a prompt).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..approval import ApprovalMode
from . import session as _session
from .custom import expand_template, load_custom_commands


@dataclass
class CommandResult:
    prompt: str | None = None
    is_exit: bool = False


@dataclass
class ReplContext:
    agent: object
    tools: list
    approval_config: object
    recorder: object
    client: object
    cfg: object
    cwd: Path
    registry: "SlashRegistry"
    history: list = field(default_factory=list)
    base_system: str = ""          # system prompt WITHOUT project memory (so /init can rebuild it)
    pretty: bool = True            # LaTeX→Unicode + markdown rendering of answers (/pretty toggles)


@dataclass
class SlashCommand:
    name: str
    description: str
    handler: Callable
    source: str = "builtin"
    argument_hint: str | None = None


class SlashRegistry:
    def __init__(self):
        self._cmds: dict = {}

    def register(self, cmd: SlashCommand) -> None:
        self._cmds[cmd.name] = cmd

    def get(self, name: str):
        return self._cmds.get(name)

    def names(self) -> list:
        return sorted(self._cmds)

    def dispatch(self, line: str, ctx: ReplContext) -> CommandResult:
        parts = line[1:].split(None, 1)
        name = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        cmd = self._cmds.get(name)
        if cmd is None:
            print(f"unknown command /{name} (try /help)")
            return CommandResult()
        return cmd.handler(ctx, arg)


# ── built-in handlers ─────────────────────────────────────────────────────────
def _help(ctx, arg):
    print("commands:")
    for n in ctx.registry.names():
        c = ctx.registry.get(n)
        hint = f" {c.argument_hint}" if c.argument_hint else ""
        tag = "" if c.source == "builtin" else f"  ({c.source})"
        print(f"  /{n}{hint:<14} {c.description}{tag}")
    return CommandResult()


def _clear(ctx, arg):
    ctx.history.clear()
    if ctx.recorder:
        ctx.recorder.write_marker("clear")
    print("(history cleared)")
    return CommandResult()


def _model(ctx, arg):
    if arg:
        ctx.agent.model = arg
        print(f"model → {arg}")
    else:
        print(f"model: {ctx.agent.model}")
    return CommandResult()


def _tools(ctx, arg):
    print("tools: " + ", ".join(t.name for t in ctx.tools))
    return CommandResult()


def _skills(ctx, arg):
    """List model-invokable skills (invocation is model-driven via use_skill; this only lists)."""
    from ..skills import load_skills
    skills = load_skills(ctx.cwd)
    if not skills:
        print("(no skills — add .genai_studio/skills/<name>/SKILL.md)")
        return CommandResult()
    print(f"skills ({len(skills)}) — the model loads one via use_skill(name, task):")
    for s in skills.values():
        tier = "isolated" if s.isolated else "in-context"
        hint = f"  when: {s.when_to_use}" if s.when_to_use else ""
        print(f"  {s.name}  [{tier}, {s.source}] — {s.description}{hint}")
    return CommandResult()


def _open_memory(ctx):
    from ..memory import open_store
    return open_store(ctx.cwd, ctx.cfg.memory_dir)


def _memory(ctx, arg):
    """List the live durable facts (and the store path)."""
    store = _open_memory(ctx)
    facts = sorted(store.live(), key=lambda f: f.ts, reverse=True)
    if not facts:
        print(f"(no memory yet — the agent writes facts via write_memory, or use /remember)\n  store: {store.path}")
        return CommandResult()
    print(f"memory ({len(facts)} facts) — {store.path}")
    for f in facts:
        tags = f"  (tags: {', '.join(map(str, f.tags))})" if f.tags else ""
        print(f"  [{f.id}] {f.text}{tags}")
    return CommandResult()


def _remember(ctx, arg):
    """Save a durable fact: /remember <text>."""
    if not (arg or "").strip():
        print("usage: /remember <fact text>")
        return CommandResult()
    f = _open_memory(ctx).add(arg.strip(), source="user")
    print(f"remembered [{f.id}]: {f.text}  (restart to refresh the always-on index)")
    return CommandResult()


def _forget(ctx, arg):
    """Forget a fact by id: /forget <id>."""
    if not (arg or "").strip():
        print("usage: /forget <fact-id>  (see /memory for ids)")
        return CommandResult()
    _open_memory(ctx).forget(arg.strip())
    print(f"forgot {arg.strip()}")
    return CommandResult()


def _pretty(ctx, arg):
    """Toggle LaTeX→Unicode + markdown rendering of answers (on by default)."""
    a = arg.strip().lower()
    ctx.pretty = (a == "on") if a in ("on", "off") else not getattr(ctx, "pretty", True)
    print(f"pretty rendering {'ON — LaTeX→Unicode + markdown' if ctx.pretty else 'OFF — raw text'}")
    return CommandResult()


def _plan(ctx, arg):
    """Toggle PLAN MODE — read-only exploration (read/grep/glob + update_plan), no writes/exec."""
    from ..approval import SandboxPolicy
    cfg = ctx.approval_config
    if cfg is None:
        print("(approval config unavailable — plan mode needs the built-in approval engine)")
        return CommandResult()
    if cfg.sandbox != SandboxPolicy.read_only:
        ctx.plan_prev_sandbox = cfg.sandbox
        cfg.set_policy(sandbox=SandboxPolicy.read_only)
        print("PLAN MODE on — read-only. Explore (read/grep/glob), lay out steps with update_plan, "
              "then run /plan again to execute the plan.")
    else:
        prev = getattr(ctx, "plan_prev_sandbox", SandboxPolicy.workspace_write)
        cfg.set_policy(sandbox=prev)
        print(f"PLAN MODE off — sandbox restored to {prev.value}; writes/exec enabled.")
    return CommandResult()


def _approvals(ctx, arg):
    cfg = ctx.approval_config
    if cfg is None:
        print("(approval config unavailable)")
        return CommandResult()
    if arg:
        try:
            cfg.set_policy(mode=ApprovalMode(arg))
            print(f"approval mode → {cfg.mode.value}")
        except ValueError:
            print("usage: /approvals [suggest|auto|full]")
    else:
        print(f"approvals: mode={cfg.mode.value} sandbox={cfg.sandbox.value} network={cfg.network}")
    return CommandResult()


def _status(ctx, arg):
    cfg = ctx.approval_config
    print(f"model={ctx.agent.model}  cwd={ctx.cwd}  history={len(ctx.history)} msgs"
          + (f"  approvals={cfg.mode.value}/{cfg.sandbox.value}" if cfg else ""))
    if ctx.recorder:
        print(f"session: {ctx.recorder.path}")
    return CommandResult()


def _diff(ctx, arg):
    try:
        out = subprocess.run(["git", "-C", str(ctx.cwd), "diff"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception as e:
        out = f"(error running git diff: {e})"
    print(out if out.strip() else "(no changes)")
    return CommandResult()


def _resume(ctx, arg):
    sessions = [s for s in _session.list_sessions(ctx.cfg.sessions_dir)
                if ctx.recorder is None or s.path != ctx.recorder.path]
    if not sessions:
        print("(no prior sessions)")
        return CommandResult()
    if arg:
        match = next((s for s in sessions if s.id.startswith(arg)), None)
    else:
        for i, s in enumerate(sessions[:10]):
            print(f"  [{i}] {s.id}  {s.model}  {s.turns} turns  {s.preview!r}")
        try:
            match = sessions[int(input("resume which? > ").strip())]
        except (ValueError, EOFError, IndexError):
            print("(cancelled)")
            return CommandResult()
    if match is None:
        print("(session not found)")
        return CommandResult()
    ctx.history[:] = _session.load_history(match.path)
    print(f"(resumed {match.id}: {len(ctx.history)} messages)")
    return CommandResult()


def _compact(ctx, arg):
    if not ctx.history:
        print("(nothing to compact)")
        return CommandResult()
    new, summary = _session.compact_history(ctx.history, ctx.client, ctx.agent.model)
    ctx.history[:] = new
    if ctx.recorder:
        ctx.recorder.write_marker("compact", summary=summary)
    print(f"(compacted to {len(ctx.history)} messages)")
    return CommandResult()


def _init(ctx, arg):
    from .memory import build_system_prompt, init_agents_md, load_project_memory
    p = init_agents_md(ctx.cwd)
    mem, files = load_project_memory(ctx.cwd)
    ctx.agent.system = build_system_prompt(ctx.base_system, mem)     # reflect it immediately
    print(f"(wrote {p}; reloaded {len(files)} memory file(s))")
    return CommandResult()


def _cost(ctx, arg):
    """Show cumulative token usage this session (gateway billing is local/free)."""
    u = getattr(ctx, "total_usage", None)
    turns = len(getattr(ctx, "turn_marks", []))
    if not u or not u.get("total"):
        print("(no token usage recorded yet)")
        return CommandResult()
    print(f"session usage: {u['total']:,} tokens  ({u['prompt']:,} prompt + {u['completion']:,} "
          f"completion) over {turns} turn(s)")
    return CommandResult()


def _retry(ctx, arg):
    """Re-run the last prompt (drops its previous turn first)."""
    lp = getattr(ctx, "last_prompt", None)
    if not lp:
        print("(nothing to retry)")
        return CommandResult()
    marks = getattr(ctx, "turn_marks", [])
    if marks:
        ctx.history[:] = ctx.history[:marks.pop()]      # replace the previous attempt
    print(f"(retrying) {lp[:70]}")
    return CommandResult(prompt=lp)


def _undo(ctx, arg):
    """Drop the last turn from the conversation history."""
    marks = getattr(ctx, "turn_marks", [])
    if not marks:
        print("(nothing to undo)")
        return CommandResult()
    mark = marks.pop()
    removed = len(ctx.history) - mark
    ctx.history[:] = ctx.history[:mark]
    print(f"(undid last turn — removed {removed} message(s); {len(ctx.history)} left)")
    return CommandResult()


def _preset(ctx, arg):
    """Switch the speed/quality preset (model + sampling) live: /preset fast|balanced|careful."""
    from ..presets import PRESETS, resolve_preset
    a = (arg or "").strip().lower()
    if a not in PRESETS:
        print(f"usage: /preset [{'|'.join(PRESETS)}]   (current model: {ctx.agent.model})")
        return CommandResult()
    model, temp, sampling = resolve_preset(a, None)
    ctx.agent.model, ctx.agent.temperature, ctx.agent.sampling = model, temp, sampling
    print(f"preset → {a}: model={model}" + ("  · greedy" if temp == 0.0 else ""))
    return CommandResult()


def _export(ctx, arg):
    """Save the transcript to a markdown file: /export [path]."""
    path = Path(arg.strip()) if (arg or "").strip() else ctx.cwd / "genai-session.md"
    lines = ["# genai-studio session\n"]
    for m in ctx.history:
        role = getattr(m, "role", "?")
        if role == "system":
            continue
        content = (getattr(m, "content", "") or "").strip()
        calls = getattr(m, "tool_calls", None)
        if not content and calls:
            content = "(tool call: " + ", ".join(getattr(c, "name", "?") for c in calls) + ")"
        if content:
            lines.append(f"\n### {role}\n\n{content}\n")
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"(exported {len(ctx.history)} message(s) → {path})")
    except OSError as e:
        print(f"(export failed: {e})")
    return CommandResult()


def _reload(ctx, arg):
    """Reload file-based custom commands + project memory without restarting."""
    from .memory import build_system_prompt, load_project_memory
    n = load_custom_into(ctx.registry, ctx.cwd)
    mem, files = load_project_memory(ctx.cwd)
    ctx.agent.system = build_system_prompt(ctx.base_system, mem)
    print(f"(reloaded: +{n} custom command(s), {len(files)} memory file(s); skills reload per call)")
    return CommandResult()


def _mcp(ctx, arg):
    """List configured MCP servers and the MCP tools active this session."""
    active = [t.name for t in ctx.tools if getattr(t, "name", "").startswith("mcp__")]
    try:
        from ..mcp.config import load_mcp_config
        cfgs = load_mcp_config(None)
    except Exception:
        cfgs = []
    if not cfgs and not active:
        print("(no MCP servers — add .genai_studio/mcp.json, then start with mcp=/allow_stdio)")
        return CommandResult()
    print(f"MCP servers configured ({len(cfgs)}):")
    for c in cfgs:
        print(f"  {c.name}  [{c.transport}]  {c.command or c.url or '?'}")
    print(f"active MCP tools ({len(active)}): " + (", ".join(active) or "(none connected this session)"))
    return CommandResult()


def _verify(ctx, arg):
    """Run an adversarial critic panel on a claim (defaults to the last answer): /verify [claim]."""
    from ..panel import critic_panel
    claim = (arg or "").strip() or next(
        (getattr(m, "content", "") for m in reversed(ctx.history)
         if getattr(m, "role", None) == "assistant" and (getattr(m, "content", "") or "").strip()), "")
    if not claim:
        print("usage: /verify [claim]  (defaults to the last answer)")
        return CommandResult()
    print("(running an adversarial critic panel…)")
    try:
        v = critic_panel(ctx.client, claim, n=3)
    except Exception as e:
        print(f"(verify failed: {e})")
        return CommandResult()
    print(f"verdict: {'UPHELD ✓' if v.survived else 'REFUTED ✗'}  "
          f"({v.n_refute} refute / {v.n_uphold} uphold / {v.n_abstain} abstain)")
    for vote in v.votes:
        tag = "REFUTE" if vote.refuted else "ABSTAIN" if vote.abstained else "UPHOLD"
        print(f"  [{vote.lens}] {tag}: {vote.reason}")
    return CommandResult()


def _doctor(ctx, arg):
    """Environment + gateway health check."""
    import importlib.util as iu
    print("environment:")
    for label, mod in (("[math] sympy", "sympy"), ("[smt] z3", "z3"), ("[mcp] mcp", "mcp"),
                       ("[datascience] pandas", "pandas"), ("[structured] pydantic", "pydantic")):
        print(f"  {'✓' if iu.find_spec(mod) else '·'} {label}")
    try:
        from ..tools.lean import lean_available
        print(f"  {'✓' if lean_available() else '·'} Lean toolchain (lean_check)")
    except Exception:
        pass
    try:
        from ..client import Message
        msg = Message.user("ping") if hasattr(Message, "user") else Message(role="user", content="ping")
        ctx.client.complete([msg], model=ctx.agent.model, max_tokens=1)
        print("  ✓ gateway reachable")
    except Exception as e:
        print(f"  · gateway: {type(e).__name__}: {str(e)[:60]}")
    print(f"session: model={ctx.agent.model}  tools={len(ctx.tools)}  history={len(ctx.history)} msgs")
    return CommandResult()


def _profile(ctx, arg):
    """Switch the tool profile live (best-effort rebuild): /profile research|coding|general."""
    a = (arg or "").strip().lower()
    if a not in ("research", "coding", "general"):
        print("usage: /profile [research|coding|general]")
        return CommandResult()
    cfg = ctx.approval_config
    try:
        from ..compose import wire_capabilities
        from ..profiles import build_tools
        from ..tool import ToolRegistry
        base, approval_guard, newcfg = build_tools(
            a, workspace_root=ctx.cwd, mode=cfg.mode, sandbox=cfg.sandbox, prompt_fn=cfg.prompt_fn)
        tools, _blocks, tool_search, _ = wire_capabilities(
            base, cwd=ctx.cwd, client=ctx.client, model=ctx.agent.model,
            memory_dir=getattr(ctx.cfg, "memory_dir", None), shared_guards=[approval_guard])
        ctx.tools = tools
        ctx.agent.tools = tools
        ctx.agent._registry = ToolRegistry(tools)
        ctx.agent.tool_search = tool_search
        if tool_search is not None:
            ctx.agent._setup_deferred()
        ctx.agent.guards = [approval_guard]
        ctx.approval_config = newcfg
        print(f"profile → {a}  ({len(tools)} tools)")
    except Exception as e:
        print(f"(could not switch profile live: {e}; restart with --profile {a})")
    return CommandResult()


def _quit(ctx, arg):
    return CommandResult(is_exit=True)


_BUILTINS = [
    ("help", "show commands", _help, None),
    ("clear", "clear the conversation history", _clear, None),
    ("model", "show or switch the model", _model, "[name]"),
    ("tools", "list available tools", _tools, None),
    ("skills", "list model-invokable skills", _skills, None),
    ("memory", "list durable memory facts", _memory, None),
    ("remember", "save a durable fact", _remember, "<text>"),
    ("forget", "forget a fact by id", _forget, "<id>"),
    ("pretty", "toggle LaTeX/markdown rendering of answers", _pretty, "[on|off]"),
    ("plan", "toggle plan mode (read-only explore + propose)", _plan, None),
    ("approvals", "show or set approval mode", _approvals, "[suggest|auto|full]"),
    ("status", "show session status", _status, None),
    ("cost", "show session token usage", _cost, None),
    ("init", "write a starter AGENTS.md and load it", _init, None),
    ("diff", "show the working-tree git diff", _diff, None),
    ("resume", "resume a prior session", _resume, "[id]"),
    ("compact", "summarize history to free up context", _compact, None),
    ("retry", "re-run the last prompt", _retry, None),
    ("undo", "drop the last turn", _undo, None),
    ("preset", "switch speed/quality preset (model+sampling)", _preset, "[fast|balanced|careful]"),
    ("profile", "switch tool profile", _profile, "[research|coding|general]"),
    ("export", "save the transcript to markdown", _export, "[path]"),
    ("reload", "reload custom commands + memory", _reload, None),
    ("mcp", "list MCP servers and tools", _mcp, None),
    ("verify", "critic-panel the last answer", _verify, "[claim]"),
    ("doctor", "environment + gateway health check", _doctor, None),
    ("quit", "exit the session", _quit, None),
]


def build_registry() -> SlashRegistry:
    r = SlashRegistry()
    for name, desc, handler, hint in _BUILTINS:
        r.register(SlashCommand(name, desc, handler, "builtin", hint))
    return r


def _make_custom_handler(cc):
    def handler(ctx, arg):
        prompt = expand_template(cc.body, arg, ctx.cwd, allow_shell=ctx.cfg.allow_shell_expansion)
        return CommandResult(prompt=prompt)
    return handler


def load_custom_into(registry: SlashRegistry, cwd) -> int:
    """Load file-based custom commands; built-ins are never shadowed."""
    custom = load_custom_commands(cwd)
    n = 0
    for name, cc in custom.items():
        if registry.get(name) is not None:           # don't shadow a built-in
            continue
        registry.register(SlashCommand(name, cc.description or "(custom command)",
                                       _make_custom_handler(cc), cc.source, cc.argument_hint))
        n += 1
    return n
