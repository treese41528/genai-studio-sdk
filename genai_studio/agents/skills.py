"""Model-invoked, file-defined skills with progressive disclosure — **P0: in-context tier**.

A skill is a directory ``<root>/skills/<name>/SKILL.md`` whose frontmatter ``description`` (and
optional ``when_to_use``) is injected ALWAYS as a one-line catalog entry in the system prompt,
while the BODY loads only when the model calls the single ``use_skill(name, task)`` meta-tool.
This is the progressive-disclosure win: the always-on cost is one line per skill; a body enters
context only on demand.

P0 ships the **in-context tier**: ``use_skill`` returns the (template-expanded) body, so a
skill's instructions enter the caller's context. Capability-bearing frontmatter (``allowed-tools``
/ ``model`` / ``sampling`` / ``scripts`` / ``isolate``) is PARSED and surfaced (``Skill.isolated``),
but the bounded sub-agent executor that ENFORCES tool-scoping/model swap is P1 — until then such a
skill's instructions run in-context under the caller's normal guards/approval.

Single root (``genai_studio``): skills live under ``~/.genai_studio/skills/`` (user) and
``./.genai_studio/skills/`` (project); project wins on a name collision. Headless-first —
``load_skills`` / ``render_skills_catalog`` / ``build_skill_tools`` need no REPL.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .frontmatter import _aslist, expand_template, parse_frontmatter
from .tool import ToolResult, tool


@dataclass(frozen=True)
class Skill:
    name: str                                     # the <name> dir (catalog key + use_skill routing)
    description: str                              # always-on one-liner the model routes on
    body: str                                     # SKILL.md body (instructions; expanded on use)
    dir: Path                                     # the skill bundle dir (base for @path + scripts)
    source: str                                   # "user" | "project"
    when_to_use: str | None = None                # optional routing hint (shown in the catalog)
    allowed_tools: tuple | None = None            # P1: narrows the isolated child's tools
    scripts: tuple = ()                           # P1: bundled scripts -> child tools
    model: str | None = None                      # P1: isolated child model override
    sampling: dict = field(default_factory=dict)  # P1: isolated child sampling override
    isolate: bool = False                         # force the isolated tier
    max_steps: int = 8                            # P1: isolated child step cap

    @property
    def isolated(self) -> bool:
        """True when the skill needs the bounded sub-agent tier (capability-bearing or isolate)."""
        return bool(self.isolate or self.allowed_tools or self.scripts or self.model or self.sampling)


@dataclass
class SkillBundle:
    tool: object | None       # the use_skill Tool (None when no skills were discovered)
    catalog: str              # the always-on catalog block (or "")
    skills: dict              # {name: Skill}


def _skill_dirs(cwd):
    # user first, project second -> project wins on a name collision (mirrors load_custom_commands)
    return [(Path.home() / ".genai_studio" / "skills", "user"),
            (Path(cwd) / ".genai_studio" / "skills", "project")]


def load_skills(cwd) -> dict:
    """Discover ``<root>/skills/<name>/SKILL.md`` (user then project; project wins).

    Fail-OPEN: a malformed / unreadable / description-less SKILL.md is skipped with a warning,
    never raising — one bad skill must not break the others or the agent. Returns ``{name: Skill}``.
    """
    skills: dict = {}
    for d, source in _skill_dirs(Path(cwd)):
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*/SKILL.md")):
            name = md.parent.name                            # the <name> dir, NOT "SKILL"
            try:
                meta, body = parse_frontmatter(md.read_text("utf-8"))
            except OSError as e:                             # unreadable -> skip, keep going
                warnings.warn(f"skill {name!r}: unreadable ({e}); skipped")
                continue
            desc = (meta.get("description") or "").strip()
            if not desc:                                     # nothing to route on -> skip
                warnings.warn(f"skill {name!r}: no description; skipped")
                continue
            samp = meta.get("sampling") if isinstance(meta.get("sampling"), dict) else {}
            skills[name] = Skill(
                name=name, description=desc, body=body, dir=md.parent, source=source,
                when_to_use=(meta.get("when_to_use") or meta.get("when-to-use") or None),
                allowed_tools=(tuple(_aslist(meta.get("allowed-tools")) or ()) or None),
                scripts=tuple(_aslist(meta.get("scripts")) or ()),
                model=(meta.get("model") or None), sampling=samp,
                isolate=bool(meta.get("isolate", False)),
                max_steps=int(meta.get("max_steps", 8) or 8),
            )
    return skills


def render_skills_catalog(skills: dict) -> str:
    """The always-on cheap block: one line per skill (name + description + optional when_to_use).
    Bodies are NOT included — the model loads one by calling ``use_skill(name, task)``."""
    if not skills:
        return ""
    lines = ["# Skills (call use_skill(name, task) to load a skill's instructions)"]
    for s in skills.values():
        hint = f" — use when: {s.when_to_use}" if s.when_to_use else ""
        lines.append(f"- {s.name}: {s.description}{hint}")
    return "\n".join(lines)


def _run_isolated_skill(skill, task, *, client, base_tools, default_model, shared_guards) -> ToolResult:
    """Run a capability-bearing skill as a BOUNDED sub-agent (the isolated tier).

    Tools are NARROWED to the skill's ``allowed-tools`` — a skill can only narrow the parent's
    tools, never escalate (names the parent lacks are dropped). Model/sampling are swapped per the
    frontmatter; the parent's ``shared_guards`` (approval etc.) are forwarded and a
    ``ToolFilterGuard`` enforces the scope. The child gets NO ``use_skill`` (built from ``base_tools``
    = the pre-skill tool list), so recursion into skills is structurally impossible. It runs via the
    audited ``Agent.as_tool`` contract and returns only its narrow result."""
    from .agent import Agent
    from .guard import ToolFilterGuard
    from .trace import NullTracer

    parent = {t.name: t for t in base_tools}
    allowed = [n for n in (skill.allowed_tools or ()) if n in parent]    # narrow only — no escalation
    child_tools = [parent[n] for n in allowed]
    if not any(t.name in ("final_answer", "finish") for t in child_tools):
        from .tools.general import final_answer                          # give the child a clean finish
        child_tools.append(final_answer)
        allowed.append("final_answer")
    child = Agent(client=client, model=(skill.model or default_model), tools=child_tools,
                  system=expand_template(skill.body, "", skill.dir, allow_shell=False),
                  sampling=(skill.sampling or {}), max_steps=skill.max_steps, tracer=NullTracer(),
                  guards=[*shared_guards, ToolFilterGuard(allow=set(allowed))])
    delegate = child.as_tool(name=f"skill_{skill.name}", description=skill.description,
                             input_field="task")
    return delegate.run({"task": task or skill.description})


def build_skill_tools(cwd, *, base_tools=(), client=None, default_model=None,
                      shared_guards=(), **_kw) -> SkillBundle:
    """Headless entry point: returns ``SkillBundle(use_skill tool, catalog, skills)``.

    ``use_skill`` dispatches by tier: an IN-CONTEXT skill (pure instructions) returns its
    template-expanded body (``@path`` resolved against the skill dir; ``task`` as inert
    ``$ARGUMENTS``/``$1..``), so the instructions enter the caller's context. An ISOLATED skill
    (capability-bearing frontmatter) runs as a bounded sub-agent (see ``_run_isolated_skill``) —
    this needs ``client``; without one it falls back to in-context. ``shared_guards`` (e.g. the
    parent approval guard) are forwarded to the isolated child so its writes stay gated.
    """
    skills = load_skills(cwd)
    if not skills:
        return SkillBundle(tool=None, catalog="", skills={})
    catalog = render_skills_catalog(skills)
    names = ", ".join(sorted(skills))

    @tool(name="use_skill",
          description=("Load and apply a skill's instructions by name, then follow them. Available "
                       f"skills: {names}. Pass the user's request text as `task`. See the Skills "
                       "section of the system prompt for each skill's description and when to use it."))
    def use_skill(name: str, task: str = "") -> ToolResult:
        skill = skills.get(name)
        if skill is None:                                    # fail-closed: unknown -> error + valid names
            return ToolResult(content="", error=f"Unknown skill {name!r}. Available: {names}")
        if skill.isolated:
            if client is None:                               # no client -> best-effort in-context
                body = expand_template(skill.body, task, skill.dir, allow_shell=False)
                return ToolResult(content=body + "\n\n[note: isolated skill run in-context "
                                  "(no client to scope a sub-agent).]")
            return _run_isolated_skill(skill, task, client=client, base_tools=base_tools,
                                       default_model=default_model, shared_guards=shared_guards)
        return ToolResult(content=expand_template(skill.body, task, skill.dir, allow_shell=False))

    return SkillBundle(tool=use_skill, catalog=catalog, skills=skills)
