"""The approval engine — a pure decision function + an interactive guard.

``assess(call, config) -> ALLOW | _ASK | deny(...)`` is UI-free and unit-testable: it
auto-approves read-only tools and known-safe shell commands, applies the sandbox
capability ceiling, consults the session "always" cache, and otherwise returns the
``_ASK`` sentinel. ``approval_guard(config)`` wraps it for the agent loop: it turns
``_ASK`` into a human prompt via an injected ``prompt_fn`` (allow / always / deny),
caches "always" decisions for the session, and FAILS CLOSED (deny) when no prompt
handler is attached. ``deny().reason`` is fed back to the model so it can adapt.

The approval engine is the PRIMARY safety control in this environment (the sandbox is a
poor-man's one). Pair the guard with a ``BudgetGuard`` to bound deny-retry loops.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .guard import ALLOW, Decision, Guard, deny, guard
from .tools._workspace import WorkspaceConfig, resolve_in_workspace


class ApprovalMode(str, Enum):
    suggest = "suggest"        # ask before every state-changing call
    auto = "auto"             # auto-approve obviously-in-workspace; ask the borderline
    full = "full"             # never ask (within the sandbox ceiling)


class SandboxPolicy(str, Enum):
    read_only = "read-only"
    workspace_write = "workspace-write"
    danger_full = "danger-full"


_MODE_RANK = {ApprovalMode.full: 0, ApprovalMode.auto: 1, ApprovalMode.suggest: 2}
_SB_RANK = {SandboxPolicy.danger_full: 0, SandboxPolicy.workspace_write: 1, SandboxPolicy.read_only: 2}

# Tools safe to run without asking (no host state change). load_table/load_dataset fetch
# model-chosen URLs — classified read-only for the HOST fs; SSRF is bounded by the http
# tools' own block_private_ips default, not by an approval prompt.
READ_ONLY_TOOLS = frozenset({
    "read_file", "web_search", "wikipedia_search", "calculator", "arxiv_search",
    "openalex_search", "http_get", "fetch_json", "sql_query", "describe_data",
    "load_table", "load_dataset", "final_answer", "finish",
    "grep", "glob",                                 # read-only codebase search
    "update_plan",                                  # working-memory task list (no state change)
    "verify_math", "verify_factorization",          # exact-math grounding (pure computation)
    "symbolic_math", "matrix_op",
    "prove", "solve_constraints",                   # sound theorem proving (SMT, pure reasoning)
    "lean_check", "grade_proof",                    # kernel-checked proof verification (pure)
    "search_lemmas",                                # mathlib lemma retrieval (read-only)
    "verify_stat",                                  # re-compute a claimed statistic (read-only)
    "check_job",                                    # poll a background job's status/output
    # read-only meta-tools (load instructions / discover tools / recall facts — no state change)
    "use_skill", "search_tools", "recall_memory",
})
STATE_CHANGING_TOOLS = frozenset({
    "write_file", "edit_file", "apply_patch", "run_shell", "run_background", "python_exec", "r_exec",
})

_ASK = Decision("ask")        # internal sentinel — NEVER returned to the Agent (guard converts it)


@dataclass
class ApprovalConfig:
    """Mutable approval state shared by the guard and the ``/approvals`` command."""

    workspace: WorkspaceConfig
    mode: ApprovalMode = ApprovalMode.suggest
    sandbox: SandboxPolicy = SandboxPolicy.workspace_write
    network: bool = False
    prompt_fn: Callable[..., str] | None = None      # (call, preview) -> "allow"|"always"|"deny"
    _session_allow: set = field(default_factory=set)

    def set_policy(self, *, mode=None, sandbox=None, network=None) -> None:
        """Update policy live (used by ``/approvals``). TIGHTENING clears the session
        'always' cache so blanket grants don't outlive a stricter policy."""
        tightening = False
        if mode is not None and _MODE_RANK[mode] > _MODE_RANK[self.mode]:
            tightening = True
        if sandbox is not None and _SB_RANK[sandbox] > _SB_RANK[self.sandbox]:
            tightening = True
        if mode is not None:
            self.mode = mode
        if sandbox is not None:
            self.sandbox = sandbox
        if network is not None:
            self.network = network
        if tightening:
            self._session_allow.clear()


# ── command-safety allow-list (ported from codex is_safe_command.rs) ──────────
_SAFE_ARGV0 = frozenset(
    "cat cd echo false grep head ls nl pwd rev seq sort stat tail tr true uname uniq wc which whoami rg".split())
_SAFE_GIT = frozenset("status log diff show branch".split())
_SAFE_GIT_FLAGS = frozenset({"--stat", "--oneline", "--name-only", "--name-status", "--no-color"})
_FIND_BAD = frozenset({"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprintf", "-fprint", "-fls"})


def is_known_safe_command(command: str) -> bool:
    """True only for commands provably read-only and side-effect-free. Conservative:
    rejects redirects/subshells/substitution and any command not on the allow-list, so a
    wrong "safe" verdict (a silent write) is hard to reach. ``&&``/``||``/``;``/``|`` are
    decomposed and EVERY segment must be safe."""
    command = (command or "").strip()
    if not command:
        return False
    probe = command.replace("&&", "").replace("||", "")
    if any(t in probe for t in (">", "<", "`", "$", "(", ")", "{", "}", "\\", "&")):
        return False                          # redirects / subshells / substitution / background
    segments = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command)
    return all(_segment_safe(s) for s in segments if s.strip())


def _segment_safe(seg: str) -> bool:
    try:
        argv = shlex.split(seg)
    except ValueError:
        return False
    if not argv:
        return False
    cmd = argv[0]
    if cmd in _SAFE_ARGV0:
        if cmd in ("find",):                  # find handled below, not here
            return False
        return True
    if cmd == "find":
        return not (set(argv) & _FIND_BAD)
    if cmd == "sed":
        return "-n" in argv and not any(a.startswith("-i") for a in argv)   # only `sed -n`
    if cmd == "git":
        if len(argv) < 2 or argv[1] not in _SAFE_GIT:
            return False
        return all((not a.startswith("-")) or a in _SAFE_GIT_FLAGS for a in argv[2:])
    return False


# ── the decision function ─────────────────────────────────────────────────────
def _cache_key(call) -> str:
    """Session-cache key. For run_shell, key on argv0 (+ git subcommand) so "always
    allow git status" doesn't blanket-bless "git push"."""
    if call.name == "run_shell":
        try:
            argv = shlex.split((call.arguments or {}).get("command", "") or "")
        except ValueError:
            argv = []
        head = argv[0] if argv else ""
        if head == "git" and len(argv) > 1:
            return f"run_shell:git {argv[1]}"
        return f"run_shell:{head}"
    return call.name


def _write_inside_workspace(call, config) -> bool | None:
    """For write_file/edit_file: True if the path resolves inside a writable root,
    False if outside/carved, None if no path argument."""
    path = (call.arguments or {}).get("path")
    if path is None:
        return None
    try:
        resolve_in_workspace(config.workspace, path, for_write=True)
        return True
    except Exception:
        return False


def assess(call, config: ApprovalConfig) -> Decision:
    """Pure policy decision: ``ALLOW`` / ``_ASK`` / ``deny(reason)`` (no UI, no I/O)."""
    name = call.name
    if name in READ_ONLY_TOOLS:
        return ALLOW
    if name == "run_shell" and is_known_safe_command((call.arguments or {}).get("command", "")):
        return ALLOW                          # codex 'UnlessTrusted' parity
    if name == "write_memory":
        # a single reversible fact to the local memory store (no path, never the workspace):
        # auto-approve outside suggest; suggest still asks so the user sees what's saved.
        return _ASK if config.mode == ApprovalMode.suggest else ALLOW
    if name not in STATE_CHANGING_TOOLS:
        return _ASK                           # unknown tool -> prompt, never silently allow

    sb, mode = config.sandbox, config.mode
    is_write = name in ("write_file", "edit_file", "apply_patch")   # path-confined file writes

    # sandbox capability ceiling (what is POSSIBLE) ---------------------------------
    if sb == SandboxPolicy.read_only:
        if mode == ApprovalMode.suggest:
            return _ASK                       # let the user consciously escalate
        return deny(f"{name} blocked: sandbox is read-only (change with /approvals)")
    if sb == SandboxPolicy.workspace_write and is_write and _write_inside_workspace(call, config) is False:
        return deny(f"{name} blocked: path is outside the writable workspace")

    # session 'always' cache --------------------------------------------------------
    if _cache_key(call) in config._session_allow:
        return ALLOW

    # mode policy under the ceiling -------------------------------------------------
    if mode == ApprovalMode.suggest:
        return _ASK
    if mode == ApprovalMode.full:
        return ALLOW                          # ceiling already permitted it
    # mode == auto:
    if sb == SandboxPolicy.danger_full:
        return ALLOW
    if is_write:
        return ALLOW                          # workspace_write + in-root (else denied above)
    return _ASK                               # non-safe shell / python_exec / r_exec -> ask


def _preview(call, config: ApprovalConfig) -> str:
    args = call.arguments or {}
    if call.name == "run_shell":
        detail = args.get("command", "")
    elif call.name in ("write_file", "edit_file", "read_file"):
        detail = args.get("path", "")
    else:
        detail = ", ".join(f"{k}={v!r}"[:60] for k, v in args.items())
    return f"{call.name}: {detail}   [mode={config.mode.value} sandbox={config.sandbox.value}]"


def approval_guard(config: ApprovalConfig) -> Guard:
    """A ``before_tool`` guard that enforces ``assess`` and prompts the human on ``_ASK``."""

    def before(call) -> Decision | None:
        d = assess(call, config)
        if d is not _ASK:
            return d                          # ALLOW / deny pass straight to the loop
        if config.prompt_fn is None:
            return deny(f"approval required for {call.name} but no prompt handler is attached")
        choice = (config.prompt_fn(call, _preview(call, config)) or "deny").strip().lower()
        if choice in ("always", "w", "a-always"):
            config._session_allow.add(_cache_key(call))
            return ALLOW
        if choice in ("allow", "a", "y", "yes"):
            return ALLOW
        return deny(f"user denied {call.name}")

    return guard(before=before)
