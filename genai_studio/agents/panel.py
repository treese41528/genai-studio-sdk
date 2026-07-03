"""Adversarial verification — a critic panel + a fail-closed verify-before-commit guard.

``critic_panel(client, claim, ...) -> Verdict`` spawns N INDEPENDENT critics (distinct model ×
lens), each a single structured ``client.complete`` asked to REFUTE the claim; the claim SURVIVES
iff fewer than the rule's threshold refute. **Abstentions never count as refutations** — a critic
must ABSTAIN, not refute, on mere uncertainty (so an unsure/hallucinated critic can't kill a good
answer). ``panel_tool`` exposes the panel as a narrow ``@tool``; ``critic_gate`` turns it into a
``before_tool`` Guard that blocks a state-changing tool unless the panel upholds it (fail-closed).

Free functions beside ``orchestrate.py`` (coordination is code, not chat). Sequential +
short-circuit by default — the gateway drops bursts. **Rate-limit invariant:** critics run through
the ONE shared ``client``; a critic Agent built from a different client warns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from .client import Message
from .guard import Guard, deny
from .tool import ToolResult, tool

DEFAULT_LENSES = ("correctness", "safety", "repro")

LENS_PROMPTS = {
    "correctness": "Focus on FACTUAL and LOGICAL correctness: is any claim false, unsupported, or a non-sequitur?",
    "safety": "Focus on SAFETY and side effects: could acting on this cause data loss, a security issue, or irreversible harm?",
    "repro": "Focus on REPRODUCIBILITY and completeness: are steps or assumptions missing, ambiguous, or unverifiable?",
}

REFUTE_PROMPT = """\
You are an adversarial reviewer. Try to REFUTE the statement below — find a concrete, specific flaw.
{lens}

Choose ONE verdict:
- REFUTE  — you found a concrete flaw (state it).
- UPHOLD  — you tried and found no flaw; the statement holds.
- ABSTAIN — you cannot tell from the information given. Do NOT refute out of mere uncertainty.

Reply with ONLY a JSON object: {{"verdict": "REFUTE|UPHOLD|ABSTAIN", "reason": "<one sentence>"}}"""


@dataclass(frozen=True)
class CriticVote:
    refuted: bool
    reason: str
    lens: str
    model: str | None = None
    abstained: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Verdict:
    survived: bool
    n_refute: int
    n_uphold: int
    n_abstain: int
    votes: tuple
    rule: str
    threshold: int

    @property
    def dissent(self) -> list:
        """The refuters' reasons — feed these back into a revise-then-re-verify loop."""
        return [v.reason for v in self.votes if v.refuted]


@dataclass
class Critic:
    lens: str = "correctness"
    model: str | None = None
    agent: Any = None                 # optional pre-built Agent (e.g. a grounded verifier)
    temperature: float | None = None


def _assign_critics(models, lenses, n) -> list:
    """n critics that maximize distinct (model, lens) pairs — round-robin lens (primary diversity)
    while cycling models. Warns if n forces duplicate pairs (correlated critics = false agreement)."""
    models = list(models) if models else [None]
    lenses = list(lenses) if lenses else list(DEFAULT_LENSES)
    pairs = [(models[i % len(models)], lenses[i % len(lenses)]) for i in range(n)]
    if len(set(pairs)) < len(pairs):
        warnings.warn("critic_panel: n exceeds distinct model×lens pairs; some critics are duplicates "
                      "(correlated critics can agree on a wrong verdict).")
    return [Critic(lens=lens, model=model) for model, lens in pairs]


def _parse_vote(text: str):
    """(verdict, reason) from a critic reply — robust to prose around the JSON. If no clear verdict
    is found, ABSTAIN (an unparseable critic must not refute)."""
    from .agent import _extract_json
    obj = _extract_json(text or "")
    if isinstance(obj, dict) and obj.get("verdict"):
        v, reason = str(obj["verdict"]).strip().upper(), str(obj.get("reason", "")).strip()
    else:
        up = (text or "").upper()
        v = "REFUTE" if "REFUTE" in up else "UPHOLD" if "UPHOLD" in up else "ABSTAIN"
        reason = (text or "").strip()[:200]
    if v not in ("REFUTE", "UPHOLD", "ABSTAIN"):
        v = "ABSTAIN"
    return v, reason


def _run_critic(client, critic: Critic, claim: str) -> CriticVote:
    prompt = REFUTE_PROMPT.format(lens=LENS_PROMPTS.get(critic.lens, ""))
    try:
        if critic.agent is not None:                          # a pre-built (e.g. grounded) critic Agent
            res = critic.agent.run(f"{prompt}\n\nStatement:\n{claim}")
            text, model = (res.text or ""), getattr(critic.agent, "model", None)
        else:
            resp = client.complete([Message.system(prompt), Message.user(f"Statement:\n{claim}")],
                                   model=critic.model, temperature=critic.temperature)
            text, model = (resp.text or ""), critic.model
    except Exception as e:                                    # a broken critic never crashes the panel
        return CriticVote(refuted=False, reason=f"critic error: {e}", lens=critic.lens,
                          model=critic.model, error=str(e))
    v, reason = _parse_vote(text)
    return CriticVote(refuted=(v == "REFUTE"), reason=reason, lens=critic.lens, model=model,
                      abstained=(v == "ABSTAIN"))


def _threshold(rule, n: int) -> int:
    if rule == "any":
        return 1
    if rule == "unanimous":
        return n
    if isinstance(rule, int):
        return rule
    return n // 2 + 1                                         # "majority" (default)


def _decide(votes, rule, n, require_uphold) -> Verdict:
    n_refute = sum(1 for v in votes if v.refuted)
    n_abstain = sum(1 for v in votes if v.abstained)
    n_uphold = sum(1 for v in votes if not v.refuted and not v.abstained)
    thr = _threshold(rule, n)
    survived = n_refute < thr
    if require_uphold and n_uphold == 0:                     # all-abstain / all-error -> fail closed
        survived = False
    return Verdict(survived=survived, n_refute=n_refute, n_uphold=n_uphold, n_abstain=n_abstain,
                   votes=tuple(votes), rule=str(rule), threshold=thr)


def _warn_stranger(client, critics) -> None:
    for c in critics:
        if c.agent is not None and getattr(c.agent, "client", client) is not client:
            warnings.warn("critic_panel: a critic Agent uses a different client than the panel — the "
                          "shared rate-limiter is bypassed (the gateway silently drops bursts).")


def critic_panel(client, claim: str, *, critics=None, models=None, lenses=DEFAULT_LENSES, n: int = 3,
                 rule="majority", require_uphold: bool = False, short_circuit: bool = True) -> Verdict:
    """Run N independent critics against ``claim``; it SURVIVES iff fewer than the rule's threshold
    refute (abstentions don't count). ``rule`` ∈ {'majority' (default), 'any', 'unanimous', int}.
    ``require_uphold`` (used by the gate) fails closed when no critic upholds. Sequential +
    short-circuit (stop once the outcome is decided) to bound cost."""
    crits = list(critics) if critics else _assign_critics(models, lenses, n)
    _warn_stranger(client, crits)
    thr = _threshold(rule, len(crits))
    votes = []
    for c in crits:
        votes.append(_run_critic(client, c, claim))
        n_ref = sum(1 for v in votes if v.refuted)
        remaining = len(crits) - len(votes)
        if short_circuit and (n_ref >= thr or n_ref + remaining < thr):
            break                                            # outcome already decided -> stop
    return _decide(votes, rule, len(crits), require_uphold)


def panel_tool(client, *, critics=None, models=None, lenses=DEFAULT_LENSES, n: int = 3,
               rule="majority", name: str = "review_claim", description: str | None = None):
    """The panel as a narrow ``@tool`` — the model passes a ``claim``; it gets back the verdict +
    each critic's vote. ``n``/``rule`` are fixed in the factory (not model-tunable, so the model
    can't weaken its own check)."""
    @tool(name=name, description=description or
          "Have an independent panel of critics review a claim, answer, or plan and report whether "
          "it survives adversarial refutation. Pass the statement to review as `claim`.")
    def review(claim: str) -> ToolResult:
        v = critic_panel(client, claim, critics=critics, models=models, lenses=lenses, n=n, rule=rule)
        head = "SURVIVED" if v.survived else "REFUTED"
        lines = "\n".join(
            f"- [{vote.lens}] {'REFUTE' if vote.refuted else 'ABSTAIN' if vote.abstained else 'UPHOLD'}: {vote.reason}"
            for vote in v.votes)
        return ToolResult(content=f"Panel verdict: {head} "
                          f"({v.n_refute} refute / {v.n_uphold} uphold / {v.n_abstain} abstain)\n{lines}")
    return review


def _default_claim(call) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in (call.arguments or {}).items())
    return f"Executing the tool call {call.name}({args}) right now is correct and safe."


def critic_gate(client, *, gate_tools, critics=None, models=None,
                lenses=("safety", "correctness", "repro"), n: int = 3, rule="majority",
                require_uphold: bool = True, claim_of=None) -> Guard:
    """A fail-closed ``before_tool`` Guard: a call to a gated tool is DENIED unless a critic panel
    upholds it. ``gate_tools`` is a set of tool names or a predicate ``(call) -> bool``. ``claim_of``
    renders the reviewed claim (default: the tool name + arguments). Denials are cached by
    (name, args) so an identical retry isn't re-paneled; pair with a ``BudgetGuard`` to bound loops."""
    is_gated = gate_tools if callable(gate_tools) else (lambda call: call.name in set(gate_tools))
    render = claim_of or _default_claim
    denied: set = set()

    class _CriticGate(Guard):
        def before_tool(self, call):
            if not is_gated(call):
                return None
            key = f"{call.name}:{sorted((call.arguments or {}).items(), key=lambda kv: kv[0])!r}"
            if key in denied:
                return deny("blocked by critic panel earlier; change the call to retry")
            v = critic_panel(client, render(call), critics=critics, models=models, lenses=lenses,
                             n=n, rule=rule, require_uphold=require_uphold)
            if v.survived:
                return None                                  # allow (never cached — state may change)
            denied.add(key)
            return deny("blocked by critic panel: " + "; ".join(v.dissent[:3] or ["no critic upheld it"]))

    return _CriticGate()
