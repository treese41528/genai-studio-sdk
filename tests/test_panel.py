"""P1 verification — critic_panel / panel_tool / critic_gate (no network, via ScriptedClient)."""

from __future__ import annotations

import types

from genai_studio.agents.panel import (Verdict, _assign_critics, _parse_vote, critic_gate,
                                        critic_panel, panel_tool)

from conftest import ScriptedClient, says


def _refute(r="a concrete flaw"):
    return says(f'{{"verdict": "REFUTE", "reason": "{r}"}}')


def _uphold():
    return says('{"verdict": "UPHOLD", "reason": "holds"}')


def _abstain():
    return says('{"verdict": "ABSTAIN", "reason": "cannot tell"}')


def _call(name, **args):
    return types.SimpleNamespace(name=name, arguments=args)


# ── the panel ────────────────────────────────────────────────────────────────
def test_survives_minority_refute():
    v = critic_panel(ScriptedClient([_refute(), _uphold(), _uphold()]), "claim", n=3)
    assert v.survived and v.n_refute == 1 and v.threshold == 2


def test_refuted_by_majority():
    v = critic_panel(ScriptedClient([_refute(), _refute()]), "claim", n=3)  # short-circuits at 2
    assert not v.survived and v.n_refute == 2


def test_abstain_never_counts_as_refute():
    v = critic_panel(ScriptedClient([_abstain(), _abstain(), _abstain()]), "claim", n=3,
                     short_circuit=False)
    assert v.survived and v.n_refute == 0 and v.n_abstain == 3


def test_gate_require_uphold_denies_all_abstain():
    # the gate flips require_uphold=True, so all-abstain -> fail closed
    v = critic_panel(ScriptedClient([_abstain(), _abstain(), _abstain()]), "claim", n=3,
                     require_uphold=True, short_circuit=False)
    assert not v.survived


def test_short_circuit_any_stops_early():
    client = ScriptedClient([_refute()])                    # only ONE response scripted
    v = critic_panel(client, "claim", n=3, rule="any")      # thr=1 -> stop after first refute
    assert not v.survived and client.i == 1                 # exactly one critic ran


def test_dissent_carries_refuter_reasons():
    v = critic_panel(ScriptedClient([_refute("2+2 is not 5"), _refute("also wrong")]), "x", n=3)
    assert not v.survived and "2+2 is not 5" in v.dissent


# ── panel_tool ───────────────────────────────────────────────────────────────
def test_panel_tool_reports_verdict():
    t = panel_tool(ScriptedClient([_refute(), _refute()]), n=3)
    out = t.run({"claim": "the sky is green"})
    assert "REFUTED" in out.content and "refute" in out.content


# ── critic_gate (a before_tool Guard) ────────────────────────────────────────
def test_gate_denies_refuted_call():
    g = critic_gate(ScriptedClient([_refute(), _refute()]), gate_tools={"deploy"})
    d = g.before_tool(_call("deploy", env="prod"))
    assert d is not None and d.action == "deny"


def test_gate_allows_ungated_tool():
    g = critic_gate(ScriptedClient([]), gate_tools={"deploy"})
    assert g.before_tool(_call("read_file", path="x")) is None   # not gated -> no panel, allow


def test_gate_allows_upheld_call():
    g = critic_gate(ScriptedClient([_uphold(), _uphold()]), gate_tools={"deploy"})
    assert g.before_tool(_call("deploy", env="prod")) is None    # panel upholds -> allow


def test_gate_caches_denial():
    client = ScriptedClient([_refute(), _refute()])              # enough for ONE panel only
    g = critic_gate(client, gate_tools={"deploy"})
    call = _call("deploy", env="prod")
    assert g.before_tool(call).action == "deny"
    # identical retry is served from the denial cache (no second panel -> client not re-consumed)
    assert g.before_tool(call).action == "deny" and client.i == 2


# ── helpers ──────────────────────────────────────────────────────────────────
def test_parse_vote_robust():
    assert _parse_vote('{"verdict":"REFUTE","reason":"x"}')[0] == "REFUTE"
    assert _parse_vote("I UPHOLD this, no flaws")[0] == "UPHOLD"
    assert _parse_vote("garbled nonsense")[0] == "ABSTAIN"        # unparseable -> abstain (never refute)


def test_assign_critics_diversifies_lenses():
    crits = _assign_critics(models=None, lenses=("correctness", "safety", "repro"), n=3)
    assert {c.lens for c in crits} == {"correctness", "safety", "repro"}
