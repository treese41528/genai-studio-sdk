"""Verification: an adversarial critic panel + a fail-closed gate.

``critic_panel`` spawns N INDEPENDENT critics (distinct lens: correctness / safety /
repro) each asked to REFUTE a claim; it SURVIVES iff fewer than a majority refute
(abstain != refute, so an unsure critic can't kill a good answer). ``critic_gate``
turns the panel into a ``before_tool`` guard that blocks a state-changing tool
unless the panel upholds it — verify-before-commit.

Run: python examples/17_verification.py
"""

from __future__ import annotations

from genai_studio.agents import critic_panel
from _common import make_client


def _show(claim, verdict):
    print(f"\n{claim!r}\n  -> survived={verdict.survived}  "
          f"({verdict.n_refute} refute / {verdict.n_uphold} uphold / {verdict.n_abstain} abstain)")
    for v in verdict.votes:
        tag = "REFUTE" if v.refuted else "ABSTAIN" if v.abstained else "UPHOLD"
        print(f"     [{v.lens}] {tag}: {v.reason}")


if __name__ == "__main__":
    client = make_client()
    for claim in ("The capital of France is Paris.",
                  "The capital of France is Berlin."):
        _show(claim, critic_panel(client, claim, n=3))

    # A gate blocks a state-changing tool unless the panel upholds the action:
    #   from genai_studio.agents import critic_gate
    #   agent = Agent(client=client, tools=[deploy, final_answer],
    #                 guards=[critic_gate(client, gate_tools={"deploy"})])
