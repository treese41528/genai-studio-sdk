"""Benchmark-informed model presets — a simple speed↔quality knob (analogous to an "effort"
choice): pick a model **and** its sampling by GOAL, instead of memorizing which model wins.

Derived from the 10-model routing study (`benchmarks/README.md`, `agentic_eval.py`): grounded
condition, n=60, rotating peer-judge panel. Reasoning-model presets use **greedy (temp=0)** — the
routing-optimal sampling for agentic tool-use (the temp=0.6 reasoning recipe *underperforms* here;
it more than doubled deepseek-r1's accuracy when switched to greedy).

    fast     → llama4:latest      cheapest + quickest, best at math (reckless on obscure facts)
    balanced → qwen2.5:72b        best all-round accuracy — the default
    careful  → deepseek-r1:32b    reasoning model @ greedy: best calibration (F), lowest hallucination
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    model: str
    temperature: float | None = None       # None -> gateway default
    sampling: dict = field(default_factory=dict)
    blurb: str = ""


PRESETS = {
    "fast": Preset(
        "fast", "llama4:latest",
        blurb="cheapest + quickest (~335 tokens/answer), strongest at math; reckless on obscure facts"),
    "balanced": Preset(
        "balanced", "qwen2.5:72b",
        blurb="best all-round accuracy — the default"),
    "careful": Preset(
        "careful", "deepseek-r1:32b", temperature=0.0, sampling={"top_p": 0.95},
        blurb="reasoning model at greedy: best calibration (highest F-score), lowest hallucination, "
              "best at single-hop facts"),
}
DEFAULT_PRESET = "balanced"


def resolve_preset(preset: str | None = None, model: str | None = None):
    """Return ``(model, temperature, sampling)`` for a preset name, with an optional ``model``
    override. An explicit ``model`` replaces the preset's model but KEEPS the preset's sampling
    (so ``--preset careful --model X`` runs X deterministically). Unknown/None preset -> default."""
    p = PRESETS.get(preset or DEFAULT_PRESET, PRESETS[DEFAULT_PRESET])
    return (model or p.model), p.temperature, dict(p.sampling)
