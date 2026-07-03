"""Benchmark-informed model presets — the speed↔quality knob for the CLI."""

from __future__ import annotations

from genai_studio.agents import PRESETS, resolve_preset


def test_default_is_balanced_qwen():
    model, temp, samp = resolve_preset()
    assert model == "qwen2.5:72b" and temp is None and samp == {}


def test_fast_is_llama4():
    model, temp, samp = resolve_preset("fast")
    assert model == "llama4:latest" and temp is None


def test_careful_is_deepseek_greedy():
    model, temp, samp = resolve_preset("careful")
    assert model == "deepseek-r1:32b" and temp == 0.0 and samp == {"top_p": 0.95}


def test_model_override_keeps_preset_sampling():
    # --preset careful --model X  ->  X, but still greedy (the preset's sampling)
    model, temp, samp = resolve_preset("careful", model="qwen2.5:72b")
    assert model == "qwen2.5:72b" and temp == 0.0 and samp == {"top_p": 0.95}


def test_bare_model_uses_default_sampling():
    # --model X (no preset)  ->  X at the gateway default (backward-compatible)
    model, temp, samp = resolve_preset(None, "my-model")
    assert model == "my-model" and temp is None and samp == {}


def test_unknown_preset_falls_back_to_default():
    assert resolve_preset("nonsense")[0] == "qwen2.5:72b"


def test_returned_sampling_is_a_copy():
    # mutating a resolved sampling dict must not corrupt the shared PRESETS
    _, _, samp = resolve_preset("careful")
    samp["top_p"] = 0.1
    assert PRESETS["careful"].sampling == {"top_p": 0.95}
