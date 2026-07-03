"""Shared helpers for the examples: build a client and pick a model."""

from __future__ import annotations

import os

from genai_studio import GenAIStudio
from genai_studio.agents import GenAIStudioClient

DEFAULT_MODEL = os.getenv("GENAI_STUDIO_MODEL", "qwen2.5:72b")


def make_client(**kw) -> GenAIStudioClient:
    """A GenAIStudioClient on the default (native-tool-calling) model."""
    studio = GenAIStudio(validate_model=False)
    return GenAIStudioClient(studio, default_model=DEFAULT_MODEL, **kw)
