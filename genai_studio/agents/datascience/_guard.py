"""Import-guard helper for the optional data-science extra.

Heavy scientific deps are imported *at call time* through ``_require`` so that
``import genai_studio.agents`` (and even ``import genai_studio.agents.datascience``)
never pulls pandas/sklearn/etc. A missing dep yields a friendly install hint
instead of a raw ImportError.
"""

from __future__ import annotations

import importlib


def _require(pkg: str, extra: str = "datascience"):
    try:
        return importlib.import_module(pkg)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        top = pkg.split(".")[0]
        raise ImportError(
            f"'{top}' is required for the data-science tools. Install the extra:\n"
            f"    pip install 'genai-studio-sdk[{extra}]'"
        ) from e
