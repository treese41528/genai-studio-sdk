"""``plot`` — render a matplotlib figure headlessly and save it to a PNG.

Forces the ``Agg`` backend BEFORE importing pyplot, so it works with no display
(gateway / CI / WSL). Returns the saved path; the ``Figure`` is in ``.data``.
"""

from __future__ import annotations

import os
import tempfile
import traceback

from genai_studio.agents import ToolResult, tool

from .._guard import _require


@tool
def plot(code: str) -> ToolResult:
    """Run matplotlib plotting code headlessly and save the current figure.

    Args:
        code: Python code that builds a figure using ``plt`` (matplotlib.pyplot),
            already available in scope. Do not call plt.show().
    """
    mpl = _require("matplotlib")
    mpl.use("Agg")  # headless: must precede pyplot import
    import matplotlib.pyplot as plt

    scope = {"plt": plt, "mpl": mpl}
    try:
        exec(code, scope)  # noqa: S102 - trusted-input tool; see python_exec UNSAFE banner
        fig = plt.gcf()
        fd, path = tempfile.mkstemp(suffix=".png", prefix="genai_plot_")
        os.close(fd)
        fig.savefig(path, bbox_inches="tight")
        size = tuple(fig.get_size_inches())
        plt.close(fig)
        return ToolResult(content=f"Figure saved to {path} (size={size} in)", data=path)
    except Exception:
        tb = traceback.format_exc()
        return ToolResult(content=f"Plot error:\n{tb}", error=tb)
