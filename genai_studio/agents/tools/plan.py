"""``update_plan`` — a working-memory task list the agent maintains (like a todo list).

On a multi-step task, an agent that keeps an explicit, updated plan follows through far better than
one improvising step to step. This tool holds the current plan in the session (a closure), echoes it
back each call (so it stays in context), and tracks progress. It is read-only working memory — it
changes no files and touches no gateway.

Pairs with **plan mode** (``/plan`` in the REPL, or ``--sandbox read-only``): explore read-only,
lay out the plan with ``update_plan``, get it approved, then execute.
"""

from __future__ import annotations

from genai_studio.agents import ToolResult, tool

_MARKERS = {"[ ]": "todo", "[~]": "in progress", "[x]": "done", "[-]": "skipped"}


def make_plan_tool() -> object:
    """Return the stateful ``update_plan`` tool (its plan persists for the session)."""
    state: dict = {"steps": []}

    @tool
    def update_plan(steps: list[str]) -> ToolResult:
        """Record or update your step-by-step plan for a multi-step task. Pass the FULL current
        ordered list of steps, each prefixed with a status marker: ``[ ]`` todo, ``[~]`` in
        progress, ``[x]`` done, ``[-]`` skipped. Call this at the start of a non-trivial task and
        again whenever a step's status changes, so your plan stays current.

        Args:
            steps: The full ordered list of plan steps, each starting with a status marker.
        """
        steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
        if not steps:
            return ToolResult(content="", error="pass a non-empty list of plan steps")
        state["steps"] = steps
        done = sum(1 for s in steps if s.startswith("[x]") or s.startswith("[-]"))
        active = next((s for s in steps if s.startswith("[~]")), None)
        head = f"plan updated — {done}/{len(steps)} complete" + (f"; now: {active[3:].strip()}" if active else "")
        return ToolResult(content=head + "\n" + "\n".join(steps), data={"steps": steps})

    return update_plan
