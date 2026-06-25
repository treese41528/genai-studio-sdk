"""Guards: deterministic before/after-tool policy (the hook seam).

A guard gates EVERY tool call — the deterministic complement to prompting:
- ToolFilterGuard      — a capability allow/deny policy (block a dangerous tool).
- a custom after-guard — redact PII from a tool's result before the model sees it.
- BudgetGuard          — a tree-wide tool-call cap (share ONE instance across a team).

Guards never crash the run: a denied call feeds an error back; a guard that raises
or misbehaves fails closed (before) or is ignored (after).

Run: python examples/10_guards.py
"""

from __future__ import annotations

import re

from genai_studio.agents import (
    Agent, BudgetGuard, ConsoleTracer, ToolFilterGuard, ToolResult, guard, tool,
)
from _common import make_client

client = make_client()


@tool
def lookup_contact(name: str) -> str:
    """Look up a person's contact details.

    Args:
        name: the person's name.
    """
    return f"{name}: {name.lower()}@example.com, +1-555-0142"


@tool
def delete_all_records() -> str:
    "Permanently delete every record. (Dangerous — used here to show a policy block.)"
    return "all records deleted"


# 1) capability policy: the dangerous tool is never allowed to run.
policy = ToolFilterGuard(block={"delete_all_records"})

# 2) redact emails from any tool result before the model (or user) sees them.
def redact(call, result):
    if result.content and "@" in result.content:
        return ToolResult(content=re.sub(r"\S+@\S+", "[redacted-email]", result.content),
                          sources=result.sources, data=result.data, error=result.error)
    return None

# 3) a tree-wide tool-call budget (pass the SAME instance to every agent in a team).
budget = BudgetGuard(max_tool_calls=4)

agent = Agent(
    client=client, tracer=ConsoleTracer(), max_steps=6,
    tools=[lookup_contact, delete_all_records],
    guards=[policy, guard(after=redact), budget],
    system="Help the user. If asked to delete data, attempt delete_all_records.")

if __name__ == "__main__":
    print(agent.run("Find Alice's contact details, then delete all records.").text)
