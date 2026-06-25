"""Multi-agent: an agent exposed AS a tool.

The minimal multi-agent primitive: ``Agent.as_tool()`` turns a specialist agent
into a single tool an orchestrator can call. The sub-agent runs in an ISOLATED
context and returns only its final answer (with sources / structured output
riding along in the ToolResult). No new framework concepts — just composition.

Run: python examples/06_multi_agent.py
"""

from __future__ import annotations

from genai_studio.agents import Agent, ConsoleTracer, NullTracer, tool
from _common import make_client

client = make_client()  # ONE shared client -> one rate-limiter across the tree

# A specialist sub-agent that only does arithmetic.
@tool
def add(a: int, b: int) -> str:
    "Add two integers.\n\nArgs:\n    a: x.\n    b: y."
    return str(a + b)


calculator = Agent(client=client, tools=[add], tracer=NullTracer(), max_steps=4,
                   name="calculate",
                   system="You are a calculator. Use the add tool and return only the number.")


# Expose the sub-agent AS a tool — one call replaces the hand-rolled @tool
# closure you'd otherwise write (calculator.run(q).text). Citations and stop
# reasons propagate automatically.
calculate = calculator.as_tool(
    description="Delegate ONE arithmetic question (as 'task') to the calculator "
                "sub-agent; it returns just the number.")


if __name__ == "__main__":
    orchestrator = Agent(
        client=client, tools=[calculate], tracer=ConsoleTracer(), max_steps=6,
        system="Break the task into arithmetic questions, delegate each to the "
               "calculate tool, then summarize.",
    )
    result = orchestrator.run(
        "I bought 3 books at $12 each and 2 pens at $4 each. What did I spend in total?"
    )
    print("\nFINAL:", result.text)
