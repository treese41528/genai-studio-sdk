"""Deferred (searchable) tools: carry many tools, pay for a catalog not schemas.

With ``Agent(tool_search=ToolSearch(...))`` the deferred tools appear only as a
name + one-line catalog in the prompt; the model calls ``search_tools(query)`` to
unlock the ones it needs, and their full JSON schemas arrive on the NEXT step. This
keeps context flat when an agent carries dozens/hundreds of tools. Default (no
``tool_search``) is unchanged — all schemas eager.

Run: python examples/16_deferred_tools.py
"""

from __future__ import annotations

from genai_studio.agents import Agent, ConsoleTracer, ToolSearch, tool
from genai_studio.agents.tools import final_answer
from _common import make_client


@tool
def celsius_to_fahrenheit(c: float) -> str:
    "Convert a temperature from Celsius to Fahrenheit.\n\nArgs:\n    c: degrees Celsius."
    return f"{c * 9 / 5 + 32} F"


@tool
def word_count(text: str) -> str:
    "Count the words in some text.\n\nArgs:\n    text: the text to count."
    return str(len(text.split()))


@tool
def reverse_string(text: str) -> str:
    "Reverse a string.\n\nArgs:\n    text: the text to reverse."
    return text[::-1]


if __name__ == "__main__":
    # imagine dozens more tools — only the relevant one's schema needs to load.
    tools = [celsius_to_fahrenheit, word_count, reverse_string, final_answer]
    agent = Agent(
        client=make_client(), tools=tools, tracer=ConsoleTracer(),
        # defer everything except the finish tool; the model must search_tools first.
        tool_search=ToolSearch(deferred=("*",), eager=("final_answer",)))

    print(agent.run("What is 100 degrees Celsius in Fahrenheit?").text)
