"""Skills: model-invoked, file-defined capabilities with progressive disclosure.

A skill is a folder ``.genai_studio/skills/<name>/SKILL.md`` whose one-line
``description`` sits in the system prompt ALWAYS (cheap), while its BODY loads only
when the model calls ``use_skill(name, task)``. Pure-instruction skills run
in-context; capability-bearing ones (``allowed-tools``/``model``) run as a bounded
sub-agent. Here we write a sample skill into a temp workspace, wire it with
``build_skill_tools``, and let an agent invoke it.

Run: python examples/14_skills.py
"""

from __future__ import annotations

import pathlib
import tempfile

from genai_studio.agents import Agent, ConsoleTracer, assemble_system
from genai_studio.agents.skills import build_skill_tools
from genai_studio.agents.tools import final_answer
from _common import make_client

SKILL = """\
---
description: Summarize text as exactly three concise bullet points.
when_to_use: when asked to summarize or condense text.
---
Summarize the text in the task as EXACTLY three concise bullet points (one line
each), then call final_answer with the three bullets.
"""

if __name__ == "__main__":
    ws = pathlib.Path(tempfile.mkdtemp())
    skill_dir = ws / ".genai_studio" / "skills" / "summarize"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL)

    client = make_client()
    bundle = build_skill_tools(ws, client=client)          # discovers the skill above
    agent = Agent(client=client, tools=[bundle.tool, final_answer], tracer=ConsoleTracer(),
                  # the always-on catalog (one line/skill) goes into the system prompt:
                  system=assemble_system("You are a helpful assistant.", bundle.catalog))

    print(agent.run(
        "Use your summarize skill on this: The mitochondrion is the powerhouse of the "
        "cell; it generates ATP through oxidative phosphorylation across the inner "
        "membrane, and it carries its own small circular genome."
    ).text)
