"""
GenAI Studio Agent Framework — a lightweight, teachable, reusable agent loop.

Four seams:
    @tool          turn a typed function into a model-callable Tool (auto JSON Schema)
    ModelClient    "chat, optionally with tools" — Purdue default, swappable
    Agent          the loop: prompt -> tool_calls -> execute -> repeat -> answer
    Tracer         structured events for every step (the teaching core)

Importing this package pulls only the light core (httpx/openai via the parent
client). The data-science tools live behind ``genai_studio.agents.datascience``
and pull the scientific stack only when imported.

    from genai_studio import GenAIStudio
    from genai_studio.agents import Agent, tool, GenAIStudioClient, ConsoleTracer

    @tool
    def add(a: int, b: int) -> str:
        "Add two integers."
        return str(a + b)

    studio = GenAIStudio()
    agent = Agent(client=GenAIStudioClient(studio, default_model="qwen2.5:72b"),
                  tools=[add], tracer=ConsoleTracer())
    print(agent.run("What is 2 + 3?").text)
"""

from __future__ import annotations

from .errors import (
    AgentError,
    BudgetExceeded,
    Cancelled,
    ToolError,
    TransientError,
)
from .tool import (
    Source,
    Tool,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    tool,
)
from .client import (
    BaseModelClient,
    GenAIStudioClient,
    Message,
    ModelClient,
    ModelResponse,
    ReActClient,
    RetryPolicy,
    ToolCall,
    Usage,
)
from .agent import (
    Agent,
    AgentResult,
    Budget,
    Cancel,
    Step,
)
from .guard import (
    ALLOW,
    BudgetGuard,
    Decision,
    Guard,
    ToolFilterGuard,
    deny,
    guard,
    modify,
)
from .orchestrate import (DELEGATION_GUIDE, EFFORT_PRESETS, ROUTED_DEFAULTS, ROUTING_GUIDE,
                          effort_policy, pipeline, routed_team, supervisor)
from .fanout import make_fanout_tool, parallel_agents
from .team import Team
from .verify import VERIFY_PROMPT, verifier
from .panel import (Critic, CriticVote, Verdict, critic_gate, critic_panel, panel_tool,
                    DEFAULT_LENSES, LENS_PROMPTS, REFUTE_PROMPT)
from .compose import assemble_agent, assemble_system, wire_capabilities
from .presets import DEFAULT_PRESET, PRESETS, Preset, resolve_preset
from .skills import Skill, SkillBundle, build_skill_tools, load_skills, render_skills_catalog
from .memory import (MemoryFact, MemoryStore, inject_memory, make_memory_tools,
                     memory_index_text, open_store, recall)
from .tool_search import ToolSearch, embedding_rank, keyword_rank, make_search_tool, render_catalog
from .embed import can_embed, make_embedder
from .events import (
    Final,
    StepFinished,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from .trace import (
    AgentEnd,
    AgentStart,
    ConsoleTracer,
    JsonlTracer,
    LLMCall,
    LLMResponse,
    LLMRetry,
    NullTracer,
    ScopedTracer,
    StepEnd,
    ToolCallEvent,
    ToolResultEvent,
    Tracer,
    TraceEvent,
)

__all__ = [
    # tool
    "tool", "Tool", "ToolSpec", "ToolResult", "Source", "ToolRegistry",
    # client
    "ModelClient", "BaseModelClient", "ModelResponse", "Message", "ToolCall",
    "Usage", "GenAIStudioClient", "ReActClient", "RetryPolicy",
    # agent
    "Agent", "AgentResult", "Step", "Budget", "Cancel",
    # guard seam (deterministic before/after-tool policy)
    "Guard", "Decision", "ALLOW", "deny", "modify", "guard",
    "BudgetGuard", "ToolFilterGuard",
    # multi-agent: orchestration topologies + grounded verifier
    "supervisor", "pipeline", "DELEGATION_GUIDE", "ROUTING_GUIDE", "Team",
    "routed_team", "ROUTED_DEFAULTS",
    "effort_policy", "EFFORT_PRESETS",
    "parallel_agents", "make_fanout_tool",
    "verifier", "VERIFY_PROMPT",
    # verification (adversarial critic panel + fail-closed gate)
    "critic_panel", "panel_tool", "critic_gate", "Verdict", "CriticVote", "Critic",
    "REFUTE_PROMPT", "LENS_PROMPTS", "DEFAULT_LENSES",
    # skills + composition
    "Skill", "SkillBundle", "load_skills", "build_skill_tools", "render_skills_catalog",
    "assemble_system", "assemble_agent", "wire_capabilities",
    # benchmark-informed model presets (speed↔quality knob)
    "Preset", "PRESETS", "resolve_preset", "DEFAULT_PRESET",
    # recall memory
    "MemoryStore", "MemoryFact", "open_store", "recall", "make_memory_tools",
    "memory_index_text", "inject_memory",
    # deferred (searchable) tools + embeddings
    "ToolSearch", "make_search_tool", "keyword_rank", "render_catalog",
    "embedding_rank", "make_embedder", "can_embed",
    # streaming events
    "TextDelta", "ToolCallStarted", "ToolCallFinished", "StepFinished", "Final",
    # trace
    "Tracer", "TraceEvent", "AgentStart", "LLMCall", "LLMResponse", "LLMRetry",
    "ToolCallEvent", "ToolResultEvent", "StepEnd", "AgentEnd",
    "ConsoleTracer", "NullTracer", "JsonlTracer", "ScopedTracer",
    # errors
    "AgentError", "TransientError", "ToolError", "BudgetExceeded", "Cancelled",
]
