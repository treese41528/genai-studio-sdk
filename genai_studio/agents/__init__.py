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
from .verify import VERIFY_PROMPT, verifier
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
    # multi-agent: grounded verifier
    "verifier", "VERIFY_PROMPT",
    # streaming events
    "TextDelta", "ToolCallStarted", "ToolCallFinished", "StepFinished", "Final",
    # trace
    "Tracer", "TraceEvent", "AgentStart", "LLMCall", "LLMResponse", "LLMRetry",
    "ToolCallEvent", "ToolResultEvent", "StepEnd", "AgentEnd",
    "ConsoleTracer", "NullTracer", "JsonlTracer",
    # errors
    "AgentError", "TransientError", "ToolError", "BudgetExceeded", "Cancelled",
]
