# Project overview

**genai-studio-sdk** — a Python SDK for Purdue's GenAI Studio gateway (`genai.rcac.purdue.edu`,
an Open WebUI + LiteLLM + Ollama stack) **plus** a full agent framework, `genai_studio.agents`.

- `genai_studio/__init__.py` — the gateway client + CLI (chat / agent / embed / rag / health).
- `genai_studio/agents/` — the agent framework: the loop (`agent.py`), tools (`tool.py`,
  `tools/`), guards (`guard.py`, `approval.py`), orchestration (`orchestrate.py`, `team.py`),
  grounding/verification (`verify.py`, `panel.py`), and the Claude-Code-style meta-capabilities
  (`skills.py`, `memory/`, `tool_search.py`, `embed.py`, `compose.py`) + the REPL (`repl/`).
- `genai_studio/agents/datascience/` — the `[datascience]` toolkit + `data_analyst` (its own README).
- `benchmarks/` — paper replications + the agentic reliability / model-routing harness.
- `examples/` — `01_…17_*.py`, runnable feature demos.

## Build & test

```bash
pip install -e '.[dev]'            # everything incl. test deps
pytest tests/ -q                  # ~360 offline unit tests (no gateway/network)
export GENAI_STUDIO_API_KEY=...    # only for live runs
export GENAI_STUDIO_RPM=20        # ALWAYS pace the gateway — it silently drops bursts (no 429s)
```

## Conventions

- **Ride the four seams** (`@tool`, `ModelClient`, `Agent`, `Tracer`) — a new capability should be a
  tool, a guard, or a system-prompt block, not a new loop. Only `Agent.tool_search` earned a new
  dataclass field (deferred tools must change per-step loop behavior).
- **Backward-compatible by default:** new features must leave existing behavior byte-identical when
  unused (e.g. `tool_search=None`, `sampling={}`).
- Style: dataclasses; free functions over classes where possible; frozen result/config objects;
  dense honest docstrings; fail-CLOSED on invocation/action, fail-OPEN on discovery/retrieval.
- **One shared client per agent tree** ⇒ one process-wide `RateLimiter`. Sub-agents (`as_tool`,
  critics, skill children, embedders) MUST reuse the parent's client — a stranger client bypasses
  the limiter and the gateway drops bursts.
- Model tags: use `llama3.2:latest` / `qwq:latest` (bare `llama3.2` / `qwq:32b` 400 on the gateway).
  Only ~15/35 models embed (NOT gemma3/llama4/qwen3/gpt-oss); default embed model `llama3.2:latest`.

## Do / don't

- **Do** add a unit test with every change and keep the suite green before committing.
- **Do** run gateway sweeps as ONE sequential process at `GENAI_STUDIO_RPM=20` — never two
  concurrent gateway processes (they don't share the limiter).
- **Don't** reintroduce course/STAT/lecture references — the SDK is course-agnostic by directive.
- **Don't** put reasoning models at temp=0.6 for agentic (tool-use) tasks — the routing study found
  **greedy (temp=0)** best; `_thinking_sampling` defaults to greedy (see `benchmarks/README.md`).
