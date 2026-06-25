# GenAI Studio SDK

A Python wrapper for Purdue's GenAI Studio API — chat completions, streaming, multi-turn conversations, embeddings, RAG pipelines, and a full CLI. Built for the Open WebUI + LiteLLM + Ollama stack at `genai.rcac.purdue.edu`.

## Setup

```bash
pip install -r requirements.txt
export GENAI_STUDIO_API_KEY="your-key"
```

Get your API key from **GenAI Studio → Settings → Account → API Keys**.

## Quick Start

```python
from genai_studio import GenAIStudio

ai = GenAIStudio()
ai.select_model("gemma3:12b")

# Chat
response = ai.chat("What is a p-value?")

# Stream
for chunk in ai.chat_stream("Explain regression"):
    print(chunk, end="", flush=True)

# RAG — ground responses in your documents
file = ai.upload_file("notes.pdf")
kb = ai.create_knowledge_base("My Notes")
ai.add_file_to_knowledge_base(kb.id, file.id)

import time; time.sleep(10)  # Wait for indexing

response = ai.chat("Summarize chapter 1", collections=[kb.id])
```

## Embeddings — model support

Probed live (2026-06-17): of the 35 models the gateway lists, **only 15 expose a
working embedding endpoint**. Embedding **dimension varies by model**, so keep one
embed model consistent across indexing and query.

⚠️ **`gemma3` (the gateway's default chat model) has no embedding endpoint.** For embeddings /
RAG indexing, use an embed-capable model, e.g. `llama3.2:latest` (3072-d),
`mistral:latest` (4096-d), or `qwen2.5:72b` (8192-d).

Full list (with dimensions) and the no-embed models: **[docs/embedding-models.md](docs/embedding-models.md)**.

## Agent Framework

A lightweight, teachable agent loop lives in `genai_studio.agents` — four seams
(`@tool`, `ModelClient`, `Agent`, `Tracer`) that build a tool-using agent in ~10
lines and scale to async/streaming/citations for production.

```python
from genai_studio import GenAIStudio
from genai_studio.agents import Agent, tool, GenAIStudioClient, ConsoleTracer

@tool
def add(a: int, b: int) -> str:
    "Add two integers."
    return str(a + b)

client = GenAIStudioClient(GenAIStudio(), default_model="qwen2.5:72b")
agent = Agent(client=client, tools=[add], tracer=ConsoleTracer())
print(agent.run("What is 21 + 21?").text)
```

Highlights: native tool-calling **and** a `ReActClient` fallback (same `Agent`);
`output_schema=` for validated pydantic results; `Budget`/`Cancel` guards; typed
transient-error retry with backoff; `run`/`arun`/`stream`/`astream`; a terminal
`final_answer`/`finish` tool; and `ConsoleTracer`/`JsonlTracer` for full
step-by-step observability.

Shipped tools:
- **General** (core — `genai_studio.agents.tools`): `final_answer`/`finish`
  (terminal), `calculator` (safe arithmetic), `web_search` + `wikipedia_search`
  (httpx, results carried as `ToolResult.sources` citations).
- **Data science** (`[datascience]` extra): `python_exec`, `load_dataset`,
  `describe_data` (EDA), `fit_model`, `hypothesis_test` (scipy t-test / χ² /
  correlation / ANOVA), `plot`, and the `data_analyst` agent.

See `examples/01_…07_*.py`, `benchmarks/` (paper replications), and the
[design doc](../genai-studio-agent-design.md).

```bash
pip install -e '.[datascience]'   # core + scientific stack for the DS agents
```

## CLI

```bash
genai-studio models                                  # List models (console script)
python -m genai_studio chat -m gemma3:12b "What is AI?"   # or: python -m genai_studio
python -m genai_studio chat -m gemma3:12b -i --stream     # Interactive
python -m genai_studio embed -m llama3.2:latest "a" "b" --similarity
python -m genai_studio rag upload notes.pdf              # RAG workflow
python -m genai_studio health                            # Connection check
```

## Tests

```bash
pytest tests/                                # Agent-framework unit tests (no network)
python test_full_suite.py --skip-rag         # Legacy live suite (needs an API key)
python test_full_suite.py --all-models       # Every available model
```

## Documentation

Full developer guide with architecture diagrams, code examples, method reference, and troubleshooting:

**[Open the Guide →](https://htmlpreview.github.io/?https://github.com/treese41528/genai-studio-sdk/blob/main/docs/guide.html)**

Or open `docs/guide.html` locally in any browser.

## What's In The Box

| File | Description |
|------|-------------|
| `genai_studio.py` | The SDK — single-file module, library + CLI |
| `test_full_suite.py` | 276-test suite across all models and file types |
| `docs/guide.html` | Interactive developer guide |
| `requirements.txt` | Dependencies: `openai`, `httpx`, `numpy` |

## License

MIT