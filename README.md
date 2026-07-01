# GenAI Studio SDK

A Python SDK for Purdue's GenAI Studio API **and** a batteries-included agent framework on top of
it. Two layers in one package:

1. **The gateway client** — chat completions, streaming, multi-turn, embeddings, RAG pipelines, and
   a full CLI over the Open WebUI + LiteLLM + Ollama stack at `genai.rcac.purdue.edu`.
2. **`genai_studio.agents`** — a teachable, production-capable agent framework: a tool-using loop,
   deterministic guards, multi-agent orchestration, grounded verification, a data-science toolkit,
   and a set of Claude-Code-style meta-capabilities (skills, recall memory, searchable tools,
   adversarial verification) — plus an interactive `genai-studio agent` REPL.

Everything targets an **OpenAI-compatible chat gateway**; any other such endpoint works by pointing
the client's base URL at it.

---

## Setup

```bash
pip install -e .                      # base SDK + agent framework core
pip install -e '.[datascience]'       # + pandas/NumPy/scikit-learn/SciPy/statsmodels/matplotlib
pip install -e '.[dev]'               # everything + test deps
export GENAI_STUDIO_API_KEY="your-key"
export GENAI_STUDIO_RPM=18            # pace the ~20 req/min gateway (silent-drop safe)
```

Get your API key from **GenAI Studio → Settings → Account → API Keys**.

**Optional extras:** `[datascience]` (scientific stack) · `[structured]` (pydantic — validated
outputs) · `[grounding]` (Data Commons) · `[knowledge]` (PyYAML — OKF bundles) · `[test]` · `[dev]`.
The heavy scientific stack imports **only when a tool actually runs**, so the core stays light.

---

## Part 1 — The gateway client

```python
from genai_studio import GenAIStudio

ai = GenAIStudio()
ai.select_model("gemma3:12b")

response = ai.chat("What is a p-value?")                       # chat
for chunk in ai.chat_stream("Explain regression"):            # stream
    print(chunk, end="", flush=True)

# RAG — ground responses in your documents
file = ai.upload_file("notes.pdf")
kb = ai.create_knowledge_base("My Notes")
ai.add_file_to_knowledge_base(kb.id, file.id)
import time; time.sleep(10)                                    # wait for indexing
response = ai.chat("Summarize chapter 1", collections=[kb.id])

vec = ai.embed("statistical significance")                    # embeddings (3072-d on llama3.2)
```

### Embeddings — model support
Probed live: of the 35 models the gateway lists, **only 15 expose a working embedding endpoint**,
and **dimension varies by model** — keep one embed model consistent across index and query.

⚠️ **`gemma3` (the default chat model) has no embedding endpoint.** For embeddings / RAG indexing use
an embed-capable model, e.g. `llama3.2:latest` (3072-d), `mistral:latest` (4096-d), or
`qwen2.5:72b` (8192-d). Full list + dimensions: **[docs/embedding-models.md](docs/embedding-models.md)**.

---

## Part 2 — The agent framework (`genai_studio.agents`)

A tool-using agent in ~10 lines, that scales to async/streaming/citations, multi-agent teams, and
policy guards for production.

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

### Core
Four seams — `@tool`, `ModelClient`, `Agent`, `Tracer` — compose the whole loop:
- **Native tool-calling *and* a `ReActClient` fallback** behind the *same* `Agent` (swap the client,
  not the loop); the gateway is auto-probed and degrades gracefully.
- **`output_schema=`** for validated pydantic results (`[structured]` extra).
- **`Budget`** (per-run tokens/steps/tool-calls) and **`Cancel`** (cooperative interruption).
- Typed **transient-error retry** with backoff; `run` / `arun` / `stream` / `astream`.
- A terminal **`final_answer`/`finish`** tool; `ConsoleTracer` / `JsonlTracer` / `ScopedTracer` for
  full step-by-step observability.

### Guards — deterministic policy around every tool call
```python
from genai_studio.agents import Agent, guard, BudgetGuard, ToolFilterGuard, deny
```
A `Guard` is the single chokepoint for cross-cutting policy: **`before_tool`** (allow / `deny(reason)`
/ `modify(args)`) and **`after_tool`** (rewrite the result). Shipped: **`BudgetGuard`** (tree-wide
tool-call cap), **`ToolFilterGuard`** (allow/block by name). Guards **never crash a run** — a raising
`before_tool` fails *closed*. An **approval** layer (`approval.py`) adds a mode × sandbox matrix
(`suggest`/`auto`/`full` × `read-only`/`workspace-write`/`danger-full`) with a known-safe-command
allowlist — the policy engine behind the REPL.

### Multi-agent — bounded, composable, not a swarm
```python
from genai_studio.agents import supervisor, pipeline, Team
```
- **`Agent.as_tool()`** — expose any agent as a tool (agents-as-tools; the one core primitive).
- **`supervisor(client, system, workers)`** — one coordinator delegating to worker agents.
- **`pipeline(stages)`** — a fixed sequence of agent stages with an optional validation gate.
- **`Team`** — correct-by-construction wiring: one shared client (⇒ one rate-limiter), scoped
  tracing, tree-wide guards.

> **Rate-limit invariant:** build every agent in a team from the **same** `ModelClient` so the whole
> tree shares one process-wide `RateLimiter` (the gateway silently drops bursts).

### Grounding & verification
- **RAG-as-a-tool** — `make_kb_search_tool(studio, collection)` runs server-side retrieval; results
  ride as `ToolResult.sources` citations.
- **Data Commons** — `make_datacommons_tool(...)` (`[grounding]` extra) for public statistics
  (RIG-as-a-tool, honest "no data").
- **`verifier(client, kb_search=, datacommons=)`** — a grounded fact-checker sub-agent (execute-then-read,
  so a parent can't finalize on an invented statistic).
- **`critic_panel(client, claim, …)`** — an adversarial panel of N independent critics (distinct
  model × lens) each asked to *refute*; the claim survives iff fewer than a majority refute
  (abstain ≠ refute). `panel_tool` exposes it as a tool; **`critic_gate`** is a fail-closed
  `before_tool` guard that blocks a state-changing tool unless the panel upholds it.

### Meta-capabilities (Claude-Code-style)
```python
from genai_studio.agents import assemble_agent, critic_panel, ToolSearch
```
- **Skills** — model-invoked, file-defined capabilities in `.genai_studio/skills/<name>/SKILL.md`,
  with **progressive disclosure**: an always-on catalog (one line/skill) in the system prompt, and
  the body loads only when the model calls `use_skill(name, task)`. Pure-instruction skills run
  in-context; capability-bearing skills (`allowed-tools`/`model`/`sampling`) run as a **bounded
  sub-agent** (tool-scoped, model-swapped).
- **Recall memory** — the agent **writes durable facts and recalls them by relevance** across runs
  (`write_memory`/`recall_memory`); a JSONL store with dedup, a capped always-on index, a keyword
  floor, and an optional embedding rerank (fail-open).
- **Deferred (searchable) tools** — carry hundreds of tools as a name+1-line catalog; full schemas
  load only when the model calls `search_tools(query)` (keyword or embedding-ranked). Opt-in via
  `Agent(tool_search=ToolSearch(...))`; default off ⇒ behavior unchanged.
- **`assemble_agent(client, profile, cwd, *, skills=, memory=, defer=, studio=)`** — one call wires
  profile tools + skills + recall-memory + optional searchable tools into a ready `Agent`.

### Shipped tools
| Group | Tools |
|---|---|
| **General** (core, `genai_studio.agents.tools`) | `final_answer`/`finish`, `calculator`, `web_search`, `wikipedia_search` |
| **Codebase** (workspace-confined) | `read_file`/`write_file`/`edit_file`, `apply_patch` (multi-hunk atomic), `grep`/`glob` (search + explore), `run_shell`, `run_background`/`check_job` |
| **Planning** | `update_plan` (working-memory task list) + `/plan` read-only plan mode |
| **Math** (`[math]` CAS · `[smt]` prover) | `verify_math` (self-verify a claim), `symbolic_math` (solve/simplify/calculus), `matrix_op` (exact linear algebra), `prove`/`solve_constraints` (sound z3 proving — proves ∀ or returns a counterexample) |
| **Web/academic** (keyless, httpx) | `arxiv_search`, `openalex_search`, `http_get`, `fetch_json` (SSRF-guarded) |
| **Grounding** | `make_kb_search_tool` (RAG), `make_datacommons_tool` (`[grounding]`) |
| **Knowledge** | OKF bundle loader (`agents/knowledge/okf.py`, `[knowledge]`) |
| **Orchestration** | `as_tool`, `supervisor`, `pipeline`, `Team`, `parallel_agents` + `fan_out` (dynamic parallel sub-agents) |
| **Data science** (`[datascience]`) | `python_exec`, `load_dataset`, `load_table`, `sql_query`, `r_exec`, `describe_data`, `fit_model`, `hypothesis_test`, `plot`, + the `data_analyst` agent — see [the DS package README](genai_studio/agents/datascience/README.md) |

### Evaluation
`genai_studio.agents.eval` — `evaluate(...)` reports **pass^k / pass@k / consistency** with L1
(`contains`/`used_tool`) and L2 (`llm_judge`) checks, resumable via `done_ids`. The `benchmarks/`
directory has faithful paper replications (DSBench, DataSciBench, ReAct on live Wikipedia) and an
agentic reliability + hallucination harness (`agentic_eval.py`) with a rotating peer-judge panel.

---

## Part 3 — The interactive agent REPL

A Claude-Code-style terminal agent — type a task, watch it use tools step-by-step, approve risky
actions, use slash commands, resume sessions.

```bash
genai-studio agent                                      # interactive (default 'balanced' preset)
genai-studio agent --preset fast "What is 100C in F?"   # one-shot, fast/cheap model
genai-studio agent -m qwen2.5:72b "Refactor utils.py"   # pick a model explicitly
```

- **Presets** (`--preset fast|balanced|careful`) — a benchmark-informed speed↔quality knob (like an
  effort choice) that picks the model **and** its sampling from the [routing study](benchmarks/README.md):
  `fast` = llama4:latest (quick/cheap), `balanced` = qwen2.5:72b (default), `careful` = deepseek-r1:32b
  at **greedy** (best calibration, lowest hallucination). `--model` overrides the model, keeps the sampling.
- **Profiles:** `--profile research|coding|general` (file/shell/web/data tools).
- **Approval × sandbox:** `--approval suggest|auto|full` × `--sandbox read-only|workspace-write|danger-full`
  — read-only tools auto-run; state-changing tools prompt allow / always / deny.
- **Skills + memory:** discovers `.genai_studio/skills/`, loads recall memory, injects both into the
  system prompt.
- **Slash commands:** `/help /tools /skills /memory /remember /forget /approvals /model /diff /init /compact /resume /clear /quit`, plus file-based custom commands (`.claude/commands/*.md`).
- **Streaming + interrupt** (Ctrl-C once = clean cancel, twice = force), **session persistence** and
  `--resume`.

---

## CLI

```bash
genai-studio models                                       # list models (console script)
python -m genai_studio chat -m gemma3:12b "What is AI?"    # chat (add -i --stream for interactive)
python -m genai_studio agent -m qwen2.5:72b "..."         # the agent REPL (Part 3)
python -m genai_studio embed -m llama3.2:latest "a" "b" --similarity
python -m genai_studio rag upload notes.pdf               # RAG workflow (upload/create-kb/link/query)
python -m genai_studio health                             # connection check
```

---

## Tests

```bash
pytest tests/                                # framework unit tests (no network) — 360+ tests
pytest tests/ -q -k "skills or panel"        # a subset
python test_full_suite.py --skip-rag         # legacy live gateway suite (needs an API key)
```

---

## Documentation

- **Interactive developer guide:** open `docs/guide.html`, or
  [view online →](https://htmlpreview.github.io/?https://github.com/treese41528/genai-studio-sdk/blob/main/docs/guide.html).
- **Endpoint catalog:** `docs/gateway-endpoints.md` · **Embeddings:** `docs/embedding-models.md`.
- **Design docs** (parent dir): `genai-studio-agent-design.md`, `genai-studio-multiagent-design.md`,
  `genai-studio-metaskills-design.md` (+ implementation companion), `genai-studio-routing-results.md`
  (a 10-model routing study), `genai-studio-work-report.md`.

## What's in the box

| Path | Description |
|---|---|
| `genai_studio/__init__.py` | The gateway SDK — library + CLI (chat/agent/embed/rag/health) |
| `genai_studio/agents/` | The agent framework: loop, tools, guards, orchestration, grounding, skills/memory/deferred-tools/verification, REPL |
| `genai_studio/agents/datascience/` | The `[datascience]` toolkit + `data_analyst` |
| `benchmarks/` | Paper replications + the agentic reliability/routing harness |
| `examples/` | `01_…13_*.py` — tools, tracing, structured output, data tools, multi-agent, guards, grounded verifier, orchestration, pass^k eval, Team, arXiv→SQL |
| `tests/` | 360+ offline unit tests |

## License

MIT
