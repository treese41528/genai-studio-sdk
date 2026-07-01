# Changelog

All notable changes to `genai-studio-sdk`. This project follows [semantic versioning](https://semver.org).

## [1.3.0] — 2026-07-01

The **agent framework** (`genai_studio.agents`) and a set of Claude-Code-style **meta-capabilities**.
All additions are backward-compatible: the base gateway SDK (chat / stream / embed / RAG / CLI) is
unchanged, and every new agent feature is byte-identical to before when unused.

### Added — agent framework core
- `Agent` loop with native tool-calling **and** a `ReActClient` fallback behind the same agent
  (auto-probed, graceful degrade); `run` / `arun` / `stream` / `astream`.
- `@tool` (auto-schema from type hints + docstring), `ToolResult`/`ToolSpec`, `ToolRegistry`.
- `output_schema=` for validated pydantic results (`[structured]`); `Budget` + `Cancel`; typed
  transient-error retry with backoff; `final_answer`/`finish` terminal tools.
- Tracers: `ConsoleTracer`, `JsonlTracer`, `ScopedTracer`, `NullTracer`.

### Added — guards & approval
- `Guard` seam (`before_tool`/`after_tool`, fail-closed) with `guard()`, `BudgetGuard`,
  `ToolFilterGuard`.
- Approval engine (`approval.py`): mode × sandbox matrix (`suggest`/`auto`/`full` ×
  `read-only`/`workspace-write`/`danger-full`) + a known-safe-command allowlist.

### Added — multi-agent
- `Agent.as_tool()`, `supervisor(...)`, `pipeline(...)`, and `Team` (correct-by-construction wiring:
  one shared client/rate-limiter, scoped tracing, tree-wide guards).

### Added — grounding & verification
- RAG-as-a-tool (`make_kb_search_tool`), Data Commons (`make_datacommons_tool`, `[grounding]`),
  a grounded `verifier` sub-agent, and OKF knowledge bundles (`[knowledge]`).
- **Adversarial verification** (`panel.py`): `critic_panel` (N independent critics, distinct
  model × lens, abstain ≠ refute), `panel_tool`, and `critic_gate` (fail-closed `before_tool` guard).

### Added — meta-capabilities (Claude-Code-style)
- **Skills** — model-invoked, file-defined `SKILL.md` capabilities with progressive disclosure and a
  bounded, tool-scoped isolated tier (`skills.py`, `frontmatter.py`).
- **Recall memory** — `write_memory`/`recall_memory` over a JSONL store with dedup, a keyword floor,
  and an optional embedding rerank (`memory/`).
- **Deferred (searchable) tools** — `Agent(tool_search=ToolSearch(...))` + `search_tools`; carry
  hundreds of tools as a catalog, load schemas on demand (`tool_search.py`).
- Shared paced/fail-open embeddings helper (`embed.py`); `assemble_agent`/`wire_capabilities`/
  `assemble_system` composition (`compose.py`). Single `~/.genai_studio/` root for skills, memory,
  and sessions (project `./.genai_studio/` overrides).

### Added — data science, eval, CLI
- `[datascience]` toolkit: `python_exec` (+ a hardened `make_sandboxed_python_exec`),
  `load_dataset`, `load_table`, `sql_query` (read-only), `r_exec`, `describe_data`, `fit_model`,
  `hypothesis_test`, `plot`, and the `data_analyst` agent.
- Web/academic tools: `web_search`, `wikipedia_search`, `arxiv_search`, `openalex_search`,
  `http_get`/`fetch_json` (SSRF-guarded), `calculator`.
- `evaluate(...)` (pass^k / pass@k / consistency) + the `benchmarks/` harnesses.
- **`genai-studio agent`** — an interactive tool-using REPL (profiles, approval × sandbox, skills +
  memory, slash commands, streaming, session persistence + `--resume`).
- **Benchmark-informed presets** (`presets.py`, `--preset fast|balanced|careful`) — a speed↔quality
  knob (like an effort choice) that picks the model **and** its sampling (greedy for reasoning
  models) from the routing study; `--model` is now optional (defaults to the preset's model).

### Findings (see `benchmarks/README.md`)
- A 10-model **routing study** (SimpleQA / GSM8K / MuSiQue, grounded, rotating peer-judge). Best
  all-round: `qwen2.5:72b`; `deepseek-r1:32b` at greedy is a close #2.
- **Reasoning models on agentic tasks want greedy (temp=0), not the temp=0.6 recipe** — the default
  `_thinking_sampling` was corrected accordingly.

## [1.2.1] — earlier

Base gateway SDK: chat completions, streaming, multi-turn, embeddings, RAG pipelines, and the CLI.
