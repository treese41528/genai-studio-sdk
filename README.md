# GenAI Studio SDK

A Python SDK for Purdue's **GenAI Studio** gateway *and* a batteries-included, teachable **agent
framework** on top of it. Three things in one package:

1. **The gateway client** (`genai_studio.GenAIStudio`) — chat, streaming, multi-turn, embeddings, and
   RAG over the Open WebUI + LiteLLM + Ollama stack at `genai.rcac.purdue.edu`. It absorbs the
   deployment's quirks (model list at `/api` not `/v1`, embedding param-stripping, the non-standard RAG
   collection shape, silent-empty-completion retries) so you never see them.
2. **The agent framework** (`genai_studio.agents`) — a tool-using loop written *once*, deterministic
   guards, multi-agent orchestration, grounded/adversarial verification, exact math + kernel-checked
   proofs, an external-tool (MCP) client, and Claude-Code-style meta-capabilities (skills, recall
   memory, searchable tools).
3. **The interactive REPL** (`genai-studio agent`) — a terminal agent: type a task, watch it use tools
   step-by-step, approve risky actions, run 27 slash commands, resume sessions.

Everything targets an **OpenAI-compatible chat endpoint**; point the client at any other such gateway
via `base_url=`. (The backend is the local Ollama/LiteLLM gateway through the OpenAI SDK — no external
model provider is involved.)

> **New in 2.0:** an external **MCP client** (untrusted-by-construction), a generalized **check≪solve**
> verified-best-of primitive, an optional **Lean 4 + mathlib** proof track with lemma retrieval, and a
> much richer REPL (27 slash commands, a welcome splash, LaTeX rendering). See [CHANGELOG.md](CHANGELOG.md).

---

## Contents
- [Install](#install) · [Environment](#environment)
- **Part I — Using it:** [the gateway client](#part-i--the-gateway-client) · [the agent REPL](#the-agent-repl) · [the CLI](#the-cli)
- **Part II — How it works & why:** [seams](#the-four-seams) · [tools](#tools-tool) · [guards & approval](#guards--the-approval-matrix) · [meta-capabilities](#meta-capabilities) · [orchestration](#multi-agent-orchestration) · [verification](#grounding--verification) · [math & proofs](#math--proof-grounding) · [MCP](#mcp--external-tools-untrusted-by-construction)
- **Part III —** [Extending it for your project](#part-iii--extending-it-for-your-project)
- [Code map](#code-map) · [Suggested reading](#suggested-reading) · [Testing](#testing)

---

## Install

```bash
pip install -e .                      # base SDK + agent-framework core (stays light)
pip install -e '.[math,smt]'          # exact CAS (sympy) + sound proving (z3)
pip install -e '.[datascience]'       # pandas/NumPy/scikit-learn/SciPy/statsmodels/matplotlib
pip install -e '.[mcp]'               # connect external MCP tool servers
pip install -e '.[dev]'               # everything + test deps
```

**Extras** (all lazy — the heavy import happens only when a tool actually runs):

| extra | pulls in | unlocks |
|---|---|---|
| `[math]` | sympy | `verify_math`, `symbolic_math`, `verify_factorization`, `matrix_op` |
| `[smt]` | z3-solver | `prove`, `solve_constraints` (sound ∀-proofs / counterexamples) |
| `[datascience]` | pandas, numpy, sklearn, scipy, statsmodels, matplotlib | the DS toolkit + `data_analyst` |
| `[structured]` | pydantic | validated typed final answers (`output_schema=`) |
| `[grounding]` | datacommons-client | the Data Commons statistics tool |
| `[knowledge]` | pyyaml | OKF knowledge-bundle loader |
| `[mcp]` | `mcp` SDK | the external MCP client |
| `[dev]` | all of the above + pytest | tests + everything |

**Lean 4 + mathlib** (the optional proof track) is *not* a pip package — it's a Lean toolchain install;
see [Math & proof grounding](#math--proof-grounding).

### Environment
```bash
export GENAI_STUDIO_API_KEY="your-key"   # GenAI Studio → Settings → Account → API Keys
export GENAI_STUDIO_RPM=20               # pace the ~20 req/min gateway (it SILENTLY drops bursts, no 429s)
```
> **Nuance worth knowing:** `GENAI_STUDIO_RPM` is read by the *agent framework's* rate limiter
> (`agents/client.py:88`), not the plain gateway client — it governs agent/benchmark runs, which is
> where burst-drops bite. Always set it for live agent work.

---

# Part I — Using it

## Part I — The gateway client
`genai_studio/__init__.py` is the whole gateway layer (one module). `GenAIStudio()` reads the API key
from `$GENAI_STUDIO_API_KEY` and talks to `base_url + '/api'`.

```python
from genai_studio import GenAIStudio

ai = GenAIStudio()
ai.select_model("gemma3:12b")

print(ai.chat("What is a p-value?"))                       # one-shot
for chunk in ai.chat_stream("Explain regression"):        # streaming
    print(chunk, end="", flush=True)

# RAG — ground answers in your documents
f  = ai.upload_file("notes.pdf")
kb = ai.create_knowledge_base("My Notes")
ai.add_file_to_knowledge_base(kb.id, f.id)
print(ai.chat("Summarize chapter 1", collections=[kb.id]))  # collections= attaches the KB

vec = ai.embed("statistical significance")                  # embeddings
print(ai.similarity("cat", "kitten"))                       # cosine, numpy w/ pure-Python fallback
```

Key methods: `chat` / `chat_complete` / `chat_stream` / `chat_messages` / `chat_conversation`,
`embed` / `embed_complete`, the RAG suite (`upload_file`, `create_knowledge_base`,
`add_file_to_knowledge_base`, …), `models` / `select_model`, `health_check`.

**Embeddings caveat:** of the 35 models the gateway lists, **only ~15 expose a working embedding
endpoint**, and **dimension varies by model** — keep one embed model across index and query.
⚠️ `gemma3` (the default chat model) has **no** embedding endpoint; use e.g. `llama3.2:latest` (3072-d).
Full list + dims: **[docs/embedding-models.md](docs/embedding-models.md)**.

## The agent REPL
A Claude-Code-style terminal agent. Run it from your project directory (skills, `AGENTS.md`, and
`.genai_studio/mcp.json` load relative to the working dir):

```bash
genai-studio agent                                       # interactive (default 'balanced' preset)
genai-studio agent --preset careful                      # reasoning model at greedy — best for proofs
genai-studio agent -m qwen2.5:72b --allow-stdio          # explicit model + connect MCP servers
genai-studio agent --preset fast "What is 100C in F?"    # one-shot, quick/cheap model
```

**Flags** (registered on the `agent` subparser, `genai_studio/__init__.py:3060`; consumed in
`run_repl`, `repl/cli.py:166`):

| flag | effect |
|---|---|
| `--preset fast\|balanced\|careful` | benchmark-informed speed↔quality: `fast`=llama4, `balanced`=qwen2.5:72b, `careful`=deepseek-r1:32b **greedy** — picks model **and** sampling ([routing study](benchmarks/README.md)) |
| `--model / -m` | pick the model explicitly (keeps the preset's sampling) |
| `--profile research\|coding\|general` | which toolset (file/shell/web/data) |
| `--approval suggest\|auto\|full` × `--sandbox read-only\|workspace-write\|danger-full` | the approval matrix (below) |
| `--mcp <config.json>` + `--allow-stdio` | connect external MCP tool servers |
| `--resume [id]` · `--max-steps N` · `--no-stream` · `--system "…"` | resume a session · step cap · disable streaming · prepend to the system prompt |

**What it can do out of the box** (auto-wired if the deps/tools are present): exact math + proofs,
Lean/mathlib theorem proving with lemma search, factorization checking, code read/edit/grep/shell,
web/arXiv search, data science, dynamic sub-agent fan-out, skills, recall memory, and any connected
MCP tools. `/doctor` shows what's live.

### The 27 slash commands
Type `/help` in the REPL, or Tab to cycle through them. Highlights:

| | |
|---|---|
| **session** | `/model` `/preset` `/profile` `/approvals` `/status` `/cost` `/retry` `/undo` `/clear` `/compact` `/resume` `/export` `/reload` `/quit` |
| **inspect** | `/tools` `/skills` `/mcp` `/memory` `/doctor` `/diff` |
| **act** | `/remember` `/forget` `/plan` (read-only explore mode) `/verify` (adversarial critic panel on the last answer) `/pretty` `/init` (write a starter `AGENTS.md`) |

Plus **file-based custom commands**: drop `./.claude/commands/<name>.md` (project) or
`~/.claude/commands/<name>.md` (user) — the body becomes the prompt, with `$ARGUMENTS`, `@file`
inlining, and gated `` !`cmd` `` expansion. Ctrl-C once = clean cancel, twice = force; sessions persist
under `~/.genai_studio/sessions/` and `--resume` replays them.

### Connecting MCP servers (quick start)
Drop a `.genai_studio/mcp.json`, then launch with `--allow-stdio`:
```json
{"mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}}}
```
Every MCP tool is namespaced `mcp__filesystem__…` and **always prompts for approval** (see
[MCP](#mcp--external-tools-untrusted-by-construction)). `/mcp` lists the live tools.

## The CLI
```bash
genai-studio models                                       # list models
python -m genai_studio chat -m gemma3:12b "What is AI?"    # chat (-i --stream for interactive)
python -m genai_studio agent -m qwen2.5:72b "..."         # the agent REPL (above)
python -m genai_studio embed -m llama3.2:latest "a" "b" --similarity
python -m genai_studio rag upload notes.pdf               # RAG workflow (upload/create-kb/link/query/test)
python -m genai_studio health                             # connection check
```

---

# Part II — How it works & why

The design bet, stated in `CLAUDE.md`: **every capability is a tool, a guard, or a system-prompt
block — never a new loop.** That single rule is why the framework stays small while doing a lot.

## The four seams
The **entire** agent loop is written once, as a pure generator `Agent._drive` (`agents/agent.py`), that
does no I/O itself — it only *yields intents* ("call the model", "run this tool", "step done") which
thin drivers fulfill. So `run`, `arun`, `stream`, and `astream` all pump the *same* generator and
behave identically. Four seams compose the whole thing:

| seam | what it is | where |
|---|---|---|
| **`@tool`** | a typed function → a JSON-Schema tool the model can call | `agents/tool.py:159` |
| **`ModelClient`** | the backend protocol (native tool-calling *or* ReAct fallback) | `agents/client.py:279` |
| **`Agent`** | the loop that ties them together | `agents/agent.py:182` |
| **`Tracer`** | step-by-step observability (console / JSONL / scoped) | `agents/trace.py` |

```python
from genai_studio import GenAIStudio
from genai_studio.agents import Agent, tool, GenAIStudioClient, ConsoleTracer

@tool
def add(a: int, b: int) -> str:
    "Add two integers."
    return str(a + b)

client = GenAIStudioClient(GenAIStudio(), default_model="qwen2.5:72b")
print(Agent(client=client, tools=[add], tracer=ConsoleTracer()).run("What is 21 + 21?").text)
```

`Agent` also gives you `output_schema=` (validated pydantic results), `Budget` (per-run token/step/call
caps), `Cancel` (cooperative interruption), typed transient-error retry, and `agent.as_tool()` — the
one primitive behind all multi-agent work. Native backends and the `ReActClient` fallback sit behind
the *same* `Agent`; the gateway is auto-probed. Models on this gateway sometimes emit a tool call as
JSON *text* — `_tool_calls_from_text` (`client.py`) recovers those so the loop never mistakes a call
for a final answer.

## Tools (`@tool`)
A tool is a typed function; the decorator derives the wire schema from your type hints + a Google-style
`Args:` docstring. Tools return a **`ToolResult`** (`tool.py:53`) — only `.content` is shown to the
model; `.sources` carries citations, `.data` structured payload, `.error` a failure the model can read
and recover from. `ToolRegistry` (`tool.py:457`) ingests any tool; pass tools straight to
`Agent(tools=[…])`, no registration ceremony.

## Guards & the approval matrix
Guards are the **deterministic** complement to prompting — one chokepoint for cross-cutting policy
(`agents/guard.py`). A `Guard` has `before_tool(call) → ALLOW | deny(reason) | modify(args)` and
`after_tool(call, result) → ToolResult`. Guards **fail closed**: a `before_tool` that raises *blocks*
the call (`agent.py:421`). Shipped: `BudgetGuard` (tree-wide call cap), `ToolFilterGuard` (allow/block
by name), and the `guard(before=, after=)` wrapper for plain callables.

The **approval engine** (`agents/approval.py`) is the policy behind the REPL: a 2-axis matrix
**`ApprovalMode`(suggest/auto/full) × `SandboxPolicy`(read-only/workspace-write/danger-full)**. Its
pure, unit-testable `assess(call, config)` returns `ALLOW / _ASK / deny(...)`: read-only tools
(`READ_ONLY_TOOLS`, `approval.py:45`) and known-safe shell commands auto-run; host-mutating tools go
through the mode×sandbox truth table; a session cache remembers "always". **Workspace confinement**
(`tools/_workspace.py`) is a tool-level floor (realpath + `.git`/`.genai_studio` carved read-only +
atomic writes), so a guard misconfig can't widen file access.

## Meta-capabilities
Claude-Code-style progressive disclosure, wired through the **single** injection point
`assemble_system` (`agents/compose.py`, one byte budget for all always-on blocks):

- **Skills** (`agents/skills.py`) — file-defined capabilities in `.genai_studio/skills/<name>/SKILL.md`.
  An always-on catalog (one line/skill) sits in the system prompt; the body loads only when the model
  calls `use_skill(name, task)`. A pure-instruction skill runs **in-context**; a skill with
  `allowed-tools`/`model`/`sampling` runs as a **bounded, tool-scoped sub-agent** (isolated). *The
  shipped `lean-prove` skill is a worked example.*
- **Recall memory** (`agents/memory/`) — the agent `write_memory(fact)`s durable facts and
  `recall_memory(query)`s them by relevance across runs: an append-only JSONL store with dedup, a capped
  always-on index, a keyword floor, and an optional embedding rerank (fail-open).
- **Deferred (searchable) tools** (`agents/tool_search.py`) — carry hundreds of tools as a name + 1-line
  catalog; full schemas load only when the model calls `search_tools(query)`. Opt in with
  `Agent(tool_search=ToolSearch(...))`; default off ⇒ byte-identical behavior. *This is the one
  capability that earned a per-step loop change.*

One call wires it all: `assemble_agent(client, profile, cwd, *, skills=, memory=, defer=, mcp=)`.

## Multi-agent orchestration
One agent is the default; reach for these only when a task genuinely decomposes (`agents/orchestrate.py`,
`team.py`, `fanout.py`). Every factory is a thin composition over `Agent.as_tool` + the same loop — **no
new control flow**:

- **`supervisor(client, system, workers)`** — one coordinator delegating to worker agents.
- **`pipeline(stages)`** — a fixed sequence of agent stages with an optional validation gate.
- **`routed_team(...)`** — a supervisor whose specialists are pre-wired to benchmark-optimal models
  (`ROUTED_DEFAULTS`) with the measured `ROUTING_GUIDE` in its prompt.
- **`parallel_agents` / `make_fanout_tool`** — dynamic parallel sub-agents (`fan_out`), the model picks
  N independent subtasks.
- **`Team`** — correct-by-construction wiring: one shared client ⇒ one process-wide rate-limiter,
  scoped tracing, tree-wide guards.

> **Rate-limit invariant:** build every agent in a tree from the **same** `ModelClient` (the gateway
> silently drops bursts). `Team` enforces this for you.

## Grounding & verification
Non-frontier models hallucinate; these make them earn their claims.
- **`verifier(client, kb_search=, datacommons=)`** (`agents/verify.py`) — a grounded fact-checker
  sub-agent (execute-then-read, so a parent can't finalize on an invented statistic).
- **`critic_panel(client, claim, n=3)`** (`agents/panel.py`) — N independent critics (distinct model ×
  lens) each asked to **refute**; the claim survives iff fewer than a majority refute. `critic_gate`
  wraps it as a fail-closed guard around a state-changing tool; `/verify` exposes it in the REPL.
- **`verified_best_of(candidates, check, complete=)`** (`agents/verified.py`) — the **check≪solve**
  primitive: when *verifying* is far cheaper than *solving*, sample k candidates, keep only what a
  **sound** checker accepts, and vote among survivors — turning pass@k into accuracy. Ships with
  `inequality_check` (z3) and `factorization_check` (CAS).

## Math & proof grounding
The catalog gives the agent real engines instead of mental arithmetic (`agents/tools/`):
- **Exact CAS** (`[math]`, `symbolic.py`): `verify_math` (symbolic→SMT→numeric verdict on any claim),
  `symbolic_math` (solve/simplify/calculus/factor), `verify_factorization`, `matrix_op`.
- **Sound proving** (`[smt]`, `smt.py`): `prove` / `solve_constraints` — a real ∀-proof over the reals
  or a concrete counterexample (z3), with a human-readable proof sketch.
- **Kernel-checked Lean 4** (`lean.py`): `lean_check` runs the Lean kernel on a model-written proof;
  `grade_proof` grades a claim + candidate proof. Needs the Lean toolchain on `PATH` (or `~/.elan`).
- **Optional mathlib track** (`mathlib.py`): with a mathlib project present
  (`$GENAI_STUDIO_LEAN_PROJECT`, else a default), `lean_check` runs *inside* it so `import Mathlib`
  resolves (ring/norm_num/linarith + the ~180k-lemma library), and **`search_lemmas`** retrieves
  relevant lemmas by concept (hybrid keyword + embedding over the scanned corpus). `setup_mathlib(dir)`
  scaffolds a project + `lake exe cache get`. See the [math-grounding design docs](#suggested-reading).

## MCP — external tools, untrusted by construction
The MCP client (`agents/mcp/`, `[mcp]`) imports tools from external servers (filesystem, git, …
spawned as local stdio subprocesses) **without trusting them**. Rather than a trust flag, every MCP
tool is namespaced `mcp__<server>__<tool>` (`mapping.py`), which keeps it *out* of the approval
allowlists — so `assess()` falls through to `_ASK` **before** the session-allow cache
(`approval.py:171`). Net effect: **MCP tools always re-prompt**, and a cached "always" can't survive a
rug-pulled definition. Plus an `MCPGuard` server allowlist (runs before approval, fails closed), a
provenance banner as the first line of each tool's description, stdio spawning **opt-in**
(`allow_stdio`), and the gateway key scrubbed from server env. `mcp_tools(config, allow_stdio=True)`
returns `(tools, MCPManager)`; the REPL wires it via `--mcp`/`--allow-stdio`.

---

# Part III — Extending it for your project

Each of these is a small, local change — the table says **what to edit and what to read**.

| You want to… | Do this | Where | Learn from |
|---|---|---|---|
| **Add a tool** | write a typed function + Google-style docstring, decorate `@tool`, return a `ToolResult` | `tools/tool.py:159` | `examples/01_tools.py` |
| **Use a tool once** | pass it in `Agent(tools=[my_tool, …])` — no registration needed | `agent.py` | `examples/02_agent_loop.py` |
| **Add a tool to a profile** | append to the family list in `build_tools`; lazy-import heavy deps via `_require('pkg','extra')` | `agents/profiles.py:19` | `datascience/_guard.py:14` |
| **Write a custom guard** | subclass `Guard` (override `before_tool`/`after_tool`) or `guard(before=, after=)` | `agents/guard.py` | `agents/mcp/guard.py`, `examples/10_guards.py` |
| **Classify a tool for approval** | add its name to `READ_ONLY_TOOLS` or `STATE_CHANGING_TOOLS` | `agents/approval.py:45/60` | `tests/test_approval.py` |
| **Add a tool profile** | extend the profile branch of `build_tools` (keep tools deduped) | `agents/profiles.py:82` | `profiles.py` itself |
| **Write a skill** | create `.genai_studio/skills/<name>/SKILL.md` with frontmatter; add `allowed-tools`/`model` to make it an isolated sub-agent | `agents/skills.py` | `examples/14_skills.py`, `.genai_studio/skills/lean-prove/` |
| **Add recall memory** | model calls `write_memory`; or `MemoryStore(path).add(...)` in code | `agents/memory/` | `examples/15_memory.py` |
| **Carry many tools cheaply** | `Agent(tool_search=ToolSearch(deferred=('*',), eager=(...)))`; swap ranker via `rank=` | `agents/tool_search.py` | `examples/16_deferred_tools.py` |
| **Add a slash command** | write `_foo(ctx, arg) -> CommandResult`, append to `_BUILTINS` | `repl/commands.py:468` | `commands.py` |
| **Add a command with no code** | drop `./.claude/commands/<name>.md` | `repl/custom.py` | `custom.py` header |
| **Use a different gateway** | `GenAIStudio(base_url=…)` — `/api` is appended | `__init__.py:199` | module docstring 1-167 |
| **Write a custom `ModelClient`** | implement the `ModelClient` Protocol; or wrap a non-native backend in `ReActClient(inner)` | `agents/client.py:279` | `examples/02_agent_loop.py`, `tests/test_react.py` |
| **Add a CLI subcommand** | register a parser under `subparsers` in `main()`, add an `elif args.command == …` branch + a `cmd_*` handler | `__init__.py:2988` | dispatch block `:3211` |
| **Add a sound checker** | write `check(candidate)->bool` (+ optional `complete`) that is SOUND & CHEAP; feed `verified_best_of` | `agents/verified.py:47` | `verified.py` checkers |
| **Add an orchestration topology** | a free function composing `Agent.as_tool` + the loop — no new loop | `agents/orchestrate.py` | `examples/09_orchestration.py` |
| **Add a critic lens / role specialist** | extend `DEFAULT_LENSES`/`LENS_PROMPTS`, or add a `_<role>_worker` + register in `_ROUTED_BUILDERS` | `panel.py:25`, `orchestrate.py:248` | `examples/17_verification.py` |
| **Connect MCP servers** | `.genai_studio/mcp.json` + `--allow-stdio`, or `mcp_tools(cfg, allow_stdio=True)` | `agents/mcp/` | `tests/test_mcp.py`, `mcp/__init__.py:1-17` |

**Golden rules** (from `CLAUDE.md`): ride the four seams (a new capability is a tool/guard/prompt-block,
not a loop); new features are byte-identical when unused; **fail closed on action, fail open on
discovery**; one shared client per agent tree; bump the version **only** at `genai_studio/__init__.py`
`__version__` (pyproject and the REPL banner read it dynamically).

---

## Code map

| Path | What lives here |
|---|---|
| `genai_studio/__init__.py` | The gateway SDK — `GenAIStudio` client + the CLI (chat/agent/embed/rag/health) |
| `genai_studio/agents/agent.py` · `tool.py` · `client.py` · `events.py` · `trace.py` | The core: loop, `@tool`, `ModelClient`, streaming events, tracers |
| `genai_studio/agents/guard.py` · `approval.py` · `profiles.py` · `tools/_workspace.py` | Guards + the approval matrix + toolset profiles + workspace confinement |
| `genai_studio/agents/compose.py` · `skills.py` · `memory/` · `tool_search.py` · `embed.py` · `frontmatter.py` | Capability wiring + meta-capabilities (skills / memory / deferred tools) |
| `genai_studio/agents/orchestrate.py` · `team.py` · `fanout.py` · `panel.py` · `verify.py` · `verified.py` · `presets.py` · `eval.py` | Multi-agent topologies + grounded/adversarial/verified-best-of verification |
| `genai_studio/agents/mcp/` | The external MCP client (`mcp_tools`, config, stdio bridge, `MCPGuard`, namespacing) |
| `genai_studio/agents/tools/` | The shipped tool catalog (math/proof, coding, web/academic, planning, grounding) |
| `genai_studio/agents/datascience/` · `knowledge/` | The `[datascience]` toolkit + `data_analyst`; OKF knowledge bundles |
| `genai_studio/agents/repl/` | The interactive REPL (`run_repl`, 27 slash commands, rendering, sessions, project memory) |
| `benchmarks/` · `examples/` · `tests/` · `docs/` | Paper replications + reliability/routing/proof harnesses · 19 runnable demos · 490+ offline tests · guides |

## Suggested reading
- **Learn the framework by example:** `examples/01…19_*.py` (each a focused concept — see
  [examples/README.md](examples/README.md)): tools, the loop, tracing, structured output, data tools,
  multi-agent, guards, grounded verifier, orchestration, pass^k eval, Team, skills, memory, deferred
  tools, verification, Lean proving, routed team.
- **The interactive developer guide:** `docs/guide.html`. **Endpoints:** `docs/gateway-endpoints.md`.
  **Embeddings:** `docs/embedding-models.md`.
- **Design rationale (the deep "why"):** the design docs — agent framework, multi-agent, meta-skills
  (+ implementation companion), MCP, the math-grounding design + literature review, and the 10-model
  routing study — plus [CHANGELOG.md](CHANGELOG.md) for the feature history.
- **The data-science package** has its own [README](genai_studio/agents/datascience/README.md).

## Testing
```bash
pytest tests/                                # 490+ offline unit tests (no network)
pytest tests/ -q -k "approval or skills"     # a subset
python test_full_suite.py --skip-rag         # live gateway suite (needs an API key)
```
The `benchmarks/` directory holds runnable harnesses — paper replications (ReAct, DSBench,
DataSciBench), the agentic reliability + routing study, the math check≪solve evals
(`root_solve_eval.py`, `factor_verify_eval.py`), the end-to-end mathlib prover
(`mathlib_prove_eval.py`), and the free-chat quality eval (`free_chat_eval.py`).

## License
MIT
