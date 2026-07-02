# Changelog

All notable changes to `genai-studio-sdk`. This project follows [semantic versioning](https://semver.org).

## [1.8.0] — 2026-07-02

Optional Lean + **mathlib** proof track — real competition-math proving, gated on a mathlib install.

### Added
- **mathlib-backed `lean_check` / `grade_proof`** — `make_lean_check(project_dir=…)` runs the kernel
  INSIDE a Lean project with mathlib (`lake env lean`), so `import Mathlib` resolves and
  `ring` / `norm_num` / `linarith` / `positivity` + the ~180k-lemma library become available. Falls
  back to Lean-core tactics when no project.
- **`search_lemmas` tool** (`tools/mathlib.py`) — retrieval over the mathlib declaration corpus so the
  model can FIND the lemma it needs before proving. Scans the source once (~180k declarations → cached
  JSON, built LAZILY on first search), keyword-ranks the whole corpus, then optionally embedding-reranks
  only the top-N (reusing `embed.py`; fails open to keyword-only). Read-only.
- **`mathlib_project()` / `mathlib_tools()` / `setup_mathlib()`** — detect a project
  (`$GENAI_STUDIO_LEAN_PROJECT` or a default), bundle the mathlib tools, and scaffold + `lake exe cache
  get` (prebuilt oleans — a large download, not a from-scratch compile). The math profile auto-uses the
  mathlib tools when a project is present.
- **`scan_declarations` / `scan_file`** — a Lean declaration scanner (name + signature + module + doc),
  robust to `@[attr]` / `protected` / `private`. Offline-tested; the live path is exercised against a
  real mathlib checkout (Lean 4.31.0).

Note: mathlib is a Lean/lake install (not a pip package) — `setup_mathlib(dir)` or `lake new <p> math`.

## [1.7.0] — 2026-07-02

check≪solve, generalized — a reusable verified best-of-n + sound checkers for new problem classes.

### Added
- **`verified_best_of` primitive** (`genai_studio.agents.verified`) — the *check≪solve* pattern
  extracted from `benchmarks/root_solve_eval.py` into a reusable, model-agnostic function: filter k
  candidates by a SOUND checker, then vote among the survivors (with a graceful `fallback-vote` and an
  honest `verified` flag). Prefers a stronger `complete` check over plain `validity` when available.
  When verifying is cheaper than solving, this converts *pass@k* into accuracy.
- **`verify_factorization` tool** + `is_factorization` / `factorization_check` — verify a claimed
  factorization soundly (expands back to the original AND is genuinely a product, not the polynomial
  restated). Wired into the math profile; read-only.
- **`inequality_check`** — the z3-backed sound checker for polynomial (in)equalities, exposed for
  generate-and-filter over candidate bounds.
- **`grade_proof` tool** (`tools/lean.py`) — Lean-certificate grading: give a claim + a candidate proof
  (term or `by` block) separately, assemble the theorem, and kernel-check it. Verifying a proof is far
  cheaper than finding one, so propose several and keep the accepted one. (Needs the Lean toolchain.)
- **`benchmarks/factor_verify_eval.py`** — factorization benchmark (bare / maj@k / verified / pass@k)
  demonstrating the check≪solve payoff beyond root-solving; grading is exact (expand + `sympy.factor`
  fixed point), the checker is `factorization_check`.

## [1.6.0] — 2026-07-02

MCP client — connect to external MCP servers, gated (P1: stdio + tools).

### Added
- **MCP client** (`genai_studio/agents/mcp/`, `[mcp]` extra, official SDK, lazy-imported): connect to
  MCP servers over **stdio**, snapshot `tools/list`, and expose them to the Agent. `mcp_tools(config,
  allow_stdio=True)` → `(tools, MCPManager)`; `assemble_agent(mcp=…)`; or in the REPL via
  **`--mcp <config.json>`** (or an auto-loaded `./.genai_studio/mcp.json`) + **`--allow-stdio`**, with
  `/mcp` listing the live tools. Fail-open on discovery (a dead server → zero tools, never crashes).
- **Untrusted-by-construction security:** every tool is namespaced `mcp__<server>__<tool>`, which keeps
  it out of the approval allowlists so `assess()` returns `_ASK` **before** the session-allow cache —
  MCP tools **always re-prompt**, and a cached "always" grant can't survive a rug-pulled definition.
  Plus an `MCPGuard` (server allowlist, runs before approval), a provenance banner as the first
  description line (line-jumping), stdio spawning **opt-in** (`allow_stdio`) with the command shown,
  and the gateway key scrubbed from server env.
- **`Agent.close()` + context manager** (generic `_closeables`) so `assemble_agent(mcp=…)` teardown is
  one call; `mcp=None` is byte-identical (no MCP import).
- **REPL splash** — a framed, welcoming banner (version + tagline, aligned session info, a capability
  summary instead of a raw tool dump, and next-step hints).
- **REPL commands (17 → 27)** + **Tab-cycling slash completion** (`menu-complete`, Shift-Tab backward,
  custom commands included): `/cost`, `/retry`, `/undo`, `/preset`, `/profile`, `/export`, `/reload`,
  `/mcp`, `/verify` (critic-panel the last answer), `/doctor` (env + gateway health). `/init` now writes
  **`AGENTS.md`** (vendor-neutral) rather than `CLAUDE.md` (both still auto-loaded).
- **`prove` shows a proof** — factors the difference (`x²+y²−2xy = (x−y)² ≥ 0`), not just "PROVEN".

### Changed
- **Python floor raised to `>=3.10`** (3.9 is EOL; required by the MCP SDK).
- **`[structured]` → `pydantic>=2`** (shared major with the MCP SDK).

## [1.5.0] — 2026-07-02

Routing hardening — the orchestrator now has the knowledge *and* the workers to route for accuracy.

### Added
- **`ROUTING_GUIDE`** — the measured decision knowledge (which model per task type, greedy sampling
  for reasoning models, compute+verify with the CAS tools, sample-filter-revote when check ≪ solve)
  appended to `supervisor`/`Team` prompts by default, so the coordinator routes itself.
- **`routed_team(client)`** — specialists pre-wired to role-optimal model + sampling + tools:
  `math_specialist`, `reasoning_specialist` (greedy), grounded `research_specialist`, and an opt-in
  `critic_specialist` (gpt-oss:120b — 4% hallucination, abstains rather than guess). All share one
  client/rate-limiter; per-role models overridable. `examples/19_routed_team.py`.

### Findings (`benchmarks/`)
- **`math_specialist` bake-off** (`math_specialist_bakeoff.py`, n=24): qwen2.5:72b and llama3.3:70b
  tie at **75% with 100% tool-use**, while llama4 — the tool-free GSM8K "champion" — **never calls the
  CAS tools** (0% tool-use, 50%). The general routing crown does not transfer to a tool-using role;
  role-specific defaults are set from this, not extrapolation.
- **CAS-verification pays off only where check ≪ solve** (root-solving `+13.3 pp`; MATH-500 `+0.0 pp`),
  and a *completeness*-aware filter converts pass@k → accuracy (`filt == pass@k` at every difficulty).

## [1.4.0] — 2026-07-01

Capability + grounding round-out (all additive). Closes the coding/agentic-harness gaps vs Claude
Code / Codex except MCP.

### Added — codebase understanding, planning, orchestration
- **Codebase search:** `grep` (ripgrep fast-path + pure-Python fallback) and `glob`, workspace-
  confined, in all profiles — the agent can now *explore* a repo, not just read known paths.
- **Planning:** an `update_plan` working-memory task list + a **`/plan`** REPL command (read-only
  plan mode: explore → propose → execute).
- **Dynamic + parallel fan-out:** `parallel_agents` + a model-facing `fan_out(subtasks)` tool — the
  model decides how many independent subtasks to spawn; they run in parallel, paced by the shared
  rate-limiter. Wired into the REPL with read-only workers.
- **Orchestrator routing knowledge:** `ROUTING_GUIDE` — the measured decision knowledge (which model
  per task type, greedy sampling for reasoning models, compute+verify with the CAS tools, and
  sample-filter-revote when check ≪ solve) — is appended to the `supervisor`/`Team` prompt by default
  so the coordinator can route well itself instead of a hard-coded classifier. **`routed_team(client)`**
  pre-wires specialists to the benchmark-optimal model + sampling + tools (`math_specialist`,
  `reasoning_specialist` at greedy, grounded `research_specialist`), all sharing one client/rate-limiter,
  so the routing knowledge maps onto real workers end-to-end.
- **Coding edits + processes:** `apply_patch` (multi-hunk atomic SEARCH/REPLACE edits) and
  `run_background`/`check_job` (long-running processes: dev servers, builds, long tests).

### Added — exact-math grounding (`[math]`) + sound proving (`[smt]`)
- `verify_math` (decide a claim TRUE/FALSE with a CAS — self-verification), `symbolic_math` (exact
  solve/simplify/factor/expand/diff/integrate/limit/series/evaluate), and `matrix_op` (exact linear
  algebra over rationals). sympy-backed, lazy-imported, safe (no `exec`), read-only. Wired into all
  profiles + `data_analyst`, with a system-prompt nudge to compute/verify rather than reason numbers
  in-head.
- **Sound theorem proving** (`[smt]`, z3): `prove` establishes a universally-quantified (in)equality
  for ALL values (negation UNSAT) or returns a concrete counterexample; `solve_constraints` finds a
  witness or proves infeasibility. `verify_math` now uses z3 when sympy is inconclusive, so it
  **proves** claims it used to return UNKNOWN for (e.g. `x²+y² ≥ 2xy`), and labels a numeric fallback
  explicitly as a check, **not** a proof. Honest limit: unbounded induction / transcendentals /
  analysis fall outside z3's fragment (→ UNKNOWN); machine-checked proofs of arbitrary theorems need
  an interactive prover (Lean/Isabelle) — deferred.

### Added — terminal math rendering
- **`prettify`** (`repl/prettify.py`): the REPL now renders **LaTeX → Unicode** instead of dumping raw
  `$\boxed{…}$`. Handles fractions/roots, Greek + ~120 operators/relations/arrows, blackboard &
  calligraphic (`ℝ ℤ ℒ`), super-/sub-scripts, **big-operator limits as ranges** (`\sum_{i=1}^{n}` →
  `∑[i=1..n]`), **combining accents** (`\vec v` → `v⃗`, `\hat x` → `x̂`, `\bar X` → `X̄`), transpose
  (`A^{\top}` → `Aᵀ`), binomials, floor/ceil/angle brackets, `\pmod`, nested **continued fractions**,
  sizing delimiters, and multi-line **matrices** (aligned bracketed grids) and **cases** (braced). Plus
  light **markdown → ANSI** (headers, bold, code, bullets). Validated against an adversarial cross-domain
  corpus. Built-in, no dependency, fails open to raw text; on by default, **`/pretty [on|off]`** toggles.

### Added — Lean 4 proving + a competition-math benchmark
- **`lean_check`** (`tools/lean.py`): the model writes a Lean 4 theorem + proof, the Lean kernel
  checks it, and the agent repairs from the compiler's errors — the loop IS the proof-repair loop.
  A wrong proof or a `sorry` is rejected. Auto-added to profiles when the Lean toolchain is present;
  read-only. A **`lean-prove` skill** teaches the core-tactic discipline (decide / omega / rfl — no
  mathlib), and `examples/18_lean_prove.py` runs the loop. (Interactive proving is now prototyped
  rather than merely deferred; mathlib-scale competition proofs remain a heavier, model-gated step.)
- **MATH-500 benchmark** (`benchmarks/math_eval.py` + `math_grade.py`): 500 competition problems,
  graded by a sympy answer-equivalence checker (dogfoods the CAS; 98% self-consistent). Runs the
  agent **bare vs math-grounded** to measure whether the exact-math tools reduce hallucination
  (the lift). Data is downloaded, not vendored.

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
