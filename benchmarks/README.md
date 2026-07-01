# Benchmarks — replicating agent experiments with the SDK

Faithful replications of four experiments from the agentic data-science
literature, used to stress-test `genai_studio.agents` against an OpenAI-compatible
LLM gateway (default model `qwen2.5:72b`, which supports native tool-calling).
The first three are **offline-gradeable** slices: each mirrors a real paper's task
design and grading, with external dependencies (Kaggle downloads, the live
Wikipedia API) removed so runs are deterministic and reproducible. The fourth,
`react_exact.py`, is the **exact** ReAct experiment run live over the real
Wikipedia API and graded with official Exact-Match, so its numbers are directly
comparable to the paper's Table 1.

```bash
pip install -e '.[datascience]'
export GENAI_STUDIO_API_KEY=...           # required (live)
export GENAI_STUDIO_MODEL=qwen2.5:72b     # optional
export GENAI_STUDIO_RPM=18                # pace the gateway (it silently drops on burst)

python benchmarks/dsbench_mini.py         # data-analysis accuracy
python benchmarks/datascibench_mini.py    # TFC: Completion Rate / Success Rate
python benchmarks/react_eval.py           # 4-condition ablation, frozen fictional corpus
python benchmarks/react_exact.py          # EXACT ReAct on real HotpotQA (live Wikipedia), 10-model sweep
python benchmarks/run_all.py              # everything
```

Traces for every run are written to `benchmarks/_traces/<suite>_<task>/trace.jsonl`
via the SDK's `JsonlTracer`.

## What each suite replicates

### 1. `dsbench_mini.py` — DSBench data analysis (Jing et al., ICLR 2025)
Each task = a long natural-language intro + a self-contained tabular dataset
(CSV) + a multiple-choice or fill-in question with one discrete ground-truth
answer. The agent (an `Agent` with `python_exec`) must **write and run code** to
compute the answer; guessing from the intro fails. Grading is deterministic and
matches the answer **value** — which is exactly what DSBench's GPT-4o "semantic
judge" does for discrete answers (the paper validated 100% judge/human agreement;
for discrete GTs the judge is just normalizing phrasing, so we check directly).

*Faithful:* the input shape (intro + table + MC/fill-in), the discrete-answer
grading, multi-table reasoning. *Adapted:* self-contained datasets instead of the
ModelOff `.xlsx` workbooks (so no external files), and our tables are **cleaner**
than ModelOff's gnarly multi-sheet spreadsheets — so a higher score than the
paper's frontier agents is expected; this suite measures *computation + answer
commitment*, not spreadsheet-parsing.

*Paper reference (data-analysis accuracy):* Llama3-70B ≈ 23%, GPT-4o ≈ 28%,
AutoGen+GPT-4o ≈ 34%, human ≈ 64%.

### 2. `datascibench_mini.py` — DataSciBench TFC evaluation (Zhang et al., 2025)
Each task asks the agent to **produce named output files** (CSV/PNG) with
`python_exec`; grading runs deterministic **checker functions** (the "C" in the
paper's Task–Function–Code framework) over the executed outputs, using the
paper's rule types: exact count `==`, error `≤` threshold, score `≥` threshold
(accuracy/F1/R²/silhouette), and file-exists. We report the paper's two headline
metrics:
- **Completion Rate (CR)** = passed checkers / total checkers (partial credit),
- **Success Rate (SR)** = fraction of tasks where **all** checkers pass.

*Faithful:* the TFC checker methodology, CR/SR metrics, threshold rules.
*Adapted:* sklearn/synthetic datasets; VLM-as-a-judge plot-quality scoring is
replaced by the paper's file-exists boolean (F4).

*Paper reference (coarse SR%):* GPT-4o ≈ 66%, Deepseek-Coder-33B ≈ 56%,
Qwen2.5-7B ≈ 44%.

### 3. `react_eval.py` — ReAct 4-condition ablation (Yao et al., ICLR 2023)
Tools `search[entity]` / `lookup[string]` with the paper's exact semantics
(first-5-sentences page summary with a similar-title fallback; stateful Ctrl-F
lookup) over a small **frozen, fictional** corpus — fictional so the multi-hop
answers cannot be recalled from the model's parameters, forcing genuine
retrieval. Multi-hop QA (Exact Match) + claim verification (FEVER-style accuracy).

The paper's four conditions map directly onto the SDK:
| Condition | Tools | Reasoning prompt | SDK setup |
|---|---|---|---|
| Standard | — | — | `Agent`, no tools, direct |
| CoT | — | "think step by step" | `Agent`, no tools |
| Act | native tool-calling | none | `GenAIStudioClient` + tools |
| ReAct | native tool-calling | "Thought before each action" | same client + reasoning system prompt |
| ReAct-injected | prompt-injected JSON actions | (same) | `ReActClient` — the framework's swap demo |

*Expected finding to reproduce:* Act/ReAct ≫ Standard/CoT (tools matter on
unknowable facts), and ReAct ≥ Act (reasoning helps chain the hops).

### 4. `react_exact.py` — ReAct EXACT on real HotpotQA (Yao et al., ICLR 2023)
Where suite 3 freezes a *fictional* corpus to stay offline, this is the **actual
paper experiment**, reproduced setting-for-setting (the mechanics live in
`paper_react.py`):

- **Data:** real HotpotQA dev questions (HuggingFace), multi-hop and **question-only**;
  default **n = 500** (the paper's dev subset).
- **Environment:** the paper's exact action space over the **live Wikipedia API** —
  `Search[entity]` returns the first sentences of a page (or the top-5 similar titles
  when missing), `Lookup[keyword]` is a stateful Ctrl-F over the current page,
  `Finish[answer]` ends the episode.
- **Prompting:** the paper's **6-shot** in-context exemplars — the *verbatim*
  `webthink` / `webact` / `cotqa` / `webqa` `_simple6` prompts from the official repo
  (vendored in `_data/react_prompts.json`), the same six cases across all four
  conditions, differing only in which fields are shown.
- **Decoding:** **greedy** (`temperature = 0`); **max 7** thought–action steps; **no
  forced answer** on exhaustion (an unfinished trajectory yields no answer).
- **Metric:** strict SQuAD-style **Exact-Match** (`==`, no substring credit).

The framework's signature move drives it: the **same `Agent`** runs all four
conditions, with only the ModelClient swapped to a **`PaperReActClient`** that speaks
the paper's text `Thought / Action: Search[…] / Finish[…]` grammar (generation is
stopped before each `Observation`, so the *environment* supplies observations — not
the model). This is swept across the **10 gateway models** spanning the gateway's
functionality: flagships (`qwen2.5:72b`, `llama3.3:70b`, `gpt-oss:120b`, `llama4:latest`), fast
mid/small models (`gemma3:27b`, `gemma3:1b`, `qwen3:4b`, `llama3.2:latest`), and slow
reasoning models (`deepseek-r1:32b`, `qwq:latest`). Wikipedia calls do **not** touch
the rate-limited gateway; only LLM calls do, paced by `GENAI_STUDIO_RPM`. The runner
checkpoints per (model, condition) cell *and* per question, so an interrupted sweep
resumes exactly where it stopped.

*Two documented deviations, both forced by chat models replacing the paper's
completion model:* (1) the gateway is chat-only (`text-davinci-002` is retired), so
the original's single growing completion is delivered as a multi-turn chat with stop
sequences cutting each turn before the observation — semantically identical 6-shot
greedy ReAct. (2) Instruct models answer bare direct-QA conversationally and ignore
the terse `Question/Answer` pattern, so the **Standard** prompt adds an explicit
"answer only, prefixed `Answer:`" cue to recover the terseness the completion format
enforced implicitly (exemplars unchanged; CoT/Act/ReAct need no cue — they end in a
scaffolded `Answer:`/`Finish[…]`). Even so, residual verbosity tends to *understate*
Standard under strict EM — an honest chat-model artifact, not a harness bug.

*Paper reference (PaLM-540B, HotpotQA EM):* Standard 28.7 / CoT 29.4 / Act 25.7 /
ReAct 27.4.

## Results (qwen2.5:72b)

### DSBench-mini — data-analysis accuracy
**7 / 9 = 77.8%** (value-graded, faithful to the paper's semantic judge).

| ✓/✗ | task | note |
|---|---|---|
| ✅ | voters_district | counted band correctly |
| ✅ | inflation_cap | `82.02` (compound interest) |
| ❌ | coin_montecarlo | answered 10-15%, true ≈17.2% (15-20%) |
| ✅ | feb_avg_usage | filtered February, averaged |
| ✅ | diabetes_ols_coef | OLS bmi coef ≈ 520 |
| ✅ | titanic_survival | female-1st-class survival 90% |
| ❌ | iris_petal_ratio | **computed "setosa", then talked itself into "virginica"** |
| ✅ | pearson_correlation | r = 0.92 |
| ✅ | multitable_revenue | joined two tables, summed West region |

Reference: Llama3-70B ≈ 23%, GPT-4o ≈ 28%, AutoGen+GPT-4o ≈ 34%, human ≈ 64% —
on ModelOff's much messier multi-sheet Excel. The two failures are instructive: a
genuine Monte-Carlo bucketing error, and a **self-doubt failure** where the agent
reached the right answer and then revised it to a wrong one.

> An early raw run scored 44% only because the grader required the answer *letter*;
> the agent had computed the right *values* (`Answer: 82.02`, `Answer: 90`) — fixed
> by value-matching, which is exactly what DSBench's LLM judge does for discrete GTs.

### DataSciBench-mini — TFC Completion / Success Rate
**Completion Rate 1.00, Success Rate 1.00 (6 / 6)** — every task produced the
required output files and passed all programmatic checkers (cleaning, stats,
classification ≥ 0.85, regression R² ≥ 0.35, clustering silhouette ≥ 0.5,
visualization PNG). Reference (coarse SR): Qwen2.5-7B ≈ 44%, GPT-4o ≈ 66%. Our 6
tasks are well-specified single/two-step pipelines; DataSciBench's 222 prompts
include much longer chains, so 100% here reflects clean tasks, not a solved
benchmark — but it shows the `python_exec` loop executes real DS workflows reliably.

### ReAct ablation — Standard / CoT / Act / ReAct (multi-hop EM + FEVER acc)

**n=10 runs per task** (90 outcomes per condition). The gateway is capacity-limited
(~20 req/min) and silently *drops* responses under burst load rather than returning
429; those drops are infrastructure, not agent error, so the fair metric excludes
them from the denominator (`fair = correct / (correct + agent_wrong)`).

| Condition | Mechanism | fair acc | correct/valid | infra drops |
|---|---|---|---|---|
| Standard | no tools, direct | **0.211** | 16/76 | 14 |
| CoT | no tools, reason | **0.225** | 20/89 | 1 |
| Act | native tool-calling | **0.733** | 66/90 | 0 |
| ReAct | native tools + reasoning | **0.767** | 69/90 | 0 |
| ReAct-injected | `ReActClient` (prompt-injected JSON actions) | **0.767** | 69/90 | 0 |

The robust, statistically clear finding is **tools ≫ no tools** (0.73–0.77 vs
0.21–0.23 — a ~0.5 gap on facts that cannot be recalled). The **ReAct ≥ Act** edge
(0.767 vs 0.733) is directional but *within noise* at n=10 (SE ≈ 0.045) — and note
that the single-run estimate (`react=0.889`) overstated it, which is exactly why
n=10 matters. The `react_injected` row shows the framework's signature move — the
**same `Agent`** driven by `ReActClient` (text JSON actions) instead of native tools
— matches the native ReAct condition. The recurring miss, `seine_claim` (model says
NOT ENOUGH INFO where the answer is REFUTES), is the genuine FEVER difficulty of
distinguishing contradiction from insufficiency.

> **Fairness & the gateway limit.** The tool conditions made 0 dropped requests
> (their per-task multi-call loops are naturally paced by tool execution); only the
> single-call Standard/CoT baselines, fired back-to-back, overran the gateway —
> which *depresses the floor*, biasing the tools-vs-no-tools gap conservatively. The
> SDK now ships a global `RateLimiter` (`GENAI_STUDIO_RPM`, paces every request incl.
> retries) so a clean run drops nothing; the table above already excludes the drops.
> **Verified:** re-running the worst-hit `standard` condition with `GENAI_STUDIO_RPM=18`
> produced **0 drops** (vs 14) and a clean 0.222 — matching the re-graded 0.211, so the
> exclusion was accurate and the conclusion is unchanged.

> **A framework bug this benchmark caught and fixed.** On multi-hop tool use, qwen
> sometimes emits its next tool call as JSON in the message *content* instead of the
> native `tool_calls` field — so the loop mistook it for a final answer (garbage
> like `{"id":"call_..","function":{"name":"lookup"...}}`). `GenAIStudioClient` now
> recovers tool calls embedded in content (`_tool_calls_from_text`, with a
> regression test); after the fix the agent correctly chains
> `search[Lutz]→search[Zubrowka]→"Zubrish"`. This is the payoff of running real
> benchmarks against the SDK.

### HotpotQA EXACT (paper-faithful) — full results (n = 500, 10 models × 4 conditions)

The setting-for-setting replication (suite 4): the paper's protocol — 6-shot verbatim
exemplars, ≤7 steps, strict SQuAD Exact-Match — over **n = 500** real dev questions,
live Wikipedia, across **10 gateway models × 4 conditions** (40 cells, all complete;
results in `_results/hotpotqa_paper_n500.json`). Fair-EM (infra drops excluded).

| model | standard | cot | act | react |
|---|---|---|---|---|
| qwen2.5:72b | 29.6 | 26.6 | 38.2 | 38.4 |
| llama3.3:70b | 32.0 | **34.8** | **42.2** | 41.4 |
| gpt-oss:120b | **32.8** | 19.4 | 35.8 | 40.6 |
| llama4:latest | 24.2 | 26.8 | 36.0 | 34.0 |
| gemma3:27b | 26.4 | 28.4 | 39.4 | 38.0 |
| gemma3:1b | 9.6 | 9.4 | 1.8 | 1.2 |
| qwen3:4b | 19.2 | 19.1 | 38.9 | 39.4 |
| llama3.2:latest | 19.4 | 11.6 | 7.2 | 2.0 |
| deepseek-r1:32b | 22.8 | 21.2 | 32.8 | 26.8 |
| qwq:latest | 29.2 | 28.0 | **44.6** | **43.8** |

*Paper PaLM-540B (2022): Standard 28.7 / CoT 29.4 / Act 25.7 / ReAct 27.4.*

**Findings.**
- **Tools ≫ no-tools holds for every capable model.** 8 of 10 score markedly higher on
  Act/ReAct than Standard/CoT (≈+10–25 pts). The two exceptions, `gemma3:1b` (1B) and
  `llama3.2` (3B), *invert* it (Act/ReAct *below* their own Standard) — a genuine
  capability floor: small models can't sustain a multi-hop tool loop. This is the
  paper's headline finding, reproduced across a 2024–2026 model zoo.
- **Modern flagships beat 2022 PaLM-540B on the tool conditions** (Act/ReAct ≈ 32–45 vs
  the paper's 25.7/27.4) — expected, and the floor models confirm it's capability, not
  harness inflation. The reasoning model **qwq is the top tool performer (Act 44.6 /
  ReAct 43.8)** once the thinking fix lets it emit clean actions.
- **ReAct ≈ Act (no consistent reasoning edge).** Across models the Act vs ReAct gap is
  small and sign-flips by model — matching the paper's own note that the ReAct-over-Act
  margin is within noise.
- Best per condition: Standard `gpt-oss` 32.8 · CoT `llama3.3` 34.8 · Act/ReAct `qwq`
  44.6 / 43.8.

**Methodology & honest caveats** (all in `paper_react.py`, documented there):
- Decoding is **greedy (temp 0)** for non-thinking models (paper-faithful). **Thinking
  models** (`qwen3:4b`, `deepseek-r1:32b`, `qwq`) degenerate to *empty* completions under
  greedy, so they use a documented exception — **sampled (temp 0.6 / top-p 0.95), no stop
  sequences, 8192-token budget, `<think>` stripped, first-action parse** — which makes
  those cells **non-deterministic** (the paper itself calls greedy "sub-optimal", fn 4).
- Two model-agnostic adherence fixes were required for chat/instruct models that don't
  follow the completion-era few-shot: a **terseness cue** on Standard (instruct models
  answer verbosely → strict EM scored ~1% until fixed) and an **`Action:` format contract
  + bounded repair loop** on Act/ReAct (e.g. `gpt-oss` emitted verbose markdown → 5–7%
  until fixed → 35.8/40.6). `qwen`/`llama` already complied (fixes are no-ops for them).
- **Strict EM understates verbose models** (especially Standard) — the answer is often
  present but not a terse span. The Standard column is a conservative floor.
- **Infra**: 107 dropped requests total across the run (gateway "server busy" congestion
  over a multi-day window + an expired-session API key mid-run), all excluded from
  fair-EM. Only one cell has materially reduced effective-n: `qwen3:4b|act` (valid 434,
  66 drops). All other cells are at/near n = 500; **no all-infra cells**.
- **Token logprobs (re-probed 2026-06-24).** The gateway *accepts* `logprobs=True` /
  `top_logprobs` without error on every model, but only **`gpt-oss:120b` returns a
  populated `logprobs` object** — covering the real answer tokens (e.g. the `Paris`
  token with its top-5 alternatives), interleaved with Harmony control tokens
  (`<|channel|>`, `<|message|>`, …) that must be filtered out. The flagship/general
  models (`qwen2.5:72b`, `llama3.3:70b`, `gemma3:27b`, `mistral`, `phi4`, `llama3.2`)
  return **none** — confirmed with non-empty completions and at the wire
  (`choices[0].logprobs` is literally `null`), so it is genuinely unsupported, not an
  empty-completion artifact. Continuous, logprob-based uncertainty (semantic entropy,
  temperature-scaling calibration) is therefore available **only for gpt-oss**; on the
  general models, uncertainty work is limited to black-box methods (discrete semantic
  entropy, self-consistency).

(A separate `hotpotqa_exact_n3.json` holds an earlier *zero-shot* smoke test, kept only
as a baseline; the paper-faithful numbers are the `hotpotqa_paper_*` files.)

---

## Agentic reliability + model routing (`agentic_eval.py`)

Beyond the four paper replications, `agentic_eval.py` (+ `real_benchmarks.py`) is a **reliability +
model-routing** harness: it runs each task *k* times per condition and reports **accuracy /
hallucination (= incorrect-rate) / abstention** (SimpleQA 3-way) plus an F-score and tokens, on top
of pass^k / consistency. It grounds the eval in real datasets — **SimpleQA** (single-hop facts),
**GSM8K** (math), **MuSiQue** (multi-hop), and **BFCL** (tool-call AST-graded) — downloaded and
cached to a gitignored `_data/benchmarks/`.

**Design.** The `grounded` condition gives the agent `web_search` + `wikipedia` tools. Grading uses a
**rotating peer-judge panel**: per task a judge is sampled (seeded) from a validated flagship pool
{gpt-oss:120b, llama3.3:70b, llama4:latest, qwen2.5:72b}, **excluding the agent's own model** (no
self-judging), with a de-biased question-free judge prompt. A **resumable matrix accumulator** keys
cells by (task-content, run, sampling-regime), so a screen can grow *n* / *k* while reusing cells.

### Finding 1 — reasoning models want GREEDY, not the temp=0.6 recipe

The documented reasoning-benchmark recipe (temperature 0.6, top-p 0.95) is the **wrong default for
agentic tasks**. A controlled same-question A/B at n=60 (only temperature changed):

| model | temp=0.6 | greedy (0.0) | Δ |
|---|---|---|---|
| **deepseek-r1:32b** | 21% (38% abstain) | **46%** (34% abstain) | **+25pp** — `<think>` looping collapsed |
| qwq:latest | 36% | 38% | +2pp (~flat) |
| qwen3:4b | 36% | 37% | +1pp (~flat) |

Greedy is best-or-tied for all three (never worse), **transformative for `deepseek-r1`**. The
framework default (`_thinking_sampling`) was corrected 0.6 → greedy. (An exploratory sweep at n=10
overstated `qwq` at 53%; the n=60 finalization corrected it to 38% — small-n screens can mislead.)

> This is the **opposite** of the paper-ReAct finding above (§ Reproducibility), where thinking
> models needed temp=0.6 — because that setting used *greedy + no stop sequences + a completion
> grammar*, where greedy degenerates to empty. Different baseline + task ⇒ opposite conclusion.

### Finding 2 — the routing table (all 10 models @ n=60, reasoning models at greedy)

| model | acc | halluc | abstain | F | tokens |
|---|---|---|---|---|---|
| **qwen2.5:72b** | **47%** | 30% | 23% | 0.53 | 3395 |
| **deepseek-r1:32b** | 46% | **19%** | 34% | **0.56** | 2477 |
| llama4:latest | 43% | 45% | 12% | 0.46 | **335** |
| gemma3:27b | 38% | 23% | 38% | 0.47 | 2697 |
| qwq:latest | 38% | 23% | 39% | 0.48 | 2729 |
| qwen3:4b | 37% | 47% | 17% | 0.40 | 2096 |
| llama3.3:70b | 34% | 33% | 33% | 0.41 | 2510 |
| gpt-oss:120b | 31% | **4%** | 65% | 0.46 | 10340 |
| llama3.2:latest | 6% | 36% | 58% | 0.08 | 3805 |
| gemma3:1b | 3% | 72% | 26% | 0.03 | 4333 |

**Route by task:** math (GSM8K) → llama4 93% (cheapest) / qwen2.5 90% / qwen3 87% · facts (SimpleQA)
→ deepseek-r1 43% / qwen2.5 32% · multi-hop (MuSiQue) → all weak (≤25%, needs *decomposition*, not a
model swap) · "must not be wrong" → gpt-oss (4% halluc, at 10k tokens + 65% abstention) · default →
qwen2.5:72b.

**Caveats.** k=1 screen (directional, ±~10pp; the peer-judge + F-score make the *ordering*
trustworthy); `grounded` condition only. A related finding: **grounding cuts hallucination when there
IS an answer but INCREASES it on unanswerable / false-premise questions** (tools make the model
confabulate from tangential hits instead of refusing). Reproduce:

```bash
python benchmarks/agentic_eval.py --benchmarks simpleqa gsm8k musique --n 60 --k 1 \
  --conditions grounded --model qwen2.5:72b \
  --judge-pool gpt-oss:120b llama3.3:70b llama4:latest qwen2.5:72b --seed 0
```

## Math grounding — MATH-500 (`math_eval.py`)

Does giving a non-frontier model the **exact-math tools** (`symbolic_math` / `verify_math` /
`matrix_op` / `prove` / `solve_constraints`) reduce math hallucination? `math_eval.py` runs
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — 500 competition problems
(algebra → number theory → precalc, 5 difficulty levels) — in two conditions and reports the **lift**:

- **bare** — the model reasons and boxes an answer (no tools). The baseline.
- **grounded** — the agent has the math tools and is told to compute/verify with them.

Answers are graded by a **sympy equivalence checker** (`math_grade.py`, dogfooding the CAS; ~98%
self-consistent on the gold), so `\frac12` ≡ `0.5` ≡ `\boxed{\frac{1}{2}}`. Reproduce:

```bash
export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
python benchmarks/math_eval.py --model qwen2.5:72b --n 60 --conditions bare,grounded
```

**Finding (n=63 stratified, qwen2.5:72b):** bare ≈ grounded (**46% vs 44%, no subject lift**). This is
the literature-predicted result — tool grounding fixes *computation*, but a strong 72B fails MATH on
*setup/insight*, not arithmetic, and forced tool use even hurts easy items (see
`../../genai-studio-math-grounding-litreview.md`). Two lessons: (1) don't use tools as an inline
*solver*; the gains come from tools as a *verifier* over samples; (2) `prove`/`lean_check` make
proofs, which an answer-matching benchmark can't credit.

### CAS-verified self-consistency (`math_selfconsist.py`)

Per Tao's division of labour (**LLMs generate, formal tools verify**) and PROVE (arXiv:2410.12608),
this is the integration style that actually lifts accuracy — three arms:

- **bare@1** — one greedy solution.
- **maj@k** — k sampled solutions, majority-vote (self-consistency; the honest baseline).
- **cas_verified** — same k samples; a CAS pass (`symbolic_math`/`verify_math`/`prove`) verifies each
  distinct candidate answer and discards the refuted ones; majority-vote the survivors.

It also structurally avoids the tool-call spin/leak (tools score finished answers instead of steering
the loop). A hard per-sample wall-clock guard + tight request timeout keep a dropped gateway response
from stalling the run. Reproduce:

```bash
export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
python benchmarks/math_selfconsist.py --model qwen2.5:72b --n 40 --k 8 --stratify subject
```

For **formal proof** (not answer-matching), `examples/18_lean_prove.py` runs the Lean 4 kernel-checked
write→check→repair loop.
