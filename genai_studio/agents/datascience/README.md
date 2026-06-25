# `genai_studio.agents.datascience` — an agentic data-science toolkit

A small, self-contained layer that turns the core agent framework
(`genai_studio.agents`) into a **data-science analyst**: a handful of typed
`@tool`s for loading data, computing statistics, fitting models, and plotting,
plus a pre-assembled `data_analyst` agent that wires them together. It is the
optional **`[datascience]`** extra — the heavy scientific stack
(pandas / NumPy / scikit-learn / SciPy / statsmodels / matplotlib) is imported
**only when a tool actually runs**, so importing the core stays light.

The whole package is ~530 lines. There is no new agent machinery here: every
tool is an ordinary `@tool`, and `data_analyst` is a thin factory over the core
`Agent`. What makes it useful is the *combination* — a persistent code
sandbox sharing a namespace with a dataset loader, fronted by a model that
decides what to compute.

---

## Install & 30-second quickstart

```bash
pip install -e '.[datascience]'        # core framework + scientific stack
export GENAI_STUDIO_API_KEY=...        # an OpenAI-compatible chat gateway key
export GENAI_STUDIO_RPM=18             # pace the gateway (see "Reference environment")
```

```python
from genai_studio import GenAIStudio
from genai_studio.agents import GenAIStudioClient
from genai_studio.agents.datascience import data_analyst

client = GenAIStudioClient(GenAIStudio(), default_model="qwen2.5:72b")
agent  = data_analyst(client)

result = agent.run(
    "Load the iris dataset and tell me which two features best separate "
    "the species. Show the evidence."
)
print(result.text)
```

The agent loads `iris`, inspects it inside a Python sandbox, computes
class-separation evidence (e.g. per-feature group means / a quick classifier),
and writes a short, evidence-backed conclusion — never guessing from the prompt.

---

## What's in the box

| Tool | Purpose |
|---|---|
| `python_exec` | Execute Python in a **persistent** namespace — a DataFrame built in one step is still there in the next. The trailing expression is summarised notebook-style; full tracebacks are fed back so the model self-corrects. |
| `load_dataset` | Load a bundled scikit-learn toy set (`iris`, `wine`, `diabetes`, `breast_cancer`). Nothing is downloaded or vendored. When sharing a namespace, the frame is injected as `df` for `python_exec`. |
| `describe_data` | One-call EDA of a CSV: shape, dtypes, missing-value counts, summary statistics, and the strongest pairwise correlations. |
| `hypothesis_test` | Classical tests on CSV columns via `scipy.stats`: two-/one-sample t-test, Pearson/Spearman correlation, chi-square independence, one-way ANOVA — each reporting the statistic, p-value, and a reject / fail-to-reject conclusion at a chosen α. |
| `fit_model` | Fit a quick model on a bundled dataset — OLS / logistic (`statsmodels`) or random forest (`scikit-learn`) — and return a text summary (coefficients or feature importances + in-sample R²). |
| `plot` | Run matplotlib code **headlessly** (forces the `Agg` backend before importing pyplot, so it works with no display) and save the figure to a PNG; the `Figure` rides along in `ToolResult.data`. |

The `data_analyst(client, *, model="qwen2.5:72b", **kw)` factory bundles all of
the above — plus the core `calculator` and `final_answer` — behind a system
prompt that enforces good habits: compute in `python_exec` (never in the
model's head), inspect data before concluding, build up state across steps,
don't repeat tool calls, and stop to write a plain-text, evidence-backed answer
once there's enough to say. `python_exec` and `load_dataset` are bound to **one
shared namespace** so a loaded frame flows directly into executed code.

```
ns = {}                          # one namespace shared by python_exec + load_dataset
data_analyst = Agent(
    client, model, system=DATA_ANALYST_SYSTEM,
    tools=[make_python_exec(ns), make_load_dataset(ns), describe_data,
           fit_model, hypothesis_test, plot, calculator, final_answer],
)
```

---

## Design notes

- **Thin over the core.** Nothing here subclasses or forks `Agent`. The tools
  are plain `@tool`s; swapping the `ModelClient` (native tool-calling ⇄ a
  ReAct text grammar) changes the *mechanism* without touching the tools or the
  loop. This client-swap is the framework's signature move and is exactly what
  the benchmark replications below exercise.
- **Lazy, friendly dependencies.** `import genai_studio.agents.datascience`
  pulls in *nothing* heavy. Each tool calls `_require(...)` at run time; a
  missing package raises a one-line install hint
  (`pip install 'genai-studio-sdk[datascience]'`) instead of a raw
  `ImportError`.
- **Compact, model-facing results.** A shared `summarize()` turns a DataFrame /
  Series / ndarray into a short string (shape + dtypes + `head()`) rather than
  dumping the whole object, so tool results don't blow up the context window.

---

## Safety model (read before pointing it at untrusted data)

`python_exec` and `plot` run model-authored code **in the current process**.
This is deliberate and transparent: it keeps the teaching/analysis path simple,
and it is fine for **trusted input** (your own prompts, your own data). It is
**not** safe the moment an agent processes shared, user-supplied, or
web-fetched content — a model (or a prompt-injected one) can run
`__import__('os').system(...)`, read files, or open sockets with your
privileges.

The seam is designed for a drop-in replacement: anything matching
`code: str -> ToolResult` can stand in. The documented hardening (in
`tools/python_exec.py`) runs the code in a subprocess that sets
`resource.setrlimit` (address space / CPU), enforces a wall-clock timeout +
`SIGKILL`, drops to a restricted user in a container/nsjail with no network and
a scratch filesystem, and ships state by pickling the namespace in and out.
Harden before untrusted data.

---

## Does the agent loop actually work? — evidence from benchmark replications

The package is validated by faithful replications of published agent
experiments (full code, methodology, and numbers in
[`benchmarks/`](../../../benchmarks/README.md)). Two of them target
data-science skill directly; two target the agentic reasoning the loop depends
on. All run live against a real LLM gateway.

### Data-science task accuracy

| Replication | Metric | Result | Reference points |
|---|---|---|---|
| **DSBench-mini** (Jing et al., ICLR 2025) — write-and-run code to answer a question over a tabular dataset | value-graded accuracy | **77.8 %** (7/9) | Llama3-70B ≈ 23 %, GPT-4o ≈ 28 %, AutoGen+GPT-4o ≈ 34 %, human ≈ 64 % (on messier source spreadsheets) |
| **DataSciBench-mini** (Zhang et al., 2025) — produce named output files, graded by deterministic checker functions | Completion / Success Rate | **CR 1.00 · SR 1.00** (6/6) | Qwen2.5-7B ≈ 44 %, GPT-4o ≈ 66 % (on longer chains) |

Both run on `qwen2.5:72b` driving `python_exec`. The point is not the headline
percentage (our slices use cleaner, well-specified tasks than the source
benchmarks, so higher scores are expected) but that **the execute-then-read
loop runs real pandas/scikit-learn workflows end-to-end and commits to a
discrete answer.** The two DSBench misses are instructive: one genuine
Monte-Carlo bucketing error, and one *self-doubt* failure where the agent
computed the right answer and then talked itself into a wrong one.

### Agentic reasoning — ReAct, replicated exactly

The headline experiment is a **setting-for-setting replication of ReAct**
(Yao et al., ICLR 2023) on **real HotpotQA** over the **live Wikipedia API**:
the paper's verbatim 6-shot exemplars, its exact `Search` / `Lookup` / `Finish`
action space, ≤7 thought–action steps, and **strict SQuAD Exact-Match**. The
*same* `Agent` runs all four conditions — only the `ModelClient` is swapped to
speak the paper's `Thought / Action / Observation` grammar. We swept **n = 500**
dev questions across **10 gateway models × 4 conditions** (40 cells, all
complete).

**HotpotQA Exact-Match %** (fair-EM; infrastructure drops excluded):

| model | Standard | CoT | Act | ReAct |
|---|---|---|---|---|
| qwen2.5:72b | 29.6 | 26.6 | 38.2 | 38.4 |
| llama3.3:70b | 32.0 | **34.8** | 42.2 | 41.4 |
| gpt-oss:120b | **32.8** | 19.4 | 35.8 | 40.6 |
| llama4:latest | 24.2 | 26.8 | 36.0 | 34.0 |
| gemma3:27b | 26.4 | 28.4 | 39.4 | 38.0 |
| gemma3:1b | 9.6 | 9.4 | 1.8 | 1.2 |
| qwen3:4b | 19.2 | 19.1 | 38.9 | 39.4 |
| llama3.2:latest | 19.4 | 11.6 | 7.2 | 2.0 |
| deepseek-r1:32b | 22.8 | 21.2 | 32.8 | 26.8 |
| qwq:latest | 29.2 | 28.0 | **44.6** | **43.8** |

*Original paper, PaLM-540B (2022): Standard 28.7 · CoT 29.4 · Act 25.7 · ReAct 27.4.*

**What the sweep shows.**

- **Tools ≫ no-tools, for every capable model.** 8 of 10 score markedly higher
  on Act/ReAct than on Standard/CoT (≈ +7–20 pts) — the paper's central
  finding, reproduced across a 2024–2026 model zoo. The two exceptions,
  `gemma3:1b` (1B) and `llama3.2` (3B), *invert* it (tool conditions fall
  **below** their own Standard): a genuine capability floor, where a model can't
  sustain a multi-hop tool loop. That the floor models fail confirms the gains
  are capability, not harness inflation.
- **Most modern models clear 2022 PaLM-540B on the tool conditions** (Act/ReAct
  ≈ 33–45 vs 25.7 / 27.4; deepseek-r1's ReAct 26.8 is the lone near-miss). The
  reasoning model **`qwq` is the top tool
  performer (Act 44.6 / ReAct 43.8)** once a thinking-model decoding fix lets it
  emit clean actions.
- **ReAct ≈ Act (no consistent reasoning edge).** The Act-vs-ReAct gap is small
  and sign-flips by model — matching the paper's own note that the margin is
  within noise.

A separate offline **ablation** on a frozen *fictional* corpus (so answers
can't be recalled from parameters), `n = 10` per item, isolates the same effect
without any Wikipedia dependency: Standard 0.21 · CoT 0.22 · Act 0.73 · ReAct
0.77 — and the prompt-injected `ReActClient` matches native tool-calling
(0.77), demonstrating the client-swap.

> **A real bug the benchmarks caught.** On multi-hop tool use, some models emit
> the next tool call as JSON in the message *content* instead of the native
> `tool_calls` field — so the loop mistook it for a final answer. The client now
> recovers tool calls embedded in content (with a regression test). Running real
> benchmarks against the toolkit is what surfaced it.

---

## Reproducibility & honest caveats

The numbers above are real but come with documented deviations, all forced by
running modern **chat** models where the paper used a completion model:

- **Decoding.** Greedy (`temperature = 0`) for non-thinking models, paper-faithful.
  **Thinking models** (`qwen3`, `deepseek-r1`, `qwq`) degenerate to *empty*
  completions under greedy, so they use a documented exception — sampled
  (`temp 0.6 / top-p 0.95`), no stop sequences, an 8192-token budget for the
  `<think>` block, `<think>` stripped, and a first-action parse. This makes
  those cells **non-deterministic** (the paper itself calls greedy "sub-optimal").
- **Adherence fixes** (model-agnostic, no-ops for already-compliant models):
  a terseness cue on **Standard** (instruct models answer verbosely → strict EM
  scored ~1 % until fixed) and an `Action:`-format contract + a bounded repair
  loop on **Act/ReAct** (e.g. one model emitted verbose markdown → 5–7 % until
  fixed → 36 / 41 %).
- **Strict EM understates verbose models**, especially **Standard** — the answer
  is often present but not a terse span. Treat the Standard column as a
  conservative floor.
- **Infrastructure.** Capacity-limited gateways silently *drop* responses under
  burst load (no HTTP 429). Those drops are infrastructure, not agent error, so
  fair-EM excludes them from the denominator. Across the full n = 500 sweep, 107
  requests dropped; only one cell has materially reduced effective-n
  (`qwen3:4b|act`, 434/500 valid). A process-wide `RateLimiter`
  (`GENAI_STUDIO_RPM`) paces every request so a clean run drops nothing.

### Token logprobs on the gateway (probed 2026-06-24)

Uncertainty methods that need per-token probabilities (continuous semantic
entropy, temperature-scaling calibration) depend on the gateway returning
logprobs. A live probe with `logprobs=True, top_logprobs=5` — recording each
model's *actual completion*, not just the logprobs field — found:

- The parameter is **accepted (no HTTP 400) on every model tested.**
- **Only `gpt-oss:120b` returns a populated `logprobs` object.** It covers the
  real answer tokens (e.g. the `Paris` token with its top-5 alternatives),
  interleaved with Harmony control tokens (`<|channel|>`, `<|message|>`, …) a
  consumer must filter out.
- The flagship/general models return **none** — confirmed on `qwen2.5:72b`,
  `llama3.3:70b`, `gemma3:27b`, `mistral`, `phi4`, and `llama3.2` with **non-empty
  completions** and at the wire (`choices[0].logprobs` is literally `null`), so it
  is genuinely unsupported, not an empty-completion artifact. (The thinking models
  `qwen3` / `deepseek-r1` / `qwq` likewise returned none in an earlier probe.)

So continuous, logprob-based uncertainty is available **only for `gpt-oss`**; on
the general models, uncertainty work is limited to **black-box** methods
(discrete semantic entropy, self-consistency over sampled generations).

The cause is the gateway stack, not the models. The OpenAI-compatible chat route
is served through LiteLLM, which does not forward the underlying Ollama runner's
per-token logprobs. The routes that *would* carry them are closed on the
reference deployment: the **raw Ollama proxy** (`/ollama/api/chat`, where Ollama
≥ 0.12.11 returns logprobs natively) is disabled (HTTP 503), the **OpenAI
passthrough** (`/openai/v1/...`) is disabled (HTTP 403,
`ENABLE_OPENAI_API_PASSTHROUGH=False`), and legacy completions are not exposed
(HTTP 405). So logprobs here are gated by **gateway configuration**, not a hard
limit — an operator who enables the native Ollama API or the OpenAI passthrough
would unlock them for every model.

---

## Reference environment

Everything here targets an **OpenAI-compatible chat gateway**. The reference
deployment is **GenAI Studio** (Open WebUI + LiteLLM + Ollama); it offers native
tool-calling on capable models (verified on `qwen2.5:72b`) and is
**capacity-limited (~20 requests/min)**, which is why a shared `RateLimiter`
(`GENAI_STUDIO_RPM`) paces every call. Wikipedia calls in the ReAct
replication do **not** touch the gateway; only LLM calls do. Any other
OpenAI-compatible endpoint works by pointing the client's base URL at it.

> **Model tags.** Use `llama3.2:latest` and `qwq:latest` (the bare `llama3.2` /
> `qwq:32b` tags 400 on the reference gateway). For embeddings/RAG, note that
> not every chat model exposes an embedding endpoint — see
> [`docs/embedding-models.md`](../../../docs/embedding-models.md).
