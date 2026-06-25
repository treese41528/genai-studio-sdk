# Examples — agentic data science

Ten runnable scripts, each a focused concept, built on `genai_studio.agents`.

## Setup

```bash
pip install -e '.[datascience]'        # core + scientific stack
export GENAI_STUDIO_API_KEY="your-key" # GenAI Studio → Settings → Account → API Keys
export GENAI_STUDIO_MODEL="qwen2.5:72b"  # optional; this model supports native tool-calling
```

## The examples

| File | Concept |
|------|---------|
| `01_tools.py` | A typed function becomes a tool — inspect the auto-generated JSON Schema, call it directly. |
| `02_agent_loop.py` | The loop — the **same** `Agent` over native tool-calling vs ReAct. |
| `03_tracing.py` | Observe every step: `ConsoleTracer`, then `JsonlTracer` to a file. |
| `04_structured_output.py` | A validated, typed final answer via `output_schema` (pydantic). |
| `05_data_tools.py` | `data_analyst` on iris (`python_exec` + `load_dataset`) + the sandbox-safety lesson. |
| `06_multi_agent.py` | `Agent.as_tool()`: an orchestrator delegates to a sub-agent in one step (isolated context, result returned). |
| `07_eval_guardrails.py` | Budgets + `JsonlTracer` → Level-1 assertions + an LLM-as-judge stub. |
| `08_grounded_verifier.py` | A grounded `verifier` sub-agent (`kb_search`) exposed via `as_tool` to fact-check claims before answering. |
| `09_orchestration.py` | The two team topologies: `supervisor` (dynamic delegation to workers-as-tools) and `pipeline` (fixed sequential stages). |
| `10_guards.py` | Deterministic before/after-tool `Guard`s: a capability policy, PII redaction, and a tree-wide tool-call budget. |

`01`–`04`, `06`, `07`, `09`, `10` need only the core; `05` needs the `[datascience]` extra; `08` needs a knowledge base on the gateway (and `[grounding]` to also ground public stats via Data Commons).
