# Embedding-capable models on Purdue GenAI Studio

**Probed live 2026-06-17** against `genai.rcac.purdue.edu` (`/api/embeddings`).
Of the **35** models the gateway lists, only **15 expose a working embedding
endpoint** — the rest return an error (no embedding endpoint). Use this when
choosing a model for `ai.embed(...)`, similarity, or RAG indexing.

> ⚠️ **Note:** `gemma3` (the gateway's default **chat** model) does **not**
> support embeddings. For any embedding / RAG work, switch to an
> embed-capable model — e.g. `llama3.2:latest` (3072-d), `mistral:latest` (4096-d),
> or `qwen2.5:72b` (8192-d).

## ✅ Embed-capable (with embedding dimension)

| Model | Dim |
|---|---|
| `llama3.2:latest` | 3072 |
| `llama3.1:latest` | 4096 |
| `llama3.1:70b-instruct-q4_K_M` | 8192 |
| `llama3.3:70b` | 8192 |
| `qwen2.5:72b` | 8192 |
| `phi4:latest` | 5120 |
| `mistral:latest` | 4096 |
| `codellama:latest` | 4096 |
| `llava:latest` | 4096 |
| `qwq:latest`, `qwq:32b-fp16` | 5120 |
| `deepseek-r1:*` (1.5b → 32b) | 1536 – 5120 (varies by size) |

## ❌ No embedding endpoint (returns an error)

- `gemma3` — **all sizes** (1b / 12b / 27b); the gateway's default chat model
- `llama4:latest`
- `qwen3` — all sizes, including `qwen3-coder` and `qwen3-vl`
- `gpt-oss` (`:latest`, `:120b`)
- `devstral-small-2:latest`
- `medgemma:27b`

## Notes
- Embedding **dimension differs by model**, so a vector store / RAG index built
  with one model is not interchangeable with another — pick one embed model and
  keep it consistent across indexing and query.
- Availability is a property of the live gateway and can change; re-probe with
  `ai.embed("test")` per model if in doubt.
