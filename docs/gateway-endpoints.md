# Gateway API surface — Open WebUI endpoints for the pipeline

The SDK talks to an **Open WebUI** gateway (Open WebUI → LiteLLM → Ollama). The
OpenAI-compatible chat endpoint is only one surface; Open WebUI also exposes its
own application API (`/api/...`, `/api/v1/...`) with retrieval, files, knowledge,
tasks, audio, and more — several of which are useful to an agentic-data-science
pipeline and are **not** the locked-down passthroughs.

This catalogs the pipeline-relevant endpoints, **what is actually reachable on
the reference deployment**, and what's worth wrapping. Status was live-probed
2026-06-24 against the reference gateway (**Open WebUI 0.9.6**) with a normal
(non-admin) user token; request/response contracts are from the Open WebUI
source. All endpoints need `Authorization: Bearer <token>` unless noted.

**Legend**
`✅ verified` — probed live, works with our token ·
`🔒 admin` — exists but returns 401/403 for a non-admin token ·
`🚫 disabled` — turned off by the operator ·
`📄 documented` — in the Open WebUI API, plausibly reachable, but not probed here
(or not present on 0.9.6).

---

## TL;DR — what to wrap (in priority order)

1. **`POST /api/v1/retrieval/query/collection`** ✅ — server-side semantic RAG:
   ranked chunks + scores + metadata, **no chat call, no client-side embeddings**.
   The single highest-value addition.
2. **`files` payload on `POST /api/chat/completions`** — turns the one working
   chat route into server-side RAG by attaching a knowledge base / file.
3. **`GET /api/config`** ✅ — cheap capability/identity probe at SDK init.
4. **`GET /api/v1/models/list`** ✅ — richer per-model metadata (capabilities,
   attached knowledge) than the flat `/api/models`, for smarter routing.

---

## Chat & models

| Method · Path | Status | Notes |
|---|---|---|
| `POST /api/chat/completions` | ✅ | The only working chat route (Open WebUI → LiteLLM). Native tool-calling on capable models. |
| `GET /api/models` | ✅ | Flat OpenAI-style list (what the SDK uses today). |
| `GET /api/v1/models/list` | ✅ | **Richer**: per-model `meta` (incl. attached `knowledge`), `params`, `access_grants`, capabilities. Use for auto-routing. |
| `GET /api/v1/models/model?id=<id>` | 📄 | Single-model detail. |
| `GET /openai/models` | ✅ | Model list via the `/openai` router (listing works even though inference passthrough is disabled — see below). |

## Server-side RAG / retrieval ⭐ (the headline find)

Query a knowledge collection directly and get chunks back, **without an LLM
round-trip**. The gateway does embedding + vector search. Both query endpoints
are verified-user (not admin), so they work for the SDK token even though the
RAG *config* endpoint is admin-gated.

| Method · Path | Status | Notes |
|---|---|---|
| `POST /api/v1/retrieval/query/collection` | ✅ | Search across **multiple** collections. |
| `POST /api/v1/retrieval/query/doc` | ✅ | Same, **single** collection. |
| `POST /api/v1/retrieval/process/{file,text,web,youtube}` | 📄 | Embed/ingest a file id, raw text, URL, or YouTube transcript into a collection (verified-user per source; not load-tested here). |
| `GET/POST /api/v1/retrieval/config`, `/embedding` | 🔒 | RAG + embedding-model config — admin (returns 401 here). |

**`query/collection` request** (`QueryCollectionsForm`):
```json
{ "collection_names": ["<knowledge_id>"], "query": "text",
  "k": 4, "k_reranker": null, "r": null, "hybrid": false }
```
`query/doc` is identical with `"collection_name": "<id>"` (singular).
**Response** (Chroma-style parallel arrays):
```json
{ "documents": [["chunk text", ...]],
  "distances": [[0.58, 0.59, ...]],
  "metadatas": [[{...}, ...]] }
```
*Caveat:* `hybrid: true` (BM25 + dense) has a known regression on some builds
(`'QueryCollectionsForm' object has no attribute 'hybrid_bm25_weight'`,
open-webui#14432) — test before relying on it; dense-only is safe.

### Chat-with-collections (server-side RAG through the chat route)

Attach a knowledge base or file to `POST /api/chat/completions` with a top-level
`files` array; the gateway retrieves and injects chunks server-side:
```json
{ "model": "<model>",
  "messages": [{"role": "user", "content": "..."}],
  "files": [ {"type": "collection", "id": "<knowledge_id>"},
             {"type": "file",       "id": "<file_id>"} ] }
```
The resource must be fully indexed first — poll `GET /api/v1/knowledge/{id}`
until its status is processed. (The SDK already merges collection IDs into the
chat `extra_body`; this is the documented payload shape.)

## Files & knowledge lifecycle

All verified-user (subject to per-resource access control). The SDK already wraps
upload / create / single-add; the **gaps** worth adding are batch-add, remove,
and update.

| Method · Path | Status | Notes |
|---|---|---|
| `POST /api/v1/files/` · `GET /api/v1/files/` | ✅ | Upload (multipart) · list. |
| `GET /api/v1/files/{id}` · `/{id}/content` · `DELETE …/{id}` | 📄 | Metadata · download · delete. |
| `POST /api/v1/knowledge/create` · `GET /api/v1/knowledge/` | ✅ | Create · list collections. |
| `GET /api/v1/knowledge/{id}` | ✅ | Collection + **processing status** (poll before referencing). |
| `POST /api/v1/knowledge/{id}/files/batch/add` | 📄 | **Batch** add (gap vs current single-add). |
| `POST /api/v1/knowledge/{id}/file/{add,remove}` · `/update` · `DELETE …/delete` | 📄 | Add / remove file · rename · delete. |

## Capability discovery

| Method · Path | Status | Notes |
|---|---|---|
| `GET /api/config` | ✅ (public) | Gateway identity + a `features` block. **On 0.9.6 this only exposed auth/UI flags** (`auth`, `enable_signup`, `enable_websocket`, …) — *not* the richer `enable_web_search` / `enable_image_generation` / `enable_code_execution` dict that newer Open WebUI returns. Treat the rich flags as version-dependent. |
| `GET /api/version` · `GET /health` | ✅ | Version (`0.9.6`) · liveness. |

On this build, the real RAG/audio/image enablement lives behind the **admin**
`*/config` endpoints (below), so a non-admin client can't read it — discover
capability by *probing the endpoint* and handling 401/403, rather than trusting
`/api/config.features`.

## Tasks — offload housekeeping LLM calls

| Method · Path | Status | Notes |
|---|---|---|
| `GET /api/v1/tasks/config` | ✅ | Reveals the task model (`TASK_MODEL = llama3.2:latest`) + prompt templates. |
| `POST /api/v1/tasks/{title,tags,queries,auto,follow_up,…}/completions` | 📄 | Gateway-side generation of chat titles, tags, **search/retrieval queries**, autocompletes, follow-ups. `queries/completions` is a handy server-side query-rewriter to feed the retrieval endpoint above. |

## Tools / functions / pipelines

| Method · Path | Status | Notes |
|---|---|---|
| `GET /api/v1/tools/` · `GET /api/v1/functions/` | ✅ | Enumerate server-registered tools / pipes (read). |
| `GET /api/v1/pipelines/` | 🔒 | Admin. |

Registration is largely admin-gated, and these **execute only inside the
chat-completions pipeline** (a tool runs when a model calls it; functions are
middleware) — not as standalone API calls. Low priority on a locked-down instance.

## Audio / images

| Method · Path | Status | Notes |
|---|---|---|
| `POST /api/v1/audio/transcriptions` (STT) · `/speech` (TTS) | 📄 | Verified-user per source, but only work if the operator configured a backend. |
| `POST /api/v1/images/generations` | 📄 | Gated behind `features.enable_image_generation` + a backend. |
| `GET/POST /api/v1/{audio,images}/config` | 🔒 | Admin (401 here). |

## Memory / prompts

| Method · Path | Status | Notes |
|---|---|---|
| `GET /api/v1/memories/` (+ `/add`, `/query`, `/{id}/update`, delete) | ✅ | Per-user vector store for persistent context (empty here). |
| `GET /api/v1/prompts/` (+ `/create`, `/command/{cmd}`) | ✅ | Reusable server-side prompt templates. |

---

## Locked down on the reference gateway (don't bother)

The operator disabled every route that would bypass Open WebUI's own pipeline:

| Method · Path | Result | Consequence |
|---|---|---|
| `POST /ollama/api/*` (raw Ollama) | 🚫 503 "Ollama API is disabled" | No native Ollama access — and **this is what blocks per-token logprobs** (Ollama ≥ 0.12.11 returns them natively, but only on this route). |
| `* /openai/v1/*` (OpenAI passthrough) | 🚫 403 `ENABLE_OPENAI_API_PASSTHROUGH=False` | No direct-passthrough chat/completions. |
| `POST /v1/completions`, `/api/completions` (legacy) | 🚫 405 | Legacy text-completions not exposed. |

See the data-science README's *Token logprobs* section for the full logprobs
finding: general models return no logprobs through `/api/chat/completions`
(LiteLLM drops them); only `gpt-oss:120b` does. Enabling either bypass above is
an operator setting, not a client-side fix.

---

## Re-probing

The probes used to produce this table are read-only `GET`s plus the retrieval
`query` (a read). To re-check after a gateway change, hit `GET /api/config`,
`GET /api/v1/models/list`, and `POST /api/v1/retrieval/query/collection` against
a known collection id from `GET /api/v1/knowledge/`. The instance's own
`GET /api/v1/docs` (Swagger UI) is the authoritative contract for the deployed
version.
