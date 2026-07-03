"""P1 embedding rerank — shared embedder, embedding_rank (deferred), memory embed-at-write.
All offline via a mock studio; embeddings always FAIL OPEN to the keyword floor."""

from __future__ import annotations

from genai_studio.agents import can_embed, embedding_rank, make_embedder
from genai_studio.agents.memory import MemoryStore, make_memory_tools


class MockStudio:
    """A stand-in for GenAIStudio.embed: maps exact strings to vectors (deterministic)."""

    def __init__(self, table=None, fail=False):
        self.table = table or {}
        self.fail = fail
        self.calls = 0

    def embed(self, text, model=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embed endpoint down")
        if isinstance(text, list):
            return [self._vec(t) for t in text]
        return self._vec(text)

    def _vec(self, t):
        return self.table.get(t, [float(len(t) % 5) + 0.1, float(sum(map(ord, t)) % 7) + 0.1, 1.0])


# ── model split ──────────────────────────────────────────────────────────────
def test_can_embed_split():
    assert can_embed("llama3.2:latest") and can_embed("qwen2.5:72b") and can_embed("mistral:latest")
    for no in ("gemma3:27b", "llama4:latest", "qwen3:4b", "gpt-oss:120b"):
        assert not can_embed(no)


# ── the shared embedder ──────────────────────────────────────────────────────
def test_make_embedder_none_and_failopen():
    assert make_embedder(None) is None                     # no studio -> no embedder (keyword mode)
    e = make_embedder(MockStudio(fail=True))
    assert e("x") is None                                  # any failure -> None (caller falls back)


# ── embedding_rank (deferred tools) ──────────────────────────────────────────
_CAT = [("read_parquet", "Read a parquet table"), ("send_email", "Send an email")]
_TABLE = {
    "read_parquet: Read a parquet table": [1.0, 0.0],
    "send_email: Send an email": [0.0, 1.0],
    "load a data table from disk": [0.9, 0.1],             # query -> closest to read_parquet
}


def test_embedding_rank_orders_by_cosine_and_caches():
    st = MockStudio(_TABLE)
    rank = embedding_rank(st)
    assert rank("load a data table from disk", _CAT, 2)[0] == "read_parquet"
    before = st.calls
    rank("load a data table from disk", _CAT, 2)           # catalog cached -> only the query re-embeds
    assert st.calls == before + 1


def test_embedding_rank_fails_open_to_keyword():
    rank = embedding_rank(MockStudio(fail=True))
    assert rank("parquet table", _CAT, 2) == ["read_parquet"]   # == keyword_rank result


def test_embedding_rank_none_studio_is_keyword():
    rank = embedding_rank(None)
    assert rank("send an email", _CAT, 2) == ["send_email"]


# ── memory embed-at-write ────────────────────────────────────────────────────
def test_memory_embeds_at_write(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    write, recall_tool = make_memory_tools(store, studio=MockStudio())
    write.run({"fact": "the deploy script needs the prod flag", "tags": ["deploy"]})
    f = store.live()[0]
    assert f.vec is not None and f.embed_model == "llama3.2:latest"   # embedded once at write
    assert "deploy" in recall_tool.run({"query": "how do I deploy", "k": 5}).content


def test_memory_keyword_only_when_no_studio(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    write, _ = make_memory_tools(store)                    # no studio -> no vectors
    write.run({"fact": "prefer tabs over spaces", "tags": ["style"]})
    assert store.live()[0].vec is None
