"""P0 recall-memory — JSONL store, dedup/supersede/forget/compact, keyword recall, index, tools."""

from __future__ import annotations

from genai_studio.agents.memory import (MemoryStore, make_memory_tools, memory_index_text,
                                         open_store, recall)


def _store(tmp_path):
    return MemoryStore(tmp_path / "mem.jsonl")


def test_add_and_live(tmp_path):
    s = _store(tmp_path)
    f = s.add("the deploy script needs --env prod", ["deploy"], ts=1.0)
    live = s.live()
    assert len(live) == 1 and live[0].text == "the deploy script needs --env prod"
    assert live[0].tags == ["deploy"] and live[0].id == f.id


def test_dedup_supersede(tmp_path):
    s = _store(tmp_path)
    s.add("the deploy script needs the prod environment flag", ["deploy"], ts=1.0)
    s.add("the deploy script needs the staging environment flag", ["deploy"], ts=2.0)  # same topic+tag
    live = s.live()
    assert len(live) == 1 and "staging" in live[0].text          # newer superseded the older


def test_distinct_facts_coexist(tmp_path):
    s = _store(tmp_path)
    s.add("the database lives in us-east-1", ["infra"], ts=1.0)
    s.add("the frontend deploys from main", ["deploy"], ts=2.0)
    assert len(s.live()) == 2


def test_forget_and_compact(tmp_path):
    s = _store(tmp_path)
    f = s.add("ephemeral note about nothing useful", ts=1.0)
    s.add("a keeper fact about the project layout", ts=2.0)
    s.forget(f.id)
    assert len(s.live()) == 1
    n_records_before = len(s.path.read_text().splitlines())
    assert s.compact() == 1
    assert len(s.path.read_text().splitlines()) == 1 < n_records_before  # history dropped


def test_recall_keyword_floor(tmp_path):
    s = _store(tmp_path)
    s.add("the deploy script needs the prod environment flag", ["deploy"], ts=1.0)
    s.add("the database lives in us-east-1", ["infra"], ts=2.0)
    hits = recall(s.live(), "how do I deploy to production", k=5)
    assert hits and hits[0][0].text.startswith("the deploy script")
    assert recall(s.live(), "completely unrelated xyzzy plugh", k=5) == []   # honest no-memory


def test_index_render_and_budget(tmp_path):
    s = _store(tmp_path)
    for i in range(5):
        s.add(f"fact number {i} with some descriptive content here", ts=float(i))
    idx = memory_index_text(s, budget_chars=120, max_facts=40)
    assert idx.startswith("# Recalled memory") and "may be stale" in idx
    assert "more; use recall_memory" in idx                       # overflow tail present


def test_index_empty_when_no_facts(tmp_path):
    assert memory_index_text(_store(tmp_path)) == ""


def test_tools_write_and_recall(tmp_path):
    s = _store(tmp_path)
    write, rec = make_memory_tools(s)
    out = write.run({"fact": "prefer pytest over unittest", "tags": ["style"]})
    assert "saved memory" in out.content and len(s.live()) == 1
    hit = rec.run({"query": "which test framework to prefer", "k": 5})
    assert "pytest" in hit.content
    assert "no relevant memory" in rec.run({"query": "zzz nothing matches qqq", "k": 5}).content


def test_open_store_is_cwd_keyed(tmp_path):
    a = open_store(tmp_path / "projA", tmp_path / "mem")
    b = open_store(tmp_path / "projB", tmp_path / "mem")
    assert a.path != b.path and a.path.parent.name == "projects"
