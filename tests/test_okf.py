"""Tests for the OKF (Open Knowledge Format) loader."""

from __future__ import annotations

import pytest

from genai_studio.agents.knowledge import OKFDoc, load_bundle
from genai_studio.agents.knowledge.okf import _kb_ready, _mini_yaml, _split_frontmatter


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ── load_bundle ──────────────────────────────────────────────────────────────
def test_load_bundle_parses_concepts_and_skips_reserved(tmp_path):
    _write(tmp_path, "population.md",
           "---\ntype: metric\ntitle: Population\nresource: https://ex.org/pop\n"
           "tags: [demography, counts]\n---\nThe number of people.")
    _write(tmp_path, "geo/place.md", "---\ntype: entity\n---\nA place.")
    _write(tmp_path, "index.md", "---\ntype: index\n---\nreserved")   # must be skipped
    _write(tmp_path, "log.md", "---\ntype: log\n---\nreserved")        # must be skipped
    _write(tmp_path, "notes.txt", "not markdown")                      # ignored

    docs = load_bundle(str(tmp_path))
    ids = [d.identity for d in docs]
    assert ids == ["geo/place", "population"]                          # sorted, posix ids, no reserved
    pop = next(d for d in docs if d.identity == "population")
    assert pop.type == "metric" and pop.title == "Population"
    assert pop.tags == ["demography", "counts"]
    assert pop.body.strip() == "The number of people."


def test_load_bundle_include_reserved(tmp_path):
    _write(tmp_path, "a.md", "---\ntype: x\n---\nbody")
    _write(tmp_path, "index.md", "---\ntype: index\n---\nidx")
    ids = {d.identity for d in load_bundle(str(tmp_path), include_reserved=True)}
    assert ids == {"a", "index"}


def test_nested_index_md_is_a_normal_concept(tmp_path):
    # 'index.md'/'log.md' are reserved at the ROOT only; sub/index.md is a real concept
    _write(tmp_path, "index.md", "---\ntype: index\n---\nroot index")
    _write(tmp_path, "sub/index.md", "---\ntype: topic\n---\nnested concept")
    ids = {d.identity for d in load_bundle(str(tmp_path))}
    assert ids == {"sub/index"}                                       # root skipped, nested kept


def test_bom_file_is_loaded_not_dropped(tmp_path):
    # Windows editors emit a UTF-8 BOM; utf-8-sig must strip it so 'type' parses
    (tmp_path / "c.md").write_text("---\ntype: metric\n---\nbody", encoding="utf-8-sig")
    docs = load_bundle(str(tmp_path))
    assert [d.type for d in docs] == ["metric"]


def test_non_utf8_file_skipped_not_crash(tmp_path):
    _write(tmp_path, "good.md", "---\ntype: ok\n---\nb")
    (tmp_path / "bad.md").write_bytes(b"---\ntype: x\n---\n\xff\xfe invalid")
    with pytest.warns(UserWarning, match="unreadable"):
        docs = load_bundle(str(tmp_path))
    assert [d.identity for d in docs] == ["good"]                     # bad skipped, good survives


def test_missing_type_is_skipped_or_strict_raises(tmp_path):
    _write(tmp_path, "good.md", "---\ntype: ok\n---\nb")
    _write(tmp_path, "bad.md", "---\ntitle: no type here\n---\nb")
    with pytest.warns(UserWarning, match="missing the required"):
        docs = load_bundle(str(tmp_path))
    assert [d.identity for d in docs] == ["good"]                      # bad skipped
    with pytest.raises(ValueError, match="missing the required"):
        load_bundle(str(tmp_path), strict=True)


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_bundle(str(tmp_path / "nope"))


def test_no_frontmatter_doc_is_skipped_nonstrict(tmp_path):
    _write(tmp_path, "plain.md", "just body, no frontmatter")
    with pytest.warns(UserWarning, match="missing the required"):
        assert load_bundle(str(tmp_path)) == []                        # no type -> skipped


# ── to_source ────────────────────────────────────────────────────────────────
def test_to_source_maps_provenance():
    doc = OKFDoc(identity="pop", type="metric",
                 body="People counted in a census.",
                 frontmatter={"title": "Population", "resource": "https://ex.org/p",
                              "tags": ["demography"], "description": "head count"})
    s = doc.to_source()
    assert s.id == "pop" and s.title == "Population" and s.url == "https://ex.org/p"
    assert s.metadata["type"] == "metric" and s.metadata["tags"] == ["demography"]
    assert s.metadata["description"] == "head count"
    assert s.snippet.startswith("People counted")


def test_to_source_defaults_title_to_identity():
    assert OKFDoc(identity="x/y", type="t").to_source().title == "x/y"


# ── frontmatter parsing ──────────────────────────────────────────────────────
def test_split_frontmatter_without_block_returns_body():
    fm, body = _split_frontmatter("no fence here\nsecond line")
    assert fm == {} and body == "no fence here\nsecond line"


def test_split_frontmatter_unclosed_fence_is_all_body():
    text = "---\ntype: x\nnever closed"
    fm, body = _split_frontmatter(text)
    assert fm == {} and body == text


def test_mini_yaml_scalars_inline_and_block_lists():
    out = _mini_yaml(
        'type: metric\n'
        'title: "Quoted Title"\n'
        'tags: [a, b, c]\n'
        'authors:\n'
        '  - Ada\n'
        '  - Grace\n'
        '# a comment\n'
    )
    assert out["type"] == "metric"
    assert out["title"] == "Quoted Title"                             # quotes stripped
    assert out["tags"] == ["a", "b", "c"]                            # inline list
    assert out["authors"] == ["Ada", "Grace"]                        # block list


def test_mini_yaml_used_when_pyyaml_absent(tmp_path, monkeypatch):
    # force the fallback by making `import yaml` fail
    import builtins
    real_import = builtins.__import__

    def no_yaml(name, *a, **k):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    _write(tmp_path, "c.md", "---\ntype: metric\ntags: [x, y]\n---\nbody")
    docs = load_bundle(str(tmp_path))
    assert docs[0].type == "metric" and docs[0].tags == ["x", "y"]


# ── ingest readiness check ───────────────────────────────────────────────────
def test_kb_ready_logic():
    class KB:
        def __init__(self, raw):
            self.raw = raw

    assert _kb_ready(KB({"files": [{"status": "processed"}, {"status": "completed"}]}), 2) is True
    assert _kb_ready(KB({"files": [{"status": "pending"}, {"status": "processed"}]}), 2) is False
    assert _kb_ready(KB({"files": [{}, {}]}), 2) is True              # no status -> present is enough
    assert _kb_ready(KB({"files": [{}]}), 2) is False                 # too few files
    assert _kb_ready(KB({}), 1) is False                              # no files key
    assert _kb_ready(KB({"files": [{"status": None}]}), 1) is False   # explicit null status -> pending
