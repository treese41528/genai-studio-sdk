"""Append-only JSONL store of durable facts with dedup → supersede / forget / compact.

The JSONL is the source of truth: each line is a fact record, OR a tombstone
``{"id":.., "op":"forget"}``. A fact may carry ``"supersedes":[ids]`` to retire near-duplicates
it replaces. ``live()`` replays the log (last-write-wins per id; forgets + supersedes remove);
``compact()`` rewrites the file with only the live facts. One bad line is skipped, never fatal.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryFact:
    id: str
    text: str
    tags: list = field(default_factory=list)
    ts: float = 0.0
    source: str | None = None
    vec: list | None = None              # cached embedding (P1; None in keyword-only P0)
    embed_model: str | None = None

    def to_record(self) -> dict:
        return {"id": self.id, "text": self.text, "tags": self.tags, "ts": self.ts,
                "source": self.source, "vec": self.vec, "embed_model": self.embed_model}


def _fact_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def _tokens(s: str) -> set:
    """Lowercased alphanumeric tokens of length > 2 (a cheap stopword-ish floor)."""
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def open_store(cwd, memory_dir) -> "MemoryStore":
    """A per-project store under the global ``memory_dir`` (keyed by absolute cwd), so memory
    survives clean checkouts and is never accidentally committed."""
    key = hashlib.sha1(str(Path(cwd).resolve()).encode("utf-8")).hexdigest()[:12]
    return MemoryStore(Path(memory_dir) / "projects" / f"{key}.jsonl")


class MemoryStore:
    def __init__(self, path, *, dup_jaccard: float = 0.6, tagged_jaccard: float = 0.3):
        self.path = Path(path)
        self.dup_jaccard = dup_jaccard         # token-Jaccard >= this -> supersede (same topic)
        self.tagged_jaccard = tagged_jaccard   # ... or a shared tag and >= this

    def _records(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                       # skip one corrupt line, keep going
        return out

    def live(self) -> list:
        """Replay the log into the current live facts (in latest-write order)."""
        facts: dict = {}
        for r in self._records():
            if r.get("op") == "forget":
                facts.pop(r.get("id"), None)
                continue
            for sid in r.get("supersedes", []):
                facts.pop(sid, None)
            facts[r["id"]] = MemoryFact(
                id=r["id"], text=r["text"], tags=r.get("tags", []), ts=r.get("ts", 0.0),
                source=r.get("source"), vec=r.get("vec"), embed_model=r.get("embed_model"))
        return list(facts.values())

    def _append(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    def add(self, text, tags=(), *, source=None, ts=None, vec=None, embed_model=None) -> MemoryFact:
        """Add a fact, superseding near-duplicates (token-Jaccard >= ``dup_jaccard``, or a shared
        tag and >= ``tagged_jaccard``) — so an updated fact replaces the stale one. Returns it."""
        text = text.strip()
        tags = list(tags or [])
        ts = time.time() if ts is None else ts
        new_tok = _tokens(text)
        fact = MemoryFact(id=_fact_id(text), text=text, tags=tags, ts=ts, source=source,
                          vec=vec, embed_model=embed_model)
        supersedes = []
        for f in self.live():
            if f.id == fact.id:
                continue
            j = _jaccard(new_tok, _tokens(f.text))
            if j >= self.dup_jaccard or (set(tags) & set(f.tags) and j >= self.tagged_jaccard):
                supersedes.append(f.id)
        rec = fact.to_record()
        if supersedes:
            rec["supersedes"] = supersedes
        self._append(rec)
        return fact

    def forget(self, fact_id: str) -> None:
        self._append({"id": fact_id, "op": "forget"})

    def compact(self) -> int:
        """Rewrite the file with only the live facts (drop superseded/forgotten history). Atomic."""
        live = self.live()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for fact in live:
                f.write(json.dumps(fact.to_record()) + "\n")
        os.replace(tmp, self.path)
        return len(live)
