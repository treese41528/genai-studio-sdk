"""
Real benchmark loaders for agentic_eval — grounds the eval in recognized datasets
instead of hand-authored tasks. Each loader downloads its (small, ungated) source
on first use, caches it under ``_data/benchmarks/``, takes a DETERMINISTIC sample,
and maps rows onto ``agentic_eval.Task``:

- **gsm8k**      — grade-school math word problems; exact numeric gold (the final
  number after ``####``). Tests reasoning + tool-use + pass^k.  [MIT, OpenAI]
- **truthfulqa** — adversarial questions targeting common misconceptions; judge-
  graded against the correct/incorrect answer sets.            [Apache-2.0]
- **simpleqa**   — short, obscure factual questions models often get wrong; the
  canonical hallucination benchmark. Hard enough that grounding can actually help.
  Judge-graded against the gold answer.                        [MIT, OpenAI]
- **musique**    — MULTI-HOP questions (2-4 reasoning hops) with short entity golds;
  tests whether retrieval/grounding lets the agent chain facts it can't recall.
  Hybrid-graded (short gold -> deterministic floor, residual -> judge).  [CC-BY-4.0]
- **bfcl** / **bfcl_irrelevance** — Berkeley Function-Calling Leaderboard v3: the agent
  gets the function(s) as tools and is graded on the CALL it emits (deterministic AST
  match vs gold; for irrelevance, correct == calling NO function). Call-graded, not
  text-graded — condition-independent (run with one condition).         [Apache-2.0]

These are JUDGE-graded by design (free-form answers); the deterministic floor is
only meaningful for gsm8k (exact numbers). Data is downloaded, not vendored, to
avoid committing answer keys; see SOURCES for provenance/licences.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random

import httpx

_DATA = os.path.join(os.path.dirname(__file__), "_data", "benchmarks")
SOURCES = {
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl",
    "truthfulqa": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
    "simpleqa": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
    "musique": "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/dev.jsonl",
}


def _fetch_url(url: str, cache_name: str) -> str:
    """Download + cache raw dataset text by URL (idempotent)."""
    os.makedirs(_DATA, exist_ok=True)
    path = os.path.join(_DATA, cache_name)
    if not os.path.exists(path):
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fetch(name: str) -> str:
    ext = os.path.splitext(SOURCES[name])[1] or ".txt"
    return _fetch_url(SOURCES[name], name + ext)


def _sample(items: list, n: int, seed: int) -> list:
    if n <= 0 or n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def load_gsm8k(n: int = 20, *, seed: int = 0) -> list:
    from agentic_eval import Task
    rows = [json.loads(ln) for ln in _fetch("gsm8k").splitlines() if ln.strip()]
    out = []
    for i, r in enumerate(_sample(rows, n, seed)):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            num = float(gold)
        except ValueError:
            continue
        out.append(Task(f"gsm_{i}", "gsm8k", r["question"], num=num, tol=1e-6, note=gold))
    return out


def load_truthfulqa(n: int = 20, *, seed: int = 0) -> list:
    from agentic_eval import Task
    rows = list(csv.DictReader(io.StringIO(_fetch("truthfulqa"))))
    out = []
    for i, r in enumerate(_sample(rows, n, seed)):
        note = (f"CORRECT answers: {r.get('Correct Answers', '').strip()} || "
                f"INCORRECT answers: {r.get('Incorrect Answers', '').strip()}")
        out.append(Task(f"tqa_{i}", "truthfulqa", r["Question"].strip(), note=note))
    return out


def load_simpleqa(n: int = 20, *, seed: int = 0) -> list:
    from agentic_eval import Task
    rows = list(csv.DictReader(io.StringIO(_fetch("simpleqa"))))
    out = []
    for i, r in enumerate(_sample(rows, n, seed)):
        ans = (r.get("answer") or "").strip()
        # accept= gives the deterministic floor a substring to match; the judge uses note.
        out.append(Task(f"sqa_{i}", "simpleqa", r["problem"].strip(),
                        accept=(ans,) if ans else (), note=ans))
    return out


def load_musique(n: int = 20, *, seed: int = 0) -> list:
    from agentic_eval import Task
    rows = [json.loads(ln) for ln in _fetch("musique").splitlines() if ln.strip()]
    out = []
    for i, r in enumerate(_sample(rows, n, seed)):
        gold = [a for a in (r.get("golden_answers") or []) if a]
        if not gold:
            continue
        out.append(Task(f"musq_{i}", "musique", r["question"].strip(),
                        accept=tuple(gold), note="; ".join(gold)))
    return out


_BFCL = ("https://huggingface.co/datasets/gorilla-llm/"
         "Berkeley-Function-Calling-Leaderboard/resolve/main")


def _bfcl_rows(cat: str, answers: bool = False) -> list:
    sub = f"possible_answer/BFCL_v3_{cat}.json" if answers else f"BFCL_v3_{cat}.json"
    cache = f"bfcl_{cat}{'_ans' if answers else ''}.json"
    return [json.loads(ln) for ln in _fetch_url(f"{_BFCL}/{sub}", cache).splitlines() if ln.strip()]


def load_bfcl(n: int = 20, *, seed: int = 0, category: str = "simple") -> list:
    """BFCL (Berkeley Function-Calling Leaderboard) v3 — tool-call correctness. The
    agent is given the function(s) as tools and graded on the CALL it emits (AST match
    vs gold for 'simple'; for 'irrelevance', correct == calling NO function)."""
    from agentic_eval import Task
    q = {r["id"]: r for r in _bfcl_rows(category)}
    if category == "irrelevance":
        out = []
        for r in _sample(list(q.values()), n, seed):
            out.append(Task(r["id"], "bfcl_irrelevance", r["question"][0][0]["content"].strip(),
                            note="no relevant function -> do not call",
                            meta={"functions": r["function"], "kind": "irrelevance"}))
        return out
    ans = {r["id"]: r for r in _bfcl_rows(category, answers=True)}
    rows = [r for r in q.values() if r["id"] in ans]
    out = []
    for r in _sample(rows, n, seed):
        gold = ans[r["id"]]["ground_truth"]
        out.append(Task(r["id"], "bfcl", r["question"][0][0]["content"].strip(),
                        note=json.dumps(gold)[:140],
                        meta={"functions": r["function"], "kind": "ast", "gold": gold}))
    return out


LOADERS = {"gsm8k": load_gsm8k, "truthfulqa": load_truthfulqa, "simpleqa": load_simpleqa,
           "musique": load_musique,
           "bfcl": lambda n, seed=0: load_bfcl(n, seed=seed, category="simple"),
           "bfcl_irrelevance": lambda n, seed=0: load_bfcl(n, seed=seed, category="irrelevance")}


if __name__ == "__main__":   # smoke: fetch + parse 2 of each, print the mapped Tasks
    for name, loader in LOADERS.items():
        tasks = loader(2, seed=1)
        print(f"\n=== {name}: {len(tasks)} tasks ===")
        for t in tasks:
            print(f"  [{t.id}] gold={t.note[:60]!r}\n        Q: {t.prompt[:90]}")
