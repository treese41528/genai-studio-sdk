"""
ReAct — EXACT replication on real HotpotQA over the live Wikipedia API
(Yao et al., ICLR 2023), swept across the gateway's models.

Unlike `react_eval.py` (a fictional mini-corpus), this faithfully reproduces the
paper's protocol (see `paper_react.py` for the mechanics):
- Real HotpotQA dev questions (HuggingFace), multi-hop, question-only; default
  n=500 (the paper's dev subset).
- The paper's exact action space over LIVE Wikipedia: search[entity] returns the
  first sentences of the page (or top-5 similar titles if missing); lookup[keyword]
  is a stateful Ctrl-F over the current page; the agent finishes via Finish[answer].
- The paper's prompting: 6-shot in-context exemplars (the verbatim webthink /
  webact / cotqa / webqa _simple6 prompts), greedy decoding (temperature=0), max 7
  thought-action steps, no forced answer on exhaustion.
- Strict SQuAD-style Exact-Match (==, no substring credit).

The SAME `genai_studio.agents.Agent` runs all four conditions; only the
ModelClient is swapped to `PaperReActClient` (text Thought/Action/Observation
grammar) — the framework's signature move. So the numbers ARE comparable to the
paper's Table 1 (PaLM-540B): Standard 28.7, CoT 29.4, Act 25.7, ReAct 27.4
(HotpotQA EM). The only change is the model — the point of the sweep.

Wikipedia calls do NOT hit the rate-limited gateway; only LLM calls do (set
GENAI_STUDIO_RPM to pace them).

Run:  python benchmarks/react_exact.py [n_questions] [model] [conditions]
e.g.: python benchmarks/react_exact.py 500 qwen2.5:72b act,react
"""

from __future__ import annotations

import json
import os
import re

import httpx

from genai_studio.agents import Agent, tool
from genai_studio.agents.trace import JsonlTracer
from _bench import Task, extract_final_text, make_client, run_suite
from paper_react import (
    PaperReActClient, build_system, USES_TOOLS, extract_answer, strict_em,
)

_UA = {"User-Agent": "genai-studio-sdk/1.0 (research; agent benchmark)"}
_CACHE = os.path.join(os.path.dirname(__file__), "_data")


# ════════════════════════════════════════════════════════════════════════════
# HotpotQA loader (real dev questions, cached locally)
# ════════════════════════════════════════════════════════════════════════════
def load_hotpotqa(n: int) -> list[dict]:
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"hotpotqa_dev_{n}.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows, offset = [], 0
    while len(rows) < n:
        r = httpx.get("https://datasets-server.huggingface.co/rows",
                      params={"dataset": "hotpotqa/hotpot_qa", "config": "fullwiki",
                              "split": "validation", "offset": offset, "length": min(100, n - len(rows))},
                      headers=_UA, timeout=60)
        r.raise_for_status()
        batch = r.json().get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item["row"]
            rows.append({"id": row.get("id", str(offset + len(rows))),
                         "question": row["question"], "answer": row["answer"],
                         "level": row.get("level")})
        offset += len(batch)
    rows = rows[:n]
    json.dump(rows, open(path, "w"))
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Live Wikipedia environment — the paper's exact search/lookup semantics
# ════════════════════════════════════════════════════════════════════════════
class WikipediaEnv:
    def __init__(self):
        self.title = None
        self.sentences: list[str] = []
        self._lookup_key = None
        self._lookup_pos = 0
        self._client = httpx.Client(headers=_UA, timeout=30, follow_redirects=True)

    @staticmethod
    def _split(text: str) -> list[str]:
        text = re.sub(r"\n+", " ", text)
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _fetch_page(self, entity: str):
        """Return (title, plaintext) for an exact Wikipedia page, or None."""
        r = self._client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
            "titles": entity, "format": "json"})
        pages = r.json().get("query", {}).get("pages", {})
        for _, p in pages.items():
            if "missing" in p or not p.get("extract"):
                return None
            return p["title"], p["extract"]
        return None

    def _similar(self, entity: str) -> list[str]:
        r = self._client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": entity,
            "srlimit": 5, "format": "json"})
        return [h["title"] for h in r.json().get("query", {}).get("search", [])]

    def search(self, entity: str) -> str:
        page = self._fetch_page(entity)
        if page is None:
            sims = self._similar(entity)
            return f"Could not find [{entity}]. Similar: {sims}"
        self.title, self.sentences = page[0], self._split(page[1])
        self._lookup_key, self._lookup_pos = None, 0
        return " ".join(self.sentences[:5])

    def lookup(self, keyword: str) -> str:
        if self.title is None:
            return "No page loaded. Use search[entity] first."
        if keyword != self._lookup_key:
            self._lookup_key, self._lookup_pos = keyword, 0
        matches = [s for s in self.sentences if keyword.lower() in s.lower()]
        if not matches:
            return f"No results for '{keyword}'."
        if self._lookup_pos >= len(matches):
            return f"No more results for '{keyword}'."
        s = matches[self._lookup_pos]
        self._lookup_pos += 1
        return f"(Result {self._lookup_pos} / {len(matches)}) {s}"


def make_wiki_tools(env: WikipediaEnv):
    @tool
    def search(entity: str) -> str:
        """Search Wikipedia for an entity and return the first sentences of its
        page (or a list of similar page titles if it does not exist).

        Args:
            entity: the exact title of the Wikipedia page to look up.
        """
        return env.search(entity)

    @tool
    def lookup(keyword: str) -> str:
        """Return the next sentence containing `keyword` on the most recently
        searched page (like Ctrl-F).

        Args:
            keyword: a substring to find on the current page.
        """
        return env.lookup(keyword)

    return [search, lookup]


# ════════════════════════════════════════════════════════════════════════════
# Grading — strict SQuAD/HotpotQA Exact-Match (no substring credit)
# ════════════════════════════════════════════════════════════════════════════
def grade_em(gold: str):
    def _g(result, workdir):
        ans = extract_answer(extract_final_text(result))
        ok = strict_em(ans, gold)
        return (1.0 if ok else 0.0, f"got={ans[:40]!r} gold={gold!r}")
    return _g


def build_tasks(n: int) -> list[Task]:
    tasks = []
    for q in load_hotpotqa(n):
        tasks.append(Task(id=q["id"][:12], prompt=f"Question: {q['question']}",
                          grade=grade_em(q["answer"]),
                          meta={"answer": q["answer"], "level": q["level"]}))
    return tasks


# ── the paper's four conditions, run via the SAME Agent + PaperReActClient ───
# Greedy decoding (temperature=0), 6-shot exemplars in the system prompt, max 7
# thought-action steps, and NO forced answer on exhaustion — exactly the paper.
# Thinking models emit long <think> blocks and DEGENERATE under greedy decoding
# (temp=0) — Qwen/DeepSeek explicitly recommend temp~0.6 — so greedy returns empty/
# repetitive completions. For these models ONLY we make a documented exception:
# sampled decoding (temp 0.6, top_p 0.95) + a large token budget so the <think> block
# can complete. Non-thinking models stay paper-faithful greedy. (gpt-oss is a reasoning
# model too but handles greedy fine on this gateway, so it is NOT in this set.)
THINKING_MODELS = ("qwen3", "deepseek-r1", "qwq")


def _is_thinking(model: str) -> bool:
    m = model.lower()
    return any(t in m for t in THINKING_MODELS)


def make_agent_for(condition: str, model: str):
    use_tools = USES_TOOLS[condition]
    thinking = _is_thinking(model)
    # Stop sequences keep NON-thinking models from hallucinating observations / running
    # into the next exemplar. Thinking models trip "\nObservation" before they emit an
    # action (→ empty completions), so they get NO stops; <think>-strip + first-action
    # parsing + the token budget keep their output clean instead.
    if thinking:
        stop = []
    else:
        stop = ["\nObservation", "\nQuestion"] if use_tools else ["\nQuestion"]
    temperature = 0.6 if thinking else 0.0       # greedy is faithful; thinking needs sampling
    top_p = 0.95 if thinking else None
    max_tokens = 8192 if thinking else 1024      # budget for the <think> block

    def factory(task, workdir):
        client = PaperReActClient(make_client(model=model), condition=condition,
                                  stop=stop, max_tokens=max_tokens, top_p=top_p)
        tools = make_wiki_tools(WikipediaEnv()) if use_tools else []
        return Agent(client=client, tools=tools, system=build_system(condition),
                     temperature=temperature, max_steps=7, force_final_answer=False,
                     tracer=JsonlTracer(os.path.join(workdir, "trace.jsonl")))
    return factory


# default model sweep (de-duplicated, ordered flagships → fast → slow reasoning)
MODELS = [
    "qwen2.5:72b", "llama3.3:70b", "gpt-oss:120b", "llama4:latest",  # large flagships
    "gemma3:27b", "gemma3:1b", "qwen3:4b", "llama3.2:latest",  # mid + small (fast)
    "deepseek-r1:32b", "qwq:latest",                       # reasoning (slow) last
]


def _results_path(n):
    os.makedirs(os.path.join(_CACHE, "..", "_results"), exist_ok=True)
    return os.path.join(os.path.dirname(__file__), "_results", f"hotpotqa_paper_n{n}.json")


def _load_results(n):
    p = _results_path(n)
    return json.load(open(p)) if os.path.exists(p) else {}


def _save_results(n, results):
    # atomic write: a kill mid-write can't corrupt the results file
    p = _results_path(n)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, p)


def _partial_path(n, model, cond):
    d = os.path.join(os.path.dirname(__file__), "_results", "partial")
    os.makedirs(d, exist_ok=True)
    safe = f"paper_n{n}_{model}__{cond}".replace(":", "-").replace("/", "-").replace("|", "-")
    return os.path.join(d, safe + ".jsonl")


def _load_partial(path) -> dict:
    """Per-question checkpoint: {question_id -> outcome dict} from a prior run."""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                o = json.loads(line)
                out[o["id"]] = o
            except (ValueError, KeyError):
                continue
    return out


def _print_grid(conditions, models, results):
    print("\n==== HotpotQA EM% (paper PaLM-540B: Std 28.7 / CoT 29.4 / Act 25.7 / ReAct 27.4) ====")
    print(f"{'model':20} " + " ".join(f"{c:>10}" for c in conditions))
    for m in models:
        cells = []
        for c in conditions:
            r = results.get(f"{m}|{c}")
            cells.append(f"{r['fair_em']*100:7.1f}% " if r else f"{'·':>9} ")
        print(f"{m:20} " + " ".join(cells))


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS
    conditions = sys.argv[3].split(",") if len(sys.argv) > 3 else ["standard", "cot", "act", "react"]
    runs = int(os.getenv("BENCH_RUNS", "1"))

    tasks = build_tasks(n)
    results = _load_results(n)  # resume: skip completed (model|condition) cells
    print(f"HotpotQA EXACT: {len(tasks)} real dev questions | {len(models)} models x "
          f"{len(conditions)} conditions | already done: {len(results)} cells")

    for model in models:
        for cond in conditions:
            key = f"{model}|{cond}"
            if key in results:
                continue  # cell already complete

            # Per-question checkpoint: resume mid-cell after an interruption.
            ppath = _partial_path(n, model, cond)
            cached = _load_partial(ppath)          # {qid -> outcome} already done
            pf = open(ppath, "a")

            def _checkpoint(o, _pf=pf):
                _pf.write(json.dumps({"id": o.task_id, "score": o.score,
                                      "passed": o.passed, "error": bool(o.error)}) + "\n")
                _pf.flush()

            rep = run_suite(f"HotpotQA[{key}]", tasks, make_agent_for(cond, model),
                            runs=runs, quiet=True, done_ids=set(cached), on_outcome=_checkpoint)
            pf.close()

            # combine resumed (cached) + freshly-run outcomes
            outs = list(cached.values()) + [
                {"passed": o.passed, "error": bool(o.error), "score": o.score}
                for o in rep["outcomes"]]
            infra = sum(1 for o in outs if o["error"])
            correct = sum(1 for o in outs if o["passed"])
            valid = len(outs) - infra
            results[key] = {"em": (correct + 0.0) / len(outs) if outs else 0.0,
                            "fair_em": (correct / valid if valid else 0.0),
                            "correct": correct, "valid": valid, "infra": infra, "n": len(outs)}
            _save_results(n, results)              # atomic; cell now durably complete
            os.remove(ppath)                       # clear the per-question checkpoint
            print(f"  -> {key}: fair_EM={results[key]['fair_em']*100:.1f}% "
                  f"(correct {correct}/{valid}, infra {infra}, resumed {len(cached)})")
        _print_grid(conditions, models, results)
    _print_grid(conditions, models, results)
