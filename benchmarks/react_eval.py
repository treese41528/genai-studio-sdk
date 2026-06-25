"""
ReAct replication — the four-condition ablation (Yao et al., ICLR 2023),
adapted to run offline against the gateway.

Faithful to the paper's design:
- Tools `search[entity]` / `lookup[string]` with the paper's exact semantics
  (first-5-sentences + similar-title fallback; stateful Ctrl-F lookup), over a
  small FROZEN local corpus instead of the live Wikipedia API.
- The corpus is FICTIONAL and internally consistent, so the answers cannot be
  recalled parametrically — the agent MUST retrieve and chain facts. This is what
  makes the ablation meaningful (real-world facts would let Standard/CoT cheat).
- Multi-hop QA (Exact Match) + claim verification (FEVER-style accuracy).

The four conditions map onto the SDK exactly:
  • standard       — no tools, direct answer (parametric only).
  • cot            — no tools, "think step by step".
  • act            — native tool-calling, NO reasoning instruction.
  • react          — native tool-calling, reasoning Thought before each action.
  • react_injected — the SAME react agent but via ReActClient (prompt-injected JSON
                     actions) instead of native tools — the framework's swap demo.

Expected headline finding to reproduce: act/react >> standard/cot (tools matter
on unknowable facts), and react ≥ act (reasoning helps chain the hops).

Run:  python benchmarks/react_eval.py [condition] [limit]
"""

from __future__ import annotations

import difflib
import re

import os

from genai_studio.agents import Agent, tool
from genai_studio.agents.trace import JsonlTracer
from _bench import extract_final_text, make_client, run_suite, Task


# ════════════════════════════════════════════════════════════════════════════
# Frozen fictional corpus (key facts in the first sentences so `search` reveals
# them; a couple of facts are deeper, to exercise `lookup`).
# ════════════════════════════════════════════════════════════════════════════
CORPUS = {
    "Lutz": "Lutz is the capital and largest city of the country of Zubrowka. "
            "It sits on the banks of the Nadel river. Lutz was founded in 1487.",
    "Zubrowka": "Zubrowka is a landlocked country in central Vorland. "
                "The official language of Zubrowka is Zubrish. Its capital is Lutz.",
    "Nadel": "The Nadel is a river in Zubrowka. It flows through the capital city "
             "Lutz. The Nadel empties into the Grey Sea.",
    "Lutenblag": "Lutenblag is the capital of the country of Molvania. "
                 "It was founded in 1612. The official language of Molvania is Molvanian.",
    "Molvania": "Molvania is a country in eastern Vorland. Its capital is Lutenblag. "
                "The official language of Molvania is Molvanian.",
    "Eska Vorn": "Dr. Eska Vorn was a Zubrowkan mathematician. She was born in "
                 "Zubrowka in 1901. Vorn is best known for writing the book "
                 "Theory of Folds.",
    "Hale Quist": "Professor Hale Quist was a Molvanian mathematician and physicist. "
                  "He was born in Molvania. Quist wrote the influential monograph "
                  "On Tessellations.",
    "The Crimson Vault": "The Crimson Vault is a 1997 mystery film. "
                         "It was directed by Mira Calo. The film is set in Lutz.",
    "Mira Calo": "Mira Calo is a film director. She was born in Zubrowka in 1959. "
                 "She is known for directing The Crimson Vault.",
    "Aurium": "Aurium is a dense yellow metallic element. Aurium has the atomic "
              "number 88. It is highly resistant to corrosion.",
    "Aurium Standard": "The Aurium Standard was a monetary system. Under it, a "
                       "currency's value was fixed to a quantity of the element "
                       "aurium. Many nations of Vorland adopted it in the 1800s.",
    "Grey Sea": "The Grey Sea is a large inland body of water in northern Vorland. "
                "Several rivers, including the Nadel, empty into it.",
}


class WikiEnv:
    """A faithful local stand-in for ReAct's Wikipedia API (search + lookup)."""

    def __init__(self, corpus: dict):
        self.corpus = corpus
        self._titles = list(corpus)
        self.page: str | None = None
        self._sentences: list[str] = []
        self._lookup_key: str | None = None
        self._lookup_pos = 0

    @staticmethod
    def _split(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _resolve(self, entity: str) -> str | None:
        norm = entity.strip().lower()
        for t in self._titles:
            if t.lower() == norm:
                return t
        return None

    def search(self, entity: str) -> str:
        title = self._resolve(entity)
        if title is None:
            similar = difflib.get_close_matches(entity, self._titles, n=5, cutoff=0.3)
            if not similar:
                similar = self._titles[:5]
            return f"Could not find [{entity}]. Similar: {similar}"
        self.page = title
        self._sentences = self._split(self.corpus[title])
        self._lookup_key = None
        self._lookup_pos = 0
        return " ".join(self._sentences[:5])

    def lookup(self, keyword: str) -> str:
        if self.page is None:
            return "No page loaded. Use search[entity] first."
        if keyword != self._lookup_key:
            self._lookup_key = keyword
            self._lookup_pos = 0
        matches = [s for s in self._sentences if keyword.lower() in s.lower()]
        if not matches:
            return f"No results for '{keyword}' on page {self.page}."
        if self._lookup_pos >= len(matches):
            return f"No more results for '{keyword}'."
        s = matches[self._lookup_pos]
        self._lookup_pos += 1
        return f"(Result {self._lookup_pos} / {len(matches)}) {s}"


def make_wiki_tools(env: WikiEnv):
    @tool
    def search(entity: str) -> str:
        """Search the encyclopedia for an entity and return the first sentences of
        its page (or a list of similar page titles if it does not exist).

        Args:
            entity: the exact title of the page to look up.
        """
        return env.search(entity)

    @tool
    def lookup(keyword: str) -> str:
        """Return the next sentence containing `keyword` on the most recently
        searched page (like Ctrl-F in a browser).

        Args:
            keyword: a substring to find on the current page.
        """
        return env.lookup(keyword)

    return [search, lookup]


# ════════════════════════════════════════════════════════════════════════════
# Tasks
# ════════════════════════════════════════════════════════════════════════════
QA = [
    ("lang_of_capital", "What language is spoken in the country whose capital is Lutz?",
     ["Zubrish"]),
    ("river_mouth", "The river that flows through the city of Lutz empties into what "
     "body of water?", ["Grey Sea", "the Grey Sea"]),
    ("same_author", "Were the authors of the books 'Theory of Folds' and "
     "'On Tessellations' the same person? Answer yes or no.", ["no"]),
    ("common_profession", "What profession do Eska Vorn and Hale Quist have in common?",
     ["mathematician"]),
    ("director_birthplace", "In what country was the director of the 1997 film "
     "'The Crimson Vault' born?", ["Zubrowka"]),
    ("element_standard", "The monetary standard based on the element with atomic "
     "number 88 is called what?", ["Aurium Standard", "the Aurium Standard"]),
]

FEVER = [
    ("seine_claim", "The Nadel river flows through Lutenblag.", "REFUTES"),
    ("author_claim", "Eska Vorn wrote the book Theory of Folds.", "SUPPORTS"),
    ("award_claim", "Mira Calo won three awards for The Crimson Vault.", "NOT ENOUGH INFO"),
]


def _answer_field(text: str) -> str:
    m = list(re.finditer(r"answer\s*[:=]\s*(.+)", text, flags=re.IGNORECASE))
    return (m[-1].group(1) if m else text).strip()


def _norm(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def grade_em(golds: list[str]):
    def _g(result, workdir):
        ans = _answer_field(extract_final_text(result))
        na = _norm(ans)
        ok = any(_norm(g) == na or _norm(g) in na for g in golds)
        return (1.0 if ok else 0.0, f"got={ans[:45]!r} want={golds}")
    return _g


def grade_label(gold: str):
    def _g(result, workdir):
        ans = _answer_field(extract_final_text(result)).upper()
        if "NOT ENOUGH" in ans or "NEI" in ans:
            got = "NOT ENOUGH INFO"
        elif "SUPPORT" in ans:
            got = "SUPPORTS"
        elif "REFUTE" in ans:
            got = "REFUTES"
        else:
            got = "?"
        return (1.0 if got == gold else 0.0, f"got={got} want={gold}")
    return _g


def build_tasks() -> list[Task]:
    tasks = []
    for tid, q, golds in QA:
        tasks.append(Task(id=tid, prompt=f"Question: {q}", grade=grade_em(golds),
                          meta={"kind": "qa"}))
    for tid, claim, label in FEVER:
        tasks.append(Task(
            id=tid,
            prompt=f"Claim: {claim}\nIs this claim SUPPORTS, REFUTES, or NOT ENOUGH "
                   f"INFO given the encyclopedia?",
            grade=grade_label(label), meta={"kind": "fever"}))
    return tasks


# ── condition -> agent ───────────────────────────────────────────────────────
_TOOL_PREAMBLE = (
    "These entities are FICTIONAL — you do NOT know them in advance, so you MUST "
    "use the tools. Chain your steps: search one entity, read the first sentences, "
    "then search the NEXT entity named in them (use lookup only to find a specific "
    "word on the page you already searched). When you have gathered the facts, call "
    "the final_answer tool with your answer (a short phrase)."
)
_SYS = {
    "standard": "Answer the question with a short, direct answer. End your reply "
                "with a final line: 'Answer: <answer>'.",
    "cot": "Think step by step to work out the answer, then end your reply with a "
           "final line: 'Answer: <answer>'.",
    "act": "Use the search and lookup tools to find the facts. " + _TOOL_PREAMBLE,
    "react": "Solve the problem step by step. Before each tool call, briefly state "
             "your reasoning as a 'Thought:'. " + _TOOL_PREAMBLE,
}


def make_agent_for(condition: str):
    from genai_studio.agents.tools import final_answer

    def factory(task, workdir):
        use_tools = condition in ("act", "react", "react_injected")
        sys = _SYS["react" if condition == "react_injected" else condition]
        # tool conditions also get a final_answer tool (terminal); the Agent ends
        # the loop when it is called (Agent.finish_tool_names).
        tools = (make_wiki_tools(WikiEnv(CORPUS)) + [final_answer]) if use_tools else []
        client = make_client(react=(condition == "react_injected"))
        return Agent(client=client, tools=tools, system=sys,
                     tracer=JsonlTracer(os.path.join(workdir, "trace.jsonl")),
                     max_steps=8)
    return factory


if __name__ == "__main__":
    import sys

    conditions = ["standard", "cot", "act", "react", "react_injected"]
    if len(sys.argv) > 1 and sys.argv[1] in conditions:
        conditions = [sys.argv[1]]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    runs = int(os.getenv("BENCH_RUNS", "1"))  # set BENCH_RUNS=10 for variance
    tasks = build_tasks()[:limit]

    summary = {}
    for cond in conditions:
        rep = run_suite(f"ReAct[{cond}]", tasks, make_agent_for(cond), runs=runs)
        summary[cond] = (rep["mean_score"], rep["success_rate"])
    print(f"\n==== ReAct ablation summary (mean EM/accuracy, runs={runs}) ====")
    for cond in conditions:
        m, sr = summary[cond]
        print(f"  {cond:16} mean={m:.3f}")
