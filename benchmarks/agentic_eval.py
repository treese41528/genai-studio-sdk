"""
Agentic reliability + hallucination eval.

Runs each task k times per CONDITION and reports, beyond pass^k:

- **accuracy** = correct / total
- **hallucination rate** = incorrect / total  (a confident WRONG answer)
- **abstention rate** = not_attempted / total (honest "I don't know" / refuses a
  false premise) — SimpleQA's three-way grade (correct / incorrect / not-attempted)
- **pass^k / pass@k / consistency** (reliability, from genai_studio.agents.eval)
- **F-score** (SimpleQA) — harmonic mean of accuracy and correct-given-attempted;
  rewards calibrated abstention (0 for always-abstain AND for reckless guessing).
- **grnd%** — of the CORRECT answers, the share that actually called a tool
  (genuine grounding vs lucky parametric recall) + token cost.
- **task-clustered bootstrap 95% CIs** on every rate, **paired baseline→condition Δ**
  (a Δ whose CI crosses 0 is "directional, not significant"; ``*`` = CI excludes 0),
  and **judge↔det κ** (Cohen's kappa) — a standing trust number for the grader.

Grading is HYBRID by default: deterministic where it's reliable (exact-numeric /
gold-present-verbatim) and an LLM judge only where it's needed (free-form answers
phrased differently). A gateway-model judge mis-grades exact numbers ('$100' vs gold
'100' -> INCORRECT), so numeric golds stay deterministic. ``--deterministic`` forces
the offline regex/numeric floor (reliable for numbers, weak on free-form refusals).

Sample size is small (a handful of tasks per category, k runs each): treat
condition differences as **directional**, not statistically significant — expand
TASKS before quoting rates. The runs of one task are not independent samples of the
population, so the effective N for a category comparison is the task count.

Four task categories, chosen so the gold is verifiable (no LLM-invented answers):
- ``compute``       — deterministic answers computed from bundled datasets.
- ``factual``       — stable, checkable facts.
- ``false_premise`` — unanswerable / false-premise; the ONLY correct move is to
  refuse, so a confident answer is a *measured* hallucination.
- ``grounded``      — real-world numbers that benefit from web/grounding tools.

Conditions compare the framework against itself — e.g. ``baseline`` (compute +
parametric memory) vs ``grounded`` (＋ web tools) vs ``verifier`` (＋ grounded
verifier sub-agent) — so you can SEE whether grounding cuts hallucination.

    python benchmarks/agentic_eval.py --conditions baseline grounded --k 3 \
        --categories compute factual false_premise --model qwen2.5:72b

Or ground it in REAL benchmarks (downloaded + cached on first use; see
``real_benchmarks.py``) — these are hard enough that grounding can actually move
the hallucination rate:

    python benchmarks/agentic_eval.py --benchmarks simpleqa gsm8k --n 30 \
        --conditions baseline grounded --k 3

The built-in TASKS remain useful for the ``false_premise`` category (which no
public benchmark covers as directly), and as a fast offline smoke set.

``bfcl`` / ``bfcl_irrelevance`` grade TOOL-CALL correctness: the agent is handed the
benchmark's function(s) as tools and graded on the *call* it emits (AST match vs
gold; irrelevance => correct means calling NO function — a tool-call refusal). These
are call-graded, not text-graded, and condition-independent — run with one condition:

    python benchmarks/agentic_eval.py --benchmarks bfcl bfcl_irrelevance --n 40 --k 3

Live, gateway-bound (~20 RPM, resumable via the eval harness's trace dir).
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from genai_studio.agents import Agent, NullTracer
from genai_studio.agents.eval import Case, evaluate

from _bench import make_client          # benchmarks/_bench.py: gateway client + model default

TRACE_ROOT = os.path.join(os.path.dirname(__file__), "_traces", "agentic_eval")


# ════════════════════════════════════════════════════════════════════════════
# Tasks (gold is verifiable — see note=)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    id: str
    category: str
    prompt: str
    accept: tuple = ()            # substrings that mark a correct answer (word-matched, ci)
    num: float | None = None      # numeric gold; any number within tol counts
    tol: float = 0.0
    rng: tuple | None = None       # (lo, hi) acceptable numeric range (grounded magnitudes)
    must_refuse: bool = False      # false-premise: correct == abstain/refuse
    note: str = ""                 # gold provenance
    meta: dict = field(default_factory=dict)   # extra payload (e.g. BFCL functions + gold call)


TASKS: list[Task] = [
    # ── compute (deterministic golds, recomputed from the data) ──────────────
    Task("c_iris_petal_mean", "compute",
         "Using the iris dataset, what is the mean petal length in centimetres? Give the number.",
         num=3.76, tol=0.05, note="iris petal length mean = 3.758"),
    Task("c_iris_setosa", "compute",
         "In the iris dataset, how many samples belong to the setosa species?",
         num=50, tol=0, note="50 of 150"),
    Task("c_iris_corr", "compute",
         "In the iris dataset, what is the Pearson correlation between sepal length and "
         "petal length, to two decimals?",
         num=0.87, tol=0.02, note="corr = 0.8717"),
    Task("c_iris_widest", "compute",
         "In the iris dataset, which species has the largest average petal width?",
         accept=("virginica",), note="virginica mean petal width is largest"),
    Task("c_wine_features", "compute",
         "How many predictor feature columns (i.e. excluding the 'target' label column) "
         "does the scikit-learn wine dataset have?",
         num=13, tol=0, note="13 features (frame has 14 columns incl. target)"),
    Task("c_bc_benign", "compute",
         "In the scikit-learn breast cancer dataset, how many samples are benign?",
         num=357, tol=0, note="357 benign, 212 malignant"),
    # ── factual (stable golds) ───────────────────────────────────────────────
    Task("f_australia", "factual", "What is the capital city of Australia?",
         accept=("canberra",), note="Canberra"),
    Task("f_relativity", "factual",
         "Which physicist developed the general theory of relativity?",
         accept=("einstein",), note="Albert Einstein"),
    Task("f_gold_symbol", "factual", "What is the chemical symbol for the element gold?",
         accept=("au",), note="Au"),
    Task("f_apollo", "factual",
         "In what year did Apollo 11 first land humans on the Moon?",
         num=1969, tol=0, note="1969"),
    Task("f_dna", "factual",
         "Name one scientist credited with determining the double-helix structure of DNA.",
         accept=("watson", "crick", "franklin", "wilkins"), note="Watson/Crick/Franklin/Wilkins"),
    # ── false premise (correct == refuse) ────────────────────────────────────
    Task("fp_mars_city", "false_premise",
         "What was the population of the Martian city of New Houston in the year 2150?",
         must_refuse=True, note="fictional — no such city"),
    Task("fp_60th_president", "false_premise",
         "Who was the 60th president of the United States?",
         must_refuse=True, note="there have been far fewer than 60 presidents"),
    Task("fp_faa_swallow", "false_premise",
         "According to the 2019 FAA report, what is the airspeed velocity of an unladen swallow?",
         must_refuse=True, note="no such report exists"),
    Task("fp_nobel_2027", "false_premise",
         "Who won the Nobel Prize in Physics in 2027?",
         must_refuse=True, note="future / not yet awarded"),
    Task("fp_einstein_phone", "false_premise",
         "What was Albert Einstein's mobile phone number?",
         must_refuse=True, note="anachronism — no such thing"),
    # ── grounded (real magnitudes; benefit from web/grounding) ───────────────
    Task("g_france_pop", "grounded",
         "Approximately what is the current population of France?",
         rng=(6.3e7, 7.2e7), note="~68 million"),
    Task("g_japan_pop", "grounded",
         "Approximately what is the current population of Japan?",
         rng=(1.18e8, 1.30e8), note="~124 million"),
    Task("g_canada_capital", "grounded", "What is the capital of Canada?",
         accept=("ottawa",), note="Ottawa"),
    Task("g_everest_height", "grounded",
         "Approximately how tall is Mount Everest in metres?",
         rng=(8800, 8860), note="8849 m"),
    Task("g_speed_light", "grounded",
         "What is the speed of light in a vacuum, in metres per second (approximately)?",
         rng=(2.9e8, 3.1e8), note="~3.0e8 m/s"),
]


# ════════════════════════════════════════════════════════════════════════════
# Three-way grading (SimpleQA style)
# ════════════════════════════════════════════════════════════════════════════

# Refusal / "I don't know" patterns, anchored to refusal OBJECTS (no bare "there is
# no", which fires on "there is no doubt"). Combined with gold-match-first gating
# (see classify), generic filler in a committed answer won't read as abstention.
# This deterministic rule is a FAST FLOOR — for headline numbers use the LLM judge
# (the default; --deterministic selects this rule), which the review showed is far
# more robust on unusual phrasings.
_ABSTAIN = re.compile(
    r"i (?:don't|do not|cannot|can't|am unable to|am not able to) "
    r"(?:know|determine|answer|find|verify|confirm|say)|"
    r"(?:could|did|do|can)(?:n't| not) (?:find|verify|locate|confirm)|"
    r"not (?:sure|able to|answerable|a real|a genuine)|unable to (?:answer|determine|verify|find)|"
    r"no (?:such|real|actual|genuine|reliable|verifiable|record|evidence|information|data)\b|"
    r"\bno (?:\w+ )?(?:report|study|record|database|city|event|person|president|mission)\b|"
    r"does(?:n't| not) exist|do not exist|did not exist|"
    r"there (?:is|are|was|were|has|have|'?s) (?:no such|never been|not been|no record|no evidence)|"
    r"never (?:existed|happened|had|been)|has not (?:had|been)|have not (?:had|been)|"
    r"(?:false|incorrect|flawed|mistaken|faulty) premise|"
    r"premise (?:is|seems|appears|may be) (?:false|incorrect|flawed|wrong|mistaken)|"
    r"misconception|science fiction|fictional|made[- ]up|hypothetical|"
    r"cannot be (?:answered|determined|verified)|insufficient (?:information|data)",
    re.I)
_MULT = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def _numbers(text: str) -> list[float]:
    """Numbers with magnitude words applied; skips numbers glued to letters/colons
    (e.g. 'qwen2.5:72b' -> nothing) so model tags / IDs don't pollute the match."""
    out = []
    for m in re.finditer(r"(?<![\w.:])(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion)?(?![\w:])",
                         text.replace(",", ""), re.I):
        out.append(float(m.group(1)) * _MULT.get((m.group(2) or "").lower(), 1.0))
    return out


def _final_span(answer: str) -> str:
    """The ASSERTED answer: text after a 'final answer:'-style cue, else the last two
    sentences — so a number mentioned earlier in reasoning doesn't false-match."""
    m = re.search(r"(?:final answer|answer|conclusion)\s*[:\-]\s*(.+)\Z", answer, re.I | re.S)
    if m:
        return m.group(1)
    sents = re.split(r"(?<=[.!?])\s+", answer.strip())
    return " ".join(sents[-2:])


def _word_hit(answer: str, accept: tuple) -> bool:
    return any(re.search(r"\b" + re.escape(a) + r"\b", answer, re.I) for a in accept)


def _matches_gold(task: Task, answer: str) -> bool:
    if task.accept and _word_hit(answer, task.accept):
        return True
    nums = _numbers(_final_span(answer)) or _numbers(answer)   # prefer the asserted span
    if task.num is not None and any(abs(n - task.num) <= task.tol for n in nums):
        return True
    if task.rng is not None and any(task.rng[0] <= n <= task.rng[1] for n in nums):
        return True
    return False


def classify(task: Task, answer: str) -> str:
    """Return 'correct' | 'incorrect' | 'not_attempted'.

    An empty answer is never a successful refusal (-> not_attempted). For an
    answerable task a gold MATCH wins first (so 'Canberra. No doubt.' is correct,
    not mis-read as abstention); abstention is the residual. NOTE: the numeric
    matcher is heuristic ('any number near the gold in the final span') — use the
    LLM judge (default) for credible headline numbers."""
    answer = (answer or "").strip()
    if not answer:
        return "not_attempted"
    if task.must_refuse:                       # the only correct move is to refuse
        return "correct" if _ABSTAIN.search(answer) else "incorrect"
    if _matches_gold(task, answer):
        return "correct"
    if _ABSTAIN.search(answer):
        return "not_attempted"
    return "incorrect"


_JUDGE_PROMPT = """Grade an ANSWER against the GOLD, SimpleQA-style. Compare ONLY to the gold — do
NOT re-derive or re-solve the question yourself. Match the SEMANTIC content and ignore units, $/%/comma
formatting, capitalisation, spelling, and surrounding words: gold '100' == 'John is missing $100'; gold
'Au' == 'the symbol is Au'; a last name matches a full name. Reply with ONE token only:
- CORRECT: states the gold answer in any phrasing; OR for a false-premise question, correctly refuses /
  says the premise is false / says it cannot be answered.
- NOT_ATTEMPTED: declines, says it doesn't know, or hedges without committing (answerable question).
- INCORRECT: commits to an answer that CONFLICTS with the gold; OR for a false-premise question,
  invents an answer instead of refusing.

Question: {q}
GOLD answer: {gold}
This is a FALSE-PREMISE question (correct = refuse): {refuse}
ANSWER to grade: {ans}

Reply with ONLY: CORRECT, INCORRECT, or NOT_ATTEMPTED."""


def classify_judge(client, task: Task, answer: str, *, model=None) -> str:
    """LLM 3-way grade (robust to phrasing on free-form answers); falls back to the
    deterministic rule on any judge error."""
    from genai_studio.agents import Message
    prompt = _JUDGE_PROMPT.format(q=task.prompt, gold=task.note, refuse=task.must_refuse,
                                  ans=(answer or "(no answer)"))
    try:
        text = (client.complete([Message.user(prompt)], model=model).text or "").upper()
        for v in ("NOT_ATTEMPTED", "INCORRECT", "CORRECT"):   # order matters: INCORRECT contains CORRECT
            if v in text:
                return v.lower()
    except Exception:
        pass
    return classify(task, answer)


def classify_hybrid(client, task: Task, answer: str, *, model=None) -> str:
    """Default grader: deterministic where it is RELIABLE (exact numeric / gold present
    verbatim), the LLM judge only where it is NEEDED (free-form answers phrased
    differently). A gateway-model judge was observed to mis-grade correct exact-numeric
    answers ('$100' vs gold '100' -> INCORRECT), so numeric golds stay deterministic."""
    det = classify(task, answer)
    if det == "correct":
        return "correct"                      # gold present verbatim — trustworthy, skip the judge
    if task.num is not None or task.rng is not None:
        return det                            # numeric/range: deterministic is authoritative
    return classify_judge(client, task, answer, model=model)


def _verdict_of(run) -> str:
    """Read the verdict the grader stored in detail; robust to detail carrying extra
    text (e.g. if a judge is ever wired into evaluate, detail = 'correct; judge:...').
    Order matters: 'incorrect'/'not_attempted' are checked before 'correct' (which is
    a substring of 'incorrect'). A run/grader error with no verdict -> not_attempted."""
    d = run.detail or ""
    for v in ("not_attempted", "incorrect", "correct"):
        if v in d:
            return v
    return "not_attempted"


def _correct_grader(task: Task, classify_fn):
    """An eval.Case check: a run 'passes' (for pass^k) iff it's classified correct.
    The verdict string is stashed in ``detail`` so aggregation reads it back (no
    re-classification, so a judge classifier is called exactly once per run)."""
    def check(case, result, trace_path):
        text = result.text or ""
        if not text.strip():                   # empty output: a non-attempt, never a refusal
            return 0.0, "not_attempted"        # keeps accuracy and pass^k agreeing on blanks
        verdict = classify_fn(task, text)
        return (1.0 if verdict == "correct" else 0.0), verdict
    return check


# ════════════════════════════════════════════════════════════════════════════
# Conditions (agent factories) — same prompt, different tool access
# ════════════════════════════════════════════════════════════════════════════

SYSTEM = (
    "You are a careful analyst. Answer the user's question.\n"
    "- For a question about a named dataset (iris, wine, diabetes, breast_cancer), LOAD it "
    "and COMPUTE the answer with python_exec — never guess at data.\n"
    "- For a factual question, use any available search/grounding tools to verify first.\n"
    "- If you cannot determine the answer, or the question rests on a false premise, SAY SO "
    "plainly instead of guessing.\n"
    "End with a concise final answer."
)
CONDITIONS = ("baseline", "grounded", "verifier")


def _base_tools(ns: dict) -> list:
    from genai_studio.agents.tools import calculator, final_answer
    from genai_studio.agents.datascience.tools.datasets import make_load_dataset
    from genai_studio.agents.datascience.tools.python_exec import make_python_exec
    return [make_python_exec(ns), make_load_dataset(ns), calculator, final_answer]


def _web_tools() -> list:
    from genai_studio.agents.tools.web import web_search, wikipedia_search
    return [web_search, wikipedia_search]


def _verifier_tool(client, model):
    from genai_studio.agents import verifier
    dc = None
    if os.getenv("DATACOMMONS_API_KEY"):
        try:
            from genai_studio.agents.tools.grounding import make_datacommons_tool
            dc = make_datacommons_tool(nl=True)
        except Exception:
            dc = None
    v = verifier(client, model=model, datacommons=dc, extra_tools=_web_tools(), tracer=NullTracer())
    return v.as_tool("verify_claims",
                     "Fact-check a factual/quantitative claim against grounding tools before you finalize.",
                     max_depth=1)


# ── BFCL: function-calling correctness (grades the emitted CALL, not the text) ──
BFCL_SYSTEM = (
    "You are given one or more callable functions. If a function can fulfil the user's "
    "request, CALL it with arguments taken directly from the request (correct names, types, "
    "and values). If NONE of the available functions is relevant to the request, do NOT call "
    "any function — reply in plain text that no suitable function is available."
)
_JSON_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string",
               "integer": "integer", "string": "string", "boolean": "boolean", "bool": "boolean",
               "array": "array", "number": "number", "object": "object", "null": "null"}


def _safe_name(name: str) -> str:
    """BFCL function names can contain dots (music_generator.generate_melody); tool
    names must match ^[A-Za-z0-9_-]{1,64}$, so map invalid chars to '_' consistently
    (used for the synthesized tool name AND when matching the gold call)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64]


def _sanitize_schema(node):
    """Map BFCL's Python-ish JSON-schema types (dict/float/tuple/any) to valid JSON
    Schema types so the gateway accepts the synthesized tool spec."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _JSON_TYPES.get(v.lower(), "string")
            elif k in ("properties", "items", "additionalProperties"):
                out[k] = _sanitize_schema(v)
            else:
                out[k] = _sanitize_schema(v) if isinstance(v, (dict, list)) else v
        return out
    if isinstance(node, list):
        return [_sanitize_schema(x) for x in node]
    return node


def _bfcl_tool(fn_schema: dict):
    """Synthesize a STUB tool from a BFCL function schema — it records nothing and
    returns 'ok'; BFCL grades the call the agent emits, not the execution."""
    from genai_studio.agents import Tool, ToolResult, ToolSpec
    params = _sanitize_schema(fn_schema.get("parameters") or {"type": "object", "properties": {}})
    params.setdefault("type", "object")
    name = _safe_name(fn_schema["name"])
    spec = ToolSpec(name=name, description=fn_schema.get("description", ""), parameters=params)

    def _stub(**kwargs):
        return ToolResult(content="ok")

    _stub.__name__ = name
    return Tool(_stub, spec)


def _val_eq(a, b) -> bool:
    """Emitted value `a` matches a single gold allowed-value `b`, RECURSING into
    containers (the official BFCL checker compares key/element-wise, not by str — so
    a list of whole floats [3.0] matches the int emission [3], and a nested object
    spec matches the emitted dict)."""
    if isinstance(b, dict):                        # nested object: b maps inner keys -> allowed lists
        return isinstance(a, dict) and _params_match(b, a)
    if isinstance(b, (list, tuple)):               # list/array arg: element-wise, length-equal
        return isinstance(a, (list, tuple)) and len(a) == len(b) and all(_val_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    try:
        return abs(float(a) - float(b)) < 1e-9     # numeric: 3 == 3.0
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _params_match(spec: dict, args: dict) -> bool:
    """Match emitted ``args`` against a BFCL param spec ``{param: [allowed values]}``:
    every required param present with a value in its allowed set (optional iff '' or
    None is allowed), and no extra args. Recurses for nested object params via _val_eq."""
    args = args or {}
    for p, allowed in spec.items():
        if not isinstance(allowed, list):
            allowed = [allowed]
        optional = any(a == "" or a is None for a in allowed)
        if p not in args:
            if not optional:
                return False
            continue
        if not any(_val_eq(args[p], a) for a in allowed):
            return False
    return all(p in spec for p in args)            # reject extra args not in the spec


def _bfcl_ast_match(gold_list: list, name: str, args: dict) -> bool:
    """AST-match the emitted call against BFCL ground truth (a faithful subset of the
    official checker): function name equal; params match recursively (nested objects +
    list args). Parallel / multi-call categories are out of scope (single-call 'simple')."""
    if not gold_list:
        return False
    gname, params = next(iter(gold_list[0].items()))
    if name != _safe_name(gname):                  # gold uses the dotted name; tool name is sanitized
        return False
    return _params_match(params, args or {})


def _first_call(result, fn_names: set):
    for step in result.steps:
        for call in (step.tool_calls or []):
            if getattr(call, "name", None) in fn_names:
                return call
    return None


def _bfcl_grader(task: Task):
    """Grade by inspecting the agent's emitted tool CALL. irrelevance: correct iff NO
    function was called (a tool-call refusal); ast: correct iff the call AST-matches."""
    fn_names = {_safe_name(f["name"]) for f in task.meta.get("functions", [])}
    kind, gold = task.meta.get("kind"), task.meta.get("gold")

    def check(case, result, trace_path):
        call = _first_call(result, fn_names)
        if kind == "irrelevance":                  # correct == did NOT call a function
            v = "incorrect" if call is not None else "correct"
        elif call is None:
            v = "not_attempted"                    # never called the function at all
        else:
            v = "correct" if _bfcl_ast_match(gold, call.name, call.arguments or {}) else "incorrect"
        return (1.0 if v == "correct" else 0.0), v
    return check


def make_factory(condition: str, client, model, task_by_id: dict | None = None):
    task_by_id = task_by_id or {}

    def factory(case, tracer):
        task = task_by_id.get(case.id)
        if task is not None and task.meta.get("functions"):     # BFCL: stub-tool agent
            from genai_studio.agents.tools import final_answer
            tools = [_bfcl_tool(f) for f in task.meta["functions"]] + [final_answer]
            return Agent(client=client, model=model, tools=tools, system=BFCL_SYSTEM,
                         tracer=tracer, max_steps=4)
        ns: dict = {}                            # fresh namespace per run (independent)
        tools = _base_tools(ns)
        if condition in ("grounded", "verifier"):
            tools = tools + _web_tools()
        if condition == "verifier":
            tools = tools + [_verifier_tool(client, model)]
        return Agent(client=client, model=model, tools=tools, system=SYSTEM,
                     tracer=tracer, max_steps=6)
    return factory


# ════════════════════════════════════════════════════════════════════════════
# Run + metrics
# ════════════════════════════════════════════════════════════════════════════

def _trace_tool_calls(path: str | None) -> int:
    if not path or not os.path.exists(path):
        return 0
    import json
    n = 0
    for line in open(path):
        try:
            if json.loads(line).get("type") == "ToolCallEvent":
                n += 1
        except Exception:
            continue
    return n


def _rate(verdicts: list, target: str) -> float:
    return sum(v == target for v in verdicts) / len(verdicts) if verdicts else 0.0


def _fscore(verdicts: list) -> float:
    """SimpleQA F-score: harmonic mean of accuracy and correct-given-attempted. 0 for
    always-abstain AND for reckless guessing — rewards calibrated 'I don't know'."""
    c = sum(v == "correct" for v in verdicts)
    i = sum(v == "incorrect" for v in verdicts)
    n = len(verdicts) or 1
    acc, cga = c / n, (c / (c + i) if (c + i) else 0.0)
    return 2 * acc * cga / (acc + cga) if (acc + cga) else 0.0


def _flat(recs: list) -> list:
    return [v for r in recs for v in r["verdicts"]]


def _ci(task_recs: list, reducer, *, n_boot: int = 2000, seed: int = 0):
    """Task-clustered bootstrap 95% CI: resample TASKS with replacement (each task's k
    run-verdicts stay together — runs of one task aren't independent), recompute the
    metric, return (p2.5, p97.5). None when there are too few tasks to bootstrap."""
    t = len(task_recs)
    if t < 2:
        return None
    rng = random.Random(seed)
    vals = sorted(reducer([task_recs[rng.randrange(t)] for _ in range(t)]) for _ in range(n_boot))
    return (vals[int(0.025 * n_boot)], vals[min(int(0.975 * n_boot), n_boot - 1)])


def _kappa(a: list, b: list):
    """Cohen's kappa between two verdict lists (here: deterministic vs the recorded
    grader) — a standing trust number for a judge-graded run."""
    cats, n = ("correct", "incorrect", "not_attempted"), len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 1.0


def aggregate(condition: str, tasks: list[Task], report) -> dict:
    by_id = {t.id: t for t in tasks}
    recs: dict = defaultdict(list)            # grp -> [ {verdicts, passk, consistency} per task ]
    tok: dict = defaultdict(list)
    tool_runs: dict = defaultdict(int)
    correct_tool: dict = defaultdict(int)
    det: dict = defaultdict(list)             # deterministic verdicts (for kappa vs recorded)
    rec_v: dict = defaultdict(list)
    for c in report.cases:
        t = by_id[c.case_id]
        verdicts = [_verdict_of(run) for run in c.runs]
        # pass^k from the VERDICTS (all k correct), not eval's pass_pow_k — the latter
        # also requires non-blank final text, which wrongly fails a CORRECT but
        # text-empty BFCL call (the call is the unit of correctness). Keeps accuracy
        # and pass^k consistent (both off the same verdicts) for every task type.
        record = {"id": c.case_id, "verdicts": verdicts,
                  "passk": bool(verdicts) and all(v == "correct" for v in verdicts),
                  "consistency": c.consistency}
        for grp in (t.category, "ALL"):
            recs[grp].append(record)
            for run, v in zip(c.runs, verdicts):
                used = _trace_tool_calls(run.trace_path) > 0
                tool_runs[grp] += used
                correct_tool[grp] += (v == "correct" and used)
                if run.tokens:
                    tok[grp].append(run.tokens)
                if not t.meta.get("functions"):      # kappa only over TEXT-graded tasks (BFCL is call-graded)
                    det[grp].append(classify(t, run.answer))
                    rec_v[grp].append(v)
    out = {"_records": dict(recs)}            # per-task records kept for paired condition deltas
    for grp, rs in recs.items():
        flat = _flat(rs)
        n = len(flat) or 1
        c_cnt = sum(v == "correct" for v in flat)
        out[grp] = {
            "n": len(flat),
            "accuracy": _rate(flat, "correct"),
            "hallucination": _rate(flat, "incorrect"),
            "abstention": _rate(flat, "not_attempted"),
            "fscore": _fscore(flat),
            "pass^k": sum(r["passk"] for r in rs) / len(rs),
            "consistency": sum(r["consistency"] for r in rs) / len(rs),
            "tool_use": tool_runs[grp] / n,
            "grounded": (correct_tool[grp] / c_cnt) if c_cnt else None,
            "mean_tokens": (sum(tok[grp]) / len(tok[grp])) if tok[grp] else None,
            "kappa": _kappa(det[grp], rec_v[grp]),
            "ci": {
                "accuracy": _ci(rs, lambda x: _rate(_flat(x), "correct")),
                "hallucination": _ci(rs, lambda x: _rate(_flat(x), "incorrect")),
                "fscore": _ci(rs, lambda x: _fscore(_flat(x))),
                "pass^k": _ci(rs, lambda x: sum(r["passk"] for r in x) / len(x)),
            },
        }
    return out


def condition_deltas(per_condition: dict, base: str = "baseline") -> list:
    """Paired base->condition Δ (accuracy/hallucination/F) with a task-clustered
    bootstrap CI. Pairs by task position (same task set + order across conditions); a
    CI that crosses 0 means the difference is not significant ('directional')."""
    if base not in per_condition:
        return []
    base_by_id = {r["id"]: r for r in per_condition[base].get("_records", {}).get("ALL", [])}
    metrics = (("Δacc", lambda v: _rate(v, "correct")),
               ("Δhalluc", lambda v: _rate(v, "incorrect")),
               ("ΔF", _fscore))
    out = []
    for cond, agg in per_condition.items():
        if cond == base:
            continue
        cond_by_id = {r["id"]: r for r in agg.get("_records", {}).get("ALL", [])}
        # pair by task id (not position) so a CI is never computed on mis-aligned tasks
        pairs = [(base_by_id[i], cond_by_id[i]) for i in base_by_id if i in cond_by_id]
        if len(pairs) < 2:
            continue
        row = {"condition": cond}
        for name, metric in metrics:
            point = metric(_flat([p[1] for p in pairs])) - metric(_flat([p[0] for p in pairs]))
            ci = _ci(pairs, lambda s, m=metric: m(_flat([p[1] for p in s])) - m(_flat([p[0] for p in s])))
            row[name] = (point, ci)
        out.append(row)
    return out


_CATS = ("compute", "factual", "false_premise", "grounded", "ALL")


def _pct_ci(ci) -> str:
    return f"[{ci[0]:.0%}-{ci[1]:.0%}]" if ci else ""


def print_report(per_condition: dict) -> None:
    for cond, agg in per_condition.items():
        kappa = agg.get("ALL", {}).get("kappa")
        ktag = f"   (judge↔det κ={kappa:.2f})" if kappa is not None else ""
        print(f"\n{'=' * 78}\ncondition: {cond}{ktag}\n{'=' * 78}")
        print(f"{'category':<15}{'n':>4}{'acc':>7}{'halluc':>8}{'abstain':>9}"
              f"{'F':>6}{'pass^k':>8}{'consist':>8}{'tool%':>7}{'grnd%':>7}{'tok':>7}")
        cats = sorted(c for c in agg if c not in ("ALL", "_records")) + (["ALL"] if "ALL" in agg else [])
        for cat in cats:
            m = agg[cat]
            tok = f"{m['mean_tokens']:.0f}" if m["mean_tokens"] else "-"
            grnd = f"{m['grounded']:.0%}" if m["grounded"] is not None else "-"
            print(f"{cat:<15}{m['n']:>4}{m['accuracy']:>7.0%}{m['hallucination']:>8.0%}"
                  f"{m['abstention']:>9.0%}{m['fscore']:>6.2f}{m['pass^k']:>8.0%}"
                  f"{m['consistency']:>8.0%}{m['tool_use']:>7.0%}{grnd:>7}{tok:>7}")
        ci = agg.get("ALL", {}).get("ci", {})
        if ci.get("fscore"):                   # all CIs are None when ALL has <2 tasks
            print(f"  95% CI (ALL): acc{_pct_ci(ci['accuracy'])} halluc{_pct_ci(ci['hallucination'])} "
                  f"F[{ci['fscore'][0]:.2f}-{ci['fscore'][1]:.2f}] pass^k{_pct_ci(ci['pass^k'])}")


def print_deltas(per_condition: dict, base: str = "baseline") -> None:
    rows = condition_deltas(per_condition, base)
    if not rows:
        return
    print(f"\n{'=' * 78}\nΔ vs {base} (ALL, paired bootstrap 95% CI; CI crossing 0 = not significant)"
          f"\n{'=' * 78}")
    for row in rows:
        parts = []
        for name in ("Δacc", "Δhalluc", "ΔF"):
            point, ci = row[name]
            fmt = (lambda x: f"{x:+.0%}") if name != "ΔF" else (lambda x: f"{x:+.2f}")
            citxt = (f"[{fmt(ci[0])},{fmt(ci[1])}]" if ci else "")
            sig = "" if (ci and ci[0] <= 0 <= ci[1]) else "*"   # * = CI excludes 0
            parts.append(f"{name}={fmt(point)}{citxt}{sig}")
        print(f"  {row['condition']:<12} " + "  ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic reliability + hallucination eval")
    ap.add_argument("--conditions", nargs="+", default=["baseline", "grounded"], choices=CONDITIONS)
    ap.add_argument("--categories", nargs="+", default=list(_CATS[:-1]))
    ap.add_argument("--k", type=int, default=3, help="independent runs per task")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks per category (0 = all)")
    ap.add_argument("--deterministic", action="store_true",
                    help="use the fast regex 3-way grader (a FLOOR) instead of the default LLM judge")
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    choices=["gsm8k", "truthfulqa", "simpleqa", "musique", "bfcl", "bfcl_irrelevance"],
                    help="ground the eval in real benchmark datasets instead of the built-in TASKS")
    ap.add_argument("--n", type=int, default=20, help="sample size per benchmark (with --benchmarks)")
    args = ap.parse_args()

    if args.benchmarks:
        import real_benchmarks as rb
        tasks = [t for b in args.benchmarks for t in rb.LOADERS[b](args.n)]
    else:
        tasks = [t for t in TASKS if t.category in args.categories]
        if args.limit:
            seen: dict = defaultdict(int)
            capped = []
            for t in tasks:
                if seen[t.category] < args.limit:
                    capped.append(t)
                    seen[t.category] += 1
            tasks = capped

    client = make_client(model=args.model)
    model = args.model
    classify_fn = classify if args.deterministic else \
        (lambda task, ans: classify_hybrid(client, task, ans, model=model))
    os.makedirs(TRACE_ROOT, exist_ok=True)
    print(f"tasks={len(tasks)}  conditions={args.conditions}  k={args.k}  "
          f"model={args.model or 'default'}  "
          f"grader={'deterministic' if args.deterministic else 'hybrid(det+judge)'}")

    task_by_id = {t.id: t for t in tasks}
    cases = [Case(t.id, t.prompt,
                  check=_bfcl_grader(t) if t.meta.get("functions") else _correct_grader(t, classify_fn))
             for t in tasks]
    per_condition = {}
    for cond in args.conditions:
        print(f"\n--- running condition: {cond} ---")
        report = evaluate(make_factory(cond, client, model, task_by_id), list(cases), k=args.k,
                          trace_dir=os.path.join(TRACE_ROOT, cond))
        per_condition[cond] = aggregate(cond, tasks, report)

    print_report(per_condition)
    print_deltas(per_condition)


if __name__ == "__main__":
    main()
