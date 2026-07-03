"""
Paper-faithful ReAct prompting for HotpotQA (Yao et al., ICLR 2023), built on
``genai_studio.agents`` by swapping in a single ``PaperReActClient`` so the SAME
``Agent`` runs the paper's text Thought/Action/Observation grammar instead of
native tool-calling. This is the framework's signature move — same Agent, a
different ModelClient — used here to reproduce the paper's exact protocol:

  * 6-shot in-context exemplars — the paper's VERBATIM HotpotQA prompts
    (``webthink``/``webact``/``cotqa``/``webqa`` ``_simple6`` from
    github.com/ysymyth/ReAct, vendored in ``_data/react_prompts.json``); the same
    six cases across all four conditions, differing only in which fields are shown.
  * greedy decoding (``temperature=0``).
  * action space ``Search[entity]`` / ``Lookup[keyword]`` / ``Finish[answer]``,
    parsed from text; generation is stopped before each ``Observation`` so the
    environment — not the model — supplies observations.
  * max 7 thought–action steps; NO forced answer on exhaustion (an unfinished
    trajectory yields no answer and is scored wrong), matching the paper.
  * strict SQuAD-style Exact-Match (``==``; no substring credit).

Documented deviation: the gateway is chat-only (the paper's
``text-davinci-002`` completion model is retired), so the original's single
growing completion is delivered as a multi-turn chat —
``system(instruction + exemplars)`` / ``user(Question)`` /
``assistant(Thought+Action)`` / ``user(Observation)`` … — with stop sequences
cutting each turn before the env's observation. This is semantically 6-shot
greedy ReAct; the only thing not reproduced is the literal single-prompt buffer.

Format-adherence layer (MODEL-AGNOSTIC — how we make resistant models follow the
protocol): a completion model followed the few-shot format implicitly; modern
chat/reasoning models (e.g. gpt-oss) often ignore it and emit a verbose markdown
answer, which strict EM then fails even when correct. Two enforcement mechanisms,
applied identically to every model so fidelity holds (one protocol, now *enforced*
not *assumed*): (1) an explicit format contract appended to the act/react
instruction headers; (2) a bounded re-ask loop in ``PaperReActClient.complete``
(``max_repair``) that, on a tool turn, re-prompts for a valid ``Action:`` line when
the model returns prose — mirroring the SDK's own ``ReActClient`` JSON repair.
Already-compliant models (qwen/llama) are unaffected; the repair never fires for them.

Thinking-model decoding exception (set in ``react_exact.make_agent_for``): thinking
models (qwen3, deepseek-r1, qwq) emit ``<think>…</think>`` blocks and DEGENERATE under
greedy decoding (Qwen/DeepSeek recommend ``temp≈0.6``) — at ``temp=0`` they return
empty/repetitive completions and score ~0. For these models ONLY, decoding switches to
sampled (``temperature=0.6``, ``top_p=0.95``) with a large ``max_tokens`` budget so the
reasoning can complete; ``<think>`` blocks are stripped (``_strip_think``) before parsing
and echoing. This is a documented, per-model-class deviation from the paper's greedy
decoding (which the paper itself calls "sub-optimal", footnote 4); all other models stay
greedy. Sampled decoding makes thinking-model cells non-deterministic.
"""

from __future__ import annotations

import json
import os
import re
import string
import uuid

from genai_studio.agents import BaseModelClient, Message, ModelResponse, ToolCall, Usage

_PROMPTS = json.load(
    open(os.path.join(os.path.dirname(__file__), "_data", "react_prompts.json"))
)

# ── instruction headers (from the official notebook; Act drops the Thought clause) ──
_TOOLS_DESC = (
    "(1) Search[entity], which searches the exact entity on Wikipedia and returns the "
    "first paragraph if it exists. If not, it will return some similar entities to search.\n"
    "(2) Lookup[keyword], which returns the next sentence containing keyword in the "
    "current passage.\n"
    "(3) Finish[answer], which returns the answer and finishes the task.\n"
)
# The paper's completion model (text-davinci-002) followed the few-shot format
# implicitly. Modern chat/reasoning models (e.g. gpt-oss) often ignore it and emit a
# verbose markdown answer instead — which strict EM fails even when correct. These
# explicit, MODEL-AGNOSTIC format contracts (applied to every model identically) make
# the protocol adherence enforced rather than assumed; paired with the repair loop in
# PaperReActClient. Compliant models (qwen/llama) are unaffected.
_FORMAT_RULE_REACT = (
    "ALWAYS reply with a single step and then stop: one `Thought:` line, then exactly "
    "one `Action:` line that is `Search[entity]`, `Lookup[keyword]`, or `Finish[answer]`. "
    "Give your final answer ONLY inside `Finish[...]` as a short span (a name, entity, "
    "date, or yes/no) — never as prose, an 'Answer:' line, or markdown.\n"
)
_FORMAT_RULE_ACT = (
    "ALWAYS reply with exactly one `Action:` line and then stop: `Search[entity]`, "
    "`Lookup[keyword]`, or `Finish[answer]`. Give your final answer ONLY inside "
    "`Finish[...]` as a short span (a name, entity, date, or yes/no) — never as prose, "
    "an 'Answer:' line, or markdown.\n"
)
_REACT_HEADER = (
    "Solve a question answering task with interleaving Thought, Action, Observation steps. "
    "Thought can reason about the current situation, and Action can be three types:\n"
    + _TOOLS_DESC + _FORMAT_RULE_REACT + "Here are some examples.\n"
)
_ACT_HEADER = (
    "Solve a question answering task with interleaving Action, Observation steps. "
    "Action can be three types:\n"
    + _TOOLS_DESC + _FORMAT_RULE_ACT + "Here are some examples.\n"
)
_COT_HEADER = "Solve a question answering task by thinking step by step. Here are some examples.\n"
# Chat/instruct models (unlike the paper's text-davinci-002 completion model) answer
# direct QA conversationally and ignore the terse Question/Answer few-shot pattern, so
# strict EM fails even when the answer is present. This explicit cue restores the
# completion-era terseness the bare prompt assumed; the exemplars are unchanged.
_STD_HEADER = (
    "Solve a question answering task. Reply with ONLY the answer — a short span (a "
    "name, entity, date, or yes/no) on one line, no explanation, prefixed with "
    "'Answer:'. Here are some examples.\n"
)

_PROMPT_KEY = {
    "standard": "webqa_simple6",
    "cot": "cotqa_simple6",
    "act": "webact_simple6",
    "react": "webthink_simple6",
}
_HEADER = {"standard": _STD_HEADER, "cot": _COT_HEADER, "act": _ACT_HEADER, "react": _REACT_HEADER}

# tool conditions act; the baselines answer directly
USES_TOOLS = {"standard": False, "cot": False, "act": True, "react": True}


def build_system(condition: str) -> str:
    """Instruction header + the paper's verbatim 6-shot exemplar block."""
    return _HEADER[condition] + _PROMPTS[_PROMPT_KEY[condition]].strip() + "\n"


# ── action parsing ──────────────────────────────────────────────────────────
# Tolerates optional step numbers: "Action 1: Search[X]" or "Action: Search[X]".
_ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*(\w+)\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_ARG_NAME = {"search": "entity", "lookup": "keyword"}


def _parse_action(text: str):
    """Return ('search'|'lookup', arg) | ('finish', answer) | None from model text."""
    if not text:
        return None
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None
    name, arg = matches[0]  # the FIRST action in this turn (robust vs fabricated trailing steps)
    return name.lower().strip(), arg.strip()


# Corrective reminder re-asked (bounded) when a model emits prose instead of an action.
_ACTION_REPAIR = (
    "Your previous reply did not contain a valid action line. Reply with EXACTLY ONE "
    "action and nothing else (you may prefix one brief `Thought:` line): "
    "`Action: Search[entity]`, `Action: Lookup[keyword]`, or `Action: Finish[answer]`. "
    "Put your final answer ONLY inside `Finish[...]` as a short span — do not write "
    "prose, markdown, or an 'Answer:' line."
)


def _usable(parsed) -> bool:
    """A parse the Agent can act on: a tool call or a terminal finish."""
    return parsed is not None and (parsed[0] == "finish" or parsed[0] in _ARG_NAME)


# Thinking models (qwen3/deepseek-r1/qwq) wrap reasoning in <think>...</think>. Strip
# it before parsing (so a tool mentioned mid-reasoning isn't taken as the action) and
# before echoing (so it doesn't bloat the next turn's context). No-op for other models.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


class PaperReActClient(BaseModelClient):
    """Drive the paper's text Thought/Action/Observation grammar over any backend.

    Wraps an inner ``ModelClient`` (a ``GenAIStudioClient``). Renders the running
    conversation as plain text the backend reads back (tool replies become
    ``Observation N:`` lines), parses ``Action: tool[arg]`` into a ``ToolCall`` the
    ``Agent`` executes, and maps ``Finish[answer]`` to a final text answer. Carries
    its own greedy ``temperature``, ``stop`` sequences, and ``max_tokens`` because
    the ``Agent`` forwards only ``temperature``.
    """

    def __init__(self, inner, *, condition: str, stop: list[str], max_tokens: int = 1024,
                 max_repair: int = 2, top_p: float | None = None):
        self._inner = inner
        self._condition = condition
        self._stop = stop
        self._max_tokens = max_tokens
        self._top_p = top_p  # set (with temp>0) for thinking models; greedy otherwise
        # On a tool condition, re-ask up to this many times for a valid action line
        # when the model replies with prose instead — the adherence mechanism that
        # coaxes format-resistant models (gpt-oss, reasoning models) into the grammar.
        self._max_repair = max_repair

    @property
    def supports_native_tools(self) -> bool:
        return True  # we *synthesize* tool-calling from text, so the Agent offers tools

    # ── render the convo into the paper's text transcript ────────────────────
    def _augment(self, messages: list[Message]) -> list[Message]:
        """system/user pass through; assistant tool-requests become their plain
        Thought/Action text; tool replies become ``Observation N:`` user turns."""
        out: list[Message] = []
        step = 0
        for m in messages:
            if m.role == "assistant":
                step += 1
                out.append(Message.assistant(m.content or ""))  # drop native tool_calls
            elif m.role == "tool":
                out.append(Message.user(f"Observation {step}: {m.content or ''}"))
            else:
                out.append(m)  # system (instruction+exemplars) and user (Question)
        return out

    def _call_inner(self, messages, *, model, temperature, on_retry, opts):
        kw = dict(opts)
        if self._top_p is not None:
            kw["top_p"] = self._top_p
        if self._stop:  # thinking models pass [] — no stop (it cuts their output to empty)
            kw["stop"] = self._stop
        return self._inner.complete(
            self._augment(list(messages)),
            model=model,
            temperature=0.0 if temperature is None else temperature,
            on_retry=on_retry,
            max_tokens=self._max_tokens,
            **kw,
        )

    def complete(self, messages, *, tools=None, model=None, temperature=None,
                 on_retry=None, **opts) -> ModelResponse:
        # Baselines (no tools): the completion IS the answer — no action to coax.
        if not tools:
            resp = self._call_inner(messages, model=model, temperature=temperature,
                                    on_retry=on_retry, opts=opts)
            return ModelResponse(text=_strip_think(resp.text or ""), tool_calls=[],
                                 usage=resp.usage, finish_reason="stop", raw=resp.raw)

        # Tool conditions: ask; if the model replies with prose instead of an action
        # line, re-ask with a corrective reminder (bounded) to coax adherence.
        convo = list(messages)
        total = Usage.zero()
        resp = None
        for attempt in range(self._max_repair + 1):
            resp = self._call_inner(convo, model=model, temperature=temperature,
                                    on_retry=on_retry, opts=opts)
            total = total + resp.usage
            if _usable(_parse_action(_strip_think(resp.text or ""))):
                break
            if attempt < self._max_repair:
                convo = convo + [Message.user(_ACTION_REPAIR)]

        text = _strip_think(resp.text or "")
        parsed = _parse_action(text)
        if _usable(parsed):
            name, arg = parsed
            if name == "finish":
                return ModelResponse(text=arg, tool_calls=[], usage=total,
                                     finish_reason="stop", raw=resp.raw)
            call = ToolCall(id=f"paper-{uuid.uuid4().hex[:6]}", name=name,
                            arguments={_ARG_NAME[name]: arg})
            # Keep the Thought+Action text (truncated at the action, so any leaked
            # chat-template/role markers are dropped) as the assistant turn — so the
            # next round's transcript reads exactly like the paper's exemplars.
            m = list(_ACTION_RE.finditer(text))
            clean = text[: m[0].end()] if m else text
            return ModelResponse(text=clean, tool_calls=[call], usage=total,
                                 finish_reason="tool_calls", raw=resp.raw)

        # Adherence failed after repairs: salvage an answer span (best effort) so a
        # stubbornly verbose model is graded on its answer, not its whole paragraph.
        return ModelResponse(text=extract_answer(text) or text, tool_calls=[],
                             usage=total, finish_reason="stop", raw=resp.raw)


# ════════════════════════════════════════════════════════════════════════════
# Official SQuAD/HotpotQA Exact-Match (mirrors ReAct's wrappers.normalize_answer)
# ════════════════════════════════════════════════════════════════════════════
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


_FINISH_RE = re.compile(r"finish\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)


def extract_answer(text: str) -> str:
    """Pull the predicted answer span from the agent's final text."""
    if not text:
        return ""
    m = _FINISH_RE.findall(text)
    if m:
        return m[0].strip()
    m = _ANSWER_RE.findall(text)
    if m:
        return m[0].splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip() if "\n" in text.strip() else text.strip()


def strict_em(pred: str, gold: str) -> bool:
    """Strict Exact-Match: normalized prediction must equal normalized gold."""
    return normalize_answer(pred) == normalize_answer(gold)
