"""
``Agent`` — the loop: prompt -> tool_calls -> execute -> repeat -> answer.

The whole loop lives in ONE generator, ``_drive``, which contains pure control
flow and *yields I/O intents* (``_CallModel`` / ``_ExecTool``). Thin drivers pump
it: ``run`` (sync) here, and ``arun`` / ``stream`` / ``astream`` (production layer)
perform the same pumping with async/streaming I/O. Because the loop body is
written once, every entry point behaves identically — which is also why it is
worth reading: this is the teachable core.

    msgs = [system?] + history + [user]
    for step in range(max_steps):
        resp = ask_model(msgs, tools)          # <- _CallModel intent
        if no tool_calls: return finalize(resp)
        append assistant(tool_calls) to msgs
        for call in resp.tool_calls:
            result = run_tool(call)            # <- _ExecTool intent
            append tool(result) to msgs        # error is fed back, not raised
    return stop('max_steps')
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

# Current delegation nesting depth, tracked across the actual call stack (a tool
# closure can't see the caller's runtime Budget, so depth rides a ContextVar
# instead). copy_context() makes it propagate into asyncio.to_thread workers and
# across awaits, and each sibling delegation gets its own copy — so concurrent
# fan-out can't corrupt the count. as_tool(max_depth=...) reads + bumps it.
_delegation_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "genai_delegation_depth", default=0)

from .client import (
    Message,
    ModelResponse,
    Usage,
    _StreamDone,
    _TextChunk,
    _ToolChunk,
    _assemble_tool_calls,
)
from .errors import BudgetExceeded, Cancelled, ToolError
from .events import Final, StepFinished, TextDelta, ToolCallFinished, ToolCallStarted
from .guard import Decision
from .tool import Tool, ToolRegistry, ToolResult, ToolSpec, _json_schema
from .trace import (
    AgentEnd,
    AgentStart,
    ConsoleTracer,
    LLMCall,
    LLMResponse,
    LLMRetry,
    StepEnd,
    ToolCallEvent,
    ToolResultEvent,
)


# ════════════════════════════════════════════════════════════════════════════
# Guards: Budget + cancellation
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Budget:
    """Caps on a run. Exceeding one raises ``BudgetExceeded`` (the loop converts
    it into a graceful ``AgentResult(stopped="budget")``)."""

    max_tokens: int | None = None
    max_steps: int | None = None
    max_tool_calls: int | None = None
    _tool_calls: int = field(default=0, init=False, repr=False)

    def observe(self, usage: Usage) -> None:
        if self.max_tokens is not None and (usage.total_tokens or 0) > self.max_tokens:
            raise BudgetExceeded("tokens", self.max_tokens, usage.total_tokens)

    def tick_tool_call(self) -> None:
        self._tool_calls += 1
        if self.max_tool_calls is not None and self._tool_calls > self.max_tool_calls:
            raise BudgetExceeded("tool_calls", self.max_tool_calls, self._tool_calls)


class Cancel:
    """Cooperative cancellation token (Postyl trips it on user navigate-away)."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled("Run cancelled.")


class _NullCancel:
    def check(self) -> None:
        pass


def _as_cancel(token):
    """Accept a ``Cancel``, any object with ``.cancelled``/``.is_set()``, or None."""
    if token is None:
        return _NullCancel()
    if isinstance(token, Cancel):
        return token

    class _Wrapped:
        def check(self_inner):
            tripped = getattr(token, "cancelled", False)
            if not tripped and hasattr(token, "is_set"):
                tripped = token.is_set()
            if tripped:
                raise Cancelled("Run cancelled.")

    return _Wrapped()


# ════════════════════════════════════════════════════════════════════════════
# Result records
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    index: int
    request_messages: list = field(default_factory=list)  # the literal prompt sent
    response: Any = None
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    usage: Any = None


@dataclass
class AgentResult:
    text: str
    output: Any = None
    messages: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped: str = "final"  # 'final'|'max_steps'|'budget'|'cancelled'|'error'
    sources: list = field(default_factory=list)
    error: BaseException | None = None


# ── internal I/O intents yielded by _drive ──────────────────────────────────
@dataclass
class _CallModel:
    messages: list
    tools: Any
    step: int


@dataclass
class _ExecTool:
    call: Any
    step: int


@dataclass
class _StepDone:
    step: int
    usage: Any


# ════════════════════════════════════════════════════════════════════════════
# Agent
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    client: Any  # a ModelClient
    tools: Sequence = ()
    model: str | None = None
    system: str | None = None
    max_steps: int = 10
    temperature: float | None = None
    sampling: dict = field(default_factory=dict)  # extra sampling opts (e.g. {"top_p": 0.95}) forwarded to every client call
    tracer: Any = field(default_factory=ConsoleTracer)
    output_schema: type | None = None
    on_tool_error: str = "recover"  # 'recover' | 'abort'
    force_final_answer: bool = True  # on max_steps, make one tool-less call for a text answer
    finish_tool_names: tuple = ("final_answer", "finish")  # calls that end the loop
    name: str | None = None  # default tool name when exposed via as_tool()
    guards: Sequence = ()  # before/after-tool hooks (deterministic policy seam)
    tool_search: Any = None  # opt-in deferred-tool config (ToolSearch); None = all tools eager

    def __post_init__(self):
        self._registry = ToolRegistry(self.tools)
        self._catalog = None
        self._rank = None
        if self.tool_search is not None:
            self._setup_deferred()
        if self.output_schema is not None:
            try:
                import pydantic  # noqa: F401
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "output_schema requires pydantic: "
                    "pip install 'genai-studio-sdk[structured]'"
                ) from e

    # ── public sync entry point ──────────────────────────────────────────────
    def run(self, prompt, *, memory=None, budget=None, cancel=None) -> AgentResult:
        """Run the agent to completion (synchronous)."""
        gen = self._drive(prompt, memory=memory, budget=budget, cancel=cancel)
        self._emit(AgentStart(prompt=prompt, tools=self._registry.specs(), model=self.model))
        sent = None
        try:
            while True:
                intent = gen.send(sent)
                sent = self._handle_sync(intent)
        except StopIteration as stop:
            result = stop.value
            self._emit(AgentEnd(result=result, stopped=result.stopped))
            return result

    # ── the loop (shared by sync + async drivers) ────────────────────────────
    def _drive(self, prompt, *, memory, budget, cancel):
        """Pure control flow for the agent loop. Yields ``_CallModel`` /
        ``_ExecTool`` / ``_StepDone`` intents and returns the final
        ``AgentResult``. Contains NO network or tracer calls — the driver that
        pumps it performs I/O and tracing. Keeping this function pure is what
        lets ``run`` and ``arun``/``astream`` share one identical loop body.
        """
        budget = budget or Budget()
        cancel = _as_cancel(cancel)
        native = bool(getattr(self.client, "supports_native_tools", False))
        deferred = self.tool_search is not None
        unlocked = OrderedDict() if deferred else None  # run-local LRU of unlocked deferred tools
        specs = self._specs(native, unlocked)
        msgs = self._build_initial(prompt, memory, native)
        steps: list[Step] = []
        usage = Usage.zero()
        output_retry_used = False

        try:
            for step in range(self.max_steps):
                # Budget may cap steps below max_steps.
                if budget.max_steps is not None and step >= budget.max_steps:
                    raise BudgetExceeded("steps", budget.max_steps, step)
                cancel.check()
                if deferred:                    # active tool schemas change as tools unlock
                    specs = self._specs(native, unlocked)

                # (1) Ask the model. The driver fills this intent with a real
                #     ModelResponse (native call, ReAct, or streamed-and-assembled).
                resp: ModelResponse = yield _CallModel(
                    messages=list(msgs), tools=specs or None, step=step
                )
                usage = usage + resp.usage
                budget.observe(usage)
                rec = Step(index=step, request_messages=list(msgs),
                           response=resp, usage=resp.usage)

                # (2a) No tool calls -> the model is answering, not acting.
                if not resp.tool_calls:
                    msgs.append(Message.assistant(resp.text or ""))
                    # JSON-mode structured output (backends without native tools).
                    if self.output_schema is not None and not native:
                        obj, err = _validate_output(self.output_schema,
                                                    _extract_json(resp.text or ""))
                        if err is None:
                            steps.append(rec)
                            return self._finalize(_render(obj), obj, msgs, steps, usage, "final")
                        if not output_retry_used:
                            output_retry_used = True  # one self-correct retry
                            msgs.append(Message.user(
                                f"That did not match the required schema ({err}). "
                                "Reply with ONLY a JSON object matching it."))
                            continue
                        steps.append(rec)
                        return self._finalize(resp.text or "", None, msgs, steps, usage, "error")
                    steps.append(rec)
                    return self._finalize(resp.text or "", None, msgs, steps, usage, "final")

                # (2b) Tool calls. The assistant turn that REQUESTS the tools must
                #      be appended BEFORE the tool replies, and every tool reply
                #      must carry a matching tool_call_id — else the gateway 400s.
                msgs.append(Message.assistant(resp.text, resp.tool_calls))
                for call in resp.tool_calls:
                    # A finish/final_answer call is TERMINAL: take its answer and
                    # stop. Honored even if the tool was never registered, so a
                    # model that "calls" final_answer (a near-universal habit)
                    # ends cleanly instead of looping on an unknown-tool error.
                    if call.name in self.finish_tool_names:
                        answer = _finish_answer(call.arguments)
                        rec.tool_calls.append(call)
                        msgs.append(self._tool_message(call, ToolResult(content=answer)))
                        steps.append(rec)
                        return self._finalize(answer, None, msgs, steps, usage, "final")

                    # Native structured output: return_result is terminal — it is
                    # validated here, not dispatched to a user tool.
                    if self.output_schema is not None and call.name == "return_result":
                        obj, err = _validate_output(self.output_schema, call.arguments)
                        if err is None:
                            rec.tool_calls.append(call)
                            steps.append(rec)
                            return self._finalize(_render(obj), obj, msgs, steps, usage, "final")
                        if not output_retry_used:
                            output_retry_used = True
                            msgs.append(self._tool_message(call, ToolResult(
                                content=f"Validation failed: {err}. "
                                "Call return_result again with valid fields.")))
                            continue
                        rec.tool_calls.append(call)
                        steps.append(rec)
                        return self._finalize(resp.text or "", None, msgs, steps, usage, "error")

                    # (3) Execute a normal tool — the driver performs the call and
                    #     returns a ToolResult (errors captured, never raised).
                    result: ToolResult = yield _ExecTool(call=call, step=step)
                    rec.tool_calls.append(call)
                    rec.tool_results.append(result)
                    msgs.append(self._tool_message(call, result))
                    budget.tick_tool_call()
                    if deferred:                # unlock searched/guessed tool schemas for next step
                        self._apply_unlock(unlocked, call, result)
                    if result.error and self.on_tool_error == "abort":
                        raise ToolError(result.error, tool_name=call.name, call_id=call.id)
                    cancel.check()

                steps.append(rec)
                yield _StepDone(step=step, usage=resp.usage)

            # Loop exhausted. If the model was still calling tools, make ONE last
            # call with NO tools offered, forcing a final text answer — this
            # rescues a usable result when a model loops on redundant tool calls.
            if self.force_final_answer and msgs and msgs[-1].role == "tool":
                final = yield _CallModel(messages=list(msgs), tools=None, step=self.max_steps)
                usage = usage + final.usage
                return self._finalize(final.text or "", None, msgs, steps, usage, "max_steps")
            return self._finalize(None, None, msgs, steps, usage, "max_steps")

        except BudgetExceeded as e:
            return self._finalize(None, None, msgs, steps, usage, "budget", error=e)
        except Cancelled as e:
            return self._finalize(None, None, msgs, steps, usage, "cancelled", error=e)
        except ToolError as e:
            return self._finalize(None, None, msgs, steps, usage, "error", error=e)

    # ── sync driver step handler ─────────────────────────────────────────────
    def _handle_sync(self, intent):
        if isinstance(intent, _CallModel):
            self._emit(LLMCall(messages=intent.messages, tools=intent.tools or [], step=intent.step))
            resp = self.client.complete(
                intent.messages, tools=intent.tools, model=self.model,
                temperature=self.temperature, on_retry=self._on_retry(intent.step),
                **self.sampling)
            self._emit(LLMResponse(response=resp, step=intent.step))
            return resp
        if isinstance(intent, _ExecTool):
            self._emit(ToolCallEvent(call=intent.call, step=intent.step))
            t0 = time.time()
            result = self._exec_sync(intent.call)
            self._emit(ToolResultEvent(call=intent.call, result=result,
                                       elapsed=time.time() - t0, step=intent.step))
            return result
        if isinstance(intent, _StepDone):
            self._emit(StepEnd(step=intent.step, usage=intent.usage))
            return None
        raise RuntimeError(f"Unknown intent: {intent!r}")  # pragma: no cover

    def _exec_sync(self, call) -> ToolResult:
        call, blocked = self._before_tool(call)
        if blocked is not None:
            return blocked
        t = self._registry.get(call.name)
        if t is not None and t.is_async:
            result = ToolResult(content="", error=(
                f"Async tool {call.name!r} cannot run in Agent.run(); use arun()."))
        else:
            result = self._registry.execute(call)
        return self._after_tool(call, result)

    # ── guard seam: deterministic before/after-tool hooks ────────────────────
    def _before_tool(self, call):
        """Run before-tool guards. Returns (possibly-modified call, blocked-result|None).

        Fails CLOSED on anything unexpected — a guard that raises, returns a
        non-``Decision``, or names an unknown action BLOCKS the call (it can never
        silently allow a call it meant to stop) and never crashes the run.
        ``modify`` rewrites only what the tool RUNS with — the recorded transcript
        still reflects the model's original request (and ``raw_arguments`` is cleared
        so the new args win if the call is ever re-serialised).
        """
        for g in self.guards:
            try:
                d = g.before_tool(call)
            except Exception as exc:
                return call, ToolResult(
                    content="", error=f"blocked: guard error: {type(exc).__name__}: {exc}")
            if d is None:
                continue
            if not isinstance(d, Decision):
                return call, ToolResult(content="", error=(
                    f"blocked: guard returned {type(d).__name__}, expected None or a "
                    "Decision (ALLOW / deny() / modify())."))
            if d.action == "allow":
                continue
            if d.action == "deny":
                return call, ToolResult(content="", error=d.reason or "blocked by a guard")
            if d.action == "modify" and d.arguments is not None:
                call = replace(call, arguments=dict(d.arguments), raw_arguments=None)
                continue
            return call, ToolResult(content="", error=(
                f"blocked: guard returned an invalid decision (action={d.action!r})."))
        return call, None

    def _after_tool(self, call, result):
        """Run after-tool guards; a guard may replace the result with a (new)
        ``ToolResult``. A raising guard OR a non-``ToolResult`` return is ignored
        (the original result stands) so a broken guard can't lose a real result or
        crash the run."""
        for g in self.guards:
            try:
                new = g.after_tool(call, result)
            except Exception:
                continue
            if isinstance(new, ToolResult):
                result = new
        return result

    # ── public async entry point (mirrors run via the same _drive) ───────────
    async def arun(self, prompt, *, memory=None, budget=None, cancel=None) -> AgentResult:
        """Run the agent to completion (asynchronous)."""
        gen = self._drive(prompt, memory=memory, budget=budget, cancel=cancel)
        self._emit(AgentStart(prompt=prompt, tools=self._registry.specs(), model=self.model))
        sent = None
        try:
            while True:
                intent = gen.send(sent)
                sent = await self._handle_async(intent)
        except StopIteration as stop:
            result = stop.value
            self._emit(AgentEnd(result=result, stopped=result.stopped))
            return result

    async def _handle_async(self, intent):
        if isinstance(intent, _CallModel):
            self._emit(LLMCall(messages=intent.messages, tools=intent.tools or [], step=intent.step))
            resp = await self.client.acomplete(
                intent.messages, tools=intent.tools, model=self.model,
                temperature=self.temperature, on_retry=self._on_retry(intent.step),
                **self.sampling)
            self._emit(LLMResponse(response=resp, step=intent.step))
            return resp
        if isinstance(intent, _ExecTool):
            self._emit(ToolCallEvent(call=intent.call, step=intent.step))
            t0 = time.time()
            result = await self._exec_async(intent.call)
            self._emit(ToolResultEvent(call=intent.call, result=result,
                                       elapsed=time.time() - t0, step=intent.step))
            return result
        if isinstance(intent, _StepDone):
            self._emit(StepEnd(step=intent.step, usage=intent.usage))
            return None
        raise RuntimeError(f"Unknown intent: {intent!r}")  # pragma: no cover

    async def _exec_async(self, call) -> ToolResult:
        call, blocked = self._before_tool(call)
        if blocked is not None:
            return blocked
        t = self._registry.get(call.name)
        if t is None:
            result = self._registry._unknown(call.name)
        else:
            try:
                if t.is_async:
                    result = await t.arun(call.arguments)
                else:
                    # Run a sync tool off the event loop so it can't block streaming.
                    result = await asyncio.to_thread(self._registry.execute, call)
            except Exception as exc:  # pragma: no cover - registry already guards
                result = ToolResult(content="", error=f"{type(exc).__name__}: {exc}")
        return self._after_tool(call, result)

    # ── streaming (sync + async) — same _drive, events emitted to the consumer ─
    def stream(self, prompt, *, memory=None, budget=None, cancel=None):
        """Synchronous streaming generator of AgentEvents, ending with Final."""
        gen = self._drive(prompt, memory=memory, budget=budget, cancel=cancel)
        self._emit(AgentStart(prompt=prompt, tools=self._registry.specs(), model=self.model))
        sent = None
        try:
            while True:
                intent = gen.send(sent)
                if isinstance(intent, _CallModel):
                    self._emit(LLMCall(messages=intent.messages, tools=intent.tools or [], step=intent.step))
                    text_parts, buf, finish, usage = [], {}, None, None
                    streamed = getattr(self.client, "supports_streaming", False)
                    if streamed:
                        for ch in self.client.stream(intent.messages, tools=intent.tools,
                                                     model=self.model, temperature=self.temperature,
                                                     **self.sampling):
                            if isinstance(ch, _TextChunk):
                                text_parts.append(ch.delta)
                                yield TextDelta(text=ch.delta, step=intent.step)
                            elif isinstance(ch, _ToolChunk):
                                _accumulate(buf, ch)
                            elif isinstance(ch, _StreamDone):
                                finish, usage = ch.finish_reason, ch.usage or usage
                    resp = self._assemble_or_degrade_sync(
                        intent, streamed, text_parts, buf, finish, usage)
                    self._emit(LLMResponse(response=resp, step=intent.step))
                    sent = resp
                elif isinstance(intent, _ExecTool):
                    yield ToolCallStarted(id=intent.call.id, name=intent.call.name,
                                          arguments=intent.call.arguments, step=intent.step)
                    self._emit(ToolCallEvent(call=intent.call, step=intent.step))
                    result = self._exec_sync(intent.call)
                    self._emit(ToolResultEvent(call=intent.call, result=result, step=intent.step))
                    yield ToolCallFinished(id=intent.call.id, name=intent.call.name,
                                           result=result, step=intent.step)
                    sent = result
                elif isinstance(intent, _StepDone):
                    self._emit(StepEnd(step=intent.step, usage=intent.usage))
                    yield StepFinished(step=intent.step, had_text=False, tool_calls=0, usage=intent.usage)
                    sent = None
        except StopIteration as stop:
            result = stop.value
            self._emit(AgentEnd(result=result, stopped=result.stopped))
            yield Final(result=result)

    async def astream(self, prompt, *, memory=None, budget=None, cancel=None):
        """Asynchronous streaming generator of AgentEvents, ending with Final."""
        gen = self._drive(prompt, memory=memory, budget=budget, cancel=cancel)
        self._emit(AgentStart(prompt=prompt, tools=self._registry.specs(), model=self.model))
        sent = None
        try:
            while True:
                intent = gen.send(sent)
                if isinstance(intent, _CallModel):
                    self._emit(LLMCall(messages=intent.messages, tools=intent.tools or [], step=intent.step))
                    text_parts, buf, finish, usage = [], {}, None, None
                    streamed = getattr(self.client, "supports_streaming", False)
                    if streamed:
                        async for ch in self.client.astream(intent.messages, tools=intent.tools,
                                                            model=self.model, temperature=self.temperature,
                                                            **self.sampling):
                            if isinstance(ch, _TextChunk):
                                text_parts.append(ch.delta)
                                yield TextDelta(text=ch.delta, step=intent.step)
                            elif isinstance(ch, _ToolChunk):
                                _accumulate(buf, ch)
                            elif isinstance(ch, _StreamDone):
                                finish, usage = ch.finish_reason, ch.usage or usage
                    resp = await self._assemble_or_degrade_async(
                        intent, streamed, text_parts, buf, finish, usage)
                    self._emit(LLMResponse(response=resp, step=intent.step))
                    sent = resp
                elif isinstance(intent, _ExecTool):
                    yield ToolCallStarted(id=intent.call.id, name=intent.call.name,
                                          arguments=intent.call.arguments, step=intent.step)
                    self._emit(ToolCallEvent(call=intent.call, step=intent.step))
                    result = await self._exec_async(intent.call)
                    self._emit(ToolResultEvent(call=intent.call, result=result, step=intent.step))
                    yield ToolCallFinished(id=intent.call.id, name=intent.call.name,
                                           result=result, step=intent.step)
                    sent = result
                elif isinstance(intent, _StepDone):
                    self._emit(StepEnd(step=intent.step, usage=intent.usage))
                    yield StepFinished(step=intent.step, had_text=False, tool_calls=0, usage=intent.usage)
                    sent = None
        except StopIteration as stop:
            result = stop.value
            self._emit(AgentEnd(result=result, stopped=result.stopped))
            yield Final(result=result)

    def _assemble_or_degrade_sync(self, intent, streamed, text_parts, buf, finish, usage):
        if not streamed:
            return self.client.complete(intent.messages, tools=intent.tools,
                                        model=self.model, temperature=self.temperature,
                                        **self.sampling)
        calls, ok = _assemble_tool_calls(buf)
        if buf and not ok:  # watch-item 2: tool deltas didn't assemble -> degrade
            return self.client.complete(intent.messages, tools=intent.tools,
                                        model=self.model, temperature=self.temperature,
                                        **self.sampling)
        text = "".join(text_parts) or None
        if not calls and text:  # streaming parity with _parse_completion: recover a text-emitted tool call
            from .client import _tool_calls_from_text
            recovered = _tool_calls_from_text(text)
            if recovered:
                calls, text = recovered, None
        return ModelResponse(text=text, tool_calls=calls, usage=usage or Usage(), finish_reason=finish)

    async def _assemble_or_degrade_async(self, intent, streamed, text_parts, buf, finish, usage):
        if not streamed:
            return await self.client.acomplete(intent.messages, tools=intent.tools,
                                               model=self.model, temperature=self.temperature,
                                               **self.sampling)
        calls, ok = _assemble_tool_calls(buf)
        if buf and not ok:
            return await self.client.acomplete(intent.messages, tools=intent.tools,
                                               model=self.model, temperature=self.temperature,
                                               **self.sampling)
        text = "".join(text_parts) or None
        if not calls and text:  # streaming parity with _parse_completion: recover a text-emitted tool call
            from .client import _tool_calls_from_text
            recovered = _tool_calls_from_text(text)
            if recovered:
                calls, text = recovered, None
        return ModelResponse(text=text, tool_calls=calls, usage=usage or Usage(), finish_reason=finish)

    # ── expose this agent AS a tool (the minimal multi-agent primitive) ──────
    def as_tool(self, name: str | None = None, description: str | None = None, *,
                input_field: str = "task", budget=None, cancel=None,
                max_depth: int | None = None, use_async: bool = False) -> Tool:
        """Wrap this agent as a :class:`Tool` so another agent can delegate to it.

        The sub-agent runs in an ISOLATED context (its own system prompt, tools,
        and step budget) and returns only its final answer — the narrow
        delegation contract (objective in, condensed result out). Its citations
        (``AgentResult.sources``) propagate into the returned ``ToolResult`` so a
        manager's ``[n]`` references still resolve; structured ``output`` rides in
        ``ToolResult.data``; and a non-final stop (budget/max_steps/cancelled) or
        error is surfaced in the content/``error`` so the manager sees *why* a
        worker came back empty.

        By default the wrapper is SYNC (calls :meth:`run`): valid under a
        manager's ``run()`` and dispatched off-thread under ``arun()``. Pass
        ``use_async=True`` for a fully-async tree (calls ``arun``; such a tool
        runs only under ``arun``/``astream``).

        Rate-limit invariant: the sub-agent reuses ITS OWN ``client`` — build
        every agent in a team from the SAME client so they share one process-wide
        ``RateLimiter`` (the gateway silently drops bursts). Forward a shared
        ``cancel`` so one cancellation stops the whole tree.

        Args:
            name: tool name (defaults to ``self.name`` or ``"agent"``).
            description: what the sub-agent does — what the manager routes on;
                make it specific (objective + expected output format).
            input_field: the single string parameter the manager fills (default ``"task"``).
            budget: optional :class:`Budget`; each delegation runs the sub-agent
                under a FRESH COPY (same caps, independent counters) so a worker
                hitting its cap degrades gracefully without aborting the manager.
            cancel: optional shared cancel token forwarded to each sub-run.
            max_depth: max delegation NESTING allowed beneath this tool. Tracked
                across the live call stack (not the Budget), so a manager→worker→
                sub-worker chain is bounded: at depth ``max_depth`` this tool
                returns an error result *without* running the sub-agent (rather
                than raising) — the unbounded-recursion backstop. ``None`` = no cap.
            use_async: emit an async wrapper (calls ``arun``) instead of sync.
        """
        tool_name = name or self.name or "agent"
        desc = description or (
            f"Delegate a subtask to the {tool_name!r} sub-agent. "
            "Provide a clear objective and the desired output format."
        )
        spec = ToolSpec(
            name=tool_name,
            description=desc,
            parameters={
                "type": "object",
                "properties": {input_field: {
                    "type": "string",
                    "description": "The objective for the sub-agent, including the "
                                   "desired output format.",
                }},
                "required": [input_field],
                "additionalProperties": False,
            },
        )

        def _child_budget():
            # Fresh per-delegation copy so a worker hitting its cap degrades
            # gracefully without ticking/aborting the manager's own budget.
            return replace(budget) if isinstance(budget, Budget) else budget

        def _to_result(res: AgentResult) -> ToolResult:
            if res.stopped == "error":
                return ToolResult(content=res.text or "", data=res.output,
                                  sources=list(res.sources),
                                  error=str(res.error) if res.error else "sub-agent error")
            content = res.text or ""
            if res.stopped != "final":  # surface truncation even when text is present
                note = f"[sub-agent {tool_name!r} stopped early: {res.stopped}]"
                content = f"{note} {content}".rstrip()
            return ToolResult(content=content, data=res.output, sources=list(res.sources))

        def _missing() -> ToolResult:
            return ToolResult(content="", error=f"missing required argument {input_field!r}")

        def _too_deep() -> ToolResult:
            return ToolResult(
                content="",
                error=f"max delegation depth {max_depth} reached; {tool_name!r} not run "
                      "(raise max_depth or flatten the agent tree).")

        if use_async:
            async def _delegate(**kwargs) -> ToolResult:
                task = kwargs.get(input_field)
                if not task:
                    return _missing()
                depth = _delegation_depth.get()
                if max_depth is not None and depth >= max_depth:
                    return _too_deep()
                token = _delegation_depth.set(depth + 1)
                try:
                    return _to_result(await self.arun(task, budget=_child_budget(), cancel=cancel))
                finally:
                    _delegation_depth.reset(token)
        else:
            def _delegate(**kwargs) -> ToolResult:
                task = kwargs.get(input_field)
                if not task:
                    return _missing()
                depth = _delegation_depth.get()
                if max_depth is not None and depth >= max_depth:
                    return _too_deep()
                token = _delegation_depth.set(depth + 1)
                try:
                    return _to_result(self.run(task, budget=_child_budget(), cancel=cancel))
                finally:
                    _delegation_depth.reset(token)

        _delegate.__name__ = tool_name
        _delegate.__doc__ = desc
        return Tool(_delegate, spec)

    # ── helpers shared by all drivers ────────────────────────────────────────
    def _specs(self, native: bool, unlocked=None) -> list[ToolSpec]:
        # unlocked is None for eager agents (full set, unchanged); a set/dict for deferred agents
        # (eager + currently-unlocked tools only — recomputed per model call as tools unlock).
        specs = list(self._registry.specs() if unlocked is None
                     else self._registry.active_specs(unlocked))
        if self.output_schema is not None and native:
            specs.append(ToolSpec(
                name="return_result",
                description="Return the final structured answer.",
                parameters=_json_schema(self.output_schema),
            ))
        return specs

    # ── deferred tools (opt-in via Agent.tool_search) ────────────────────────
    def _setup_deferred(self) -> None:
        """Partition the registry into eager + deferred and register the search_tools meta-tool.
        The catalog (name + 1-line) is always-on; deferred schemas load only on unlock."""
        from .tool_search import keyword_rank, make_search_tool
        ts = self.tool_search
        keep = (set(ts.eager) | set(self.finish_tool_names) | {ts.tool_name, "return_result"})
        names = [t.name for t in self._registry.specs()]            # current (user) tool names
        defer = ([n for n in names if n not in keep] if ts.deferred in (None, ("*",))
                 else [n for n in ts.deferred if n not in keep])
        self._registry.mark_deferred(*defer)
        self._catalog = self._registry.catalog()                   # user tools only (search not added yet)
        self._rank = ts.rank or keyword_rank                       # embedding_rank if configured, else keyword
        self._registry.add(make_search_tool(self._catalog, rank=self._rank,
                                            tool_name=ts.tool_name, search_limit=ts.search_limit))

    def _apply_unlock(self, unlocked, call, result) -> None:
        """After a tool runs under deferral: unlock matches from search_tools (or, if a guard
        stripped result.data, re-rank from the query), or auto-unlock a name-guessed deferred
        tool. LRU-evict beyond max_active; touching a tool refreshes it."""
        ts = self.tool_search
        if call.name == ts.tool_name:
            names = (result.data or {}).get("unlock") if isinstance(result.data, dict) else None
            if names is None:                                      # after-guard dropped .data -> re-rank
                q = (call.arguments or {}).get("query", "")
                names = self._rank(q, self._catalog, ts.search_limit) if q else []
        elif self._registry.is_deferred(call.name):
            names = [call.name]                                    # name-guessed deferred tool self-unlocks
        else:
            return
        for n in names:
            unlocked.pop(n, None)                                  # refresh-on-use (move to MRU end)
            unlocked[n] = True
        while len(unlocked) > ts.max_active:
            unlocked.popitem(last=False)                           # evict least-recently-used

    def _build_initial(self, prompt, memory, native: bool) -> list[Message]:
        msgs: list[Message] = []
        if self.system:
            msgs.append(Message.system(self.system))
        if self.output_schema is not None:
            if native:
                msgs.append(Message.system(
                    "When you have the final answer, call the return_result tool "
                    "with the structured result."))
            else:
                schema = json.dumps(_json_schema(self.output_schema))
                msgs.append(Message.system(
                    "When you have the final answer, reply with ONLY a JSON object "
                    f"matching this schema: {schema}"))
        if memory:
            msgs.extend(_coerce_messages(memory))
        if isinstance(prompt, str):
            msgs.append(Message.user(prompt))
        else:
            msgs.extend(_coerce_messages(prompt))
        return msgs

    @staticmethod
    def _tool_message(call, result: ToolResult) -> Message:
        content = ("ERROR: " + result.error) if result.error else (result.content or "")
        return Message(role="tool", tool_call_id=call.id, name=call.name, content=content)

    def _finalize(self, text, output, msgs, steps, usage, stopped, error=None) -> AgentResult:
        return AgentResult(
            text=text or "",
            output=output,
            messages=msgs,
            steps=steps,
            usage=usage,
            stopped=stopped,
            sources=_aggregate_sources(steps),
            error=error,
        )

    def _on_retry(self, step):
        def _cb(attempt, delay, te):
            self._emit(LLMRetry(attempt=attempt, delay=delay,
                                status=getattr(te, "status_code", None),
                                error=str(te), step=step))
        return _cb

    def _emit(self, event) -> None:
        """Emit a trace event; a broken tracer must never crash a run."""
        try:
            self.tracer.on_event(event)
        except Exception:  # pragma: no cover - defensive
            pass


# ════════════════════════════════════════════════════════════════════════════
# Module helpers
# ════════════════════════════════════════════════════════════════════════════

def _accumulate(buf: dict, ch) -> None:
    """Accumulate a streamed ``_ToolChunk`` fragment into the per-index buffer."""
    b = buf.setdefault(ch.index, {"id": None, "name": None, "args": ""})
    if ch.id:
        b["id"] = ch.id
    if ch.name:
        b["name"] = ch.name
    b["args"] += ch.args_fragment or ""


def _finish_answer(arguments: dict) -> str:
    """Extract the answer string from a finish/final_answer tool call."""
    if not isinstance(arguments, dict) or not arguments:
        return ""
    for key in ("answer", "final_answer", "value", "text", "result", "response"):
        if key in arguments and arguments[key] is not None:
            return str(arguments[key])
    # fall back to the first scalar value
    for v in arguments.values():
        if isinstance(v, (str, int, float, bool)):
            return str(v)
    return str(arguments)


def _coerce_messages(items) -> list[Message]:
    out = []
    for m in items:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, dict):
            out.append(Message(role=m.get("role", "user"), content=m.get("content", "")))
        else:  # ChatMessage-like with role/content
            out.append(Message(role=getattr(m, "role", "user"),
                               content=getattr(m, "content", str(m))))
    return out


def _aggregate_sources(steps) -> list:
    """Ordered, de-duplicated union of every tool result's sources -> [n] map."""
    seen = set()
    out = []
    for st in steps:
        for res in getattr(st, "tool_results", []) or []:
            for src in getattr(res, "sources", []) or []:
                key = src.id or src.url or src.title or id(src)
                if key in seen:
                    continue
                seen.add(key)
                out.append(src)
    return out


def _validate_output(schema, data):
    """Validate ``data`` against a pydantic model (v1 or v2).

    Returns ``(instance, None)`` on success or ``(None, error_str)`` on failure.
    """
    if not isinstance(data, dict):
        return None, "expected a JSON object"
    try:
        if hasattr(schema, "model_validate"):  # pydantic v2
            return schema.model_validate(data), None
        return schema.parse_obj(data), None  # pydantic v1
    except Exception as e:
        return None, str(e)


def _extract_json(text: str) -> dict:
    """Best-effort: pull the first balanced JSON object out of model text."""
    if not text:
        return {}
    start = text.find("{")
    if start == -1:
        return {}
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else {}
                except (TypeError, ValueError):
                    return {}
    return {}


def _render(obj) -> str:
    """Human-readable text for a structured output instance."""
    try:
        if hasattr(obj, "model_dump_json"):  # pydantic v2
            return obj.model_dump_json()
        if hasattr(obj, "json"):  # pydantic v1
            return obj.json()
    except Exception:  # pragma: no cover
        pass
    return str(obj)
