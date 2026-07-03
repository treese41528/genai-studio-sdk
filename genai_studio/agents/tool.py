"""
``@tool`` — turn a typed Python function into a model-callable :class:`Tool`.

This is the first of the framework's four seams. A decorated function keeps
working as an ordinary callable (so students can unit-test it directly), and
gains a ``.spec`` (:class:`ToolSpec`) whose ``parameters`` are a JSON Schema
derived from the function's type hints + Google-style docstring. That schema is
MCP-compatible by construction and is exactly what the model sees.

Teaching note: ``my_tool.spec.parameters`` is meant to be *printed* — students
watch their function become a tool definition.
"""

from __future__ import annotations

import enum
import inspect
import json
import typing
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ``X | None`` (PEP 604) produces ``types.UnionType`` on 3.10+; absent on 3.9.
try:  # pragma: no cover - trivial version guard
    from types import UnionType as _UnionType  # type: ignore
except ImportError:  # pragma: no cover
    _UnionType = ()  # sentinel that never matches an isinstance/identity check

if typing.TYPE_CHECKING:  # avoid a tool<->client import cycle at runtime
    from .client import ToolCall


# ════════════════════════════════════════════════════════════════════════════
# Result / spec dataclasses
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Source:
    """Provenance for one piece of evidence (Postyl ``[n]`` citations).

    Every field is optional so the basic path never has to think about it.
    """

    id: str | None = None
    title: str | None = None
    url: str | None = None
    snippet: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a tool returns. Only ``content`` is ever shown to the model.

    - ``content``  : the string the model sees (always).
    - ``sources``  : provenance, carried through the loop to ``AgentResult.sources``
                     for citations; ignored by the basic path.
    - ``data``     : a structured payload (a DataFrame/figure/dict) for UI/DS
                     rendering; never sent to the model.
    - ``error``    : set when the tool failed; fed back so the model can recover.
    """

    content: str
    sources: list[Source] = field(default_factory=list)
    data: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def from_return(cls, value: Any) -> "ToolResult":
        """Coerce a tool's raw return value into a ``ToolResult``.

        ``ToolResult`` -> as-is; ``str`` -> ``content``; anything else ->
        ``json.dumps`` (falling back to ``repr`` for non-JSON-able objects),
        with the original object preserved in ``data`` so a UI can still render it.
        """
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, str):
            return cls(content=value)
        if value is None:
            return cls(content="")
        try:
            return cls(content=json.dumps(value, default=str), data=value)
        except (TypeError, ValueError):
            return cls(content=repr(value), data=value)


@dataclass(frozen=True)
class ToolSpec:
    """A tool's wire definition: name, description, and JSON-Schema parameters."""

    name: str
    description: str
    parameters: dict  # JSON Schema (object) — MCP-compatible

    def to_openai(self) -> dict:
        """Render as an OpenAI ``tools=[...]`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ════════════════════════════════════════════════════════════════════════════
# The Tool wrapper + @tool decorator
# ════════════════════════════════════════════════════════════════════════════

class Tool:
    """Callable wrapper around a function, carrying its :class:`ToolSpec`.

    ``Tool`` stays transparently callable (``my_tool(x=1)`` runs the function),
    which keeps the teaching path simple and tools unit-testable in isolation.
    """

    def __init__(self, func: Callable, spec: ToolSpec):
        self.func = func
        self.spec = spec
        self.is_async = inspect.iscoroutinefunction(func)
        # Preserve identity for nicer reprs / introspection.
        self.__name__ = getattr(func, "__name__", spec.name)
        self.__doc__ = getattr(func, "__doc__", None)

    @property
    def name(self) -> str:
        return self.spec.name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def run(self, arguments: dict) -> ToolResult:
        """Invoke the (sync) function with kwargs and coerce to ``ToolResult``.

        Exceptions are NOT caught here — the :class:`ToolRegistry` owns the
        capture-into-``error`` policy so the loop sees a uniform result.
        """
        return ToolResult.from_return(self.func(**(arguments or {})))

    async def arun(self, arguments: dict) -> ToolResult:
        """Async counterpart used by ``Agent.arun``/``astream``."""
        out = self.func(**(arguments or {}))
        if inspect.isawaitable(out):
            out = await out
        return ToolResult.from_return(out)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "async " if self.is_async else ""
        return f"<{kind}Tool {self.spec.name!r}>"


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable | Tool:
    """Decorator turning a typed function into a :class:`Tool`.

    Usage::

        @tool
        def search(query: str, limit: int = 10) -> str:
            \"\"\"Search the literature.

            Args:
                query: Search terms.
                limit: Maximum number of results.
            \"\"\"
            ...

        @tool(name="lookup", description="...")
        def search(...): ...

    Works on sync *and* async functions (``Tool.is_async`` records which).
    """

    def wrap(fn: Callable) -> Tool:
        spec = _build_spec(fn, name=name, description=description)
        return Tool(fn, spec)

    return wrap(func) if func is not None else wrap


# ════════════════════════════════════════════════════════════════════════════
# Schema derivation
# ════════════════════════════════════════════════════════════════════════════

def _build_spec(
    fn: Callable, *, name: str | None = None, description: str | None = None
) -> ToolSpec:
    """Derive a :class:`ToolSpec` from a function's signature + docstring."""
    tool_name = name or getattr(fn, "__name__", "tool")
    doc = inspect.getdoc(fn) or ""
    summary, arg_docs = _parse_docstring(doc)
    tool_desc = description if description is not None else summary

    # Resolve annotations, tolerating string hints (`from __future__ import
    # annotations`) and unresolvable forward refs.
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:  # pragma: no cover - defensive
        hints = getattr(fn, "__annotations__", {}) or {}

    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            warnings.warn(
                f"@tool {tool_name!r}: *args/**kwargs ({pname}) cannot be "
                "represented in a JSON Schema and were skipped.",
                stacklevel=3,
            )
            continue

        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            schema: dict = {}  # unannotated -> "any JSON"
            optional = False
        else:
            schema = _type_to_schema(annotation, tool_name=tool_name)
            optional = _is_optional(annotation)

        if pname in arg_docs and "description" not in schema:
            schema = {**schema, "description": arg_docs[pname]}

        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            # Record JSON-able defaults; helps the model and is valid schema.
            try:
                json.dumps(param.default)
                schema = {**schema, "default": param.default}
            except (TypeError, ValueError):
                pass

        properties[pname] = schema
        if not has_default and not optional:
            required.append(pname)

    parameters = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required

    return ToolSpec(name=tool_name, description=tool_desc, parameters=parameters)


def _is_optional(tp: Any) -> bool:
    """True for ``Optional[T]`` / ``T | None`` (i.e. NoneType in a Union)."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or (origin is not None and origin is _UnionType):
        return type(None) in typing.get_args(tp)
    return False


def _type_to_schema(tp: Any, *, tool_name: str = "tool") -> dict:
    """Map a Python type hint to a JSON Schema fragment."""
    if tp is None or tp is type(None):
        return {"type": "null"}
    if tp is Any:
        return {}

    origin = typing.get_origin(tp)

    # Union / Optional / X | None -> drop NoneType, anyOf the rest.
    if origin is typing.Union or (origin is not None and origin is _UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        schemas = [_type_to_schema(a, tool_name=tool_name) for a in members]
        if len(schemas) == 1:
            return schemas[0]
        return {"anyOf": schemas}

    # typing.Literal[...] -> enum (type inferred from members).
    if origin is typing.Literal:
        values = list(typing.get_args(tp))
        schema = {"enum": values}
        jtypes = {_PRIMITIVE_JSON.get(type(v)) for v in values}
        jtypes.discard(None)
        if len(jtypes) == 1:
            schema = {"type": jtypes.pop(), "enum": values}
        return schema

    # list / List[T]
    if origin in (list, typing.List):
        args = typing.get_args(tp)
        if args:
            return {"type": "array", "items": _type_to_schema(args[0], tool_name=tool_name)}
        return {"type": "array"}

    # tuple / set -> array
    if origin in (tuple, set, frozenset, typing.Tuple, typing.Set):
        return {"type": "array"}

    # dict / Dict[str, V]
    if origin in (dict, typing.Dict):
        args = typing.get_args(tp)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _type_to_schema(args[1], tool_name=tool_name)}
        return {"type": "object"}

    # Plain classes.
    if isinstance(tp, type):
        if issubclass(tp, bool):  # MUST precede int (bool subclasses int)
            return {"type": "boolean"}
        if issubclass(tp, int):
            return {"type": "integer"}
        if issubclass(tp, float):
            return {"type": "number"}
        if issubclass(tp, str):
            return {"type": "string"}
        if issubclass(tp, enum.Enum):
            values = [e.value for e in tp]
            jtypes = {_PRIMITIVE_JSON.get(type(v)) for v in values}
            jtypes.discard(None)
            if len(jtypes) == 1:
                return {"type": jtypes.pop(), "enum": values}
            return {"enum": [e.name for e in tp]}
        if _is_pydantic_model(tp):
            return _json_schema(tp)

    warnings.warn(
        f"@tool {tool_name!r}: could not map type hint {tp!r} to JSON Schema; "
        "using an open schema. Annotate with a supported type for a tighter spec.",
        stacklevel=4,
    )
    return {}


_PRIMITIVE_JSON = {str: "string", bool: "boolean", int: "integer", float: "number"}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {arg: description}).

    Recognises Google-style ``Args:`` / ``Arguments:`` / ``Parameters:`` blocks
    with ``name: desc`` or ``name (type): desc`` lines plus indented continuations.
    """
    if not doc:
        return "", {}

    lines = doc.splitlines()
    section_headers = ("args:", "arguments:", "parameters:", "returns:", "return:",
                       "raises:", "yields:", "examples:", "example:", "note:", "notes:")

    # Summary = everything before the first recognised section header.
    summary_lines: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() in section_headers:
            break
        summary_lines.append(lines[i])
        i += 1
    summary = " ".join(s.strip() for s in summary_lines).strip()

    # Find the Args-style block and parse its entries.
    arg_docs: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low in ("args:", "arguments:", "parameters:"):
            in_args = True
            current = None
            continue
        if in_args and low in section_headers:  # a new (non-args) section ends the block
            break
        if not in_args:
            continue
        if not stripped:
            continue
        # `name: desc` or `name (type): desc`
        head, sep, rest = stripped.partition(":")
        looks_like_param = sep and " " not in head.split("(")[0].strip()
        if looks_like_param:
            pname = head.split("(")[0].strip()
            arg_docs[pname] = rest.strip()
            current = pname
        elif current is not None:  # continuation line
            arg_docs[current] = (arg_docs[current] + " " + stripped).strip()
    return summary, arg_docs


# ════════════════════════════════════════════════════════════════════════════
# pydantic schema shim (v1 + v2) — shared with structured output (M2)
# ════════════════════════════════════════════════════════════════════════════

def _is_pydantic_model(tp: Any) -> bool:
    try:
        import pydantic
    except ImportError:
        return False
    return isinstance(tp, type) and issubclass(tp, pydantic.BaseModel)


def _json_schema(model: Any) -> dict:
    """JSON Schema for a pydantic model, supporting **both v1 and v2**, with
    nested ``$defs`` / ``definitions`` **inlined** (some OpenAI-compatible
    gateways choke on ``$ref``). See build watch-item 1.
    """
    if hasattr(model, "model_json_schema"):  # pydantic v2
        schema = model.model_json_schema()
    elif hasattr(model, "schema"):  # pydantic v1
        schema = model.schema()
    else:
        raise TypeError(f"{model!r} is not a pydantic BaseModel")
    return _inline_refs(schema)


def _inline_refs(schema: dict) -> dict:
    """Resolve ``$ref`` against the schema's own ``$defs``/``definitions`` and
    drop the defs tables. Recursion-guarded; a cyclic ref degrades to ``{}``."""
    defs = {}
    for key in ("$defs", "definitions"):
        defs.update(schema.get(key, {}) or {})

    def resolve(node, seen: frozenset):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                name = ref.split("/")[-1]
                if name in seen or name not in defs:
                    return {}  # cycle or missing -> open schema
                merged = resolve(defs[name], seen | {name})
                # Carry sibling keys (e.g. description) alongside the resolved ref.
                extra = {k: v for k, v in node.items() if k != "$ref"}
                if extra:
                    merged = {**merged, **{k: resolve(v, seen) for k, v in extra.items()}}
                return merged
            return {k: resolve(v, seen) for k, v in node.items()
                    if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve(schema, frozenset())


# ════════════════════════════════════════════════════════════════════════════
# Tool registry
# ════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Holds the agent's tools and dispatches calls to them.

    ``execute`` / ``aexecute`` **always** return a ``ToolResult`` — a raising
    tool becomes ``ToolResult.error`` and an unknown tool name returns an error
    listing the valid names (so a model — especially under ReAct — can self-correct).
    """

    def __init__(self, tools: Iterable = ()):
        self._tools: dict[str, Tool] = {}
        self._deferred: set[str] = set()        # names whose schema is withheld until unlocked
        for t in tools:
            self.add(t)

    def add(self, t) -> None:
        tool_obj = t if isinstance(t, Tool) else tool(t)
        if tool_obj.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool_obj.name!r}")
        self._tools[tool_obj.name] = tool_obj

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def openai_tools(self) -> list[dict]:
        return [t.spec.to_openai() for t in self._tools.values()]

    # ── deferred-tool partition (opt-in, used only when Agent.tool_search is set) ──
    def mark_deferred(self, *names: str) -> None:
        """Withhold these tools' schemas until they are unlocked (they still execute if called)."""
        self._deferred.update(n for n in names if n in self._tools)

    def is_deferred(self, name: str) -> bool:
        return name in self._deferred

    def catalog(self) -> list:
        """``(name, first-line-of-description)`` for every registered tool — the always-on
        lightweight listing the model searches over."""
        out = []
        for name, t in self._tools.items():
            lines = (t.spec.description or "").strip().splitlines()
            out.append((name, lines[0] if lines else ""))
        return out

    def active_specs(self, unlocked) -> list[ToolSpec]:
        """Specs for the eager (non-deferred) tools plus the currently-``unlocked`` deferred ones."""
        return [t.spec for name, t in self._tools.items()
                if name not in self._deferred or name in unlocked]

    def _unknown(self, name: str) -> ToolResult:
        available = ", ".join(self._tools) or "(none)"
        return ToolResult(
            content="",
            error=f"Unknown tool {name!r}. Available tools: {available}.",
        )

    def execute(self, call: "ToolCall") -> ToolResult:
        t = self.get(call.name)
        if t is None:
            return self._unknown(call.name)
        try:
            return t.run(call.arguments)
        except Exception as exc:  # never let a tool crash the loop
            return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")

    async def aexecute(self, call: "ToolCall") -> ToolResult:
        t = self.get(call.name)
        if t is None:
            return self._unknown(call.name)
        try:
            return await t.arun(call.arguments)
        except Exception as exc:
            return ToolResult(content="", error=f"{type(exc).__name__}: {exc}")
