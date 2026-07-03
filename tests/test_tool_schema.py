"""@tool JSON-Schema derivation from type hints + docstrings."""

from __future__ import annotations

import enum
from typing import Literal, Optional

import pytest

from genai_studio.agents import tool


# Types referenced in annotations must be module-level so that, under
# `from __future__ import annotations`, get_type_hints can resolve the strings
# (exactly how a real user / notebook cell defines them).
class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


try:
    import pydantic

    class Point(pydantic.BaseModel):
        x: int
        y: int

    class Shape(pydantic.BaseModel):
        name: str
        origin: Point
except ImportError:  # pragma: no cover
    Point = Shape = None


def test_basic_types_and_required():
    @tool
    def search(query: str, limit: int = 10) -> str:
        """Search the literature.

        Args:
            query: search terms.
            limit: maximum number of results.
        """
        return ""

    p = search.spec.parameters
    assert search.spec.name == "search"
    assert search.spec.description == "Search the literature."
    assert p["properties"]["query"]["type"] == "string"
    assert p["properties"]["limit"]["type"] == "integer"
    assert p["properties"]["limit"]["default"] == 10
    assert p["required"] == ["query"]  # limit has a default -> optional
    assert "search terms" in p["properties"]["query"]["description"]


def test_bool_before_int_and_float():
    @tool
    def f(flag: bool, x: float) -> str:
        "f."
        return ""

    props = f.spec.parameters["properties"]
    assert props["flag"]["type"] == "boolean"  # not "integer"
    assert props["x"]["type"] == "number"


def test_optional_and_list_and_literal():
    @tool
    def g(tags: list[str], mode: Literal["fast", "slow"] = "fast",
          note: Optional[str] = None) -> str:
        "g."
        return ""

    props = g.spec.parameters["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["mode"]["enum"] == ["fast", "slow"]
    assert props["mode"]["type"] == "string"
    assert g.spec.parameters["required"] == ["tags"]  # mode/note optional


def test_enum_uses_values():
    @tool
    def h(c: Color) -> str:
        "h."
        return ""

    assert h.spec.parameters["properties"]["c"]["enum"] == ["red", "green"]


def test_overrides_and_unannotated():
    @tool(name="renamed", description="custom")
    def f(x) -> str:  # x unannotated -> open schema
        "ignored summary."
        return ""

    assert f.spec.name == "renamed"
    assert f.spec.description == "custom"
    assert f.spec.parameters["properties"]["x"] == {}


def test_tool_still_callable():
    @tool
    def add(a: int, b: int) -> str:
        "add."
        return str(a + b)

    assert add(a=2, b=3) == "5"  # @tool does not break direct calls


def test_pydantic_nested_refs_inlined():
    pytest.importorskip("pydantic")

    @tool
    def draw(shape: Shape) -> str:
        "draw."
        return ""

    schema = draw.spec.parameters["properties"]["shape"]
    # $ref / $defs / definitions must be inlined away (watch-item 1).
    blob = str(schema)
    assert "$ref" not in blob and "$defs" not in blob and "definitions" not in blob
    assert schema["properties"]["origin"]["properties"]["x"]["type"] == "integer"
