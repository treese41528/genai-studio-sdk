"""Client robustness: recovering tool calls a model emits as text content."""

from __future__ import annotations

from genai_studio.agents.client import _tool_calls_from_text


def test_recovers_openai_shaped_tool_call_in_content():
    # The exact shape observed on the Purdue gateway with qwen.
    content = ('{"id": "call_abc", "type": "function", "function": '
               '{"name": "lookup", "arguments": {"keyword": "Zubrowka"}}}')
    calls = _tool_calls_from_text(content)
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert calls[0].arguments == {"keyword": "Zubrowka"}


def test_recovers_bare_name_arguments():
    calls = _tool_calls_from_text('{"name": "search", "arguments": {"entity": "Lutz"}}')
    assert len(calls) == 1 and calls[0].name == "search"
    assert calls[0].arguments == {"entity": "Lutz"}


def test_recovers_arguments_as_json_string():
    calls = _tool_calls_from_text('{"function": {"name": "f", "arguments": "{\\"a\\": 1}"}}')
    assert calls[0].name == "f" and calls[0].arguments == {"a": 1}


def test_recovers_tool_calls_list_wrapper():
    calls = _tool_calls_from_text('{"tool_calls": [{"function": {"name": "g", "arguments": {}}}]}')
    assert len(calls) == 1 and calls[0].name == "g"


def test_recovers_function_as_name_string():
    # observed with MCP tool calls: "function" holds the NAME directly (not a nested object)
    calls = _tool_calls_from_text(
        '{"function": "mcp__filesystem__list_allowed_directories", "arguments": {}}')
    assert len(calls) == 1 and calls[0].name == "mcp__filesystem__list_allowed_directories"


def test_recovers_underscoreless_toolcalls_alias():
    # observed on MATH-500 grounded runs: qwen emits {"toolcalls": [...]} (no underscore),
    # which used to leak into the final answer instead of being executed
    calls = _tool_calls_from_text(
        '{"toolcalls": [{"name": "symbolic_math", "args": {"operation": "solve", "expression": "x-1"}}]}')
    assert len(calls) == 1 and calls[0].name == "symbolic_math"
    assert calls[0].arguments == {"operation": "solve", "expression": "x-1"}


def test_plain_answer_is_not_a_tool_call():
    assert _tool_calls_from_text("The answer is 42.") == []
    assert _tool_calls_from_text('{"summary": "just data", "n": 3}') == []  # no name/function
    assert _tool_calls_from_text("") == []
    assert _tool_calls_from_text(None) == []


def test_rate_limiter_spaces_requests():
    import time

    from genai_studio.agents.client import RateLimiter
    rl = RateLimiter(rpm=1200)  # 0.05s spacing
    t0 = time.monotonic()
    for _ in range(6):
        rl.acquire()
    # 6 requests => 5 gaps of 0.05s = 0.25s minimum
    assert time.monotonic() - t0 >= 0.25 - 0.02


def test_disabled_limiter_does_not_wait():
    import time

    from genai_studio.agents.client import RateLimiter
    rl = RateLimiter(rpm=0)  # disabled
    t0 = time.monotonic()
    for _ in range(1000):
        rl.acquire()
    assert time.monotonic() - t0 < 0.05
