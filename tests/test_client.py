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


# ── text-emitted call formats beyond bare JSON (fences, tags, pythonic, prose) ─

def test_recovers_hermes_tool_call_tags():
    calls = _tool_calls_from_text(
        'Okay.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>')
    assert len(calls) == 1
    assert calls[0].name == "read_file" and calls[0].arguments == {"path": "a.py"}


def test_recovers_multiple_tool_call_tag_blocks():
    calls = _tool_calls_from_text(
        '<tool_call>{"name": "f", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "g", "arguments": {"x": 1}}</tool_call>')
    assert [c.name for c in calls] == ["f", "g"]


def test_recovers_unclosed_tool_call_tag():
    # truncated output (max_tokens) loses the closing tag
    calls = _tool_calls_from_text('<tool_call>{"name": "grep", "arguments": {"pattern": "TODO"}}')
    assert len(calls) == 1 and calls[0].name == "grep"


def test_recovers_llama_function_tag():
    calls = _tool_calls_from_text('<function=web_search>{"query": "purdue"}</function>')
    assert len(calls) == 1
    assert calls[0].name == "web_search" and calls[0].arguments == {"query": "purdue"}


def test_recovers_python_tag_prefix():
    calls = _tool_calls_from_text('<|python_tag|>{"name": "run_shell", "arguments": {"command": "ls"}}')
    assert len(calls) == 1 and calls[0].name == "run_shell"


def test_recovers_json_fenced_call():
    calls = _tool_calls_from_text('```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```')
    assert len(calls) == 1 and calls[0].name == "read_file"


def test_recovers_pythonic_call_listing():
    calls = _tool_calls_from_text('[read_file(path="src/app.py"), grep(pattern="main", glob="*.py")]')
    assert [c.name for c in calls] == ["read_file", "grep"]
    assert calls[0].arguments == {"path": "src/app.py"}
    calls = _tool_calls_from_text('read_file(path="a.py")')       # bare, unbracketed
    assert len(calls) == 1 and calls[0].arguments == {"path": "a.py"}


def test_pythonic_rejects_positional_and_code():
    assert _tool_calls_from_text('print("hi")') == []             # positional args = ordinary code
    assert _tool_calls_from_text('f(x) = x**2') == []             # math, not a call
    assert _tool_calls_from_text('[1, 2, 3]') == []


def test_recovers_call_after_short_prose_preamble():
    calls = _tool_calls_from_text(
        "Let's read the file to see what it contains.\n"
        '{"name": "read_file", "arguments": {"path": "notes.md"}}')
    assert len(calls) == 1 and calls[0].name == "read_file"


def test_recovers_fenced_call_after_short_prose_preamble():
    calls = _tool_calls_from_text(
        "I'll search the codebase.\n```json\n"
        '{"name": "grep", "arguments": {"pattern": "def main"}}\n```')
    assert len(calls) == 1 and calls[0].name == "grep"


def test_trailing_json_without_args_key_stays_an_answer():
    # data that merely ends an answer must not become a call
    assert _tool_calls_from_text('Here is the config:\n{"name": "my-app"}') == []


def test_trailing_json_after_long_answer_stays_an_answer():
    long_answer = "word " * 100                                   # > 300-char preamble
    assert _tool_calls_from_text(
        long_answer + '\n{"name": "read_file", "arguments": {"path": "x"}}') == []


def test_tool_alias_requires_args_key():
    calls = _tool_calls_from_text('{"tool": "read_file", "tool_input": {"path": "x"}}')
    assert len(calls) == 1
    assert calls[0].name == "read_file" and calls[0].arguments == {"path": "x"}
    assert _tool_calls_from_text('{"tool": "hammer", "price": 3}') == []  # data, not a call


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
