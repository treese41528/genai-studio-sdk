"""P0 deferred (searchable) tools — catalog/keyword_rank + the _drive unlock surgery (no network)."""

from __future__ import annotations

from genai_studio.agents import Agent, ToolResult, tool
from genai_studio.agents.tool_search import ToolSearch, keyword_rank, render_catalog
from genai_studio.agents.trace import NullTracer

from conftest import ScriptedClient, calls_tool, says


@tool
def read_parquet(path: str) -> str:
    """Read a parquet table from disk.

    Args:
        path: file path.
    """
    return f"read {path}"


@tool
def send_email(to: str) -> str:
    """Send an email to a recipient.

    Args:
        to: recipient address.
    """
    return f"sent to {to}"


@tool
def final_answer(answer: str) -> str:
    """Finish with the answer.

    Args:
        answer: the final answer.
    """
    return answer


def _names(tools):
    return {t.name for t in (tools or [])}


def _agent(client, **kw):
    return Agent(client=client, tools=[read_parquet, send_email, final_answer],
                 tracer=NullTracer(), **kw)


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_keyword_rank():
    cat = [("read_parquet", "Read a parquet table from disk"), ("send_email", "Send an email")]
    assert keyword_rank("read a parquet file", cat, 5) == ["read_parquet"]
    assert keyword_rank("xyzzy nothing", cat, 5) == []


def test_render_catalog_caps():
    big = [(f"t{i}", f"tool number {i}") for i in range(10)]
    txt = render_catalog(big, limit=3)
    assert "t0:" in txt and "and 7 more" in txt and "search_tools" in txt


# ── the loop surgery ─────────────────────────────────────────────────────────
def test_default_no_deferral_sends_all_tools():
    client = ScriptedClient([calls_tool("read_parquet", {"path": "x"}), says("ok")])
    _agent(client).run("go")
    t1 = _names(client.calls[0]["tools"])
    assert {"read_parquet", "send_email", "final_answer"} <= t1 and "search_tools" not in t1


def test_deferred_then_search_unlocks_one():
    client = ScriptedClient([
        calls_tool("search_tools", {"query": "read a parquet table"}),
        calls_tool("read_parquet", {"path": "/d/x.parquet"}),
        says("done"),
    ])
    res = _agent(client, tool_search=ToolSearch(deferred=("*",), eager=("final_answer",))).run("go")
    t1 = _names(client.calls[0]["tools"])                   # only eager + meta; deferred hidden
    assert "search_tools" in t1 and "final_answer" in t1
    assert "read_parquet" not in t1 and "send_email" not in t1
    t2 = _names(client.calls[1]["tools"])                   # read_parquet unlocked; send_email still hidden
    assert "read_parquet" in t2 and "send_email" not in t2
    assert res.text == "done"


def test_name_guessed_deferred_executes_and_autounlocks():
    client = ScriptedClient([calls_tool("send_email", {"to": "a@b.com"}), says("done")])
    res = _agent(client, tool_search=ToolSearch(deferred=("*",), eager=("final_answer",))).run("go")
    assert "send_email" not in _names(client.calls[0]["tools"])   # schema withheld...
    assert res.text == "done"                                      # ...but the call still executed
    assert "send_email" in _names(client.calls[1]["tools"])        # and auto-unlocked for next step


def test_lru_eviction_at_max_active():
    client = ScriptedClient([
        calls_tool("search_tools", {"query": "parquet table"}),
        calls_tool("search_tools", {"query": "send email recipient"}),
        says("done"),
    ])
    _agent(client, tool_search=ToolSearch(deferred=("*",), eager=("final_answer",), max_active=1)).run("go")
    t3 = _names(client.calls[2]["tools"])
    assert "send_email" in t3 and "read_parquet" not in t3        # 1st unlock evicted (LRU, cap=1)
