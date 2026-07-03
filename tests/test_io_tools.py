"""Tests for load_table + sql_query (real local files; pandas available in dev)."""

from __future__ import annotations

import sqlite3

import pytest

from genai_studio.agents.datascience.tools.io_tools import (
    _ext, _is_readonly, make_load_table, make_sql_query,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def test_ext_handles_paths_and_urls():
    assert _ext("/data/x.CSV") == ".csv"
    assert _ext("https://h.org/d/file.parquet?token=abc") == ".parquet"
    assert _ext("noext") == ""


def test_is_readonly_accepts_reads_rejects_writes():
    assert _is_readonly("SELECT * FROM t")
    assert _is_readonly("  with cte as (select 1) select * from cte")
    assert _is_readonly("EXPLAIN QUERY PLAN SELECT 1")
    assert not _is_readonly("DELETE FROM t")
    assert not _is_readonly("DROP TABLE t")
    assert not _is_readonly("insert into t values (1)")


# ── load_table ───────────────────────────────────────────────────────────────
def test_load_table_csv_registers_in_namespace(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    ns = {}
    load_table = make_load_table(ns)
    res = load_table(str(p), name="mydf")
    assert res.error is None
    assert "mydf" in ns and "df" in ns
    assert list(ns["mydf"].columns) == ["a", "b"] and len(ns["mydf"]) == 2
    assert "available in python_exec" in res.content


def test_load_table_json_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"x": 1}\n{"x": 2}\n')
    res = make_load_table()(str(p))
    assert res.error is None and len(res.data) == 2


def test_load_table_bad_file_is_error_not_crash(tmp_path):
    res = make_load_table()(str(tmp_path / "missing.csv"))
    assert res.error and "load_table failed" in res.error


def test_load_table_unsupported_format_is_explicit_error(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><body>not a table</body></html>")
    res = make_load_table()(str(p))
    assert res.error and "unsupported or undetectable format" in res.error   # no silent CSV guess


def test_load_table_rejects_non_http_scheme():
    res = make_load_table()("ftp://host/data.csv")
    assert res.error and "unsupported URL scheme" in res.error
    res2 = make_load_table()("file:///etc/passwd")
    assert res2.error and "unsupported URL scheme" in res2.error


# ── sql_query ────────────────────────────────────────────────────────────────
def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (name TEXT, age INT)")
    con.executemany("INSERT INTO people VALUES (?, ?)", [("Ada", 36), ("Grace", 85)])
    con.commit()
    con.close()


def test_sql_query_select_returns_rows(tmp_path):
    db = tmp_path / "people.db"
    _make_db(str(db))
    ns = {}
    sql_query = make_sql_query(str(db), namespace=ns)
    res = sql_query("SELECT name, age FROM people ORDER BY age")
    assert res.error is None
    assert list(res.data["name"]) == ["Ada", "Grace"]
    assert "result" in ns


def test_sql_query_rejects_writes(tmp_path):
    db = tmp_path / "people.db"
    _make_db(str(db))
    sql_query = make_sql_query(str(db))
    res = sql_query("DELETE FROM people")
    assert res.error and "read-only" in res.error


def test_sql_query_readonly_open_blocks_writes_at_driver(tmp_path):
    # even a sneaky write that passed the prefix check would fail: file is mode=ro.
    db = tmp_path / "people.db"
    _make_db(str(db))
    sql_query = make_sql_query(str(db))
    # 'with' prefix is allowed but the statement can't write under mode=ro
    res = sql_query("SELECT count(*) AS n FROM people")
    assert res.error is None and int(res.data["n"][0]) == 2


def test_sql_query_missing_db_is_error(tmp_path):
    res = make_sql_query(str(tmp_path / "nope.db"))("SELECT 1")
    assert res.error  # cannot open (mode=ro requires the file to exist)
