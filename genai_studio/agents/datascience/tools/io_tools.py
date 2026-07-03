"""
Data I/O tools: ``load_table`` (read a tabular file) + ``sql_query`` (read-only SQL).

These turn the SDK from "analyze bundled toy data" into "analyze *your* data":

- ``load_table`` — read CSV/TSV/Parquet/Excel/JSON from a local path or http(s) URL
  into a pandas DataFrame, register it in the shared ``python_exec`` namespace, and
  return a compact summary. Format-specific engines (pyarrow/openpyxl) are required
  only on demand, with a clear install hint.
- ``sql_query`` — run a READ-ONLY SQL ``SELECT`` against a SQLite database file
  (opened ``mode=ro``), returning the rows as a DataFrame.

SAFETY: ``load_table`` will fetch a model-chosen URL — an SSRF surface. Gate it with
a :class:`Guard` (or use only local paths) when the agent processes untrusted input.
"""

from __future__ import annotations

import os
import urllib.parse

from genai_studio.agents import ToolResult, tool

from .._format import MAX_CHARS, summarize
from .._guard import _require


def _ext(source: str) -> str:
    # strip any query/fragment whether or not there's a scheme (urlparse only does
    # so for full URLs), so 'f.parquet?token=...' resolves to '.parquet', not garbage.
    if "://" in source:
        path = urllib.parse.urlsplit(source).path
    else:
        path = source.split("?", 1)[0].split("#", 1)[0]
    return os.path.splitext(path)[1].lower()


def _read_any(pd, source: str, sheet):
    ext = _ext(source)
    if ext in (".csv", ".txt"):
        return pd.read_csv(source)
    if ext == ".tsv":
        return pd.read_csv(source, sep="\t")
    if ext in (".parquet", ".pq"):
        _require("pyarrow")                       # pandas' parquet engine
        return pd.read_parquet(source)
    if ext in (".xlsx", ".xls"):
        _require("openpyxl")                      # pandas' excel engine
        return pd.read_excel(source, sheet_name=0 if sheet is None else sheet)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(source, lines=True)
    if ext == ".json":
        return pd.read_json(source)
    # No blind read_csv fallback: a wrong guess silently loads garbage the model
    # then reasons on. Make the failure explicit instead.
    raise ValueError(f"unsupported or undetectable format (ext={ext!r}); pass a "
                     "path/URL ending in .csv/.tsv/.parquet/.xlsx/.json")


def make_load_table(namespace: dict | None = None):
    """Build a ``load_table`` tool, optionally sharing a namespace with ``python_exec``."""

    @tool(name="load_table",
          description="Load a tabular file (CSV/TSV/Parquet/Excel/JSON) from a local "
                      "path or http(s) URL into a DataFrame and summarise it.")
    def load_table(source: str, name: str = "df", sheet: str | None = None) -> ToolResult:
        """Load tabular data into a pandas DataFrame.

        The shared ``df`` alias always points at the MOST RECENTLY loaded table.

        Args:
            source: a local path or http(s) URL to a .csv/.tsv/.parquet/.xlsx/.json file.
            name: variable name to register the DataFrame under (default 'df').
            sheet: worksheet name or index for Excel files (optional).
        """
        if "://" in source and urllib.parse.urlsplit(source).scheme not in ("http", "https"):
            return ToolResult(content="",   # enforce the advertised contract: no file://, ftp://, s3://…
                              error=f"unsupported URL scheme in {source!r}; use a local path or http(s) URL")
        pd = _require("pandas")
        try:
            df = _read_any(pd, source, sheet)
        except ImportError:
            raise                                 # _require's install hint must reach the user
        except Exception as exc:
            return ToolResult(content="", error=f"load_table failed for {source!r}: {exc}")
        if namespace is not None:
            namespace[name] = df
            namespace["df"] = df
        hint = f"\n(available in python_exec as `{name}` and `df`)" if namespace is not None else ""
        return ToolResult(content=f"Loaded {source!r} -> {name}:\n{summarize(df)}{hint}", data=df)

    return load_table


_RO_PREFIXES = ("select", "with", "explain", "values")


def _is_readonly(query: str) -> bool:
    q = query.strip().lstrip("(").lstrip().lower()
    return q.startswith(_RO_PREFIXES)


def make_sql_query(database: str, *, namespace: dict | None = None, max_rows: int = 1000):
    """Build a ``sql_query`` tool bound to a SQLite ``database`` file (opened read-only).

    Only read statements (SELECT/WITH/EXPLAIN/VALUES) are accepted, and the file is
    opened ``mode=ro`` so writes are impossible at the driver level too. For SQL over
    in-memory DataFrames, use duckdb inside ``python_exec`` instead.
    """

    @tool(name="sql_query",
          description="Run a READ-ONLY SQL SELECT against the configured SQLite "
                      "database and return the rows.")
    def sql_query(query: str) -> ToolResult:
        """Execute a read-only SQL query (SQLite).

        Args:
            query: a SELECT / WITH / EXPLAIN statement.
        """
        if not _is_readonly(query):
            return ToolResult(content="", error="only read-only SELECT/WITH queries are allowed")
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{os.path.abspath(database)}?mode=ro", uri=True)
        except Exception as exc:
            return ToolResult(content="", error=f"cannot open database {database!r}: {exc}")
        try:
            try:
                pd = _require("pandas")
                df = pd.read_sql_query(query, con)
                if namespace is not None:
                    namespace["result"] = df
                note = f"\n({len(df)} rows; in python_exec as `result`)" if namespace is not None else \
                    f"\n({len(df)} rows)"
                return ToolResult(content=f"{summarize(df)}{note}", data=df)
            except ImportError:
                cur = con.execute(query)
                cols = [d[0] for d in (cur.description or [])]
                rows = cur.fetchmany(max_rows)
                lines = [" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows]
                return ToolResult(content="\n".join(lines)[:MAX_CHARS],
                                  data={"columns": cols, "rows": rows})
        except Exception as exc:
            return ToolResult(content="", error=f"sql_query failed: {exc}")
        finally:
            con.close()

    return sql_query
