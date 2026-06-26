"""Research pipeline: arXiv -> SQLite -> SQL queries -> R analysis.

A realistic agentic-data-science loop. The data plumbing needs NO LLM:

1. ``arxiv_search`` pulls papers for several topics (keyless, real network).
2. We INGEST them into a SQLite table. Note the read/write split: writes live in
   trusted setup code, because ``sql_query`` is READ-ONLY by design — so the tool
   you hand an agent can never mutate (or DROP) the database.
3. ``sql_query`` answers questions over the database (read-only SELECT).
4. ``r_exec`` runs R on a CSV export (base R, no extra packages) for statistics.

The read/write split is the security lesson: the *source-of-truth* DB is written
ONLY by trusted setup code; the Agent is deliberately handed read-only tools
(``sql_query`` + ``r_exec``), so it can analyse but never mutate or DROP it. Any
agent-driven *derived* tables would go through ``python_exec`` against a SEPARATE
scratch DB — never this one.

The optional final section hands those read-only tools to an Agent so it answers a
natural-language question itself (needs the gateway / GENAI_STUDIO_API_KEY).

Run: python examples/13_arxiv_to_sql.py
"""

from __future__ import annotations

import atexit
import csv as csvlib
import os
import shutil
import sqlite3
import tempfile

from genai_studio.agents.datascience.tools.io_tools import make_sql_query
from genai_studio.agents.datascience.tools.r_exec import make_r_exec
from genai_studio.agents.tools.academic import arxiv_search

TOPICS = ["retrieval augmented generation", "graph neural networks", "diffusion models"]
WORKDIR = tempfile.mkdtemp(prefix="arxiv_demo_")
atexit.register(lambda: shutil.rmtree(WORKDIR, ignore_errors=True))   # don't leak the tempdir
DB = os.path.join(WORKDIR, "papers.db")
CSV = os.path.join(WORKDIR, "papers.csv")


def ingest(db_path: str) -> int:
    """Fetch papers per topic and WRITE them into SQLite. Writes belong in trusted
    setup code — ``sql_query`` is read-only, so an agent never mutates this DB."""
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE IF NOT EXISTS papers "
                "(topic TEXT, title TEXT, year INTEGER, n_authors INTEGER, url TEXT, abstract TEXT)")
    rows = []
    for topic in TOPICS:
        res = arxiv_search(topic, max_results=8)
        if res.error:
            print(f"  ! {topic}: {res.error}")
            continue
        for p in res.data:
            # a non-parseable/missing year becomes NULL; such rows are excluded from the R export
            year = int(p["year"]) if (p.get("year") or "").isdigit() else None
            rows.append((topic, p["title"], year, len(p.get("authors", [])),
                         p.get("url"), (p.get("abstract") or "")[:500]))
        print(f"  + {topic}: {len(res.data)} papers")
    con.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return len(rows)


def export_csv(db_path: str, csv_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.execute("SELECT topic, year, n_authors FROM papers WHERE year IS NOT NULL")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    con.close()
    with open(csv_path, "w", newline="") as f:
        w = csvlib.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(f"workdir: {WORKDIR}\n")

    print("1) Fetch arXiv papers + ingest into SQLite (writes via trusted code)")
    n = ingest(DB)
    print(f"   ingested {n} papers\n")
    if n == 0:
        raise SystemExit("no papers fetched (network down?) — nothing to query.")

    print("2) Query the database — read-only sql_query")
    sql_query = make_sql_query(DB)
    summary = sql_query("SELECT topic, COUNT(*) AS n, ROUND(AVG(year), 1) AS avg_year "
                        "FROM papers GROUP BY topic ORDER BY avg_year DESC")
    print(summary.content)
    blocked = sql_query("DELETE FROM papers")              # the read-only guarantee in action
    print(f"   write attempt -> refused: {blocked.error}\n")

    print("3) R analysis on a CSV export (base R, no extra packages)")
    n_rows = export_csv(DB, CSV)          # aggregate() errors on 0 rows, so skip R when empty
    if n_rows:
        r_exec = make_r_exec()
        r = r_exec(
            f'df <- read.csv({CSV!r})\n'
            'cat("papers per year:\\n"); print(table(df$year))\n'
            'cat("\\nmean authors per topic:\\n")\n'
            'print(aggregate(n_authors ~ topic, data = df, FUN = mean))\n'
        )
        print(r.content if r.error is None else f"   R error: {r.error}\n   {r.content}")
    print()

    print("4) Let an Agent answer a natural-language question (sql_query + r_exec)")
    if os.getenv("GENAI_STUDIO_API_KEY"):
        from genai_studio.agents import Agent, ConsoleTracer
        from genai_studio.agents.tools import final_answer
        from _common import make_client

        agent = Agent(
            client=make_client(),
            tools=[make_sql_query(DB), make_r_exec(), final_answer],
            # sql_query summarises its result to head() (first rows), so steer the model
            # to AGGREGATE server-side (GROUP BY) — the answer then fits in the tool output.
            system="You answer questions about a SQLite 'papers' table (columns: topic, "
                   "title, year, n_authors, url, abstract). Use sql_query (read-only) to "
                   "fetch data — prefer GROUP BY aggregates over row dumps — and r_exec for "
                   "statistics. State the final answer plainly.",
            tracer=ConsoleTracer(), max_steps=6,
        )
        print(agent.run("For each topic, what is the average publication year and the "
                        "average number of authors? Which topic is newest on average?").text)
    else:
        print("   (skipped — set GENAI_STUDIO_API_KEY to run the agent section)")
