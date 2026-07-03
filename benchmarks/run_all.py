"""Run all benchmark suites and print a combined summary.

Usage:
    python benchmarks/run_all.py            # everything (slow; many live calls)
    python benchmarks/run_all.py dsbench    # one suite: dsbench | datascibench | react
"""

from __future__ import annotations

import sys

import dsbench_mini
import datascibench_mini
import react_eval
from _bench import run_suite


def run_dsbench():
    return run_suite("DSBench-mini", dsbench_mini.build_tasks(), dsbench_mini.make_agent)


def run_datascibench():
    return run_suite("DataSciBench-mini", datascibench_mini.build_tasks(),
                     datascibench_mini.make_agent)


def run_react(conditions=("standard", "act", "react")):
    tasks = react_eval.build_tasks()
    out = {}
    for cond in conditions:
        rep = run_suite(f"ReAct[{cond}]", tasks, react_eval.make_agent_for(cond))
        out[cond] = rep
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dsbench"):
        run_dsbench()
    if which in ("all", "datascibench"):
        run_datascibench()
    if which in ("all", "react"):
        run_react()
