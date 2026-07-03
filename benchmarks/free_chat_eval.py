"""Free-chat eval — exercise the FULL agent (as the REPL builds it) on diverse, realistic prompts.

Unlike the scored proving harness (fixed theorem names, skill body as system prompt), this runs the
model in free chat: the REPL's BASE_SYSTEM + the always-on skills catalog + the general toolset (math,
proofs, mathlib, factorization, search, code). It captures each answer + the tools used + flags likely
issues (errored, terse proof, ungrounded math) so we can find and close problems before 2.0.

  export GENAI_STUDIO_API_KEY=...; export GENAI_STUDIO_RPM=20
  python benchmarks/free_chat_eval.py --model qwen2.5:72b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from genai_studio import GenAIStudio
from genai_studio.agents import Agent, NullTracer
from genai_studio.agents.approval import ApprovalMode, SandboxPolicy
from genai_studio.agents.client import GenAIStudioClient
from genai_studio.agents.compose import assemble_system, wire_capabilities
from genai_studio.agents.profiles import build_tools
from genai_studio.agents.repl.cli import BASE_SYSTEM

# (id, kind, prompt) — kind drives the heuristics. "proof"/"math" answers must be grounded + explained.
TASKS = [
    ("ineq_proof",   "proof", "Prove that for all real numbers a and b, a^2 + b^2 >= 2ab. Explain the proof."),
    ("cauchy",       "proof", "Prove that (a*c + b*d)^2 <= (a^2+b^2)*(c^2+d^2) for all reals, and explain why."),
    ("factor_verify","math",  "Factor x^4 - 16 completely, and verify your factorization is correct."),
    ("solve_roots",  "math",  "Find all real roots of x^3 - 6x^2 + 11x - 6 = 0 and verify each one."),
    ("derivative",   "math",  "What is the derivative of x*sin(x)? Show the steps."),
    ("float_eq",     "math",  "Is it exactly true that 0.1 + 0.2 = 0.3? Explain."),
    ("eigen",        "math",  "Find the eigenvalues of the matrix [[2,1],[1,2]]. Show your reasoning."),
    ("lean_core",    "proof", "Prove in Lean that for every natural number n, n + 0 = n."),
    ("lean_mathlib", "proof", "Prove in Lean with mathlib that (a+b)^2 = a^2 + 2*a*b + b^2 for real a, b. Explain the tactic."),
    ("lean_sqrt2",   "proof", "Prove in Lean with mathlib that the square root of 2 is irrational."),
    ("gauss",        "proof", "Prove that the sum 1+2+...+n equals n(n+1)/2. Explain the argument."),
    ("logic",        "reason","If all Bloops are Razzies and all Razzies are Lazzies, does it follow that all Bloops are Lazzies? Explain."),
    ("word_problem", "reason","A car travels 150 miles in 2.5 hours. What is its average speed? Show the calculation."),
    ("code_glob",    "code",  "How many Python files are in the genai_studio/agents/tools directory of this project?"),
]

GROUNDING = {"prove", "verify_math", "verify_factorization", "symbolic_math", "matrix_op",
             "solve_constraints", "lean_check", "grade_proof", "search_lemmas"}


def _tools_used(res):
    used = []
    for m in getattr(res, "messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            used.append(tc.name)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="/tmp/free_chat_eval.json")
    args = ap.parse_args()
    if not os.environ.get("GENAI_STUDIO_API_KEY"):
        sys.exit("set GENAI_STUDIO_API_KEY (and GENAI_STUDIO_RPM=20)")

    cwd = str(Path.cwd())
    ai = GenAIStudio(validate_model=False, timeout=180)
    client = GenAIStudioClient(ai, default_model=args.model)
    tools, guard, _cfg = build_tools("general", workspace_root=cwd, mode=ApprovalMode("suggest"),
                                     sandbox=SandboxPolicy("workspace-write"),
                                     prompt_fn=lambda call, preview: "allow")
    tools, blocks, tool_search, n_skills = wire_capabilities(
        tools, cwd=cwd, client=client, model=args.model, memory_dir=None,
        skills=True, memory=False, defer=False, shared_guards=[guard], studio=ai)
    system = assemble_system(BASE_SYSTEM, *blocks)
    print(f"agent: {len(tools)} tools, {n_skills} skills, model={args.model}\n", flush=True)

    todo = [t for t in TASKS if not args.only or t[0] in args.only.split(",")]
    rows = []
    for tid, kind, prompt in todo:
        agent = Agent(client=client, tools=tools, system=system, tracer=NullTracer(),
                      guards=[guard], model=args.model, max_steps=args.max_steps,
                      temperature=0.0, tool_search=tool_search)
        t0 = time.time()
        try:
            res = agent.run(prompt)
            ans, err = res.text or "", ""
        except Exception as e:
            ans, err, res = "", f"{type(e).__name__}: {e}", None
        dt = time.time() - t0
        used = _tools_used(res) if res is not None else []
        # heuristic issue flags
        flags = []
        if err:
            flags.append("ERROR")
        if not (ans or "").strip():
            flags.append("empty")
        if kind in ("proof", "math") and not (GROUNDING & set(used)):
            flags.append("ungrounded")                       # did math without a grounding tool
        if kind == "proof" and len((ans or "").strip()) < 220:
            flags.append("terse")                            # proof with no step-by-step explanation
        rows.append({"id": tid, "kind": kind, "prompt": prompt, "answer": ans, "tools": used,
                     "steps": len(getattr(res, "steps", []) or []), "secs": round(dt), "flags": flags})
        print(f"[{tid:14}] {dt:4.0f}s  tools={sorted(set(used))}  flags={flags or 'ok'}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    clean = sum(1 for r in rows if not r["flags"])
    print(f"\n=== {clean}/{len(rows)} clean (no flags) — full outputs in {args.out} ===")
    for r in rows:
        if r["flags"]:
            print(f"  ⚠ {r['id']}: {r['flags']}")


if __name__ == "__main__":
    main()
