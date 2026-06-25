"""
General-purpose tools (core — stdlib only, no scientific stack).

- ``final_answer`` / ``finish`` — terminal tools. The Agent loop intercepts a
  call to either (see ``Agent.finish_tool_names``) and ends with the given
  answer, so a model that "calls" a finish action terminates cleanly instead of
  looping. Shipping them as real tools also puts them in the model's tool list.
- ``calculator`` — safe arithmetic (AST-evaluated; no eval/imports), so an agent
  never has to trust the model's mental math.
"""

from __future__ import annotations

import ast
import math
import operator

from genai_studio.agents import ToolResult, tool


@tool
def final_answer(answer: str) -> str:
    """Provide your final answer and end the task.

    Args:
        answer: the final answer to return to the user.
    """
    return answer  # the Agent loop terminates on this call; this is a fallback


@tool
def finish(answer: str) -> str:
    """Finish the task and return the final answer.

    Args:
        answer: the final answer to return to the user.
    """
    return answer


# ── safe calculator ──────────────────────────────────────────────────────────
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {k: getattr(math, k) for k in (
    "sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan", "asin", "acos",
    "atan", "floor", "ceil", "factorial", "comb", "perm", "gcd", "fabs", "degrees",
    "radians") if hasattr(math, k)}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only math functions are allowed")
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> ToolResult:
    """Evaluate an arithmetic expression and return the numeric result.

    Supports + - * / ** % // , parentheses, the constants pi/e/tau, and math
    functions (sqrt, log, exp, sin, factorial, comb, ...). No variables or
    arbitrary code — safe to expose to a model.

    Args:
        expression: e.g. "70 * 1.02 ** 8" or "comb(10, 7) / 2 ** 10".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return ToolResult(content=str(_safe_eval(tree)))
    except Exception as exc:
        return ToolResult(content="", error=f"calculator error: {exc}")
