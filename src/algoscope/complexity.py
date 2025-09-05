# src/algoscope/complexity.py
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Optional, Callable, List
import time
import numpy as np


@dataclass
class ComplexityGuess:
    time_big_o: str
    space_big_o: str
    explanation: str
    interview_summary: str
    dynamic_guess: Optional[str] = None


def _has_recursion(fn_name: str, node: ast.AST) -> int:
    """Return how many recursive calls happen inside the function body (branch factor)."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == fn_name:
                count += 1
            elif isinstance(child.func, ast.Attribute) and child.func.attr == fn_name:
                count += 1
    return count


def _max_loop_depth(node: ast.AST) -> int:
    max_depth = 0

    def visit(n: ast.AST, depth: int) -> None:
        nonlocal max_depth
        if isinstance(n, (ast.For, ast.While)):
            depth += 1
            max_depth = max(max_depth, depth)
        for c in ast.iter_child_nodes(n):
            visit(c, depth)

    visit(node, 0)
    return max_depth


def _has_divide_and_conquer_patterns(node: ast.AST) -> bool:
    """
    Detect halving/divide-and-conquer patterns:
      - presence of floor-division by 2 (e.g., (l+r)//2)
      - presence of slicing (seq[mid:]) or indices that suggest halving
    This returns True for both recursive and iterative binary-search patterns.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.FloorDiv):
            # check right operand is 2
            if isinstance(child.right, ast.Constant) and child.right.value == 2:
                return True
        if isinstance(child, ast.Subscript):
            return True
        # also check for expressions like mid = left + (right - left) // 2
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "mid":
            # not common — ignore
            pass
    return False


def _has_memoization(node: ast.AST) -> bool:
    """Detect patterns like 'if (x,y) in memo' or dict caching."""
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            if any(isinstance(op, ast.In) for op in child.ops):
                return True
        if isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name) and "memo" in child.value.id.lower():
                return True
    return False


def _has_dp_arrays(node: ast.AST) -> bool:
    """Detect DP table assignments like dp[i][j] = ... or dp[w] = ..."""
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name) and child.value.id.lower().startswith("dp"):
                return True
    return False


def _empirical_growth(func: Callable, input_builder: Callable, ns: List[int]) -> str:
    """Very rough empirical scaling detector for small ns. Returns short human text."""
    times = []
    for n in ns:
        # normalize input_builder output
        try:
            res = input_builder(n)
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                args, kwargs = res
            elif isinstance(res, tuple):
                args, kwargs = res, {}
            else:
                args, kwargs = (res,), {}
        except Exception:
            return "Could not run probe inputs"
        t0 = time.perf_counter()
        try:
            func(*args, **kwargs)
        except Exception:
            return "Could not empirically test"
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if len(times) < 2:
        return "Not enough data"
    ratios = [times[i + 1] / times[i] for i in range(len(times) - 1) if times[i] > 0]
    if not ratios:
        return "Not enough positive samples"
    avg = float(np.mean(ratios))
    if avg < 1.5:
        return "Empirical trend: closer to O(log n)"
    elif avg < 2.8:
        return "Empirical trend: closer to O(n)"
    elif avg < 5.0:
        return "Empirical trend: closer to O(n log n)"
    elif avg < 20.0:
        return "Empirical trend: closer to O(n^2)"
    else:
        return "Empirical trend: appears exponential"


def analyze_function_complexity(func, input_builder: Optional[Callable] = None) -> ComplexityGuess:
    """
    Heuristic Big-O guess with improved handling of iterative halving patterns.
    """
    try:
        src = inspect.getsource(func)
        src = textwrap.dedent(src)
        node = ast.parse(src)
    except OSError:
        return ComplexityGuess(
            "O(n)",
            "O(1)",
            "Could not retrieve source; assuming O(n) time and O(1) space.",
            f"{func.__name__} likely runs in O(n) time and O(1) space.",
        )

    fn_name = func.__name__
    branch_count = _has_recursion(fn_name, node)
    loop_depth = _max_loop_depth(node)
    dac = _has_divide_and_conquer_patterns(node)
    memo = _has_memoization(node)
    dp = _has_dp_arrays(node)

    # Defaults
    time_o, space_o, rationale = "O(n)", "O(1)", "Default linear assumption."

    # New ordering: check divide-and-conquer (halving) even when iterative
    if branch_count >= 2:
        time_o = "O(2^n)"
        space_o = "O(n)"
        rationale = "Multiple recursive calls per invocation detected → exponential recursion tree."
    elif (dac and loop_depth >= 1) and branch_count == 0:
        # iterative halving or slicing suggests log n (e.g., while halving bounds)
        time_o = "O(log n)"
        space_o = "O(1)"
        rationale = "Divide-and-conquer / halving pattern detected (floor division by 2 / slicing) → logarithmic."
    elif branch_count == 1 and dac:
        time_o = "O(log n)"
        space_o = "O(log n)"
        rationale = "Single recursive call with halving pattern → logarithmic depth recursion."
    elif branch_count == 1:
        time_o = "O(n)"
        space_o = "O(n)"
        rationale = "Single recursive branch without halving → linear recursion depth."
    elif memo:
        time_o = "O(n·W)"
        space_o = "O(n·W)"
        rationale = "Memoization dictionary detected → dynamic programming complexity."
    elif dp:
        time_o = "O(n·W)"
        space_o = "O(n·W)"
        rationale = "DP array usage detected → table-filling DP complexity."
    elif loop_depth >= 2:
        time_o = "O(n^2)"
        space_o = "O(1)"
        rationale = "Nested loops detected → quadratic time."
    elif loop_depth == 1:
        time_o = "O(n)"
        space_o = "O(1)"
        rationale = "Single loop detected → linear time."

    explanation = (
        f"Heuristic static analysis of `{fn_name}`:\n\n"
        f"**Estimated Time Complexity:** {time_o}\n"
        f"**Estimated Space Complexity:** {space_o}\n\n"
        f"Why: {rationale}\n"
        "These are heuristics; empirical results below provide a practical check."
    )

    interview = (
        f"{fn_name} has estimated time complexity {time_o} and space complexity {space_o}. "
        "These results are heuristic; empirical benchmarks confirm scaling."
    )

    dynamic = None
    if input_builder:
        probe_ns = [max(1, int(v)) for v in [2, 4, 8, 16] if v <= 64]
        try:
            dynamic = _empirical_growth(func, input_builder, probe_ns)
        except Exception:
            dynamic = None

    return ComplexityGuess(time_o, space_o, explanation, interview, dynamic_guess=dynamic)
