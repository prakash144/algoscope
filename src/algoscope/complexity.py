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


def _find_function_node(node: ast.AST, fn_name: str) -> Optional[ast.FunctionDef]:
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef) and child.name == fn_name:
            return child
    return None


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
      - floor-division by 2 (e.g., (l+r)//2)
      - explicit slicing that uses ast.Slice nodes
    This intentionally does NOT treat general indexing (Subscript) as halving.
    """
    for child in ast.walk(node):
        # direct floor-division by 2 like (a + b) // 2
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.FloorDiv):
            if isinstance(child.right, ast.Constant) and child.right.value == 2:
                return True
        # slicing node like arr[a:b] -> ast.Slice
        if isinstance(child, ast.Slice):
            return True

        # assignment where RHS contains a // 2 (covers mid = (l+r)//2)
        if isinstance(child, ast.Assign):
            rhs = child.value
            for sub in ast.walk(rhs):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.FloorDiv):
                    if isinstance(sub.right, ast.Constant) and sub.right.value == 2:
                        return True
    return False


def _has_memoization(node: ast.AST, fn_name: str) -> bool:
    """
    Detect memoization cache patterns:
      - explicit 'memo' dict lookups: memo[(...)] or memo[...] detection
      - decorator usage like @lru_cache, @cache
      - checks using 'in' to test existence in a dict (if (x,y) in memo)
    """
    # check for decorator @lru_cache / @cache
    fnode = _find_function_node(node, fn_name)
    if fnode:
        for deco in getattr(fnode, "decorator_list", []):
            if isinstance(deco, ast.Name) and deco.id.lower().endswith("lru_cache"):
                return True
            if isinstance(deco, ast.Attribute) and deco.attr.lower().endswith("lru_cache"):
                return True
            if isinstance(deco, ast.Name) and deco.id.lower() == "cache":
                return True

    for child in ast.walk(node):
        # "if (x,y) in memo" or other 'in' checks
        if isinstance(child, ast.Compare):
            if any(isinstance(op, ast.In) for op in child.ops):
                return True
        # subscript on a value named 'memo' or containing 'memo'
        if isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name) and "memo" in child.value.id.lower():
                return True
        # variable named 'memo' referenced
        if isinstance(child, ast.Name) and "memo" == child.id.lower():
            return True
    return False


def _has_dp_arrays(node: ast.AST) -> bool:
    """Detect DP table usage like 'dp' arrays (dp[i][j], dp[w], etc.)."""
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name) and child.value.id.lower().startswith("dp"):
                return True
    return False


def _empirical_growth(func: Callable, input_builder: Callable, ns: List[int]) -> str:
    """Very rough empirical scaling detector for small n."""
    times = []
    for n in ns:
        args, kwargs = input_builder(n)
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
    avg = np.mean(ratios) if ratios else 0
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
    Enhanced heuristic Big-O guess with improved ordering:
    - Check for memoization / DP first (these often defeat exponential recursion).
    - Detect iterative divide-and-conquer patterns safely.
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
    memo = _has_memoization(node, fn_name)
    dp = _has_dp_arrays(node)

    # Default guesses
    time_o, space_o, rationale = "O(n)", "O(1)", "Default linear assumption."

    # Important: check memo/dp before naively declaring recursion exponential.
    if memo:
        time_o = "O(n·W) (memoized / DP states)"
        space_o = "O(n·W)"
        rationale = "Memoization/cache detected → dynamic programming style complexity proportional to number of states."
    elif dp:
        time_o = "O(n·W)"
        space_o = "O(n·W)"
        rationale = "DP table usage detected → table-filling DP complexity."
    elif branch_count >= 2:
        time_o = "O(2^n)"
        space_o = "O(n)"
        rationale = "Multiple recursive calls per invocation detected → exponential recursion tree."
    elif (dac and loop_depth >= 1) and branch_count == 0:
        time_o = "O(log n)"
        space_o = "O(1)"
        rationale = "Divide-and-conquer / halving pattern detected (floor division by 2 / explicit slicing) → logarithmic."
    elif branch_count == 1 and dac:
        time_o = "O(log n)"
        space_o = "O(log n)"
        rationale = "Single recursive call with halving pattern → logarithmic depth recursion."
    elif branch_count == 1:
        time_o = "O(n)"
        space_o = "O(n)"
        rationale = "Single recursive branch without halving → linear recursion depth."
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
        # small probe
        probe_ns = [max(1, int(v)) for v in [2, 4, 8, 16] if v <= 64]
        try:
            dynamic = _empirical_growth(func, input_builder, probe_ns)
        except Exception:
            dynamic = None

    return ComplexityGuess(time_o, space_o, explanation, interview, dynamic_guess=dynamic)
