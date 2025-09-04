from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComplexityGuess:
    time_big_o: str
    space_big_o: str
    explanation: str
    interview_summary: str


def _has_recursion(fn_name: str, node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == fn_name:
                return True
            if isinstance(child.func, ast.Attribute) and child.func.attr == fn_name:
                return True
    return False


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
    Heuristic: evidence of mid = (lo + hi) // 2, slicing halves, or len(x)//2.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.FloorDiv):
            if isinstance(child.right, ast.Constant) and child.right.value == 2:
                return True
        if isinstance(child, ast.Slice):
            return True
    return False


def analyze_function_complexity(func) -> ComplexityGuess:
    """
    Static heuristic-based Big-O guess. Designed to be beginner-friendly.

    NOTE: improved handling for divide-and-conquer patterns: even if recursion
    is not explicit, the presence of halving or slicing strongly suggests O(log n).
    """
    try:
        src = inspect.getsource(func)
        node = ast.parse(src)
    except OSError:
        # likely built-in or interactive; provide conservative default
        return ComplexityGuess(
            "O(n)", "O(1)",
            "Could not retrieve source; assuming a single pass over input (O(n)) with constant extra space.",
            f"{func.__name__} likely runs in O(n) time and O(1) extra space in a typical implementation."
        )

    fn_name = func.__name__
    recursion = _has_recursion(fn_name, node)
    loop_depth = _max_loop_depth(node)
    dac = _has_divide_and_conquer_patterns(node)

    # Heuristic order:
    # 1) If divide-and-conquer (slicing or halving) -> likely logarithmic behavior
    # 2) If recursion combined with dac -> logn with stack usage
    # 3) else detect loops/recursion depth

    if dac:
        # Halving or slicing patterns strongly indicate O(log n) behaviour
        time_o = "O(log n)"
        # If recursion present then stack space is O(log n) else O(1)
        space_o = "O(log n)" if recursion else "O(1)"
        rationale = (
            "Detected divide-and-conquer hints (index halving, slicing, floor division by 2). "
            "This suggests logarithmic time. If implemented recursively, expect O(log n) stack space; "
            "iterative implementations usually use O(1) extra space."
        )
    elif recursion and loop_depth == 0:
        time_o = "O(n)"
        space_o = "O(n)"
        rationale = (
            "Detected recursion without loops and no clear halving pattern. "
            "This commonly implies linear recursion depth and O(n) time/space."
        )
    elif loop_depth == 0:
        time_o = "O(1)"
        space_o = "O(1)"
        rationale = "No loops or recursion detected; likely constant time and space."
    elif loop_depth == 1:
        time_o = "O(n)"
        space_o = "O(1)"
        rationale = "One level of looping detected; typical single pass suggests O(n) time and O(1) space."
    else:  # loop_depth >= 2
        time_o = "O(n^2)"
        space_o = "O(1)"
        rationale = "Nested loops detected; suggests quadratic time with constant extra space."

    explanation = (
        f"Static analysis of `{fn_name}`:\n"
        f"- Recursion: {'yes' if recursion else 'no'}\n"
        f"- Max loop nesting depth: {loop_depth}\n"
        f"- Divide-and-conquer hints: {'yes' if dac else 'no'}\n\n"
        f"**Estimated Time Complexity:** {time_o}\n"
        f"**Estimated Space Complexity:** {space_o}\n\n"
        f"Why: {rationale}\n"
        "These are heuristics; empirical results below provide a practical check."
    )

    interview = (
        f"{fn_name} has an estimated time complexity of {time_o} and space complexity of {space_o}. "
        "For recursive implementations that halve the search space (e.g., binary search), "
        "space is O(log n) due to recursion depth; iterative versions are O(1) space."
    )

    return ComplexityGuess(time_o, space_o, explanation, interview)
