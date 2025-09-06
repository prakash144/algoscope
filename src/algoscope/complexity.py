# src/algoscope/complexity.py
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
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
    confidence: float = 0.0  # 0.0 to 1.0 confidence in the analysis
    patterns_detected: List[str] = field(default_factory=list)  # List of detected patterns
    interview_tips: List[str] = field(default_factory=list)  # Google SWE interview tips
    optimization_suggestions: List[str] = field(default_factory=list)  # Suggestions for optimization


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


def _detect_algorithm_patterns(node: ast.AST) -> List[str]:
    """Detect common algorithm patterns for better complexity analysis."""
    patterns = []
    
    # Binary search patterns
    if _has_binary_search_patterns(node):
        patterns.append("binary_search")
    
    # Two pointers pattern
    if _has_two_pointers_pattern(node):
        patterns.append("two_pointers")
    
    # Sliding window pattern
    if _has_sliding_window_pattern(node):
        patterns.append("sliding_window")
    
    # Hash map usage
    if _has_hash_map_usage(node):
        patterns.append("hash_map")
    
    # Tree traversal patterns
    if _has_tree_traversal_patterns(node):
        patterns.append("tree_traversal")
    
    # Graph algorithms
    if _has_graph_patterns(node):
        patterns.append("graph_algorithm")
    
    # Sorting algorithms
    if _has_sorting_patterns(node):
        patterns.append("sorting")
    
    return patterns


def _has_binary_search_patterns(node: ast.AST) -> bool:
    """Detect binary search patterns: left, right, mid variables."""
    has_left_right = False
    has_mid = False
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if child.id in ['left', 'right', 'lo', 'hi']:
                has_left_right = True
            elif child.id in ['mid', 'middle']:
                has_mid = True
    
    return has_left_right and has_mid


def _has_two_pointers_pattern(node: ast.AST) -> bool:
    """Detect two pointers pattern: i, j variables with increment/decrement."""
    has_i_j = False
    has_increment = False
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if child.id in ['i', 'j', 'start', 'end', 'left', 'right']:
                has_i_j = True
        elif isinstance(child, ast.AugAssign):
            if isinstance(child.op, (ast.Add, ast.Sub)):
                has_increment = True
    
    return has_i_j and has_increment


def _has_sliding_window_pattern(node: ast.AST) -> bool:
    """Detect sliding window pattern: window variables and size tracking."""
    has_window = False
    has_size = False
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if 'window' in child.id.lower() or child.id in ['start', 'end']:
                has_window = True
            elif child.id in ['size', 'length', 'count']:
                has_size = True
    
    return has_window and has_size


def _has_hash_map_usage(node: ast.AST) -> bool:
    """Detect hash map usage: dict, set, defaultdict."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                if child.func.id in ['dict', 'set', 'defaultdict', 'Counter']:
                    return True
        elif isinstance(child, ast.Name):
            if child.id in ['dict', 'set', 'hashmap', 'mapping']:
                return True
    return False


def _has_tree_traversal_patterns(node: ast.AST) -> bool:
    """Detect tree traversal patterns: node.left, node.right."""
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if child.attr in ['left', 'right', 'children']:
                return True
    return False


def _has_graph_patterns(node: ast.AST) -> bool:
    """Detect graph algorithm patterns: adjacency lists, BFS, DFS."""
    has_adjacency = False
    has_bfs_dfs = False
    
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if 'adj' in child.id.lower() or 'graph' in child.id.lower():
                has_adjacency = True
            elif child.id in ['bfs', 'dfs', 'queue', 'stack']:
                has_bfs_dfs = True
    
    return has_adjacency or has_bfs_dfs


def _has_sorting_patterns(node: ast.AST) -> bool:
    """Detect sorting algorithm patterns: comparisons, swaps."""
    has_comparison = False
    has_swap = False
    
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            has_comparison = True
        elif isinstance(child, ast.Assign):
            # Look for swap patterns: a, b = b, a
            if isinstance(child.value, ast.Tuple):
                has_swap = True
    
    return has_comparison and has_swap


def _analyze_data_structures(node: ast.AST) -> List[str]:
    """Analyze what data structures are being used."""
    structures = []
    
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                if child.func.id in ['list', 'tuple']:
                    structures.append("array")
                elif child.func.id in ['dict', 'defaultdict']:
                    structures.append("hash_map")
                elif child.func.id in ['set']:
                    structures.append("set")
                elif child.func.id in ['deque']:
                    structures.append("deque")
        elif isinstance(child, ast.Name):
            if child.id in ['heapq', 'PriorityQueue']:
                structures.append("heap")
    
    return list(set(structures))


def _calculate_confidence(patterns: List[str], loop_depth: int, branch_count: int, 
                         has_memo: bool, has_dp: bool) -> float:
    """Calculate confidence in the complexity analysis."""
    confidence = 0.5  # Base confidence
    
    # Pattern-based confidence
    if patterns:
        confidence += 0.2
    
    # Clear complexity indicators
    if has_memo or has_dp:
        confidence += 0.2
    
    # Loop depth gives clear indication
    if loop_depth > 0:
        confidence += 0.1
    
    # Recursion patterns
    if branch_count > 0:
        confidence += 0.1
    
    return min(1.0, confidence)


def _generate_interview_tips(patterns: List[str], time_o: str, space_o: str) -> List[str]:
    """Generate Google SWE interview tips based on detected patterns."""
    tips = []
    
    if "binary_search" in patterns:
        tips.append("💡 Binary search is O(log n) - perfect for sorted array problems")
        tips.append("🎯 Mention that binary search reduces search space by half each iteration")
    
    if "two_pointers" in patterns:
        tips.append("💡 Two pointers technique is O(n) - great for sorted array problems")
        tips.append("🎯 Explain how you're eliminating half the search space each step")
    
    if "sliding_window" in patterns:
        tips.append("💡 Sliding window is O(n) - optimal for substring/subarray problems")
        tips.append("🎯 Mention that you're maintaining a window of valid elements")
    
    if "hash_map" in patterns:
        tips.append("💡 Hash map gives O(1) average lookup - perfect for frequency counting")
        tips.append("🎯 Trade space for time - mention the space complexity trade-off")
    
    if "tree_traversal" in patterns:
        tips.append("💡 Tree traversal is O(n) where n is number of nodes")
        tips.append("🎯 Mention DFS vs BFS and when to use each")
    
    if "graph_algorithm" in patterns:
        tips.append("💡 Graph algorithms are typically O(V + E) for adjacency lists")
        tips.append("🎯 Mention BFS for shortest path, DFS for connectivity")
    
    if "sorting" in patterns:
        tips.append("💡 Sorting is O(n log n) - consider if you can avoid it")
        tips.append("🎯 Mention that sorting might not be necessary for the problem")
    
    # Complexity-specific tips
    if "O(1)" in time_o:
        tips.append("🚀 O(1) is optimal! This is the best possible time complexity")
    elif "O(log n)" in time_o:
        tips.append("🚀 O(log n) is excellent! This scales very well with input size")
    elif "O(n)" in time_o:
        tips.append("✅ O(n) is good! Linear time is often the best you can do")
    elif "O(n log n)" in time_o:
        tips.append("⚠️ O(n log n) is acceptable but consider if you can do better")
    elif "O(n^2)" in time_o:
        tips.append("⚠️ O(n²) might be too slow for large inputs - consider optimization")
    elif "O(2^n)" in time_o or "exponential" in time_o.lower():
        tips.append("❌ Exponential time is usually too slow - this needs optimization!")
    
    return tips


def _generate_optimization_suggestions(patterns: List[str], time_o: str, space_o: str, 
                                     has_memo: bool, has_dp: bool) -> List[str]:
    """Generate optimization suggestions based on current complexity."""
    suggestions = []
    
    # Time complexity optimizations
    if "O(n^2)" in time_o and "two_pointers" not in patterns:
        suggestions.append("🔄 Consider two pointers technique to reduce to O(n)")
    
    if "O(n^2)" in time_o and "hash_map" not in patterns:
        suggestions.append("🔄 Use hash map to reduce lookup time from O(n) to O(1)")
    
    if "O(2^n)" in time_o and not has_memo and not has_dp:
        suggestions.append("🔄 Add memoization to avoid recalculating subproblems")
        suggestions.append("🔄 Consider dynamic programming approach")
    
    if "O(n log n)" in time_o and "sorting" in patterns:
        suggestions.append("🔄 Consider if sorting is necessary - can you solve without it?")
    
    # Space complexity optimizations
    if "O(n)" in space_o and "hash_map" in patterns:
        suggestions.append("🔄 Consider if you can reduce space by using two pointers")
    
    if "O(n)" in space_o and has_dp:
        suggestions.append("🔄 Consider space-optimized DP (rolling array technique)")
    
    # General suggestions
    if not patterns:
        suggestions.append("🔍 Consider using standard algorithms: binary search, two pointers, sliding window")
    
    if "O(n^2)" in time_o:
        suggestions.append("🔍 Look for opportunities to eliminate nested loops")
    
    if "O(2^n)" in time_o:
        suggestions.append("🔍 This is likely a brute force approach - look for patterns to optimize")
    
    return suggestions


def _generate_google_swe_explanation(time_o: str, space_o: str, patterns: List[str], 
                                   confidence: float) -> str:
    """Generate Google SWE interview-style explanation."""
    explanation_parts = []
    
    # Start with the complexity
    explanation_parts.append(f"**Time Complexity:** {time_o}")
    explanation_parts.append(f"**Space Complexity:** {space_o}")
    explanation_parts.append("")
    
    # Add pattern-based explanation
    if patterns:
        explanation_parts.append("**Algorithm Patterns Detected:**")
        for pattern in patterns:
            pattern_explanations = {
                "binary_search": "Binary search reduces search space by half each iteration",
                "two_pointers": "Two pointers technique eliminates half the search space",
                "sliding_window": "Sliding window maintains a valid subarray/substring",
                "hash_map": "Hash map provides O(1) average lookup time",
                "tree_traversal": "Tree traversal visits each node exactly once",
                "graph_algorithm": "Graph algorithm processes each vertex and edge",
                "sorting": "Sorting algorithm arranges elements in order"
            }
            explanation_parts.append(f"- {pattern}: {pattern_explanations.get(pattern, 'Pattern detected')}")
        explanation_parts.append("")
    
    # Add confidence level
    if confidence > 0.8:
        explanation_parts.append("**Confidence Level:** High - Clear algorithmic patterns detected")
    elif confidence > 0.6:
        explanation_parts.append("**Confidence Level:** Medium - Some patterns detected")
    else:
        explanation_parts.append("**Confidence Level:** Low - Limited pattern detection")
    
    explanation_parts.append("")
    explanation_parts.append("**Interview Note:** This is a heuristic analysis. Always verify with empirical testing and consider edge cases.")
    
    return "\n".join(explanation_parts)


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
    Enhanced heuristic Big-O analysis with Google SWE interview focus:
    - Advanced pattern detection for common algorithms
    - Confidence scoring for analysis reliability
    - Interview-specific tips and optimization suggestions
    - Comprehensive explanation generation
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
            confidence=0.1,
            patterns_detected=[],
            interview_tips=["⚠️ Could not analyze source code - manual review needed"],
            optimization_suggestions=["🔍 Review the code manually for complexity analysis"]
        )

    fn_name = func.__name__
    
    # Enhanced analysis
    branch_count = _has_recursion(fn_name, node)
    loop_depth = _max_loop_depth(node)
    dac = _has_divide_and_conquer_patterns(node)
    memo = _has_memoization(node, fn_name)
    dp = _has_dp_arrays(node)
    patterns = _detect_algorithm_patterns(node)
    data_structures = _analyze_data_structures(node)
    
    # Calculate confidence
    confidence = _calculate_confidence(patterns, loop_depth, branch_count, memo, dp)
    
    # Enhanced complexity detection with pattern-based analysis
    time_o, space_o, rationale = _determine_complexity(
        patterns, branch_count, loop_depth, dac, memo, dp, data_structures
    )
    
    # Generate comprehensive explanation
    explanation = _generate_google_swe_explanation(time_o, space_o, patterns, confidence)
    
    # Generate interview summary
    interview = _generate_interview_summary(fn_name, time_o, space_o, patterns, confidence)
    
    # Generate tips and suggestions
    interview_tips = _generate_interview_tips(patterns, time_o, space_o)
    optimization_suggestions = _generate_optimization_suggestions(patterns, time_o, space_o, memo, dp)
    
    # Empirical analysis
    dynamic = None
    if input_builder:
        probe_ns = [max(1, int(v)) for v in [2, 4, 8, 16] if v <= 64]
        try:
            dynamic = _empirical_growth(func, input_builder, probe_ns)
        except Exception:
            dynamic = None

    return ComplexityGuess(
        time_big_o=time_o,
        space_big_o=space_o,
        explanation=explanation,
        interview_summary=interview,
        dynamic_guess=dynamic,
        confidence=confidence,
        patterns_detected=patterns,
        interview_tips=interview_tips,
        optimization_suggestions=optimization_suggestions
    )


def _determine_complexity(patterns: List[str], branch_count: int, loop_depth: int, 
                         dac: bool, memo: bool, dp: bool, data_structures: List[str]) -> tuple:
    """Determine time and space complexity based on detected patterns."""
    
    # Pattern-based complexity detection
    if "binary_search" in patterns:
        return "O(log n)", "O(1)", "Binary search pattern detected - halves search space each iteration"
    
    if "two_pointers" in patterns:
        return "O(n)", "O(1)", "Two pointers pattern detected - single pass through array"
    
    if "sliding_window" in patterns:
        return "O(n)", "O(1)", "Sliding window pattern detected - single pass with window maintenance"
    
    if "hash_map" in patterns:
        if "tree_traversal" in patterns:
            return "O(n)", "O(n)", "Hash map with tree traversal - O(n) for both time and space"
        return "O(n)", "O(n)", "Hash map pattern detected - O(n) time, O(n) space for storage"
    
    if "tree_traversal" in patterns:
        return "O(n)", "O(h)", "Tree traversal pattern detected - O(n) time, O(h) space for recursion depth"
    
    if "graph_algorithm" in patterns:
        return "O(V + E)", "O(V)", "Graph algorithm pattern detected - visits each vertex and edge once"
    
    if "sorting" in patterns:
        return "O(n log n)", "O(1)", "Sorting pattern detected - comparison-based sorting"
    
    # Fallback to traditional analysis
    if memo:
        return "O(n·W)", "O(n·W)", "Memoization detected - dynamic programming with state caching"
    elif dp:
        return "O(n·W)", "O(n·W)", "Dynamic programming table detected - fills table based on states"
    elif branch_count >= 2:
        return "O(2^n)", "O(n)", "Multiple recursive calls detected - exponential recursion tree"
    elif (dac and loop_depth >= 1) and branch_count == 0:
        return "O(log n)", "O(1)", "Divide-and-conquer with iteration - logarithmic time"
    elif branch_count == 1 and dac:
        return "O(log n)", "O(log n)", "Single recursive call with halving - logarithmic depth"
    elif branch_count == 1:
        return "O(n)", "O(n)", "Single recursive call - linear recursion depth"
    elif loop_depth >= 2:
        return "O(n^2)", "O(1)", "Nested loops detected - quadratic time"
    elif loop_depth == 1:
        return "O(n)", "O(1)", "Single loop detected - linear time"
    else:
        return "O(n)", "O(1)", "Default assumption - likely linear time"


def _generate_interview_summary(fn_name: str, time_o: str, space_o: str, 
                               patterns: List[str], confidence: float) -> str:
    """Generate Google SWE interview summary."""
    summary_parts = [f"{fn_name} has time complexity {time_o} and space complexity {space_o}."]
    
    if patterns:
        summary_parts.append(f"Uses {', '.join(patterns)} pattern(s).")
    
    if confidence > 0.8:
        summary_parts.append("High confidence in analysis.")
    elif confidence > 0.6:
        summary_parts.append("Medium confidence in analysis.")
    else:
        summary_parts.append("Low confidence - manual review recommended.")
    
    return " ".join(summary_parts)
