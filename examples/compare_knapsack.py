# examples/compare_knapsack.py
from __future__ import annotations

from typing import Any, Tuple
import random
import sys

# Optional: when running examples from the repo root, prefer local src
# (Uncomment if you need to force importing local package during development)
# import os
# THIS_FILE = os.path.abspath(__file__)
# REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
# SRC_PATH = os.path.join(REPO_ROOT, "src")
# if SRC_PATH not in sys.path:
#     sys.path.insert(0, SRC_PATH)

from algoscope import analyze_functions


# -------------------------
# Implementations to test
# -------------------------

def knapsack_bruteforce(weights, values, W):
    n = len(weights)
    best = 0
    # brute-force all subsets (2^n)
    for mask in range(1 << n):
        total_w = total_v = 0
        for i in range(n):
            if mask & (1 << i):
                total_w += weights[i]
                total_v += values[i]
        if total_w <= W:
            best = max(best, total_v)
    return best


def knapsack_recursive(weights, values, W, n=None):
    if n is None:
        n = len(weights)
    if n == 0 or W == 0:
        return 0
    if weights[n - 1] > W:
        return knapsack_recursive(weights, values, W, n - 1)
    return max(
        values[n - 1] + knapsack_recursive(weights, values, W - weights[n - 1], n - 1),
        knapsack_recursive(weights, values, W, n - 1),
    )


def knapsack_memo(weights, values, W, n=None, memo=None):
    # safe memo initialization (avoid sharing dicts across independent calls)
    if memo is None:
        memo = {}
    if n is None:
        n = len(weights)
    key = (n, W)
    if key in memo:
        return memo[key]
    if n == 0 or W == 0:
        res = 0
    elif weights[n - 1] > W:
        res = knapsack_memo(weights, values, W, n - 1, memo)
    else:
        res = max(
            values[n - 1] + knapsack_memo(weights, values, W - weights[n - 1], n - 1, memo),
            knapsack_memo(weights, values, W, n - 1, memo),
        )
    memo[key] = res
    return res


def knapsack_tab(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        row_prev = dp[i - 1]
        row_cur = dp[i]
        for w in range(1, W + 1):
            if wi <= w:
                row_cur[w] = max(vi + row_prev[w - wi], row_prev[w])
            else:
                row_cur[w] = row_prev[w]
    return dp[n][W]


def knapsack_1d(weights, values, W):
    n = len(weights)
    dp = [0] * (W + 1)
    for i in range(n):
        wi = weights[i]
        vi = values[i]
        for w in range(W, wi - 1, -1):
            dp[w] = max(dp[w], vi + dp[w - wi])
    return dp[W]


# -------------------------
# Input builder(s)
# -------------------------

def build_input(n: int) -> Tuple[tuple, dict] | Any:
    """
    Build deterministic-but-varied inputs for a given n:
      - uses an isolated RNG seeded from n so results are reproducible per n
      - returns ((weights, values, W), {})
    Note: no brute-force guard here; do that from main before running heavy functions.
    """
    rng = random.Random(n)  # deterministic for each n
    weights = [rng.randint(1, max(1, n)) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    W = max(1, n * 2)
    return (weights, values, W), {}


def grid_input_builder(n_items: int, capacity: int) -> Tuple[tuple, dict] | Any:
    """
    Builder for grid-mode sweeps: deterministic by (n_items, capacity).
    Returns ((weights, values, capacity), {}).
    """
    rng = random.Random(n_items * 10007 + capacity)
    weights = [rng.randint(1, max(1, n_items)) for _ in range(n_items)]
    values = [rng.randint(1, 100) for _ in range(n_items)]
    return (weights, values, capacity), {}


# -------------------------
# Run examples
# -------------------------
if __name__ == "__main__":
    # choose algorithms to compare
    functions_to_compare = [
        knapsack_bruteforce,
        knapsack_recursive,
        knapsack_memo,
        knapsack_tab,
        knapsack_1d,
    ]

    # For safety: if brute-force is included, keep n small (it explodes quickly)
    include_bruteforce = any(f.__name__ == "knapsack_bruteforce" for f in functions_to_compare)
    if include_bruteforce:
        input_sizes = [5, 7, 9, 11]  # keep small when brute-force is present
    else:
        input_sizes = [10, 20, 40, 60, 80, 100]

    # --- Example 1: standard 1D sweep (vary n, keep W = 2*n) ---
    print("Running single-parameter knapsack comparison (varying n)...")
    results = analyze_functions(
        funcs=functions_to_compare,
        input_builder=build_input,
        ns=input_sizes,
        repeats=3,
        warmup=1,
        ci_method="t",
        reference_curves=("1", "n", "n**2", "2**n"),
        normalize_ref_at="max",
        html_out="examples/reports/report_knapsack.html",
        title="0/1 Knapsack Approaches",
        notes="Comparison of brute force, recursion, memoization, tabulation, and 1D DP.",
        mem_backend="tracemalloc",
    )
    print(f"Report written to: {results.html_path}")

    # --- Example 2: (optional) 2D grid sweep over (n, W) to generate heatmaps ---
    # Uncomment to run. Grid sweeps are heavier but give a nice heatmap showing where each
    # algorithm starts to dominate / explode. Useful for DP vs exponential comparisons.
    #
    grid_x = [5, 7, 9, 11, 13, 15]          # varying item count
    grid_y = [5, 10, 20, 40, 80]            # varying capacity
    print("Running 2D grid sweep (this may take longer)...")
    results_grid = analyze_functions(
        funcs=[knapsack_memo, knapsack_tab, knapsack_1d],
        input_builder=build_input,   # required by signature but unused in grid mode
        ns=[5],                     # placeholder
        repeats=3,
        warmup=1,
        ci_method="t",
        mem_backend="tracemalloc",
        html_out="examples/reports/report_knapsack_grid.html",
        title="Knapsack Grid Sweep (n x W)",
        notes="Grid sweep for knapsack: items (n) vs capacity (W).",
        grid_x=grid_x,
        grid_y=grid_y,
        grid_input_builder=grid_input_builder,
        grid_log_color=True,
    )
    print(f"Grid report written to: {results_grid.html_path}")
