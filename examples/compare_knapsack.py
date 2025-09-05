from __future__ import annotations

from typing import Any, Tuple
import random

from algoscope import analyze_functions


# 1. Brute force (generate all subsets) -----------------------
def knapsack_bruteforce(weights, values, W):
    n = len(weights)
    best = 0
    for mask in range(1 << n):
        total_w = total_v = 0
        for i in range(n):
            if mask & (1 << i):
                total_w += weights[i]
                total_v += values[i]
        if total_w <= W:
            best = max(best, total_v)
    return best


# 2. Recursion -----------------------------------------------
def knapsack_recursive(weights, values, W, n=None):
    if n is None:
        n = len(weights)
    if n == 0 or W == 0:
        return 0
    if weights[n - 1] > W:
        return knapsack_recursive(weights, values, W, n - 1)
    else:
        return max(
            values[n - 1] + knapsack_recursive(weights, values, W - weights[n - 1], n - 1),
            knapsack_recursive(weights, values, W, n - 1),
        )


# 3. Recursion + Memoization ---------------------------------
def knapsack_memo(weights, values, W, n=None, memo=None):
    if n is None:
        n = len(weights)
        memo = {}
    if (n, W) in memo:
        return memo[(n, W)]
    if n == 0 or W == 0:
        return 0
    if weights[n - 1] > W:
        res = knapsack_memo(weights, values, W, n - 1, memo)
    else:
        res = max(
            values[n - 1] + knapsack_memo(weights, values, W - weights[n - 1], n - 1, memo),
            knapsack_memo(weights, values, W, n - 1, memo),
        )
    memo[(n, W)] = res
    return res


# 4. Tabulation (2D DP) --------------------------------------
def knapsack_tab(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][W]


# 5. Tabulation (1D DP optimization) --------------------------
def knapsack_1d(weights, values, W):
    n = len(weights)
    dp = [0] * (W + 1)
    for i in range(n):
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
    return dp[W]


# Input builder for benchmarking ------------------------------
def build_input(n: int) -> Tuple[tuple, dict] | Any:
    if n > 20 and any(f.__name__ == "knapsack_bruteforce" for f in functions_to_compare):
        raise ValueError("Brute force not allowed for n > 20")
    random.seed(n)  # stable per input size
    weights = [random.randint(1, n) for _ in range(n)]
    values = [random.randint(1, 100) for _ in range(n)]
    W = n * 2
    return (weights, values, W), {}


if __name__ == "__main__":
    functions_to_compare = [
        knapsack_bruteforce,
        knapsack_recursive,
        knapsack_memo,
        knapsack_tab,
        knapsack_1d,
    ]

    # Bruteforce explodes fast -> keep input sizes small
    input_sizes = [5, 7, 9, 11, 13, 15]

    results = analyze_functions(
        funcs=functions_to_compare,
        input_builder=build_input,
        ns=input_sizes,
        repeats=3,
        ci_method="t",
        reference_curves=("1", "n", "n**2", "2**n"),
        normalize_ref_at="max",
        html_out="report_knapsack.html",
        title="0/1 Knapsack Approaches",
        notes="Comparison of brute force, recursion, memoization, tabulation, and 1D DP.",
    )
