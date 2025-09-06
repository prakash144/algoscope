from __future__ import annotations

from typing import Any, Tuple
import random
from itertools import permutations

from algoscope import analyze_functions


def brute_force_sort(arr):
    target = sorted(arr)
    for perm in permutations(arr):
        if list(perm) == target:
            return list(perm)
    return None


def merge_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr[:]
    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    i = j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    if i < len(left):
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])
    return merged


def build_input(n: int) -> Tuple[tuple, dict] | Any:
    # build a random array of size n
    nums = random.sample(range(n * 10), n)
    return (nums,), {}


if __name__ == "__main__":
    functions_to_compare = [brute_force_sort, merge_sort]
    # keep sizes very small for factorial algorithm
    input_sizes = [3, 4, 5, 6, 7, 8]

    results = analyze_functions(
        funcs=functions_to_compare,
        input_builder=build_input,
        ns=input_sizes,
        repeats=5,
        ci_method="t",
        reference_curves=("1", "n", "nlogn", "n**2"),
        normalize_ref_at="max",
        html_out="examples/reports/report.html",
        title="Brute Force Sort vs. Merge Sort",
        notes="Comparison on random integer arrays.",
    )
    print(f"Analysis complete. Report saved to {results.html_path}")
