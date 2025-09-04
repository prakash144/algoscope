from __future__ import annotations

from typing import Any, Tuple
import random

from algoscope import analyze_functions
from examples.binary_search_example import binary_search


def linear_search(nums, target):
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1


def build_input(n: int) -> Tuple[tuple, dict] | Any:
    # build a sorted array of size n and pick a target
    nums = list(range(n))
    target = random.choice(nums)  # deterministic effect is not required for lookup
    # Return (args, kwargs)
    return (nums, target), {}


if __name__ == "__main__":
    functions_to_compare = [linear_search, binary_search]
    input_sizes = [1000, 2000, 4000, 8000, 16000, 32000]

    results = analyze_functions(
        funcs=functions_to_compare,
        input_builder=build_input,
        ns=input_sizes,
        repeats=11,
        ci_method="t",
        reference_curves=("1", "logn", "n", "nlogn", "n**2"),
        normalize_ref_at="max",
        html_out="report.html",
        title="Linear Search vs. Binary Search",
        notes="Comparison on uniformly increasing integer arrays with random targets.",
    )
    print(f"Analysis complete. Report saved to {results.html_path}")
