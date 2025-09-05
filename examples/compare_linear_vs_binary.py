# examples/compare_linear_vs_binary.py
from __future__ import annotations

from typing import Any, Tuple
import os
import sys
import random
import warnings

# Prefer local 'src' during development
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Import the library (should resolve to local src/algoscope)
from algoscope import analyze_functions

# Optional: quiet check for psutil (only warn once if missing)
try:
    import psutil  # noqa: F401
    _PSUTIL_PRESENT = True
except Exception:
    _PSUTIL_PRESENT = False
    warnings.warn(
        "psutil not found. For accurate RSS memory measurement with mem_backend='rss', "
        "install psutil: pip install psutil",
        RuntimeWarning,
    )

# Example functions
def linear_search(nums, target):
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1

# Import an example binary search if present
try:
    from examples.binary_search_example import binary_search
except Exception:
    def binary_search(nums, target):
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

# Input builder: returns ((args...), kwargs)
def build_input(n: int) -> Tuple[tuple, dict] | Any:
    nums = list(range(n))
    target = None if n == 0 else random.choice(nums)
    return (nums, target), {}

if __name__ == "__main__":
    random.seed(42)

    functions_to_compare = [linear_search, binary_search]
    input_sizes = [1000, 2000, 4000, 8000, 16000, 32000]

    results = analyze_functions(
        funcs=functions_to_compare,
        input_builder=build_input,
        ns=input_sizes,
        repeats=11,
        warmup=2,
        ci_method="t",
        confidence=0.95,
        mem_backend="rss" if _PSUTIL_PRESENT else "tracemalloc",
        reference_curves=("1", "logn", "n", "nlogn", "n**2"),
        normalize_ref_at="max",
        html_out="report.html",
        title="Linear vs Binary Search",
        notes="Comparison on integer arrays with random targets.",
        timeout=10.0,
        run_in_subprocess=True,
    )
