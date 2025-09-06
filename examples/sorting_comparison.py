#!/usr/bin/env python3
"""
Sorting Algorithm Comparison - Algoscope Pro
============================================

This example demonstrates how to use Algoscope Pro to compare different sorting algorithms,
showing the clear performance differences between O(n²) and O(n log n) algorithms.

Perfect for:
- Understanding sorting algorithm complexity
- Visualizing performance differences
- Learning when to use which sorting algorithm
- Interview preparation with real data

Author: Algoscope Pro Team
"""

from __future__ import annotations

import os
import sys
import random
from typing import Any, Tuple, List

# Add src to path for development
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from algoscope import analyze_functions

# ============================================================================
# SORTING ALGORITHMS - BRUTE FORCE vs OPTIMAL
# ============================================================================

def bubble_sort(arr: List[int]) -> List[int]:
    """Bubble Sort - O(n²) time complexity"""
    arr = arr.copy()  # Don't modify original
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def insertion_sort(arr: List[int]) -> List[int]:
    """Insertion Sort - O(n²) time complexity"""
    arr = arr.copy()  # Don't modify original
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr: List[int]) -> List[int]:
    """Merge Sort - O(n log n) time complexity"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr: List[int]) -> List[int]:
    """Quick Sort - O(n log n) average, O(n²) worst case"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def build_sorting_input(n: int) -> Tuple[tuple, dict]:
    """Build input for sorting algorithms"""
    # Create a list of random integers
    arr = [random.randint(1, 1000) for _ in range(n)]
    return (arr,), {}

def main():
    """Run sorting algorithm comparison"""
    print("🔄 Sorting Algorithm Comparison")
    print("Powered by Algoscope Pro")
    print("=" * 50)
    
    # Define functions to compare
    functions = [
        bubble_sort,
        insertion_sort, 
        merge_sort,
        quick_sort
    ]
    
    # Input sizes - keep reasonable for sorting
    input_sizes = [100, 200, 400, 800, 1600, 3200]
    
    print("🔍 Analyzing Sorting Algorithms...")
    print("Comparing: Bubble Sort, Insertion Sort, Merge Sort, Quick Sort")
    print("Input sizes:", input_sizes)
    
    # Run analysis
    results = analyze_functions(
        funcs=functions,
        input_builder=build_sorting_input,
        ns=input_sizes,
        repeats=5,
        warmup=2,
        ci_method="t",
        confidence=0.95,
        mem_backend="tracemalloc",
        reference_curves=("1", "n", "n**2", "nlogn"),
        normalize_ref_at="max",
        html_out="sorting_comparison.html",
        title="Sorting Algorithms: O(n²) vs O(n log n)",
        notes="Compare the performance of different sorting algorithms. Notice how O(n²) algorithms (Bubble, Insertion) scale poorly compared to O(n log n) algorithms (Merge, Quick).",
        timeout=60.0,
        run_in_subprocess=False,  # Disable subprocess to avoid pickle issues
    )
    
    print(f"✅ Analysis complete! Report saved to: {results.html_path}")
    print("\n📊 Key Insights:")
    print("• Bubble Sort & Insertion Sort: O(n²) - Poor performance on large datasets")
    print("• Merge Sort & Quick Sort: O(n log n) - Much better scaling")
    print("• Quick Sort: Generally fastest in practice")
    print("• Merge Sort: Consistent O(n log n) performance, stable")
    
    return results

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    main()
