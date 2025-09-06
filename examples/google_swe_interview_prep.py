#!/usr/bin/env python3
"""
Google SWE Interview Preparation Tool - Algoscope Pro
====================================================

This example demonstrates how to use Algoscope Pro for Google SWE interview preparation,
focusing on comparing brute force vs optimal solutions across common interview problems.

Perfect for:
- Understanding algorithm complexity trade-offs
- Visualizing performance differences
- Interview preparation with real data
- Learning when to use which approach

Author: Algoscope Pro Team
"""

from __future__ import annotations

import os
import sys
import random
import time
from typing import Any, Tuple, List
from itertools import combinations, permutations

# Add src to path for development
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from algoscope import analyze_functions

# ============================================================================
# GOOGLE SWE INTERVIEW PROBLEMS - BRUTE FORCE vs OPTIMAL
# ============================================================================

def two_sum_brute_force(nums: List[int], target: int) -> List[int]:
    """Brute force O(n²) solution - check every pair"""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_optimal(nums: List[int], target: int) -> List[int]:
    """Optimal O(n) solution using hash map"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def max_subarray_brute_force(nums: List[int]) -> int:
    """Brute force O(n³) solution - check all subarrays"""
    max_sum = float('-inf')
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            current_sum = sum(nums[i:j+1])
            max_sum = max(max_sum, current_sum)
    return max_sum

def max_subarray_optimal(nums: List[int]) -> int:
    """Kadane's algorithm O(n) solution"""
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

def longest_common_subsequence_brute_force(text1: str, text2: str) -> int:
    """Brute force O(2^n) solution - generate all subsequences"""
    def get_subsequences(s):
        if not s:
            return [""]
        first = s[0]
        rest = get_subsequences(s[1:])
        return rest + [first + sub for sub in rest]
    
    subs1 = set(get_subsequences(text1))
    subs2 = set(get_subsequences(text2))
    common = subs1.intersection(subs2)
    return max(len(sub) for sub in common) if common else 0

def longest_common_subsequence_optimal(text1: str, text2: str) -> int:
    """DP solution O(m*n)"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def coin_change_brute_force(coins: List[int], amount: int) -> int:
    """Brute force recursive solution O(amount^coins)"""
    def dfs(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        min_coins = float('inf')
        for coin in coins:
            result = dfs(remaining - coin)
            if result != float('inf'):
                min_coins = min(min_coins, 1 + result)
        return min_coins
    
    result = dfs(amount)
    return result if result != float('inf') else -1

def coin_change_optimal(coins: List[int], amount: int) -> int:
    """DP solution O(amount * coins)"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def word_break_brute_force(s: str, word_dict: List[str]) -> bool:
    """Brute force recursive solution O(2^n)"""
    def dfs(start):
        if start == len(s):
            return True
        
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_dict and dfs(end):
                return True
        return False
    
    return dfs(0)

def word_break_optimal(s: str, word_dict: List[str]) -> bool:
    """DP solution O(n²)"""
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]

# ============================================================================
# INPUT BUILDERS FOR DIFFERENT PROBLEMS
# ============================================================================

def build_two_sum_input(n: int) -> Tuple[tuple, dict]:
    """Build input for Two Sum problem"""
    nums = random.sample(range(-1000, 1000), n)
    target = random.choice(nums) + random.choice(nums)
    return (nums, target), {}

def build_max_subarray_input(n: int) -> Tuple[tuple, dict]:
    """Build input for Maximum Subarray problem"""
    nums = [random.randint(-100, 100) for _ in range(n)]
    return (nums,), {}

def build_lcs_input(n: int) -> Tuple[tuple, dict]:
    """Build input for Longest Common Subsequence problem"""
    chars = 'abcdefghijklmnopqrstuvwxyz'
    text1 = ''.join(random.choices(chars, k=n))
    text2 = ''.join(random.choices(chars, k=n))
    return (text1, text2), {}

def build_coin_change_input(n: int) -> Tuple[tuple, dict]:
    """Build input for Coin Change problem"""
    coins = [1, 3, 4, 5, 7, 10, 12, 15, 20, 25]
    amount = n * 10  # Scale amount with n
    return (coins, amount), {}

def build_word_break_input(n: int) -> Tuple[tuple, dict]:
    """Build input for Word Break problem"""
    words = ['cat', 'cats', 'and', 'sand', 'dog', 'dogs', 'code', 'leet', 'leetcode']
    s = 'leetcode' * (n // 8 + 1)  # Scale string length
    return (s, words), {}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_two_sum():
    """Analyze Two Sum: Brute Force vs Optimal"""
    print("🔍 Analyzing Two Sum Problem...")
    
    functions = [two_sum_brute_force, two_sum_optimal]
    input_sizes = [100, 200, 400, 800, 1600, 3200]
    
    results = analyze_functions(
        funcs=functions,
        input_builder=build_two_sum_input,
        ns=input_sizes,
        repeats=5,
        warmup=2,
        ci_method="t",
        reference_curves=("1", "n", "n**2"),
        normalize_ref_at="max",
        html_out="examples/reports/google_swe_two_sum.html",
        title="Two Sum: Brute Force vs Optimal",
        notes="Classic Google interview problem comparing O(n²) vs O(n) solutions",
        timeout=30.0,
        run_in_subprocess=False,  # Disable subprocess to avoid pickle issues
    )
    
    print(f"✅ Two Sum analysis complete: {results.html_path}")
    return results

def analyze_max_subarray():
    """Analyze Maximum Subarray: Brute Force vs Kadane's Algorithm"""
    print("🔍 Analyzing Maximum Subarray Problem...")
    
    functions = [max_subarray_brute_force, max_subarray_optimal]
    input_sizes = [50, 100, 200, 400, 800, 1600]
    
    results = analyze_functions(
        funcs=functions,
        input_builder=build_max_subarray_input,
        ns=input_sizes,
        repeats=5,
        warmup=2,
        ci_method="t",
        reference_curves=("1", "n", "n**2", "n**3"),
        normalize_ref_at="max",
        html_out="examples/reports/google_swe_max_subarray.html",
        title="Maximum Subarray: Brute Force vs Kadane's Algorithm",
        notes="Dynamic programming classic: O(n³) vs O(n) solutions",
        timeout=30.0,
        run_in_subprocess=False,  # Disable subprocess to avoid pickle issues
    )
    
    print(f"✅ Maximum Subarray analysis complete: {results.html_path}")
    return results

def analyze_lcs():
    """Analyze Longest Common Subsequence: Brute Force vs DP"""
    print("🔍 Analyzing Longest Common Subsequence Problem...")
    
    functions = [longest_common_subsequence_brute_force, longest_common_subsequence_optimal]
    input_sizes = [5, 8, 10, 12, 15, 18]  # Keep small due to exponential complexity
    
    results = analyze_functions(
        funcs=functions,
        input_builder=build_lcs_input,
        ns=input_sizes,
        repeats=3,
        warmup=1,
        ci_method="t",
        reference_curves=("1", "n", "n**2", "2**n"),
        normalize_ref_at="max",
        html_out="examples/reports/google_swe_lcs.html",
        title="Longest Common Subsequence: Brute Force vs DP",
        notes="Exponential vs polynomial: O(2^n) vs O(m*n) solutions",
        timeout=60.0,
        run_in_subprocess=False,  # Disable subprocess to avoid pickle issues
    )
    
    print(f"✅ LCS analysis complete: {results.html_path}")
    return results

def analyze_coin_change():
    """Analyze Coin Change: Brute Force vs DP"""
    print("🔍 Analyzing Coin Change Problem...")
    
    functions = [coin_change_brute_force, coin_change_optimal]
    input_sizes = [10, 20, 30, 40, 50, 60]
    
    results = analyze_functions(
        funcs=functions,
        input_builder=build_coin_change_input,
        ns=input_sizes,
        repeats=3,
        warmup=1,
        ci_method="t",
        reference_curves=("1", "n", "n**2", "2**n"),
        normalize_ref_at="max",
        html_out="examples/reports/google_swe_coin_change.html",
        title="Coin Change: Brute Force vs Dynamic Programming",
        notes="Classic DP problem: exponential recursion vs O(amount*coins) solution",
        timeout=60.0,
        run_in_subprocess=False,  # Disable subprocess to avoid pickle issues
    )
    
    print(f"✅ Coin Change analysis complete: {results.html_path}")
    return results

def analyze_word_break():
    """Analyze Word Break: Brute Force vs DP"""
    print("🔍 Analyzing Word Break Problem...")
    
    functions = [word_break_brute_force, word_break_optimal]
    input_sizes = [8, 16, 24, 32, 40, 48]
    
    results = analyze_functions(
        funcs=functions,
        input_builder=build_word_break_input,
        ns=input_sizes,
        repeats=3,
        warmup=1,
        ci_method="t",
        reference_curves=("1", "n", "n**2", "2**n"),
        normalize_ref_at="max",
        html_out="examples/reports/google_swe_word_break.html",
        title="Word Break: Brute Force vs Dynamic Programming",
        notes="String processing classic: exponential vs O(n²) solutions",
        timeout=60.0,
        run_in_subprocess=False,  # Disable subprocess to avoid pickle issues
    )
    
    print(f"✅ Word Break analysis complete: {results.html_path}")
    return results

def run_comprehensive_analysis():
    """Run comprehensive analysis of all Google SWE interview problems"""
    print("🚀 Starting Comprehensive Google SWE Interview Analysis...")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    results = []
    
    try:
        # Run all analyses
        results.append(analyze_two_sum())
        results.append(analyze_max_subarray())
        results.append(analyze_lcs())
        results.append(analyze_coin_change())
        results.append(analyze_word_break())
        
        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\n📊 Generated Reports:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.title}")
            print(f"     📄 {result.html_path}")
        
        print(f"\n💡 Key Insights for Google SWE Interviews:")
        print("  • Brute force solutions are O(n²) to O(2^n) - avoid in interviews!")
        print("  • Optimal solutions use hash maps, DP, or greedy algorithms")
        print("  • Always start with brute force, then optimize")
        print("  • Know when to use each approach based on constraints")
        print("  • Practice explaining the trade-offs clearly")
        
        print(f"\n🎯 Interview Tips:")
        print("  • Start with brute force to show you understand the problem")
        print("  • Identify the bottleneck and optimize step by step")
        print("  • Explain time/space complexity for each approach")
        print("  • Consider edge cases and constraints")
        print("  • Use the visualizations to understand scaling behavior")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return results

if __name__ == "__main__":
    print("🎓 Google SWE Interview Preparation Tool")
    print("Powered by Algoscope Pro")
    print("=" * 50)
    
    # Run the comprehensive analysis
    results = run_comprehensive_analysis()
    
    print(f"\n✨ Ready for your Google SWE interview!")
    print("Open the generated HTML files to explore the interactive reports.")

