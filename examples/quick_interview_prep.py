#!/usr/bin/env python3
"""
Quick Google SWE Interview Prep - Algoscope Pro
==============================================

A simple CLI tool for quick algorithm analysis during interview preparation.
Perfect for rapid prototyping and understanding algorithm trade-offs.

Usage:
    python quick_interview_prep.py two_sum
    python quick_interview_prep.py max_subarray
    python quick_interview_prep.py all
"""

import sys
import os
from typing import List, Tuple, Any

# Add src to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from algoscope import analyze_functions

# Import the Google SWE interview problems
from google_swe_interview_prep import (
    two_sum_brute_force, two_sum_optimal,
    max_subarray_brute_force, max_subarray_optimal,
    longest_common_subsequence_brute_force, longest_common_subsequence_optimal,
    coin_change_brute_force, coin_change_optimal,
    word_break_brute_force, word_break_optimal,
    build_two_sum_input, build_max_subarray_input, build_lcs_input,
    build_coin_change_input, build_word_break_input
)

def quick_analysis(problem: str, max_size: int = 1000):
    """Run quick analysis for a specific problem"""
    
    print(f"🚀 Quick Analysis: {problem.upper()}")
    print("=" * 50)
    
    if problem == "two_sum":
        functions = [two_sum_brute_force, two_sum_optimal]
        input_builder = build_two_sum_input
        input_sizes = [100, 200, 400, 800, min(1600, max_size)]
        title = "Two Sum: Brute Force vs Optimal"
        notes = "Quick analysis: O(n²) vs O(n) solutions"
        
    elif problem == "max_subarray":
        functions = [max_subarray_brute_force, max_subarray_optimal]
        input_builder = build_max_subarray_input
        input_sizes = [50, 100, 200, 400, min(800, max_size)]
        title = "Maximum Subarray: Brute Force vs Kadane's"
        notes = "Quick analysis: O(n³) vs O(n) solutions"
        
    elif problem == "lcs":
        functions = [longest_common_subsequence_brute_force, longest_common_subsequence_optimal]
        input_builder = build_lcs_input
        input_sizes = [5, 8, 10, 12, min(15, max_size)]
        title = "Longest Common Subsequence: Brute Force vs DP"
        notes = "Quick analysis: O(2^n) vs O(m*n) solutions"
        
    elif problem == "coin_change":
        functions = [coin_change_brute_force, coin_change_optimal]
        input_builder = build_coin_change_input
        input_sizes = [10, 20, 30, 40, min(50, max_size)]
        title = "Coin Change: Brute Force vs DP"
        notes = "Quick analysis: exponential vs O(amount*coins) solutions"
        
    elif problem == "word_break":
        functions = [word_break_brute_force, word_break_optimal]
        input_builder = build_word_break_input
        input_sizes = [8, 16, 24, 32, min(40, max_size)]
        title = "Word Break: Brute Force vs DP"
        notes = "Quick analysis: exponential vs O(n²) solutions"
        
    else:
        print(f"❌ Unknown problem: {problem}")
        print("Available problems: two_sum, max_subarray, lcs, coin_change, word_break")
        return None
    
    print(f"📊 Analyzing {len(functions)} algorithms...")
    print(f"📏 Input sizes: {input_sizes}")
    print(f"⏱️  This may take a few moments...")
    
    try:
        results = analyze_functions(
            funcs=functions,
            input_builder=input_builder,
            ns=input_sizes,
            repeats=3,  # Quick analysis with fewer repeats
            warmup=1,
            ci_method="t",
            reference_curves=("1", "n", "n**2", "2**n"),
            normalize_ref_at="max",
            html_out=f"examples/reports/quick_{problem}_analysis.html",
            title=title,
            notes=notes,
            timeout=30.0,
            verbose=True
        )
        
        print(f"✅ Analysis complete!")
        print(f"📄 Report saved to: {results.html_path}")
        print(f"🌐 Open the HTML file in your browser to view the interactive report")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def show_help():
    """Show help information"""
    print("🎓 Quick Google SWE Interview Prep - Algoscope Pro")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python quick_interview_prep.py <problem> [max_size]")
    print()
    print("Available Problems:")
    print("  two_sum      - Two Sum problem (O(n²) vs O(n))")
    print("  max_subarray - Maximum Subarray problem (O(n³) vs O(n))")
    print("  lcs          - Longest Common Subsequence (O(2^n) vs O(m*n))")
    print("  coin_change  - Coin Change problem (exponential vs O(amount*coins))")
    print("  word_break   - Word Break problem (exponential vs O(n²))")
    print("  all          - Run all problems (comprehensive analysis)")
    print()
    print("Examples:")
    print("  python quick_interview_prep.py two_sum")
    print("  python quick_interview_prep.py max_subarray 500")
    print("  python quick_interview_prep.py all")
    print()
    print("💡 Tips:")
    print("  • Start with smaller max_size for faster analysis")
    print("  • Use 'all' for comprehensive interview preparation")
    print("  • Open the generated HTML files for interactive reports")
    print("  • Compare brute force vs optimal solutions visually")

def main():
    """Main CLI function"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    problem = sys.argv[1].lower()
    max_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    if problem == "help" or problem == "-h" or problem == "--help":
        show_help()
        return
    
    if problem == "all":
        print("🚀 Running comprehensive analysis for all problems...")
        print("=" * 60)
        
        problems = ["two_sum", "max_subarray", "lcs", "coin_change", "word_break"]
        results = []
        
        for p in problems:
            print(f"\n📊 Analyzing {p}...")
            result = quick_analysis(p, max_size)
            if result:
                results.append(result)
        
        print(f"\n🎉 Comprehensive analysis complete!")
        print(f"📊 Generated {len(results)} reports:")
        for result in results:
            print(f"  • {result.title}")
            print(f"    📄 {result.html_path}")
        
        print(f"\n💡 Interview Tips:")
        print("  • Brute force solutions are O(n²) to O(2^n) - avoid in interviews!")
        print("  • Optimal solutions use hash maps, DP, or greedy algorithms")
        print("  • Always start with brute force, then optimize step by step")
        print("  • Explain the trade-offs clearly to your interviewer")
        
    else:
        quick_analysis(problem, max_size)

if __name__ == "__main__":
    main()

