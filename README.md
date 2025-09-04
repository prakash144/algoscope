# Algoscope
🚀 Think of algoscope as your algorithm performance lab — it measures, visualizes, and explains complexity so you can focus on problem-solving, not benchmarking boilerplate.

**algoscope** analyzes Python functions/algorithms for time and space complexity and produces a **single, self-contained HTML report** with beginner-friendly explanations and professional, interactive Plotly visualizations.

## Features

- Manual Big-O static analysis heuristics (loops, recursion, divide-and-conquer hints)
- Empirical benchmarking (repeats, warmup, GC control) with **95% CIs** (t-intervals or bootstrap)
- Peak memory per call (tracemalloc by default; optional memory_profiler)
- Combined runtime and memory plots with **reference curves** (O(1), O(log n), O(n), O(n log n), custom)
- Side-by-side multi-function comparison, summary table & mean ranks
- Turnkey API with a single `analyze_functions(...)` call
- Self-contained HTML report (embedded Plotly JS)

## Installation (editable)

```bash
pip install -U pip
pip install plotly jinja2 numpy memory-profiler psutil
pip install -e .
```

## Usage

See `examples/compare_linear_vs_binary.py` for a full example comparing linear vs. binary search.

### Quickstart

Set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -U pip
pip install plotly jinja2 numpy memory-profiler psutil
```

Install algoscope in editable mode:
*(After you have created the package files from the assistant's response)*

```bash
pip install -e .
```

Run the example comparison:
Create a Python file named `run_analysis.py` with the following content:

```python
from algoscope import analyze_functions
from examples.compare_linear_vs_binary import build_input, linear_search
from examples.binary_search_example import binary_search

# 1. Define the functions to compare
functions_to_compare = [linear_search, binary_search]

# 2. Define the input sizes
input_sizes = [1000, 2000, 4000, 8000, 16000, 32000]

# 3. Run the analysis
analysis_results = analyze_functions(
    funcs=functions_to_compare,
    input_builder=build_input,
    ns=input_sizes,
    repeats=11,
    ci_method="t",
    reference_curves=("1", "logn", "n", "nlogn", "n**2"),
    normalize_ref_at="max",
    html_out="report.html",
    title="Linear Search vs. Binary Search"
)

print(f"Analysis complete. Report saved to {analysis_results.html_path}")

# In a Jupyter notebook, you could simply have the 'analysis_results'
# object as the last line in a cell to display the report inline.
```

Then, run the script from your terminal:

```bash
python run_analysis.py
```

This will generate `report.html` in your current directory.

## Methods

- **Runtime** via `time.perf_counter()` with configurable repeats, warmups, and GC control.
- **Peak memory** via `tracemalloc` (default) or `memory_profiler` if installed.
- **Confidence intervals** via t-intervals (approximate criticals) or non-parametric bootstrap (2,000 resamples).
- **Reference curves** scaled at min or max n to match empirical magnitudes for meaningful overlays.

```
Analysis complete. Report saved to /.../algoscope/report.html
```

*(Opening `report.html` shows two interactive charts, manual Big-O text, beginner/interview summaries, tables with ±95% CIs, and the comparison section.)*

---

## Generated Interview-Style Summary (for the required example)

- **linear_search** — *"linear_search has an estimated time complexity of O(n) and space complexity of O(1). For recursive implementations that halve the search space (e.g., binary search), space is O(log n) due to recursion depth; iterative versions are O(1) space."*

- **binary_search** — *"binary_search has a time complexity of O(log n). The iterative implementation uses O(1) extra space; the recursive variant uses O(log n) stack space."*

And the beginner summary line you'll see for each:

- **linear_search** — "When the input size n doubles, the runtime grows by approximately ~2×, which suggests performance closer to O(n)."
- **binary_search** — "When the input size n doubles, the runtime grows by approximately ~1.3×, which suggests performance closer to O(log n)."

*(Exact multipliers vary a bit by machine; the report computes them from your measurements.)*

---

## Extensibility Guide

- **Add custom reference curves:**  
  Pass strings in `reference_curves`, e.g. `("1", "logn", "n", "nlogn", "n**2", "n**3")`. The package safely evaluates expressions like `"n**k"` using only `n`, `np`, `log`, `log2`.

- **Compare more functions:**  
  Add them to the `funcs` list and ensure your `input_builder(n)` returns inputs all functions accept. The report scales lines and CI bands per function and updates the comparison table automatically.

- **Switch CI method:**  
  Use `ci_method="bootstrap"` to get non-parametric CIs (documented in Methods).

- **Change normalization point for curves:**  
  Use `normalize_ref_at="min"` to align reference curves at the smallest n instead of the largest.

- **Use memory_profiler:**  
  `mem_backend="memory_profiler"` (requires `pip install memory-profiler`). It samples process RSS; tracemalloc measures Python heap allocations.

---

## Why this design (concise)

- **Clarity & pedagogy:** Static Big-O heuristics + empirics + reference curves anchor intuition and numbers in one place.
- **Rigor with pragmatism:** t-intervals with approximated criticals (no SciPy dep) or bootstrap with fixed seed.
- **UX:** One function call → a single, self-contained HTML artifact, interactive, with embedded Plotly JS.
- **Maintainability:** Modular files, type hints, docstrings, tests, and a clean `pyproject.toml`.