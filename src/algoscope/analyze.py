# src/algoscope/analyze.py
from __future__ import annotations

import gc
import inspect
import os
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .complexity import analyze_function_complexity
from .memory import measure_peak_memory
from .plotting import build_reference_curves, runtime_figure, memory_figure, overview_figure
from .report import build_report_html, ReportSections
from .utils import (
    t_confidence_interval,
    bootstrap_confidence_interval,
    CIResult,
    human_time,
    human_bytes,
    rank,
    ensure_args_kwargs,
    is_picklable,
    run_and_monitor_subprocess,
)


@dataclass
class FunctionStats:
    label: str
    times: Dict[int, List[float]] = field(default_factory=dict)   # n -> list of seconds
    mems: Dict[int, List[int]] = field(default_factory=dict)      # n -> list of bytes
    time_ci: Dict[int, CIResult] = field(default_factory=dict)
    mem_ci: Dict[int, CIResult] = field(default_factory=dict)
    manual_explanation: str = ""
    interview_summary: str = ""
    dynamic_guess: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class ResultObject:
    html: str
    html_path: Optional[str]
    title: str
    ns: List[int]
    func_stats: Dict[str, FunctionStats]

    def _repr_html_(self) -> str:  # Jupyter-friendly
        return self.html


def _maybe_label(func: Callable) -> str:
    return getattr(func, "__name__", repr(func))


def _beginner_summary_from_scaling(ns: List[int], means: List[float]) -> str:
    ratios = []
    for i in range(len(ns) - 1):
        if ns[i] > 0 and means[i] > 0 and not math.isnan(means[i]) and not math.isnan(means[i + 1]):
            ratios.append(means[i + 1] / means[i])
    if not ratios:
        return "Not enough data to estimate scaling when n doubles."

    avg = float(np.mean(ratios))
    if avg < 1.5:
        trend = "closer to O(log n)"
    elif avg < 2.8:
        trend = "closer to O(n)"
    elif avg < 5.0:
        trend = "closest to O(n log n)"
    else:
        trend = "suggests superlinear (possibly O(n^2) or higher)"

    return (
        f"When the input size n doubles, the runtime grows by approximately {avg:.2f}×, "
        f"which suggests performance {trend}."
    )


def _empirical_slope(ns: List[int], means: List[float]) -> Optional[float]:
    """
    Fit log-log slope. Returns slope (exponent) or None if not enough data or invalid.
    """
    arr = np.asarray(means, dtype=float)
    if arr.size < 2:
        return None
    mask = np.isfinite(arr) & (arr > 0)
    if mask.sum() < 2:
        return None
    xs = np.log(np.asarray(ns)[mask])
    ys = np.log(arr[mask])
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope)
    except Exception:
        return None


def analyze_functions(
    funcs: List[Callable],
    input_builder: Callable[[int], Tuple[tuple, dict] | Any],
    ns: List[int],
    repeats: int = 7,
    warmup: int = 2,
    ci_method: str = "t",  # "t" or "bootstrap"
    confidence: float = 0.95,
    mem_backend: str = "tracemalloc",  # "tracemalloc", "memory_profiler", or "rss"
    reference_curves: Tuple[str, ...] = ("1", "logn", "n", "nlogn"),
    normalize_ref_at: str = "max",  # "min" or "max"
    html_out: Optional[str] = "report.html",
    title: str = "Algorithm Analysis Report",
    notes: Optional[str] = None,
    timeout: Optional[float] = 10.0,
    run_in_subprocess: bool = True,
) -> ResultObject:
    """
    Runs benchmarks and memory analysis, computes confidence intervals,
    builds plots, writes the final HTML report, and returns a results object.
    If html_out is None, the report is not saved to a file.
    """
    # Input validation
    if not callable(input_builder):
        raise TypeError("input_builder must be a callable that accepts n and returns args/kwargs.")
    if not isinstance(ns, list) or not ns:
        raise ValueError("ns must be a non-empty list of input sizes (ints).")
    for n in ns:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("All values in ns must be positive integers.")

    # Pre-analysis (heuristic static analysis + dynamic probe)
    stats: Dict[str, FunctionStats] = {}
    for f in funcs:
        label = _maybe_label(f)
        guess = analyze_function_complexity(f, input_builder if callable(input_builder) else None)
        stats[label] = FunctionStats(
            label=label,
            manual_explanation=guess.explanation,
            interview_summary=guess.interview_summary,
            dynamic_guess=guess.dynamic_guess,
        )

    # Warmups (time only)
    for f in funcs:
        for _ in range(max(0, warmup)):
            for n in ns[: min(2, len(ns))]:
                args, kwargs = ensure_args_kwargs(input_builder(n))
                try:
                    if run_in_subprocess and is_picklable(f):
                        _ = run_and_monitor_subprocess(f, args=tuple(args), kwargs=kwargs, timeout=timeout)
                    else:
                        # in-process warmup
                        f(*args, **kwargs)
                except Exception:
                    # warmup failures are non-fatal; just record
                    label = _maybe_label(f)
                    stats[label].errors.append(f"warmup failed for n={n}")

    # Main timed + memory runs
    for f in funcs:
        label = _maybe_label(f)
        # Determine whether we can realistically capture RSS for this function
        can_rss = (mem_backend == "rss") and run_in_subprocess and is_picklable(f)
        if mem_backend == "rss" and not can_rss:
            stats[label].errors.append(
                "Requested mem_backend='rss' but subprocess isolation is unavailable for this function; "
                "falling back to tracemalloc for in-process memory measurements."
            )

        for n in ns:
            times = []
            mems = []
            for _ in range(repeats):
                gc.collect()
                gc.enable()
                args, kwargs = ensure_args_kwargs(input_builder(n))

                # Prefer single-run subprocess that provides duration + peak RSS (works for RSS measurement)
                if run_in_subprocess and is_picklable(f):
                    info = run_and_monitor_subprocess(f, args=tuple(args), kwargs=kwargs, timeout=timeout)
                    if info.get("timed_out"):
                        stats[label].errors.append(f"n={n}: timed out (>{timeout}s)")
                        times.append(float("nan"))
                        mems.append(0)
                        continue
                    if not info.get("success"):
                        tb = info.get("traceback") or info.get("exception") or "Unknown error"
                        stats[label].errors.append(f"n={n}: exception: {tb}")
                        times.append(float("nan"))
                        mems.append(0)
                        continue
                    duration = info.get("duration", float("nan"))
                    peak_rss = int(info.get("peak_rss", 0) or 0)
                    times.append(duration)
                    # if user explicitly asked for rss, use the peak_rss; otherwise still record peak_rss for visibility
                    mems.append(peak_rss)
                else:
                    # fallback: in-process measurement (no reliable RSS unless underlying mem_backend supports it)
                    try:
                        t0 = time.perf_counter()
                        f(*args, **kwargs)
                        duration = time.perf_counter() - t0
                        times.append(duration)
                    except Exception as e:
                        stats[label].errors.append(f"n={n}: exception: {repr(e)}")
                        times.append(float("nan"))

                    # Choose effective in-process mem backend (cannot do rss reliably here)
                    effective_mem_backend = mem_backend
                    if mem_backend == "rss":
                        # cannot do RSS without subprocess monitoring; fall back to tracemalloc
                        effective_mem_backend = "tracemalloc"

                    try:
                        m = measure_peak_memory(f, input_builder, n, mem_backend=effective_mem_backend)
                    except Exception as me:
                        stats[label].errors.append(f"n={n}: memory measurement failed: {repr(me)}")
                        m = 0
                    mems.append(m)

            stats[label].times[n] = times
            stats[label].mems[n] = mems

    # Compute confidence intervals (per function, per n), ignoring NaNs
    for s in stats.values():
        for n in ns:
            samples_t = [x for x in s.times.get(n, []) if isinstance(x, (int, float)) and math.isfinite(x)]
            samples_m = [x for x in s.mems.get(n, []) if isinstance(x, (int, float)) and math.isfinite(x)]
            if ci_method == "t":
                t_ci = t_confidence_interval(samples_t, confidence)
                m_ci = t_confidence_interval(samples_m, confidence)
            elif ci_method == "bootstrap":
                t_ci = bootstrap_confidence_interval(samples_t, confidence)
                m_ci = bootstrap_confidence_interval(samples_m, confidence)
            else:
                raise ValueError("ci_method must be 't' or 'bootstrap'")
            s.time_ci[n] = t_ci
            s.mem_ci[n] = m_ci

    # Aggregate means / bounds for plotting
    time_means: Dict[str, List[float]] = {}
    time_lowers: Dict[str, List[float]] = {}
    time_uppers: Dict[str, List[float]] = {}

    mem_means: Dict[str, List[float]] = {}
    mem_lowers: Dict[str, List[float]] = {}
    mem_uppers: Dict[str, List[float]] = {}

    for label, s in stats.items():
        time_means[label] = [s.time_ci[n].mean for n in ns]
        time_lowers[label] = [s.time_ci[n].lower for n in ns]
        time_uppers[label] = [s.time_ci[n].upper for n in ns]

        mem_means[label] = [s.mem_ci[n].mean for n in ns]
        mem_lowers[label] = [s.mem_ci[n].lower for n in ns]
        mem_uppers[label] = [s.mem_ci[n].upper for n in ns]

    # Reference curves normalized at min or max n
    anchor_idx = 0 if normalize_ref_at == "min" else -1
    anchor_values = [time_means[label][anchor_idx] for label in time_means if len(time_means[label]) > 0]
    y_anchor = float(np.mean(anchor_values)) if anchor_values else 1.0
    ref_curves = build_reference_curves(ns, reference_curves, y_anchor, normalize_at=normalize_ref_at)

    # Build runtime & memory figures (Plotly)
    runtime_fig = runtime_figure(ns, time_means, time_lowers, time_uppers, ref_curves, title)
    memory_fig = memory_figure(ns, mem_means, mem_lowers, mem_uppers, title)

    # Comparison rows + mean ranks
    comparison_rows: List[Dict[str, Any]] = []
    labels = list(stats.keys())
    ranks_accum: Dict[str, List[float]] = {label: [] for label in labels}
    for i, n in enumerate(ns):
        means = [time_means[label][i] for label in labels]
        # use rank on raw means (smaller is better)
        r = rank(means)
        for idx, label in enumerate(labels):
            ranks_accum[label].append(r[idx])
        # use nan-safe argmin/argmax
        best_label = labels[int(np.nanargmin(means))]
        worst_label = labels[int(np.nanargmax(means))]
        comparison_rows.append({"n": n, "best_runtime": best_label, "worst_runtime": worst_label})

    mean_ranks = {label: float(np.mean(ranks)) for label, ranks in ranks_accum.items()}

    # Build overview figure
    try:
        ov_fig = overview_figure(ns, time_means, mem_means, mean_ranks)
    except Exception:
        ov_fig = None

    # Build tables
    runtime_table: List[Dict[str, Any]] = []
    for n in ns:
        row: Dict[str, Any] = {"n": n}
        for label in stats.keys():
            ci = stats[label].time_ci[n]
            row[label] = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper}
        runtime_table.append(row)

    memory_table: List[Dict[str, Any]] = []
    for n in ns:
        row: Dict[str, Any] = {"n": n}
        for label in stats.keys():
            ci = stats[label].mem_ci[n]
            row[label] = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper}
        memory_table.append(row)

    # Summaries & dynamic empirical slopes
    beginner_summaries = {label: _beginner_summary_from_scaling(ns, time_means[label]) for label in labels}

    manual_complexities_full = {label: s.manual_explanation for label, s in stats.items()}
    interview_summaries = {label: s.interview_summary for label, s in stats.items()}

    dynamic_guesses = {}
    for label in labels:
        slope = _empirical_slope(ns, time_means[label])
        if slope is not None:
            # map slope to human-friendly families
            if slope < 0.3:
                family = "sub-logarithmic / decreasing"
            elif slope < 0.8:
                family = "O(log n)"
            elif slope < 1.2:
                family = "O(n)"
            elif slope < 1.8:
                family = "O(n log n) / slightly superlinear"
            elif slope < 2.5:
                family = "O(n^2)"
            else:
                family = "super-polynomial / exponential-like"
            dynamic_guesses[label] = f"Empirical slope ≈ {slope:.2f} -> suggests {family}."
        else:
            dynamic_guesses[label] = stats[label].dynamic_guess or "No dynamic hint"

    methods_text = (
        f"- **Runtime** measured with `time.perf_counter()`; each (function, n) pair ran "
        f"{repeats} times after {warmup} warmup iterations. GC was enabled and collected before runs.\n"
        f"- **Memory** peak measured with `{mem_backend}` backend. When `rss` is selected and `psutil` is installed, "
        "a child process is monitored for RSS (recommended for realistic peak memory).\n"
        f"- **Confidence Intervals:** {ci_method.upper()} at {int(100 * confidence)}% confidence."
    )

    sections = ReportSections(
        manual_complexities=manual_complexities_full,
        interview_summaries=interview_summaries,
        beginner_summaries=beginner_summaries,
        methods_text=methods_text,
        dynamic_guesses=dynamic_guesses,
    )

    html = build_report_html(
        title=title,
        notes=notes,
        ns=ns,
        runtime_table=runtime_table,
        memory_table=memory_table,
        comparison_rows=[{**row, "mean_ranks": mean_ranks} for row in comparison_rows],
        runtime_fig=runtime_fig,
        memory_fig=memory_fig,
        sections=sections,
        overview_fig=ov_fig,
        html_path=os.path.abspath(html_out) if html_out else None,
        func_stats=stats,
    )

    if html_out:
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(html)

    return ResultObject(html=html, html_path=os.path.abspath(html_out) if html_out else None, title=title, ns=ns, func_stats=stats)


# Convenience wrapper for analyzing a single function
def analyze_function(
    func: Callable,
    input_builder: Callable[[int], Tuple[tuple, dict] | Any],
    ns: List[int],
    **kwargs: Any,
) -> ResultObject:
    """
    Thin wrapper to analyze a single function.
    Simply calls analyze_functions([func], ...).
    """
    return analyze_functions([func], input_builder, ns, **kwargs)
