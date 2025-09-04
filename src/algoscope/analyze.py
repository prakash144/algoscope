from __future__ import annotations

import gc
import inspect
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .complexity import analyze_function_complexity
from .memory import measure_peak_memory, _ensure_args_kwargs
from .plotting import build_reference_curves, runtime_figure, memory_figure
from .report import build_report_html, ReportSections
from .utils import (
    t_confidence_interval,
    bootstrap_confidence_interval,
    CIResult,
    human_time,
    human_bytes,
    rank,
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


def _run_once_timed(func: Callable, args: tuple, kwargs: dict) -> float:
    t0 = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - t0


def _beginner_summary_from_scaling(ns: List[int], means: List[float]) -> str:
    # Compute average ratio for doubling steps (n_i -> n_{i+1} where n_{i+1} ~ 2*n_i)
    ratios = []
    for i in range(len(ns) - 1):
        if ns[i] > 0:
            ratios.append(means[i + 1] / means[i])
    if not ratios:
        return "Not enough data to estimate scaling when n doubles."

    avg = float(np.mean(ratios))
    # crude mapping
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


def analyze_functions(
    funcs: List[Callable],
    input_builder: Callable[[int], Tuple[tuple, dict] | Any],
    ns: List[int],
    repeats: int = 7,
    warmup: int = 2,
    ci_method: str = "t",  # "t" or "bootstrap"
    confidence: float = 0.95,
    mem_backend: str = "tracemalloc",  # "tracemalloc" or "memory_profiler"
    reference_curves: Tuple[str, ...] = ("1", "logn", "n", "nlogn"),
    normalize_ref_at: str = "max",  # "min" or "max"
    html_out: Optional[str] = "report.html",
    title: str = "Algorithm Analysis Report",
    notes: Optional[str] = None,
) -> ResultObject:
    """
    Runs benchmarks and memory analysis, computes confidence intervals,
    builds plots, writes the final HTML report, and returns a results object.
    If html_out is None, the report is not saved to a file.
    """
    if not callable(input_builder):
        raise TypeError("input_builder must be a callable that accepts n and returns args/kwargs.")
    if not isinstance(ns, list) or not ns:
        raise ValueError("ns must be a non-empty list of input sizes (ints).")
    for n in ns:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("All values in ns must be positive integers.")

    # pre-analysis (manual)
    stats: Dict[str, FunctionStats] = {}
    for f in funcs:
        label = _maybe_label(f)
        guess = analyze_function_complexity(f)
        stats[label] = FunctionStats(
            label=label,
            manual_explanation=guess.explanation,
            interview_summary=guess.interview_summary,
        )

    # benchmarking
    for f in funcs:
        label = _maybe_label(f)
        # Warmups (time only)
        for _ in range(max(0, warmup)):
            for n in ns[: min(2, len(ns))]:
                args, kwargs = _ensure_args_kwargs(input_builder(n))
                _run_once_timed(f, args, kwargs)

        # Main runs
        for n in ns:
            times = []
            mems = []
            for _ in range(repeats):
                # GC control: enable+collect to minimize noise (documented in Methods)
                gc.collect()
                gc.enable()

                args, kwargs = _ensure_args_kwargs(input_builder(n))
                t = _run_once_timed(f, args, kwargs)
                times.append(t)

                # Memory measured on a separate call to avoid double-counting timing run
                m = measure_peak_memory(f, input_builder, n, mem_backend=mem_backend)
                mems.append(m)

            stats[label].times[n] = times
            stats[label].mems[n] = mems

    # CI computation
    for s in stats.values():
        for n in ns:
            samples_t = s.times[n]
            samples_m = s.mems[n]
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

    # Prepare data for plots
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

    # Reference curves normalized at min or max n to the average time across functions
    anchor_idx = 0 if normalize_ref_at == "min" else -1
    anchor_values = [time_means[label][anchor_idx] for label in time_means]
    y_anchor = float(np.mean(anchor_values))
    ref_curves = build_reference_curves(ns, reference_curves, y_anchor, normalize_at=normalize_ref_at)

    # Build figures
    runtime_fig = runtime_figure(ns, time_means, time_lowers, time_uppers, ref_curves, title)
    memory_fig = memory_figure(ns, mem_means, mem_lowers, mem_uppers, title)

    # Tables
    runtime_table = []
    for i, n in enumerate(ns):
        row = {"n": n}
        for label in stats.keys():
            ci = stats[label].time_ci[n]
            row[label] = {
                "mean": ci.mean,
                "lower": ci.lower,
                "upper": ci.upper,
            }
        runtime_table.append(row)

    memory_table = []
    for i, n in enumerate(ns):
        row = {"n": n}
        for label in stats.keys():
            ci = stats[label].mem_ci[n]
            row[label] = {
                "mean": ci.mean,
                "lower": ci.lower,
                "upper": ci.upper,
            }
        memory_table.append(row)

    # Comparison rows + mean ranks
    comparison_rows = []
    # mean rank on runtime across ns
    labels = list(stats.keys())
    ranks_accum = {label: [] for label in labels}
    for i, n in enumerate(ns):
        means = [time_means[label][i] for label in labels]
        r = rank(means)
        for idx, label in enumerate(labels):
            ranks_accum[label].append(r[idx])
        best_label = labels[int(np.argmin(means))]
        worst_label = labels[int(np.argmax(means))]
        comparison_rows.append(
            {"n": n, "best_runtime": best_label, "worst_runtime": worst_label}
        )
    mean_ranks = {label: float(np.mean(ranks)) for label, ranks in ranks_accum.items()}

    # Summaries
    beginner_summaries = {
        label: _beginner_summary_from_scaling(ns, time_means[label]) for label in labels
    }
    manual_complexities = {label: s.manual_explanation for label, s in stats.items()}
    interview_summaries = {label: s.interview_summary for label, s in stats.items()}

    methods_text = (
        f"- **Runtime** measured with `time.perf_counter()`; each (function, n) pair ran "
        f"{repeats} times after {warmup} warmup iterations. GC was enabled and collected before runs.\n"
        f"- **Memory** peak measured with `{mem_backend}` backend (`tracemalloc` by default).\n"
        f"- **Confidence Intervals:** {ci_method.upper()} at {int(100 * confidence)}% confidence. "
        "T-intervals use standard t critical values; bootstrap uses 2,000 resamples with a fixed seed.\n"
        f"- **Reference curves** (O(1), O(log n), O(n), O(n log n), plus any custom) were normalized at "
        f"the {'largest' if normalize_ref_at == 'max' else 'smallest'} n to match the average observed runtime scale."
    )

    sections = ReportSections(
        manual_complexities=manual_complexities,
        interview_summaries=interview_summaries,
        beginner_summaries=beginner_summaries,
        methods_text=methods_text,
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
    )

    if html_out:
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(html)

    return ResultObject(
        html=html,
        html_path=os.path.abspath(html_out) if html_out else None,
        title=title,
        ns=ns,
        func_stats=stats,
    )


def analyze_function(
    func: Callable,
    input_builder: Callable[[int], Tuple[tuple, dict] | Any],
    ns: List[int],
    **kwargs: Any,
) -> ResultObject:
    """
    Thin wrapper to analyze a single function.
    """
    return analyze_functions([func], input_builder, ns, **kwargs)
