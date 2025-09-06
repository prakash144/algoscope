# src/algoscope/analyze.py
from __future__ import annotations

import gc
import os
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .complexity import analyze_function_complexity
from .memory import measure_peak_memory
from .plotting import build_reference_curves, runtime_figure, memory_figure, overview_figure, heatmap_figure
from .report import build_report_html, ReportSections
from .utils import (
    t_confidence_interval,
    bootstrap_confidence_interval,
    CIResult,
    rank,
    ensure_args_kwargs,
    is_picklable,
    run_and_monitor_subprocess,
)
from typing import Sequence

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
    # Google SWE interview specific attributes
    interview_tips: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    confidence: Optional[float] = None


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


def _is_multi_param_input_example_from_builder(input_builder: Callable, ns: List[int]) -> bool:
    """
    Heuristic: call input_builder for a small set of ns and inspect args/kwargs.
    We consider the input 'multi-parameter' only if there are multiple collection-like
    arguments (list/tuple/set/dict/ndarray) or if arguments besides the first are
    collection-like. Scalar extras (int/float/None/str/bool) are *not* considered
    to indicate multi-parameter experiments (this avoids false positives like linear_search).
    """
    try:
        # pick two different sizes if available to see shape behaviour (but we don't rely on variation)
        n0 = ns[0]
        n1 = ns[min(1, len(ns) - 1)]
        a0, kw0 = ensure_args_kwargs(input_builder(n0))
        a1, kw1 = ensure_args_kwargs(input_builder(n1))
    except Exception:
        # Can't probe — be conservative and say False (no warning)
        return False

    def is_collection_like(x) -> bool:
        if x is None:
            return False
        if isinstance(x, (list, tuple, set, dict)):
            return True
        # numpy arrays
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return True
        # other large iterables (but exclude strings)
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return True
        return False

    # Count collection-like positional args in the first call
    seq_count = sum(1 for arg in a0 if is_collection_like(arg))
    # Also count any collection-like kwargs
    seq_count += sum(1 for v in kw0.values() if is_collection_like(v))

    # If there is more than one collection-like argument, it's multi-parameter (likely)
    if seq_count > 1:
        return True

    # If there is exactly one collection-like arg but some kwargs are collections, flag it
    if seq_count == 1:
        # If there are kwargs that are collections -> multi
        if any(is_collection_like(v) for v in kw0.values()):
            return True
        # If other positional args besides the first are collection-like on the second probe => multi
        # (This handles cases where input_builder might change shape across n)
        other_pos_seq = any(is_collection_like(arg) for arg in a0[1:])
        other_pos_seq |= any(is_collection_like(arg) for arg in a1[1:])
        if other_pos_seq:
            return True

    # Otherwise do not treat it as multi-parameter (scalars like 'target' are OK)
    return False


def analyze_functions(
    funcs: List[Callable],
    input_builder: Callable[[int], Tuple[tuple, dict] | Any],
    ns: List[int],
    repeats: int = 7,
    warmup: int = 2,
    ci_method: str = "t",
    confidence: float = 0.95,
    mem_backend: str = "tracemalloc",
    reference_curves: Tuple[str, ...] = ("1", "logn", "n", "nlogn"),
    normalize_ref_at: str = "max",
    html_out: Optional[str] = "report.html",
    title: str = "Algorithm Analysis Report",
    notes: Optional[str] = None,
    timeout: Optional[float] = 10.0,
    run_in_subprocess: bool = True,
    verbose: bool = True,
    # grid params (optional)
    grid_x: Optional[List[int]] = None,
    grid_y: Optional[List[int]] = None,
    grid_input_builder: Optional[Callable[[int, int], Tuple[tuple, dict] | Any]] = None,
    grid_log_color: bool = False,
) -> ResultObject:
    """
    Analyze functions: timing, memory, heuristics, plotting.
    Optional grid sweep mode: provide grid_x, grid_y and grid_input_builder(x, y)
    to produce per-function heatmaps embedded in the report.
    """
    if verbose:
        print("🔍 Analyzing algorithms... please wait, this may take some time ⏳")

    # Validation
    if not callable(input_builder) and not (grid_x and grid_y and grid_input_builder):
        raise TypeError("input_builder must be a callable (unless running grid mode with grid_input_builder).")
    if not isinstance(ns, list) or not ns:
        raise ValueError("ns must be a non-empty list of integers.")
    for n in ns:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("All values in ns must be positive integers.")

    # Setup
    stats: Dict[str, FunctionStats] = {}
    func_map: Dict[str, Callable] = {}
    picklable_cache: Dict[str, bool] = {}

    for f in funcs:
        label = _maybe_label(f)
        guess = analyze_function_complexity(f, input_builder if callable(input_builder) else None)
        stats[label] = FunctionStats(
            label=label,
            manual_explanation=guess.explanation,
            interview_summary=guess.interview_summary,
            dynamic_guess=guess.dynamic_guess,
            interview_tips=guess.interview_tips,
            optimization_suggestions=guess.optimization_suggestions,
            patterns_detected=guess.patterns_detected,
            confidence=guess.confidence,
        )
        func_map[label] = f
        picklable_cache[label] = is_picklable(f)

        # Heuristic multi-param input detection (improved)
        try:
            is_multi_param = _is_multi_param_input_example_from_builder(input_builder, ns) if callable(
                input_builder) else False
        except Exception:
            is_multi_param = False

        if is_multi_param:
            for s in stats.values():
                s.errors.append(
                    "Input appears multi-parameter (multiple collection-like inputs). "
                    "For correct empirical scaling (e.g. knapsack where both n and W matter), "
                    "consider using grid mode (grid_x/grid_y/grid_input_builder) or run separate experiments."
                )

    # Warmup
    for f in funcs:
        label = _maybe_label(f)
        for _ in range(max(0, warmup)):
            for n in ns[: min(2, len(ns))]:
                try:
                    args, kwargs = ensure_args_kwargs(input_builder(n))
                except Exception as e:
                    stats[label].errors.append(f"input_builder failed for warmup n={n}: {repr(e)}")
                    continue
                try:
                    if run_in_subprocess and picklable_cache[label]:
                        _ = run_and_monitor_subprocess(f, args=tuple(args), kwargs=kwargs, timeout=timeout)
                    else:
                        f(*args, **kwargs)
                except Exception:
                    stats[label].errors.append(f"warmup failed for n={n}")

    # Main timed + memory runs
    for f in funcs:
        label = _maybe_label(f)
        can_rss = (mem_backend == "rss") and run_in_subprocess and picklable_cache[label]
        if mem_backend == "rss" and not can_rss:
            stats[label].errors.append(
                "Requested mem_backend='rss' but subprocess isolation is unavailable; falling back to tracemalloc."
            )

        for n in ns:
            times: List[float] = []
            mems: List[int] = []
            for _ in range(repeats):
                gc.collect()
                gc.enable()
                try:
                    args, kwargs = ensure_args_kwargs(input_builder(n))
                except Exception as e:
                    stats[label].errors.append(f"input_builder failed for n={n}: {repr(e)}")
                    times.append(float("nan"))
                    mems.append(0)
                    continue

                if run_in_subprocess and picklable_cache[label]:
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
                    mems.append(peak_rss)
                else:
                    try:
                        t0 = time.perf_counter()
                        f(*args, **kwargs)
                        duration = time.perf_counter() - t0
                        times.append(duration)
                    except Exception as e:
                        stats[label].errors.append(f"n={n}: exception during run: {repr(e)}")
                        times.append(float("nan"))

                    effective_mem_backend = mem_backend
                    if mem_backend == "rss":
                        effective_mem_backend = "tracemalloc"

                    try:
                        m = measure_peak_memory(f, input_builder, n, mem_backend=effective_mem_backend)
                    except Exception as me:
                        stats[label].errors.append(f"n={n}: memory measurement failed: {repr(me)}")
                        m = 0
                    mems.append(m)

            stats[label].times[n] = times
            stats[label].mems[n] = mems

    # Confidence intervals
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

    # Reference curves
    anchor_idx = 0 if normalize_ref_at == "min" else -1
    anchor_values = []
    for label in time_means:
        vals = time_means[label]
        if len(vals) > 0:
            v = vals[anchor_idx]
            if np.isfinite(v):
                anchor_values.append(v)
    y_anchor = float(np.mean(anchor_values)) if anchor_values else 1.0
    ref_curves = build_reference_curves(ns, reference_curves, y_anchor, normalize_at=normalize_ref_at)

    # Figures
    runtime_fig = runtime_figure(ns, time_means, time_lowers, time_uppers, ref_curves, title)
    memory_fig = memory_figure(ns, mem_means, mem_lowers, mem_uppers, title)

    # Comparison rows + ranks
    comparison_rows: List[Dict[str, Any]] = []
    labels = list(stats.keys())
    ranks_accum: Dict[str, List[float]] = {label: [] for label in labels}
    for i, n in enumerate(ns):
        means = [time_means[label][i] for label in labels]
        r = rank(means)
        for idx, label in enumerate(labels):
            ranks_accum[label].append(r[idx])
        
        # Handle NaN values in means
        if all(np.isnan(means)):
            best_label = "N/A"
            worst_label = "N/A"
        else:
            # Filter out NaN values for comparison
            valid_means = [m for m in means if not np.isnan(m)]
            if valid_means:
                best_idx = np.nanargmin(means)
                worst_idx = np.nanargmax(means)
                best_label = labels[best_idx] if not np.isnan(means[best_idx]) else "N/A"
                worst_label = labels[worst_idx] if not np.isnan(means[worst_idx]) else "N/A"
            else:
                best_label = "N/A"
                worst_label = "N/A"
        
        comparison_rows.append({"n": n, "best_runtime": best_label, "worst_runtime": worst_label})

    mean_ranks = {label: float(np.mean(ranks)) for label, ranks in ranks_accum.items()}

    # Overview figure
    try:
        ov_fig = overview_figure(ns, time_means, mem_means, mean_ranks)
    except Exception:
        ov_fig = None

    # Build tables defensively
    runtime_table: List[Dict[str, Any]] = []
    for n in ns:
        row: Dict[str, Any] = {"n": n}
        for label in stats.keys():
            ci = stats[label].time_ci.get(n)
            if ci is None:
                row[label] = {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
            else:
                row[label] = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper}
        runtime_table.append(row)

    memory_table: List[Dict[str, Any]] = []
    for n in ns:
        row: Dict[str, Any] = {"n": n}
        for label in stats.keys():
            ci = stats[label].mem_ci.get(n)
            if ci is None:
                row[label] = {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
            else:
                row[label] = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper}
        memory_table.append(row)

    # Summaries & dynamic guesses
    beginner_summaries: Dict[str, str] = {}
    dynamic_guesses: Dict[str, str] = {}
    for label in labels:
        slope = _empirical_slope(ns, time_means[label])
        if slope is None:
            beginner_summaries[label] = "Not enough reliable data to estimate scaling."
            dynamic_guesses[label] = stats[label].dynamic_guess or "No dynamic hint"
        else:
            if slope < 0.3:
                msg = "Runtime appears sub-logarithmic / nearly constant."
                family = "sub-logarithmic / decreasing"
            elif slope < 0.8:
                msg = "Runtime scales like O(log n)."
                family = "O(log n)"
            elif slope < 1.2:
                msg = "Runtime scales like O(n)."
                family = "O(n)"
            elif slope < 1.8:
                msg = "Runtime scales like O(n log n) or slightly superlinear."
                family = "O(n log n) / slightly superlinear"
            elif slope < 2.5:
                msg = "Runtime scales like O(n^2)."
                family = "O(n^2)"
            else:
                msg = "Runtime appears super-polynomial / exponential."
                family = "super-polynomial / exponential-like"

            beginner_summaries[label] = f"Empirical slope ≈ {slope:.2f} — {msg}"
            dynamic_guesses[label] = f"Empirical slope ≈ {slope:.2f} -> suggests {family}."

    manual_complexities_full = {label: s.manual_explanation for label, s in stats.items()}
    interview_summaries = {label: s.interview_summary for label, s in stats.items()}

    methods_text = (
        f"- **Runtime** measured with `time.perf_counter()`; each (function, n) pair ran "
        f"{repeats} times after {warmup} warmup iterations. GC was enabled and collected before runs.\n"
        f"- **Memory** peak measured with `{mem_backend}` backend. When `rss` is selected and `psutil` is installed, "
        "a child process is monitored for RSS (recommended for realistic peak memory).\n"
        f"- **Confidence Intervals:** {ci_method.upper()} at {int(100 * confidence)}% confidence."
    )

    # Extract Google SWE interview specific data
    interview_tips = {}
    optimization_suggestions = {}
    patterns_detected = {}
    confidence_scores = {}
    
    for label, stat in stats.items():
        if hasattr(stat, 'interview_tips') and stat.interview_tips:
            interview_tips[label] = stat.interview_tips
        if hasattr(stat, 'optimization_suggestions') and stat.optimization_suggestions:
            optimization_suggestions[label] = stat.optimization_suggestions
        if hasattr(stat, 'patterns_detected') and stat.patterns_detected:
            patterns_detected[label] = stat.patterns_detected
        if hasattr(stat, 'confidence') and stat.confidence is not None:
            confidence_scores[label] = stat.confidence

    sections = ReportSections(
        manual_complexities=manual_complexities_full,
        interview_summaries=interview_summaries,
        beginner_summaries=beginner_summaries,
        methods_text=methods_text,
        dynamic_guesses=dynamic_guesses,
        interview_tips=interview_tips,
        optimization_suggestions=optimization_suggestions,
        patterns_detected=patterns_detected,
        confidence_scores=confidence_scores,
    )

    # Optional grid sweep heatmaps
    heatmap_divs: Dict[str, str] = {}
    if grid_x and grid_y and grid_input_builder:
        gx = list(grid_x)
        gy = list(grid_y)
        # prepare mapping of label->function for safe dispatch
        for label, func in func_map.items():
            z = np.full((len(gy), len(gx)), np.nan, dtype=float)
            for j, yv in enumerate(gy):
                for i, xv in enumerate(gx):
                    samples = []
                    for rep in range(repeats):
                        try:
                            args, kwargs = ensure_args_kwargs(grid_input_builder(xv, yv))
                        except Exception as e:
                            stats[label].errors.append(f"grid input_builder failed for (x={xv}, y={yv}): {repr(e)}")
                            samples.append(float("nan"))
                            continue
                        try:
                            if run_in_subprocess and picklable_cache[label]:
                                info = run_and_monitor_subprocess(func, args=tuple(args), kwargs=kwargs, timeout=timeout)
                                if info.get("success"):
                                    samples.append(float(info.get("duration", float("nan"))))
                                else:
                                    samples.append(float("nan"))
                            else:
                                t0 = time.perf_counter()
                                func(*args, **kwargs)
                                samples.append(time.perf_counter() - t0)
                        except Exception:
                            samples.append(float("nan"))
                    arr = np.array([x for x in samples if np.isfinite(x)], dtype=float)
                    z[j, i] = float(np.mean(arr)) if arr.size > 0 else float("nan")
            try:
                fig = heatmap_figure(gx, gy, z, title=f"{title} — {label} (grid sweep)", x_label="X", y_label="Y", z_label="Time (s)", log_color=grid_log_color)
                # convert without embedding plotly.js (template includes CDN)
                import plotly.io as _pio
                heatmap_divs[label] = _pio.to_html(fig, include_plotlyjs=False, full_html=False)
            except Exception as e:
                stats[label].errors.append(f"failed to build heatmap for {label}: {repr(e)}")

    # Render HTML
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
        heatmap_divs=heatmap_divs,
    )

    if html_out:
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(html)

    if verbose:
        if html_out:
            print(f"✅ Analysis complete. Report saved to: {os.path.abspath(html_out)}")
        else:
            print("✅ Analysis complete. (No report file saved)")

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
    return analyze_functions([func], input_builder, ns, **kwargs)
