# src/algoscope/memory.py
from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Tuple
from .utils import ensure_args_kwargs, run_and_monitor_subprocess


def measure_peak_memory_tracemalloc(
    func: Callable, args: tuple, kwargs: dict
) -> int:
    """
    Returns peak allocated memory in bytes during a single invocation (Python-level allocations).
    """
    tracemalloc.start()
    try:
        func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def measure_peak_memory_memory_profiler(
    func: Callable, args: tuple, kwargs: dict
) -> int:
    """
    Uses memory_profiler.memory_usage to sample RSS and returns peak in bytes.
    """
    try:
        from memory_profiler import memory_usage
    except Exception:
        raise RuntimeError(
            "memory_profiler not installed. Install with `pip install memory-profiler` "
            "or use mem_backend='tracemalloc' or 'rss'."
        )
    # returns in MiB
    mem_seq = memory_usage(
        (func, args, kwargs), interval=0.01, timeout=None, max_iterations=1, retval=False
    )
    peak_mib = max(mem_seq) if mem_seq else 0.0
    return int(peak_mib * (1024**2))


def measure_peak_memory_rss(
    func: Callable, args: tuple, kwargs: dict, timeout: float = 10.0, poll_interval: float = 0.02
) -> Tuple[int, float, dict]:
    """
    Run func in a subprocess, monitor RSS (peak) and elapsed time, return (peak_bytes, duration_seconds, info_dict).
    Uses utils.run_and_monitor_subprocess under the hood.
    """
    info = run_and_monitor_subprocess(func, args=args, kwargs=kwargs, timeout=timeout, poll_interval=poll_interval)
    peak = int(info.get("peak_rss", 0) or 0)
    duration = info.get("duration", None)
    return peak, duration, info


def measure_peak_memory(
    func: Callable, input_builder: Callable[[int], Any], n: int, mem_backend: str, timeout: float = 10.0
) -> int:
    args, kwargs = ensure_args_kwargs(input_builder(n))
    if mem_backend == "tracemalloc":
        return measure_peak_memory_tracemalloc(func, args, kwargs)
    elif mem_backend == "memory_profiler":
        return measure_peak_memory_memory_profiler(func, args, kwargs)
    elif mem_backend == "rss":
        peak, _, info = measure_peak_memory_rss(func, args, kwargs, timeout=timeout)
        return peak
    else:
        raise ValueError("mem_backend must be 'tracemalloc', 'memory_profiler' or 'rss'")
