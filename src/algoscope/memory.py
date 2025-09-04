from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Dict, Tuple


def _ensure_args_kwargs(result: Any) -> Tuple[tuple, dict]:
    """
    Normalize input_builder output into (args, kwargs).
    Accepts:
      - (args_tuple, kwargs_dict)
      - args_tuple
      - single object (becomes (obj,), {})
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result  # (args, kwargs)
    if isinstance(result, tuple):
        return result, {}
    return (result,), {}


def measure_peak_memory_tracemalloc(
    func: Callable, args: tuple, kwargs: dict
) -> int:
    """
    Returns peak allocated memory in bytes during a single invocation.
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
            "or use mem_backend='tracemalloc'."
        )
    # returns in MiB
    mem_seq = memory_usage(
        (func, args, kwargs), interval=0.01, timeout=None, max_iterations=1, retval=False
    )
    peak_mib = max(mem_seq) if mem_seq else 0.0
    return int(peak_mib * (1024**2))


def measure_peak_memory(
    func: Callable, input_builder: Callable[[int], Any], n: int, mem_backend: str
) -> int:
    args, kwargs = _ensure_args_kwargs(input_builder(n))
    if mem_backend == "tracemalloc":
        return measure_peak_memory_tracemalloc(func, args, kwargs)
    elif mem_backend == "memory_profiler":
        return measure_peak_memory_memory_profiler(func, args, kwargs)
    else:
        raise ValueError("mem_backend must be 'tracemalloc' or 'memory_profiler'")
