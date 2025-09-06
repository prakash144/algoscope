# src/algoscope/utils.py
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Any, Optional
import traceback
import time
import multiprocessing as mp
import pickle
import sys

import numpy as np

@dataclass
class CIResult:
    mean: float
    std: float
    n: int
    lower: float
    upper: float
    method: str  # "t" or "bootstrap"


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 50: 2.009, 60: 2.000,
    }
    if df <= 30:
        return table[df]
    for bound in (40, 50, 60):
        if df <= bound:
            return table[bound]
    return 1.96


def t_confidence_interval(samples: Iterable[float], confidence: float = 0.95) -> CIResult:
    xs = [float(x) for x in samples]
    n = len(xs)
    mean = statistics.fmean(xs) if n > 0 else float("nan")
    if n <= 1:
        std = 0.0  # No standard deviation for 0 or 1 data points
        return CIResult(mean, std, n, mean, mean, "t")
    std = statistics.stdev(xs)
    if abs(confidence - 0.95) > 1e-9:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        half = z * std / math.sqrt(n)
    else:
        tcrit = _t_critical_95(n - 1)
        half = tcrit * std / math.sqrt(n)
    return CIResult(mean, std, n, mean - half, mean + half, "t")


def bootstrap_confidence_interval(
    samples: Iterable[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> CIResult:
    xs = np.asarray(list(samples), dtype=float)
    n = xs.size
    mean = float(np.mean(xs)) if n > 0 else float("nan")
    std = float(np.std(xs, ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return CIResult(mean, std, n, mean, mean, "bootstrap")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = xs[rng.integers(0, n, size=n)]
        boots[i] = float(np.mean(draw))
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boots, alpha))
    upper = float(np.quantile(boots, 1.0 - alpha))
    return CIResult(mean, std, n, lower, upper, "bootstrap")


def human_time(seconds: float) -> str:
    if seconds is None or not (isinstance(seconds, (int, float)) and math.isfinite(seconds)):
        return "—"
    if seconds < 1e-6:
        return f"{seconds*1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f} µs"
    if seconds < 1.0:
        return f"{seconds*1e3:.2f} ms"
    return f"{seconds:.3f} s"


def human_bytes(nbytes: float) -> str:
    try:
        s = float(nbytes)
    except Exception:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while s >= 1024.0 and idx < len(units) - 1:
        s /= 1024.0
        idx += 1
    return f"{s:.2f} {units[idx]}"


def rank(values: List[float]) -> List[int]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])
    ranks = [0] * len(values)
    r = 1
    for i, (orig_idx, _) in enumerate(indexed):
        ranks[orig_idx] = r
        r += 1
    return ranks


# ----------------------
# Input normalization
# ----------------------
def ensure_args_kwargs(result: Any) -> Tuple[tuple, dict]:
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


def is_picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


# ----------------------
# Module-level child worker (picklable)
# ----------------------
def _child_worker(func, args, kwargs, q):
    """
    Top-level worker function for child process.
    Puts a 'started' message immediately, then runs func, and finally puts a 'done' outcome.
    """
    try:
        q.put({"phase": "started"})
    except Exception:
        # queue might be broken in some rare envs; continue
        pass
    try:
        func(*args, **(kwargs or {}))
        try:
            q.put({"phase": "done", "success": True})
        except Exception:
            pass
    except Exception:
        try:
            q.put({"phase": "done", "success": False, "traceback": traceback.format_exc()})
        except Exception:
            pass


# ----------------------
# Subprocess monitor runner (uses module-level worker)
# ----------------------
def run_and_monitor_subprocess(
    func: Any,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: Optional[float] = None,
    poll_interval: float = 0.02,
) -> dict:
    """
    Run `func(*args, **(kwargs or {}))` in a subprocess. Monitor elapsed time and peak RSS (if psutil available).
    Returns dict:
      {
        "success": bool,
        "duration": float (seconds) or None,
        "peak_rss": int (bytes) or 0,
        "timed_out": bool,
        "exception": str or None,
        "traceback": str or None
      }
    """
    if kwargs is None:
        kwargs = {}

    q: mp.Queue = mp.Queue()

    # Create process using module-level worker
    p = mp.Process(target=_child_worker, args=(func, args, kwargs, q))

    t0 = None
    peak = 0
    timed_out = False
    outcome = {"success": False, "exception": "no result", "traceback": None}

    try:
        p.start()
        # parent can access p.pid immediately after start()
        t0 = time.perf_counter()

        # try to import psutil for RSS monitoring
        try:
            import psutil  # type: ignore
        except Exception:
            psutil = None

        # Wait for 'started' marker with a small timeout; not strictly necessary but useful
        try:
            msg = q.get(timeout=1.0)
            # if the child wrote started, continue; otherwise put msg back handling later
        except Exception:
            # no started marker; continue anyway
            pass

        if psutil and p.pid is not None:
            proc = psutil.Process(p.pid)
            while p.is_alive():
                try:
                    rss = proc.memory_info().rss
                    if rss > peak:
                        peak = rss
                except Exception:
                    # race: process may have ended between checks
                    pass
                elapsed = time.perf_counter() - t0
                if timeout is not None and elapsed >= timeout:
                    timed_out = True
                    break
                time.sleep(poll_interval)
        else:
            # no psutil: wait for process with timeout
            p.join(timeout)
            if p.is_alive():
                timed_out = True

        # handle timeout
        if timed_out and p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
            p.join(timeout=1.0)

        # final time
        t1 = time.perf_counter()
        duration = None if timed_out else (t1 - t0 if t0 is not None else None)

        # Attempt to drain the queue for final 'done' message(s)
        try:
            while True:
                msg = q.get_nowait()
                # prefer the last 'done' message if present
                if isinstance(msg, dict) and msg.get("phase") == "done":
                    outcome = msg
        except Exception:
            # nothing more in queue
            pass

        # If outcome was started-only and no 'done' message, check exitcode
        if outcome.get("phase") != "done":
            # if process exited with 0, assume success
            if p.exitcode == 0:
                outcome = {"success": True}
            elif p.exitcode is not None:
                outcome = {"success": False, "exception": f"process exit code {p.exitcode}"}
            else:
                # still unknown
                outcome = outcome

        result = {
            "success": bool(outcome.get("success")),
            "duration": duration,
            "peak_rss": int(peak or 0),
            "timed_out": bool(timed_out),
            "exception": outcome.get("exception"),
            "traceback": outcome.get("traceback"),
        }
        return result

    finally:
        # ensure process cleanup
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
            p.join(timeout=0.5)
