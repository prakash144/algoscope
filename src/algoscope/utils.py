from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List

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
    """
    Return approximate two-sided t critical value for 95% confidence.
    Uses a small lookup for common dfs, falls back to normal 1.96 above 60.
    """
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
    xs = list(samples)
    n = len(xs)
    mean = statistics.fmean(xs) if n > 0 else float("nan")
    std = statistics.pstdev(xs) if n <= 1 else statistics.stdev(xs)
    if n <= 1:
        return CIResult(mean, std, n, mean, mean, "t")
    if abs(confidence - 0.95) > 1e-9:
        # fallback: use normal quantile approximation for other confidences
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
    if seconds < 1e-6:
        return f"{seconds*1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f} µs"
    if seconds < 1.0:
        return f"{seconds*1e3:.2f} ms"
    return f"{seconds:.3f} s"


def human_bytes(nbytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(nbytes)
    idx = 0
    while s >= 1024.0 and idx < len(units) - 1:
        s /= 1024.0
        idx += 1
    return f"{s:.2f} {units[idx]}"


def rank(values: List[float]) -> List[int]:
    """
    Return 1-based ranks (smaller is better). Stable for equal values.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])
    ranks = [0] * len(values)
    r = 1
    for i, (_, _) in enumerate(indexed):
        ranks[indexed[i][0]] = r
        r += 1
    return ranks
