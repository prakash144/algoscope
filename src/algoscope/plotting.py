from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go


def _reference_funcs():
    return {
        "1": lambda n: np.ones_like(n, dtype=float),
        "logn": lambda n: np.log2(np.maximum(n, 2)),
        "n": lambda n: n.astype(float),
        "nlogn": lambda n: n.astype(float) * np.log2(np.maximum(n, 2)),
    }


def _eval_custom_curve(expr: str):
    """
    Safely evaluate expressions like 'n**2', 'n**3'. Provide a minimal namespace.
    """
    def f(n: np.ndarray) -> np.ndarray:
        local_ns = {"n": n.astype(float), "np": np, "log": np.log, "log2": np.log2}
        return eval(expr, {"__builtins__": {}}, local_ns)  # noqa: S307 (controlled)
    return f


def build_reference_curves(
    ns: List[int],
    ref_specs: Tuple[str, ...],
    y_anchor: float,
    normalize_at: str = "max",
):
    """
    Returns dict: name -> np.ndarray of scaled values matching the anchor scale.
    """
    n_arr = np.array(ns, dtype=float)
    funcs = _reference_funcs()
    curves = {}
    for spec in ref_specs:
        if spec in funcs:
            raw = funcs[spec](n_arr)
        else:
            raw = _eval_custom_curve(spec)(n_arr)
        raw = np.maximum(raw, 1e-12)
        if normalize_at == "min":
            idx = 0
        else:
            idx = len(n_arr) - 1
        scale = y_anchor / raw[idx]
        curves[spec] = raw * scale
    return curves


def runtime_figure(
    ns: List[int],
    means: Dict[str, List[float]],
    lowers: Dict[str, List[float]],
    uppers: Dict[str, List[float]],
    reference_curves: Dict[str, List[float]],
    title: str,
) -> go.Figure:
    x = ns
    fig = go.Figure()
    # error bands
    for label, y_mean in means.items():
        y_lo = lowers[label]
        y_hi = uppers[label]
        fig.add_trace(go.Scatter(
            x=x, y=y_hi, line=dict(width=0), hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_lo, fill="tonexty", line=dict(width=0),
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, opacity=0.2
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=label
        ))

    # reference curves
    for rname, ry in reference_curves.items():
        fig.add_trace(go.Scatter(
            x=x, y=ry, mode="lines", name=f"ref: {rname}", line=dict(dash="dash")
        ))

    fig.update_layout(
        title=title + " — Runtime",
        xaxis_title="Input size (n)",
        yaxis_title="Time (seconds)",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="log")  # log Y to compare growth cleanly
    return fig


def memory_figure(
    ns: List[int],
    means: Dict[str, List[float]],
    lowers: Dict[str, List[float]],
    uppers: Dict[str, List[float]],
    title: str,
) -> go.Figure:
    x = ns
    fig = go.Figure()
    for label, y_mean in means.items():
        y_lo = lowers[label]
        y_hi = uppers[label]
        fig.add_trace(go.Scatter(
            x=x, y=y_hi, line=dict(width=0), hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_lo, fill="tonexty", line=dict(width=0),
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, opacity=0.2
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=label
        ))

    fig.update_layout(
        title=title + " — Peak Memory",
        xaxis_title="Input size (n)",
        yaxis_title="Bytes",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="linear")
    return fig
