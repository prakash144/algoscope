# src/algoscope/plotting.py
from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Optional

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
    Safely evaluate simple expressions like 'n**2', 'n**3'. Provide a minimal namespace.
    """
    def f(n: np.ndarray) -> np.ndarray:
        local_ns = {"n": n.astype(float), "np": np, "log": np.log, "log2": np.log2}
        return eval(expr, {"__builtins__": {}}, local_ns)  # controlled eval
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
        idx = 0 if normalize_at == "min" else len(n_arr) - 1
        scale = (y_anchor / raw[idx]) if raw[idx] != 0 else 1.0
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
    # error bands and traces
    for label, y_mean in means.items():
        y_lo = lowers[label]
        y_hi = uppers[label]
        # upper band
        fig.add_trace(go.Scatter(
            x=x, y=y_hi, line=dict(width=0), hoverinfo="skip", showlegend=False
        ))
        # lower band with fill
        fig.add_trace(go.Scatter(
            x=x, y=y_lo, fill="tonexty", line=dict(width=0),
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, opacity=0.15
        ))
        # mean line
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=label, marker=dict(size=6)
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        margin=dict(l=60, r=20, t=60, b=60),
        height=520
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
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, opacity=0.12
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=label, marker=dict(size=6)
        ))

    fig.update_layout(
        title=title + " — Peak Memory",
        xaxis_title="Input size (n)",
        yaxis_title="Bytes",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        margin=dict(l=60, r=20, t=60, b=60),
        height=520
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="linear")
    return fig


def overview_figure(ns, time_means, mem_means, mean_ranks):
    """
    Build a grouped bar chart showing normalized runtime, memory, and ranking for each function.
    """
    labels = list(time_means.keys())

    # Normalize runtime and memory (lower is better)
    values_time = []
    values_mem = []
    for l in labels:
        vtime = time_means[l]
        vmem = mem_means[l]
        values_time.append(min(vtime) / max(vtime) if max(vtime) > 0 else 0)
        values_mem.append(min(vmem) / max(vmem) if max(vmem) > 0 else 0)

    # Ranking (invert so higher is better)
    values_rank = [1.0 / mean_ranks[l] if mean_ranks[l] > 0 else 0 for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values_time, name="Runtime (normalized)"))
    fig.add_trace(go.Bar(x=labels, y=values_mem, name="Memory (normalized)"))
    fig.add_trace(go.Bar(x=labels, y=values_rank, name="Ranking (1/mean rank)"))

    fig.update_layout(
        barmode="group",
        title="Performance Overview",
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
        template="plotly_white"
    )
    return fig


def heatmap_figure(
    x_vals: Sequence[int],
    y_vals: Sequence[int],
    z_matrix: np.ndarray,
    title: str,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Time (s)",
    log_color: bool = False,
    show_values: bool = False,
) -> go.Figure:
    """
    Build a Plotly heatmap for a grid sweep.

    - x_vals: values along x axis (e.g. n)
    - y_vals: values along y axis (e.g. W)
    - z_matrix: 2D numpy array with shape (len(y_vals), len(x_vals)) representing measured means.
      Note: rows correspond to y (vertical axis), columns correspond to x (horizontal axis).
    """
    z = np.array(z_matrix, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_matrix must be 2D (len(y) x len(x))")

    # If log_color requested but zeros/nans present, set them to nan for color mapping
    z_for_color = z.copy()
    if log_color:
        z_for_color = np.where((z_for_color <= 0) | (~np.isfinite(z_for_color)), np.nan, z_for_color)
        colorscale = "Viridis"
    else:
        colorscale = "Viridis"

    heat = go.Heatmap(
        x=list(x_vals),
        y=list(y_vals),
        z=z_for_color,
        colorscale=colorscale,
        colorbar=dict(title=z_label),
        hovertemplate="x=%{x}<br>y=%{y}<br>" + z_label + "=%{z:.6f}<extra></extra>",
    )

    fig = go.Figure(data=[heat])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=520,
        margin=dict(l=80, r=20, t=60, b=80),
    )

    if show_values:
        annotations = []
        for i_y, yv in enumerate(y_vals):
            for i_x, xv in enumerate(x_vals):
                val = z[i_y, i_x]
                text = "—" if (val is None or (isinstance(val, float) and np.isnan(val))) else f"{val:.3g}"
                annotations.append(
                    dict(x=xv, y=yv, text=text, showarrow=False, font=dict(color="black", size=10))
                )
        fig.update_layout(annotations=annotations)

    return fig
