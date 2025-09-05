# src/algoscope/plotting.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _reference_funcs():
    return {
        "1": lambda n: np.ones_like(n, dtype=float),
        "logn": lambda n: np.log2(np.maximum(n, 2)),
        "n": lambda n: n.astype(float),
        "nlogn": lambda n: n.astype(float) * np.log2(np.maximum(n, 2)),
    }


def _eval_custom_curve(expr: str):
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
        scale = y_anchor / raw[idx] if raw[idx] != 0 else 1.0
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
    for label, y_mean in means.items():
        y_lo = lowers[label]
        y_hi = uppers[label]
        fig.add_trace(go.Scatter(
            x=x, y=y_hi, line=dict(width=0), hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_lo, fill="tonexty", line=dict(width=0),
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, opacity=0.15
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=label
        ))

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
        transition={'duration': 450, 'easing': 'cubic-in-out'},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.2, 'xanchor': 'left', 'x': 0}
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="log")
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
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, opacity=0.15
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
        transition={'duration': 450, 'easing': 'cubic-in-out'},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.2, 'xanchor': 'left', 'x': 0}
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="linear")
    return fig


def overview_figure(ns, time_means, mem_means, mean_ranks):
    labels = list(time_means.keys())

    # Normalize runtime and memory (lower is better) using min/max
    values_time = []
    values_mem = []
    for l in labels:
        vtime = time_means[l]
        vmem = mem_means[l]
        if max(vtime) > 0:
            values_time.append(min(vtime) / max(vtime))
        else:
            values_time.append(0)
        if max(vmem) > 0:
            values_mem.append(min(vmem) / max(vmem))
        else:
            values_mem.append(0)

    # Ranking (invert so higher is better)
    inv_rank_values = []
    for l in labels:
        r = mean_ranks.get(l, 1.0)
        inv_rank_values.append(1.0 / r if r > 0 else 0.0)

    # Pie values from inv_rank_values scaled
    pie_vals = [v for v in inv_rank_values]
    pie_labels = labels

    # Build a subplot canvas: bars on left, donut on right
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(x=labels, y=values_time, name="Runtime (normalized)"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=values_mem, name="Memory (normalized)"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=inv_rank_values, name="Ranking (1/mean rank)"), row=1, col=1)

    fig.add_trace(go.Pie(labels=pie_labels, values=pie_vals, hole=0.55, name="Ranking share", showlegend=False), row=1, col=2)

    fig.update_layout(
        barmode="group",
        title="Performance Overview",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        template="plotly_white",
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.15, 'xanchor': 'left', 'x': 0}
    )
    return fig
