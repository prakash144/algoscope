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
    
    # Handle case where y_anchor is NaN or invalid
    if not np.isfinite(y_anchor) or y_anchor <= 0:
        y_anchor = 1.0  # Default fallback
    
    for spec in ref_specs:
        if spec in funcs:
            raw = funcs[spec](n_arr)
        else:
            raw = _eval_custom_curve(spec)(n_arr)
        raw = np.maximum(raw, 1e-12)
        
        # Check if raw values are valid
        if not np.any(np.isfinite(raw)):
            raw = np.ones_like(n_arr) * 1e-12
        
        idx = 0 if normalize_at == "min" else len(n_arr) - 1
        if idx < len(raw) and np.isfinite(raw[idx]) and raw[idx] != 0:
            scale = y_anchor / raw[idx]
        else:
            scale = 1.0
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
    
    # Google SWE Interview Color Palette
    colors = [
        '#4285f4',  # Google Blue
        '#ea4335',  # Google Red
        '#34a853',  # Google Green
        '#fbbc04',  # Google Yellow
        '#9c27b0',  # Purple
        '#00bcd4',  # Cyan
        '#ff9800',  # Orange
        '#4caf50',  # Light Green
        '#e91e63',  # Pink
        '#795548',  # Brown
    ]
    
    # error bands and traces
    for i, (label, y_mean) in enumerate(means.items()):
        y_lo = lowers[label]
        y_hi = uppers[label]
        color = colors[i % len(colors)]
        
        # Determine if this is brute force or optimal for Google SWE interviews
        is_brute_force = any(keyword in label.lower() for keyword in ['brute', 'naive', 'force', 'recursive'])
        is_optimal = any(keyword in label.lower() for keyword in ['optimal', 'dp', 'memoized', 'kadane', 'binary'])
        
        if is_brute_force:
            line_style = dict(width=4, dash="dash", color=color)
            marker_style = dict(size=10, color=color, line=dict(width=2, color='white'), symbol='diamond')
            legend_name = f"🔴 {label} (Brute Force)"
        elif is_optimal:
            line_style = dict(width=4, dash="solid", color=color)
            marker_style = dict(size=10, color=color, line=dict(width=2, color='white'), symbol='star')
            legend_name = f"🟢 {label} (Optimal)"
        else:
            line_style = dict(width=3, dash="solid", color=color)
            marker_style = dict(size=8, color=color, line=dict(width=2, color='white'))
            legend_name = label
        
        # upper band
        fig.add_trace(go.Scatter(
            x=x, y=y_hi, line=dict(width=0), hoverinfo="skip", showlegend=False,
            fillcolor=color, opacity=0.1
        ))
        # lower band with fill
        fig.add_trace(go.Scatter(
            x=x, y=y_lo, fill="tonexty", line=dict(width=0),
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, 
            fillcolor=color, opacity=0.2
        ))
        # mean line with enhanced styling
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=legend_name, 
            marker=marker_style,
            line=line_style,
            hovertemplate=f"<b>{label}</b><br>" +
                         "Input size: %{x}<br>" +
                         "Time: %{y:.6f}s<br>" +
                         f"Type: {'Brute Force' if is_brute_force else 'Optimal' if is_optimal else 'Standard'}<br>" +
                         "<extra></extra>"
        ))

    # reference curves with enhanced styling
    ref_colors = ['#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0']
    for i, (rname, ry) in enumerate(reference_curves.items()):
        fig.add_trace(go.Scatter(
            x=x, y=ry, mode="lines", name=f"O({rname})", 
            line=dict(dash="dot", width=2, color=ref_colors[i % len(ref_colors)]),
            opacity=0.7,
            hovertemplate=f"<b>O({rname})</b><br>" +
                         "Input size: %{x}<br>" +
                         "Reference: %{y:.6f}<br>" +
                         "<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=title + " — Runtime Analysis",
            font=dict(size=28, color='#1e293b', family="Inter, sans-serif"),
            x=0.5,
            pad=dict(t=30, b=30)
        ),
        xaxis_title="Input Size (n)",
        yaxis_title="Time (seconds)",
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.05, 
            xanchor="left", 
            x=0.01,
            font=dict(size=14, family="Inter, sans-serif"),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(66, 133, 244, 0.3)',
            borderwidth=2,
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        margin=dict(l=100, r=60, t=120, b=100),
        height=750,  # Increased height
        width=1200,  # Increased width
        font=dict(family="Inter, sans-serif", size=13, color='#374151'),
        # Enhanced animations and transitions
        transition=dict(duration=500, easing="cubic-in-out"),
        # Add annotations for better UX
        annotations=[
            dict(
                text="💡 Hover over data points for detailed information",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=11, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(66, 133, 244, 0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    fig.update_xaxes(
        type="linear",
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        title_font=dict(size=16, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        mirror=True,
        # Add minor grid lines for better readability
        minor=dict(
            gridcolor='rgba(66, 133, 244, 0.05)',
            gridwidth=0.5
        )
    )
    fig.update_yaxes(
        type="log",
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        title_font=dict(size=16, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        mirror=True,
        # Enhanced log scale formatting
        tickformat=".1e",
        # Add minor grid lines for better readability
        minor=dict(
            gridcolor='rgba(66, 133, 244, 0.05)',
            gridwidth=0.5
        )
    )
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
    
    # Google SWE Interview Color Palette
    colors = [
        '#4285f4',  # Google Blue
        '#ea4335',  # Google Red
        '#34a853',  # Google Green
        '#fbbc04',  # Google Yellow
        '#9c27b0',  # Purple
        '#00bcd4',  # Cyan
        '#ff9800',  # Orange
        '#4caf50',  # Light Green
        '#e91e63',  # Pink
        '#795548',  # Brown
    ]
    
    for i, (label, y_mean) in enumerate(means.items()):
        y_lo = lowers[label]
        y_hi = uppers[label]
        color = colors[i % len(colors)]
        
        # Determine if this is brute force or optimal for Google SWE interviews
        is_brute_force = any(keyword in label.lower() for keyword in ['brute', 'naive', 'force', 'recursive'])
        is_optimal = any(keyword in label.lower() for keyword in ['optimal', 'dp', 'memoized', 'kadane', 'binary'])
        
        if is_brute_force:
            line_style = dict(width=4, dash="dash", color=color)
            marker_style = dict(size=10, color=color, line=dict(width=2, color='white'), symbol='diamond')
            legend_name = f"🔴 {label} (Brute Force)"
        elif is_optimal:
            line_style = dict(width=4, dash="solid", color=color)
            marker_style = dict(size=10, color=color, line=dict(width=2, color='white'), symbol='star')
            legend_name = f"🟢 {label} (Optimal)"
        else:
            line_style = dict(width=3, dash="solid", color=color)
            marker_style = dict(size=8, color=color, line=dict(width=2, color='white'))
            legend_name = label
        
        fig.add_trace(go.Scatter(
            x=x, y=y_hi, line=dict(width=0), hoverinfo="skip", showlegend=False,
            fillcolor=color, opacity=0.1
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_lo, fill="tonexty", line=dict(width=0),
            name=f"{label} 95% CI", hoverinfo="skip", showlegend=False, 
            fillcolor=color, opacity=0.2
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers", name=legend_name, 
            marker=marker_style,
            line=line_style,
            hovertemplate=f"<b>{label}</b><br>" +
                         "Input size: %{x}<br>" +
                         "Memory: %{y:.0f} bytes<br>" +
                         f"Type: {'Brute Force' if is_brute_force else 'Optimal' if is_optimal else 'Standard'}<br>" +
                         "<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=title + " — Memory Analysis",
            font=dict(size=28, color='#1e293b', family="Inter, sans-serif"),
            x=0.5,
            pad=dict(t=30, b=30)
        ),
        xaxis_title="Input Size (n)",
        yaxis_title="Peak Memory (bytes)",
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.05, 
            xanchor="left", 
            x=0.01,
            font=dict(size=14, family="Inter, sans-serif"),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(66, 133, 244, 0.3)',
            borderwidth=2,
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        margin=dict(l=100, r=60, t=120, b=100),
        height=750,  # Increased height
        width=1200,  # Increased width
        font=dict(family="Inter, sans-serif", size=13, color='#374151'),
        # Enhanced animations and transitions
        transition=dict(duration=500, easing="cubic-in-out"),
        # Add annotations for better UX
        annotations=[
            dict(
                text="💾 Memory usage patterns help identify space complexity",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=11, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(66, 133, 244, 0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    fig.update_xaxes(
        type="linear",
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        title_font=dict(size=16, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        mirror=True,
        # Add minor grid lines for better readability
        minor=dict(
            gridcolor='rgba(66, 133, 244, 0.05)',
            gridwidth=0.5
        )
    )
    fig.update_yaxes(
        type="linear",
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        title_font=dict(size=16, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        mirror=True,
        # Enhanced formatting for memory values
        tickformat=".2s",
        # Add minor grid lines for better readability
        minor=dict(
            gridcolor='rgba(66, 133, 244, 0.05)',
            gridwidth=0.5
        )
    )
    return fig


def overview_figure(ns, time_means, mem_means, mean_ranks):
    """
    Build a grouped bar chart showing normalized runtime, memory, and ranking for each function.
    Enhanced with Google Material Design styling and animations.
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

    # Google SWE Interview Color Palette
    colors = [
        '#4285f4',  # Google Blue
        '#ea4335',  # Google Red
        '#34a853',  # Google Green
        '#fbbc04',  # Google Yellow
        '#9c27b0',  # Purple
        '#00bcd4',  # Cyan
        '#ff9800',  # Orange
        '#4caf50',  # Light Green
    ]

    fig = go.Figure()
    
    # Add traces with enhanced styling
    fig.add_trace(go.Bar(
        x=labels, 
        y=values_time, 
        name="⚡ Runtime (normalized)",
        marker=dict(
            color=colors[0],
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        hovertemplate="<b>%{x}</b><br>Runtime: %{y:.3f}<br><extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=labels, 
        y=values_mem, 
        name="💾 Memory (normalized)",
        marker=dict(
            color=colors[1],
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        hovertemplate="<b>%{x}</b><br>Memory: %{y:.3f}<br><extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=labels, 
        y=values_rank, 
        name="🏆 Ranking (1/mean rank)",
        marker=dict(
            color=colors[2],
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        hovertemplate="<b>%{x}</b><br>Ranking: %{y:.3f}<br><extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="📊 Performance Overview",
            font=dict(size=20, color='#1e293b', family="Inter, sans-serif"),
            x=0.5
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=500,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color='#374151'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Inter, sans-serif"),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(66, 133, 244, 0.2)',
            borderwidth=2
        ),
        # Enhanced animations
        transition=dict(duration=500, easing="cubic-in-out"),
        # Add annotations
        annotations=[
            dict(
                text="📈 Lower values are better for runtime and memory",
                xref="paper", yref="paper",
                x=0.5, y=0.95,
                showarrow=False,
                font=dict(size=11, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(66, 133, 244, 0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    # Enhanced axes
    fig.update_xaxes(
        title_font=dict(size=14, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        showgrid=True,
        gridwidth=1
    )
    
    fig.update_yaxes(
        title="Normalized Performance",
        title_font=dict(size=14, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        showgrid=True,
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(66, 133, 244, 0.2)',
        zerolinewidth=2
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
        title=dict(
            text=title,
            font=dict(size=20, color='#1e293b', family="Inter, sans-serif"),
            x=0.5
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=580,
        margin=dict(l=80, r=40, t=80, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color='#374151'),
        # Enhanced animations
        transition=dict(duration=500, easing="cubic-in-out"),
        # Add annotations
        annotations=[
            dict(
                text="🔥 Heatmap shows performance across different parameter combinations",
                xref="paper", yref="paper",
                x=0.5, y=0.95,
                showarrow=False,
                font=dict(size=11, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(66, 133, 244, 0.2)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    # Enhanced axes
    fig.update_xaxes(
        title_font=dict(size=14, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        showgrid=True,
        gridwidth=1
    )
    
    fig.update_yaxes(
        title_font=dict(size=14, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        showgrid=True,
        gridwidth=1
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


def performance_comparison_figure(
    ns: List[int],
    means: Dict[str, List[float]],
    title: str = "Performance Comparison",
    show_animations: bool = True
) -> go.Figure:
    """
    Create an animated performance comparison chart with Google SWE interview styling.
    Shows the performance difference between algorithms over time.
    """
    # Google SWE Interview Color Palette
    colors = [
        '#4285f4',  # Google Blue
        '#ea4335',  # Google Red
        '#34a853',  # Google Green
        '#fbbc04',  # Google Yellow
        '#9c27b0',  # Purple
        '#00bcd4',  # Cyan
        '#ff9800',  # Orange
        '#4caf50',  # Light Green
    ]
    
    fig = go.Figure()
    
    # Add traces for each algorithm
    for i, (label, y_mean) in enumerate(means.items()):
        is_brute_force = any(keyword in label.lower() for keyword in ['brute', 'naive', 'force', 'recursive'])
        is_optimal = any(keyword in label.lower() for keyword in ['optimal', 'dp', 'memoized', 'kadane', 'binary'])
        
        if is_brute_force:
            marker_style = dict(size=12, color=colors[i % len(colors)], symbol='diamond', line=dict(width=2, color='white'))
            line_style = dict(width=4, dash="dash", color=colors[i % len(colors)])
            legend_name = f"🔴 {label} (Brute Force)"
        elif is_optimal:
            marker_style = dict(size=12, color=colors[i % len(colors)], symbol='star', line=dict(width=2, color='white'))
            line_style = dict(width=4, dash="solid", color=colors[i % len(colors)])
            legend_name = f"🟢 {label} (Optimal)"
        else:
            marker_style = dict(size=10, color=colors[i % len(colors)], symbol='circle', line=dict(width=2, color='white'))
            line_style = dict(width=3, dash="solid", color=colors[i % len(colors)])
            legend_name = label
        
        fig.add_trace(go.Scatter(
            x=ns,
            y=y_mean,
            mode="lines+markers",
            name=legend_name,
            marker=marker_style,
            line=line_style,
            hovertemplate=f"<b>{label}</b><br>" +
                         "Input size: %{x}<br>" +
                         "Time: %{y:.6f}s<br>" +
                         f"Type: {'Brute Force' if is_brute_force else 'Optimal' if is_optimal else 'Standard'}<br>" +
                         "<extra></extra>",
            # Add animation frames if enabled
            visible=True if not show_animations else "legendonly"
        ))
    
    # Add performance difference area
    if len(means) == 2:
        labels = list(means.keys())
        y1 = means[labels[0]]
        y2 = means[labels[1]]
        diff = [abs(y2[i] - y1[i]) for i in range(len(ns))]
        
        fig.add_trace(go.Scatter(
            x=ns,
            y=diff,
            mode="lines",
            name="📊 Performance Gap",
            line=dict(width=2, dash="dot", color="#ff6b6b"),
            fill="tonexty",
            fillcolor="rgba(255, 107, 107, 0.2)",
            hovertemplate="<b>Performance Gap</b><br>" +
                         "Input size: %{x}<br>" +
                         "Difference: %{y:.6f}s<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(
            text=f"🚀 {title}",
            font=dict(size=28, color='#1e293b', family="Inter, sans-serif"),
            x=0.5,
            pad=dict(t=30, b=30)
        ),
        xaxis_title="Input Size (n)",
        yaxis_title="Time (seconds)",
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0.01,
            font=dict(size=14, family="Inter, sans-serif"),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(66, 133, 244, 0.3)',
            borderwidth=2,
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        margin=dict(l=100, r=60, t=120, b=100),
        height=750,  # Increased height
        width=1200,  # Increased width
        font=dict(family="Inter, sans-serif", size=13, color='#374151'),
        # Enhanced animations
        transition=dict(duration=800, easing="cubic-in-out"),
        # Add annotations
        annotations=[
            dict(
                text="🎯 Compare algorithm performance across different input sizes",
                xref="paper", yref="paper",
                x=0.5, y=0.97,
                showarrow=False,
                font=dict(size=12, color="#64748b"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(66, 133, 244, 0.3)",
                borderwidth=1,
                borderpad=6
            )
        ]
    )
    
    # Enhanced axes
    fig.update_xaxes(
        type="linear",
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        title_font=dict(size=16, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        mirror=True
    )
    
    fig.update_yaxes(
        type="log",
        gridcolor='rgba(66, 133, 244, 0.1)',
        linecolor='rgba(66, 133, 244, 0.3)',
        title_font=dict(size=16, color='#1e293b', family="Inter, sans-serif"),
        tickfont=dict(size=12, color='#64748b', family="Inter, sans-serif"),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=2,
        mirror=True,
        tickformat=".1e"
    )
    
    return fig
