# src/algoscope/report.py
from __future__ import annotations

import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from jinja2 import Environment, BaseLoader
from markupsafe import escape
import plotly.io as pio
import re
import json

from .utils import human_time, human_bytes


def load_template_text() -> str:
    """
    Load the Jinja2 template text from package resources. This tries the
    packaged templates location first, and falls back to a sibling templates
    directory for development setups.
    """
    try:
        tmpl = pkg_resources.files("algoscope.templates").joinpath("report.html.j2")
        return tmpl.read_text(encoding="utf-8")
    except Exception:
        # fallback: older layout where templates/ is adjacent to package
        return pkg_resources.files("algoscope").joinpath("../templates/report.html.j2").read_text(encoding="utf-8")


def fig_to_div(fig, include_js: bool = False) -> str:
    """
    Convert a Plotly figure to an HTML div. When include_js=True, embed
    Plotly JS inline (used once per report).
    """
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        default_width="100%",
        default_height="750px",  # Increased height for better visibility
    )


def simple_markdown_to_html(text: str) -> str:
    """
    Convert a small subset of markdown-like syntax -> safe HTML.

    - Convert **bold** -> <strong>, `code` -> <code>
    - Convert lines starting with '- ' into <ul><li>...</li></ul>
    - Convert "Label: value" into <div><strong>Label:</strong> value</div>
    - Heading-like lines ending with ':' -> <h4>
    - Ensure we escape arbitrary text but preserve our intentionally created tags.
    """
    if not text:
        return ""

    # Normalize and trim
    text = text.strip()

    # 1) Perform markdown-like transformations first (on raw text)
    transformed = text
    transformed = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", transformed)
    transformed = re.sub(r"`(.+?)`", r"<code>\1</code>", transformed)

    # 2) Protect our safe tags with placeholders so they won't be escaped:
    placeholders = {
        "<strong>": "§§STRONG_OPEN§§",
        "</strong>": "§§STRONG_CLOSE§§",
        "<code>": "§§CODE_OPEN§§",
        "</code>": "§§CODE_CLOSE§§",
        "<ul>": "§§UL_OPEN§§",
        "</ul>": "§§UL_CLOSE§§",
        "<li>": "§§LI_OPEN§§",
        "</li>": "§§LI_CLOSE§§",
        "<h4>": "§§H4_OPEN§§",
        "</h4>": "§§H4_CLOSE§§",
    }
    for k, v in placeholders.items():
        transformed = transformed.replace(k, v)

    # 3) Escape the rest safely
    escaped = escape(transformed)

    # 4) Restore placeholders back to real tags
    for k, v in placeholders.items():
        escaped = escaped.replace(v, k)

    # 5) Now build HTML structure line-by-line (lists, kv pairs, headings)
    lines = escaped.splitlines()
    out_lines = []
    in_list = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            continue

        # Bullet -> list
        if line.startswith("- "):
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            out_lines.append("<li>" + line[2:].strip() + "</li>")
            continue

        if in_list:
            out_lines.append("</ul>")
            in_list = False

        # Heading-like (ends with colon)
        if re.match(r"^[A-Za-z0-9 _`/()\-]+:$", line):
            htext = line[:-1].strip()
            out_lines.append(f"<h4 style=\"margin:6px 0;\">{htext}</h4>")
            continue

        # Label: value
        m = re.match(r"^([^:]{1,60}):\s*(.+)$", line)
        if m:
            label = m.group(1).strip()
            val = m.group(2).strip()
            out_lines.append(f"<div><strong>{label}:</strong> {val}</div>")
            continue

        # Plain paragraph
        out_lines.append(f"<p>{line}</p>")

    if in_list:
        out_lines.append("</ul>")

    result = "\n".join(out_lines)

    # Defensive wrap: if there exist <li> elements but no surrounding <ul>, wrap them
    if "<li" in result and "<ul" not in result:
        result = re.sub(r"(<li>.*?</li>(?:\s*<li>.*?</li>)*)", r"<ul>\1</ul>", result, flags=re.S)
        result = re.sub(r"</ul>\s*<ul>", "\n", result)

    # --- Targeted unescape for intentionally generated tags ---
    replacements = {
        "&lt;ul&gt;": "<ul>",
        "&lt;/ul&gt;": "</ul>",
        "&lt;li&gt;": "<li>",
        "&lt;/li&gt;": "</li>",
        "&lt;strong&gt;": "<strong>",
        "&lt;/strong&gt;": "</strong>",
        "&lt;code&gt;": "<code>",
        "&lt;/code&gt;": "</code>",
        "&lt;h4&gt;": "<h4>",
        "&lt;/h4&gt;": "</h4>",
    }
    for k, v in replacements.items():
        if k in result:
            result = result.replace(k, v)
    # --- end unescape ---

    return result


def _concise_manual_html(explanation: str) -> str:
    """
    Convert a verbose explanation into a compact HTML summary:
    - Extract Time Complexity and Space Complexity
    - Extract patterns and confidence information
    - Return a comprehensive HTML string with Google SWE styling.
    """
    if not explanation:
        return ""

    time_o = None
    space_o = None
    patterns = []
    confidence = "Medium"

    # Extract time and space complexity
    t_match = re.search(r"\*\*Time Complexity:\*\*\s*([^\n]+)", explanation)
    s_match = re.search(r"\*\*Space Complexity:\*\*\s*([^\n]+)", explanation)
    
    if t_match:
        time_o = escape(t_match.group(1).strip())
    if s_match:
        space_o = escape(s_match.group(1).strip())

    # Extract patterns
    patterns_match = re.search(r"\*\*Algorithm Patterns Detected:\*\*\s*\n((?:- .+\n?)+)", explanation)
    if patterns_match:
        patterns_text = patterns_match.group(1)
        patterns = [line.strip("- ").strip() for line in patterns_text.split("\n") if line.strip()]

    # Extract confidence
    conf_match = re.search(r"\*\*Confidence Level:\*\*\s*([^\n]+)", explanation)
    if conf_match:
        confidence = conf_match.group(1).strip()

    # Build HTML
    parts = []
    if time_o:
        parts.append(f"<strong>Time</strong>: {time_o}")
    if space_o:
        parts.append(f"<strong>Space</strong>: {space_o}")

    if parts:
        summary = " &bull; ".join(parts)
    else:
        first_line = explanation.strip().splitlines()[0]
        summary = escape(first_line)

    html = f"<p style='margin:6px 0;'>{summary}</p>"
    
    # Add patterns if available
    if patterns:
        patterns_html = " &bull; ".join([f"<span class='badge' style='font-size:10px; padding:2px 6px;'>{p}</span>" for p in patterns[:3]])
        html += f"<p style='margin:4px 0; font-size:12px;'>{patterns_html}</p>"
    
    # Add confidence indicator
    conf_color = "#10b981" if "High" in confidence else "#f59e0b" if "Medium" in confidence else "#ef4444"
    html += f"<p style='margin:4px 0; font-size:11px; color:{conf_color};'><strong>Confidence:</strong> {confidence}</p>"

    return html


@dataclass
class ReportSections:
    manual_complexities: Dict[str, str]
    interview_summaries: Dict[str, str]
    beginner_summaries: Dict[str, str]
    methods_text: str
    # optional dynamic_guesses may be added at runtime
    dynamic_guesses: Optional[Dict[str, str]] = None
    # Google SWE interview specific content
    interview_tips: Optional[Dict[str, List[str]]] = None
    optimization_suggestions: Optional[Dict[str, List[str]]] = None
    patterns_detected: Optional[Dict[str, List[str]]] = None
    confidence_scores: Optional[Dict[str, float]] = None


def build_report_html(
    title: str,
    notes: Optional[str],
    ns: List[int],
    runtime_table: List[Dict[str, Any]],
    memory_table: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    runtime_fig,
    memory_fig,
    sections: ReportSections,
    overview_fig=None,
    html_path: Optional[str] = None,
    func_stats: Optional[Dict[str, Any]] = None,
    heatmap_divs: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build the final HTML by rendering the Jinja2 template. This prepares
    small HTML fragments for manual complexities and methods text to keep
    the UI concise.
    """
    env = Environment(loader=BaseLoader())
    env.filters["human_time"] = human_time
    env.filters["human_bytes"] = human_bytes

    tpl = env.from_string(load_template_text())

    # Convert manual complexities into concise HTML fragments
    manual_html_concise = {}
    manual_html_full = {}
    # dynamic_guesses may be None or dict; guard accordingly
    dyn_map = getattr(sections, "dynamic_guesses", None) or {}

    for label, explanation in sections.manual_complexities.items():
        base_html = _concise_manual_html(explanation)
        heuristic_html = simple_markdown_to_html(explanation)
        dynamic_html = escape(str(dyn_map.get(label, "")))
        manual_html_concise[label] = base_html
        manual_html_full[label] = {"heuristic": heuristic_html, "dynamic": dynamic_html}

    # Methods text: render as HTML using our simple converter
    methods_html = simple_markdown_to_html(sections.methods_text)

    # Convert figures to divs (do not inline plotly.js; template includes CDN)
    runtime_div = pio.to_html(runtime_fig, include_plotlyjs=False, full_html=False) if runtime_fig is not None else ""
    memory_div = pio.to_html(memory_fig, include_plotlyjs=False, full_html=False) if memory_fig is not None else ""
    overview_div = pio.to_html(overview_fig, include_plotlyjs=False, full_html=False) if overview_fig is not None else ""

    # Ensure heatmap_divs is a dict
    heatmap_divs = heatmap_divs or {}

    html = tpl.render(
        title=title,
        notes=notes,
        ns=ns,
        runtime_table=runtime_table,
        memory_table=memory_table,
        comparison_rows=comparison_rows,
        runtime_div=runtime_div,
        memory_div=memory_div,
        overview_div=overview_div,
        manual_complexities_concise=manual_html_concise,
        manual_complexities_full=manual_html_full,
        interview_summaries=sections.interview_summaries,
        beginner_summaries=sections.beginner_summaries,
        methods_text=methods_html,
        html_path=html_path,
        runtime_table_json=json.dumps(runtime_table),
        memory_table_json=json.dumps(memory_table),
        comparison_rows_json=json.dumps(comparison_rows),
        heatmap_divs=heatmap_divs,
        # *** Fix: supply dynamic_guesses explicitly for template usage ***
        dynamic_guesses=dyn_map,
        # optionally expose func_stats for diagnostics in template
        func_stats=func_stats or {},
    )
    return html
