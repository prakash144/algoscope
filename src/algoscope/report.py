# src/algoscope/report.py
from __future__ import annotations

import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

from jinja2 import Environment, BaseLoader
from markupsafe import escape
import plotly.io as pio
import re

from .utils import human_time, human_bytes


def load_template_text() -> str:
    try:
        tmpl = pkg_resources.files("algoscope.templates").joinpath("report.html.j2")
        return tmpl.read_text(encoding="utf-8")
    except Exception:
        return pkg_resources.files("algoscope").joinpath("../templates/report.html.j2").read_text(encoding="utf-8")


def fig_to_div(fig, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=False,  # We'll include a single Plotly script in the template head (CDN)
        full_html=False,
        default_width="100%",
        default_height="620px",
    )


def simple_markdown_to_html(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    transformed = text
    transformed = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", transformed)
    transformed = re.sub(r"`(.+?)`", r"<code>\1</code>", transformed)

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

    escaped = escape(transformed)

    for k, v in placeholders.items():
        escaped = escaped.replace(v, k)

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

        if line.startswith("- "):
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            out_lines.append("<li>" + line[2:].strip() + "</li>")
            continue

        if in_list:
            out_lines.append("</ul>")
            in_list = False

        if re.match(r"^[A-Za-z0-9 _`/()\-]+:$", line):
            htext = line[:-1].strip()
            out_lines.append(f"<h4 style=\"margin:6px 0;\">{htext}</h4>")
            continue

        m = re.match(r"^([^:]{1,60}):\s*(.+)$", line)
        if m:
            label = m.group(1).strip()
            val = m.group(2).strip()
            out_lines.append(f"<div><strong>{label}:</strong> {val}</div>")
            continue

        out_lines.append(f"<p>{line}</p>")

    if in_list:
        out_lines.append("</ul>")

    result = "\n".join(out_lines)

    if "<li" in result and "<ul" not in result:
        result = re.sub(r"(<li>.*?</li>(?:\s*<li>.*?</li>)*)", r"<ul>\1</ul>", result, flags=re.S)
        result = re.sub(r"</ul>\s*<ul>", "\n", result)

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

    return result


def _concise_manual_html(explanation: str) -> str:
    if not explanation:
        return ""

    time_o = None
    space_o = None

    t_match = re.search(r"\*\*Estimated Time Complexity:\*\*\s*([^\s\*\n]+)", explanation)
    s_match = re.search(r"\*\*Estimated Space Complexity:\*\*\s*([^\s\*\n]+)", explanation)

    if t_match:
        time_o = escape(t_match.group(1).strip())
    if s_match:
        space_o = escape(s_match.group(1).strip())

    why = ""
    why_m = re.search(r"Why:\s*(.+?)(?:\n\n|$)", explanation, flags=re.S)
    if why_m:
        why_raw = why_m.group(1).strip()
        why = escape(why_raw)
        if len(why) > 240:
            why = why[:237].rstrip() + "..."

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
    if why:
        html += f"<p class='muted' style='margin:6px 0 0 0; font-size:13px;'>Why: {why}</p>"

    return html


@dataclass
class ReportSections:
    manual_complexities: Dict[str, str]
    interview_summaries: Dict[str, str]
    beginner_summaries: Dict[str, str]
    methods_text: str
    dynamic_guesses: Optional[Dict[str, str]] = None


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
) -> str:
    env = Environment(loader=BaseLoader())
    env.filters["human_time"] = human_time
    env.filters["human_bytes"] = human_bytes

    tpl = env.from_string(load_template_text())

    manual_html_concise = {}
    manual_html_full = {}
    dyn_map = getattr(sections, "dynamic_guesses", {}) or {}
    for label, explanation in sections.manual_complexities.items():
        conc = _concise_manual_html(explanation)
        manual_html_concise[label] = conc
        heuristic_html = simple_markdown_to_html(explanation)
        dynamic_html = escape(str(dyn_map.get(label, "")))
        manual_html_full[label] = {"heuristic": heuristic_html, "dynamic": dynamic_html}

    methods_html = simple_markdown_to_html(sections.methods_text)

    # generate divs without embedding Plotly JS; template will include CDN
    runtime_div = fig_to_div(runtime_fig, include_js=False)
    memory_div = fig_to_div(memory_fig, include_js=False)
    overview_div = fig_to_div(overview_fig, include_js=False) if overview_fig is not None else ""

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
        # <-- expose dynamic_guesses to template (always a dict)
        dynamic_guesses=dyn_map,
    )
    return html
