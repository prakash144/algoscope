from __future__ import annotations

import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader
import plotly.io as pio

from .utils import human_time, human_bytes


def load_template_text() -> str:
    # template file located at templates/report.html.j2 (one level up)
    with pkg_resources.as_file(pkg_resources.files("algoscope.templates").joinpath("report.html.j2")) as p:
        return p.read_text(encoding="utf-8")


def fig_to_div(fig, include_js=False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_js else False,
        full_html=False,
        default_width="100%",
        default_height="580px",
    )


@dataclass
class ReportSections:
    manual_complexities: Dict[str, str]
    interview_summaries: Dict[str, str]
    beginner_summaries: Dict[str, str]
    methods_text: str


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
) -> str:
    env = Environment(loader=BaseLoader())
    env.filters["human_time"] = human_time
    env.filters["human_bytes"] = human_bytes

    tpl = env.from_string(load_template_text())

    # Put Plotly JS once (inline) using the runtime figure
    runtime_div = fig_to_div(runtime_fig, include_js=True)
    memory_div = fig_to_div(memory_fig, include_js=False)

    html = tpl.render(
        title=title,
        notes=notes,
        ns=ns,
        runtime_table=runtime_table,
        memory_table=memory_table,
        comparison_rows=comparison_rows,
        runtime_div=runtime_div,
        memory_div=memory_div,
        manual_complexities=sections.manual_complexities,
        interview_summaries=sections.interview_summaries,
        beginner_summaries=sections.beginner_summaries,
        methods_text=sections.methods_text,
    )
    return html
