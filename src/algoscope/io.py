# src/algoscope/io.py
from __future__ import annotations
import json
from typing import Any
from pathlib import Path
from .analyze import ResultObject
from .utils import human_time, human_bytes

def export_results_json(result: ResultObject, out_path: str | Path) -> None:
    """
    Write a JSON file that contains structured numeric data and human-readable text
    for use by external frontends (Next.js, Angular, etc).
    """
    out_path = Path(out_path)
    data: dict[str, Any] = {
        "title": result.title,
        "html_path": result.html_path,
        "ns": result.ns,
        "functions": {},
    }

    for label, fs in result.func_stats.items():
        # time_ci and mem_ci are CIResult objects with attributes: mean, std, n, lower, upper, method
        time_ci_serial = {n: {
            "mean": getattr(ci, "mean", None),
            "std": getattr(ci, "std", None),
            "n": getattr(ci, "n", None),
            "lower": getattr(ci, "lower", None),
            "upper": getattr(ci, "upper", None),
            "method": getattr(ci, "method", None),
            "mean_human": human_time(getattr(ci, "mean", 0)) if getattr(ci, "mean", None) is not None else None,
        } for n, ci in fs.time_ci.items()}

        mem_ci_serial = {n: {
            "mean": getattr(ci, "mean", None),
            "std": getattr(ci, "std", None),
            "n": getattr(ci, "n", None),
            "lower": getattr(ci, "lower", None),
            "upper": getattr(ci, "upper", None),
            "method": getattr(ci, "method", None),
            "mean_human": human_bytes(getattr(ci, "mean", 0)) if getattr(ci, "mean", None) is not None else None,
        } for n, ci in fs.mem_ci.items()}

        data["functions"][label] = {
            "label": label,
            "times_raw": fs.times,      # raw per-repeat samples (n -> [sec,...])
            "mems_raw": fs.mems,        # raw per-repeat samples (n -> [bytes,...])
            "time_ci": time_ci_serial,
            "mem_ci": mem_ci_serial,
            "manual_explanation": fs.manual_explanation,
            "interview_summary": fs.interview_summary,
            "dynamic_guess": fs.dynamic_guess,
            "errors": fs.errors,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write prettified for local debugging; frontends can fetch & parse easily
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
