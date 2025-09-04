from __future__ import annotations

import os
from algoscope import analyze_functions


def f_linear(arr):
    s = 0
    for x in arr:
        s += x
    return s


def build_input(n: int):
    return ([0] * n,), {}


def test_basic_api(tmp_path):
    out = tmp_path / "t.html"
    res = analyze_functions(
        funcs=[f_linear],
        input_builder=build_input,
        ns=[100, 200, 400],
        repeats=3,
        ci_method="t",
        html_out=str(out),
        title="Test Report",
    )
    assert out.exists()
    assert "Manual Big-O Analysis" in res.html
    assert "Runtime Benchmarks" in res.html
