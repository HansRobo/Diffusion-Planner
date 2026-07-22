from pathlib import Path

from human_match_prototype.multi_human_report import diagnose_mismatch, render_location_report


def test_diagnose_planner_deficiency():
    row = {"mismatch_4s": 1.0, "dp_human_coverage_4s": 0.1, "n_humans": 20}
    assert diagnose_mismatch(row) == "planner_deficiency"


def test_diagnose_unusual_human():
    row = {"mismatch_4s": 1.0, "dp_human_coverage_4s": 0.9, "n_humans": 20}
    assert diagnose_mismatch(row) == "unusual_test_human"


def test_diagnose_insufficient_data():
    row = {"mismatch_4s": 1.0, "dp_human_coverage_4s": 0.5, "n_humans": 3}
    assert diagnose_mismatch(row) == "insufficient_data"


def test_diagnose_no_mismatch():
    row = {"mismatch_4s": 0.0, "dp_human_coverage_4s": 0.9, "n_humans": 20}
    assert diagnose_mismatch(row) == "no_mismatch"


def test_render_location_report_empty_rows(tmp_path):
    out_html = tmp_path / "report.html"
    render_location_report([], [], out_html)
    content = out_html.read_text()
    assert "No mismatches found" in content
