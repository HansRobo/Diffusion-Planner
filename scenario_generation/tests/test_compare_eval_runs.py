import json
from pathlib import Path

import pytest
from scenario_generation.compare_eval_runs import compare_runs, load_run_data


def test_compare_runs(tmp_path: Path):
    base_dir = tmp_path / "base"
    treat_dir = tmp_path / "treat"
    base_dir.mkdir()
    treat_dir.mkdir()

    # Create dummy segment rows
    base_rows = [
        {"route": "scen_1", "result_kind": "Failure", "object": {"collision_count": 1, "clearance_min_m": 0.0}, "max_speed_mps": 5.0, "progress_m": 10.0, "n_steps_run": 50},
        {"route": "scen_2", "result_kind": "Pass", "object": {"collision_count": 0, "clearance_min_m": 3.0}, "max_speed_mps": 8.0, "progress_m": 100.0, "n_steps_run": 200},
        {"route": "scen_3", "result_kind": "Failure", "object": {"collision_count": 0, "clearance_min_m": 50.0}, "max_speed_mps": 0.2, "progress_m": 1.0, "n_steps_run": 100},
    ]
    treat_rows = [
        {"route": "scen_1", "result_kind": "Pass", "object": {"collision_count": 0, "clearance_min_m": 2.5}, "max_speed_mps": 6.0, "progress_m": 100.0, "n_steps_run": 150},
        {"route": "scen_2", "result_kind": "Failure", "object": {"collision_count": 1, "clearance_min_m": 0.0}, "max_speed_mps": 8.0, "progress_m": 50.0, "n_steps_run": 100},
        {"route": "scen_3", "result_kind": "Pass", "object": {"collision_count": 0, "clearance_min_m": 10.0}, "max_speed_mps": 5.0, "progress_m": 80.0, "n_steps_run": 120},
    ]

    (base_dir / "segments.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n")
    (treat_dir / "segments.jsonl").write_text("\n".join(json.dumps(r) for r in treat_rows) + "\n")

    report = compare_runs(base_dir, treat_dir, "ModelA", "ModelB")
    assert "クローズドループ シナリオ評価 比較レポート" in report
    assert "改善ケース (Wins: Fail → Pass)" in report
    assert "悪化ケース (Losses: Pass → Fail)" in report
    assert "失敗要因の排他分類" in report
    assert "ハザード提示検知と妥当性分析" in report
    assert "scen_1" in report
    assert "scen_2" in report


def test_build_comparison_html_report(tmp_path: Path):
    from scenario_generation.scenario_comparison_html_report import build_comparison_html_report, build_comparison_payload

    base_dir = tmp_path / "base"
    treat_dir = tmp_path / "treat"
    base_dir.mkdir()
    treat_dir.mkdir()

    base_rows = [
        {"route": "scen_1", "result_kind": "Failure", "object": {"collision_count": 1, "clearance_min_m": 0.0}, "max_speed_mps": 5.0, "progress_m": 10.0, "n_steps_run": 50},
        {"route": "scen_2", "result_kind": "Pass", "object": {"collision_count": 0, "clearance_min_m": 3.0}, "max_speed_mps": 8.0, "progress_m": 100.0, "n_steps_run": 200},
    ]
    treat_rows = [
        {"route": "scen_1", "result_kind": "Pass", "object": {"collision_count": 0, "clearance_min_m": 2.5}, "max_speed_mps": 6.0, "progress_m": 100.0, "n_steps_run": 150},
        {"route": "scen_2", "result_kind": "Failure", "object": {"collision_count": 1, "clearance_min_m": 0.0}, "max_speed_mps": 8.0, "progress_m": 50.0, "n_steps_run": 100},
    ]

    (base_dir / "segments.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n")
    (treat_dir / "segments.jsonl").write_text("\n".join(json.dumps(r) for r in treat_rows) + "\n")

    # Create dummy video and colormap
    (base_dir / "scen_1.mp4").write_text("dummy video")
    (treat_dir / "scen_1_trajcolormap_clearance.png").write_text("dummy colormap")

    out_html = tmp_path / "comparison.html"
    res = build_comparison_html_report(
        base_dir=base_dir,
        treat_dir=treat_dir,
        out_path=out_html,
        title_base="Model Alpha",
        title_treat="Model Beta",
        title="Custom Model Comparison Report",
        subtitle="Benchmark test runs",
    )

    assert res.is_file()
    html = res.read_text(encoding="utf-8")
    assert "Custom Model Comparison Report" in html
    assert "Benchmark test runs" in html
    assert "Model Alpha" in html
    assert "Model Beta" in html
    assert "scen_1" in html
    assert "scen_2" in html
    assert "kpiPassRate" in html
    assert "failureTable" in html
    assert "hazardTable" in html
    assert "metricsTable" in html
    assert "btn-filter" in html

