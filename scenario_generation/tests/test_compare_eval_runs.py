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
    assert "Closed-Loop Scenario Evaluation Comparison" in report
    assert "Wins (Improved Fail → Pass)" in report
    assert "Losses (Regressed Pass → Fail)" in report
    assert "Failure Root-Cause Breakdown" in report
    assert "Hazard Engagement & Validity Analysis" in report
    assert "scen_1" in report
    assert "scen_2" in report
