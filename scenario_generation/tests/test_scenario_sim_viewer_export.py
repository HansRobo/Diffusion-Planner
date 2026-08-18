"""Contract tests for the scenario_sim viewer export.

These assert the shape a consumer outside this repo depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenario_generation.scenario_sim_viewer_export import export, key_aliases, load_submitted

_MAP = "/suite/assets/map/proj/2231/2231-0001/lanelet2_map.osm"
_MAP_ID = "2231"
_SC1 = "5b071057-5e4d-4585-a0c6-2a3e649fb28c"
_SC2 = "8dd3a7fd-6af8-4dcc-9f10-ce46d09b0e88"


def _rel(scenario_id: str, case: int = 5) -> str:
    return f"proj/{scenario_id}/{case}/scenario_0.xosc"


def _row() -> dict:
    """A segment row with the block layout ``aggregate`` rolls up."""
    clearance = {
        "miss_thresh_m": 1.0,
        "collision_steps": 0,
        "collision_count": 0,
        "miss_steps": 0,
        "miss_count": 0,
        "clearance_min_m": 2.0,
        "clearance_mean_m": 3.0,
        "clearance_p5_m": 2.1,
        "clearance_finite_steps": 3,
    }
    return {
        "n_steps_run": 3,
        "terminated": "max_steps",
        "progress_m": 10.0,
        "object": dict(clearance),
        "road_border": dict(clearance),
        "red_light_violation": {"steps": 0, "count": 0, "measured": False},
        # inf is the in-band "no braking event" value the rollout writes.
        "strong_brake": {"thresh_mps2": -2.5, "strongest_mps2": float("inf"), "steps": 0, "count": 0},
        "reproducer": {"expand_count": 0, "snap_count": 0, "repeat_steps": 0, "normal_steps": 3},
        "map_path": _MAP,
    }


def _make_run(root: Path, rels: list[str], *, submitted: list[str] | None = None) -> Path:
    """A run directory in the shape the suite driver leaves behind."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_context.txt").write_text("jobs=4 max_steps=1700 draw_every=4\n")
    (root / "work.json").write_text(
        json.dumps([[str(root / key_aliases(r)[0]), r] for r in (submitted or rels)])
    )
    trace = [json.dumps({"k": k, "ego": [float(k), 0.0], "speed": 1.0, "clearance_m": 2.0, "collision": False}) for k in range(3)]
    for rel in rels:
        case = root / key_aliases(rel)[0]
        case.mkdir(parents=True, exist_ok=True)
        (case / "row.json").write_text(json.dumps(_row()))
        (case / "rollout.jsonl").write_text("\n".join(trace) + "\n")
        (case / f"{case.name}.mp4").write_bytes(b"\x00")
    return root


def test_layout_matches_the_viewer_contract(tmp_path):
    export(_make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC1, case=6)]), tmp_path / "out")
    out = tmp_path / "out"

    assert set(json.loads((out / "groups_summary.json").read_text())["sub_groups"]) == {_MAP_ID}
    site = out / _MAP_ID / _SC1
    rows = [json.loads(ln) for ln in (site / "segments.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        stem = row["route"]
        assert (site / f"{stem}.mp4").is_file()
        assert (site / stem / "rollout.jsonl").is_file()
        assert (site / f"{stem}_trajcolormap_clearance.png").is_file()


def test_exported_json_carries_no_non_json_constants(tmp_path):
    """A consumer outside Python cannot read a file containing Infinity or NaN."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")

    def reject(token):
        raise AssertionError(f"non-JSON constant in output: {token}")

    for path in (tmp_path / "out").rglob("*.json"):
        json.loads(path.read_text(), parse_constant=reject)
    for path in (tmp_path / "out").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line, parse_constant=reject)


def test_unmeasured_families_are_not_reported_as_zero(tmp_path):
    """A family nobody observed must stay distinguishable from one measured at zero."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")
    summary = json.loads((tmp_path / "out" / _MAP_ID / _SC1 / "summary.json").read_text())

    assert "mean_route_completion" not in summary
    assert summary["red_light_violation"]["measured"] is False
    assert "mean_route_completion" in summary["unmeasured_keys"]
    # Rows keep the key absent rather than null: the viewer rounds it, and round(None) raises.
    row = json.loads((tmp_path / "out" / _MAP_ID / _SC1 / "segments.jsonl").read_text().splitlines()[0])
    assert "route_completion" not in row


def test_missing_rows_reach_the_manifest_the_viewer_reads(tmp_path):
    """Failures are absent rows, and a count only the export report carries is invisible."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)], submitted=[_rel(_SC1), _rel(_SC1, case=6)])
    export(run, tmp_path / "out")

    summary = json.loads((tmp_path / "out" / _MAP_ID / _SC1 / "summary.json").read_text())
    assert "produced no row" in summary["error"]
    assert json.loads((tmp_path / "out" / "export_meta.json").read_text())["submitted_cases"] == 2


def test_a_site_mode_that_fits_nothing_fails(tmp_path):
    """A suite with no scenario id in its paths is a wrong mode, not an empty run."""
    with pytest.raises(SystemExit, match="resolved no site"):
        export(_make_run(tmp_path / "run", ["cat/a.xosc"]), tmp_path / "out", group_mode="flat")


def test_re_export_keeps_groups_written_by_an_earlier_pass(tmp_path):
    export(_make_run(tmp_path / "r1", [_rel(_SC1)]), tmp_path / "out")
    export(_make_run(tmp_path / "r2", [_rel(_SC2)]), tmp_path / "out", group_mode="flat")
    manifest = json.loads((tmp_path / "out" / "groups_summary.json").read_text())
    assert set(manifest["sub_groups"]) == {_MAP_ID, "all"}


def test_work_list_entries_resolve_against_the_manifest(tmp_path):
    """Entries resolve against the manifest's own directory; absolute ones pass through."""
    run = tmp_path / "run"
    run.mkdir()
    absolute = str(tmp_path / "suite" / _rel(_SC2))
    (run / "work.json").write_text(
        json.dumps([[str(run / "case_a"), _rel(_SC1)], [str(run / "case_b"), absolute]])
    )
    got = dict(load_submitted(run))
    assert got["case_a"] == str(run / _rel(_SC1))
    assert got["case_b"] == absolute
