"""Contract tests for the scenario_sim viewer export.

These assert the shape a consumer outside this repo depends on.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scenario_generation.scenario_sim_viewer_export import export, key_aliases, load_submitted
from scenario_generation.scene_trace import asset_dir

_MAP = "/suite/assets/map/proj/2231/2231-0001/lanelet2_map.osm"
_SIDECAR = {
    "categories": {"C": "交差点"},
    "scenarios": {},  # _make_run が埋める
}
_MAP_ID = "2231"
_SC1 = "5b071057-5e4d-4585-a0c6-2a3e649fb28c"
_SC2 = "8dd3a7fd-6af8-4dcc-9f10-ce46d09b0e88"
_MAP_REF = "0123456789abcdef"


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


def _make_suite(root: Path, rels: list[str]) -> None:
    """The sidecar the suite carries: display names and the category labels they use."""
    import re

    scenarios = {}
    for rel in rels:
        m = re.search(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", rel)
        if m:
            scenarios[m.group(1)] = f"C-01-1000{len(scenarios)}_case01_dp"
    (root / "scenario_names.json").write_text(
        json.dumps({"categories": _SIDECAR["categories"], "scenarios": scenarios})
    )


def _make_run(root: Path, rels: list[str], *, submitted: list[str] | None = None) -> Path:
    """A run directory in the shape the suite driver leaves behind."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_context.txt").write_text("jobs=4 max_steps=1700 draw_every=4\n")
    suite = root.parent / "suite"
    suite.mkdir(parents=True, exist_ok=True)
    _make_suite(suite, submitted or rels)
    (root / "work.json").write_text(
        json.dumps(
            [[str(root / key_aliases(r)[0]), str(suite / "scenarios" / r)] for r in (submitted or rels)]
        )
    )
    trace = [json.dumps({"k": k, "ego": [float(k), 0.0], "speed": 1.0, "clearance_m": 2.0, "collision": False}) for k in range(3)]
    for rel in rels:
        case = root / key_aliases(rel)[0]
        case.mkdir(parents=True, exist_ok=True)
        (case / "row.json").write_text(json.dumps(_row()))
        (case / "rollout.jsonl").write_text("\n".join(trace) + "\n")
        (case / f"{case.name}.mp4").write_bytes(b"\x00")
        with gzip.open(case / "scene_trace.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(json.dumps({"event": "header", "map_ref": _MAP_REF}) + "\n")
    # One asset for the whole run, written where the rollout puts it.
    assets = asset_dir(root)
    assets.mkdir(parents=True, exist_ok=True)
    with gzip.open(assets / f"{_MAP_REF}.json.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"version": 1, "lanes": [], "road_borders": []}))
    return root


def test_layout_is_three_files_plus_media(tmp_path):
    """The listing must not cost a read per scenario."""
    export(_make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC1, case=6)]), tmp_path / "out")
    out = tmp_path / "out"

    for name in ("run.json", "scenarios.json", "cases.jsonl"):
        assert (out / name).is_file()
    assert not list(out.glob("*/summary.json")), "per-scenario summary files are back"

    cases = [json.loads(ln) for ln in (out / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 2
    media = out / "media" / _SC1
    for case in cases:
        assert case["scenario"] == _SC1
        stem = case["route"]
        assert not (media / f"{stem}.mp4").exists()
        assert (media / f"{stem}.rollout.jsonl").is_file()
        assert not list(media.glob(f"{stem}.*.png"))


def test_grouping_keys_are_fields_not_directories(tmp_path):
    """Category, map and version travel as data so a new grouping needs no re-export."""
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), tmp_path / "out")
    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]

    assert entry["category"] == "C"
    # The label comes from the suite's sidecar, never from this repo.
    assert entry["category_name"] == "交差点"
    assert entry["map"] == _MAP_ID
    assert entry["version"] == "5"
    assert entry["name"].startswith("C-01-")


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
    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]

    assert "mean_route_completion" not in entry["summary"]
    assert entry["summary"]["red_light_violation"]["measured"] is False
    assert {"mean_route_completion", "reproducer"} <= set(entry["unmeasured_keys"])
    case = json.loads((tmp_path / "out" / "cases.jsonl").read_text().splitlines()[0])
    assert "route_completion" not in case


def test_missing_cases_are_stated_against_the_submitted_list(tmp_path):
    """Failures are absent rows; the artifacts on disk cannot reveal them."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)], submitted=[_rel(_SC1), _rel(_SC1, case=6)])
    export(run, tmp_path / "out")

    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]
    assert "produced no row" in entry["error"]
    assert json.loads((tmp_path / "out" / "run.json").read_text())["submitted_cases"] == 2


_JUNIT_FAILURE = """<?xml version="1.0"?>
<testsuites name="s" failures="1" errors="0" tests="1">
  <testsuite name="5" failures="1" errors="0" tests="1">
    <testcase name="scenario_0">
      <failure type="SimulationFailure" message="CustomCommandAction typed &quot;exitFailure&quot; \
was triggered by the anonymous Condition (OpenSCENARIO.Storyboard.Story[0]): Is [ego] \
colliding with Npc1?&#10;Unmet success conditions:&#10;  - &quot;goal_position&quot;" />
    </testcase>
  </testsuite>
</testsuites>
"""


def test_a_case_that_never_decided_is_not_a_failure(tmp_path):
    """Reaching a verdict and being cut off before one are different states.

    The row's ``result_kind`` cannot tell them apart: the interpreter presets it to a timeout
    failure, so a case that never decided still reads ``Failure``.
    """
    run = _make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC1, case=6)])
    osp_out = run / key_aliases(_rel(_SC1))[0] / "osp_out"
    osp_out.mkdir(parents=True)
    (osp_out / "result.junit.xml").write_text(_JUNIT_FAILURE)
    export(run, tmp_path / "out")

    verdicts = [
        json.loads(line)["verdict"]
        for line in (tmp_path / "out" / "cases.jsonl").read_text().splitlines()
    ]
    decided = [v for v in verdicts if v["decided"]]
    assert [v for v in verdicts if not v["decided"]] == [{"decided": False}]
    assert decided[0]["kind"] == "Failure"
    assert decided[0]["unmet"] == ["goal_position"]

    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]
    assert entry["verdicts"] == {"pass": 0, "failure": 1, "error": 0, "undecided": 1}
    assert sum(entry["verdicts"].values()) == entry["n_cases"]


def test_a_site_mode_that_fits_nothing_fails(tmp_path):
    """A suite with no scenario id in its paths is a wrong mode, not an empty run."""
    with pytest.raises(SystemExit, match="resolved nothing"):
        export(_make_run(tmp_path / "run", ["cat/a.xosc"]), tmp_path / "out")


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


def test_the_map_the_rollout_wrote_is_the_map_the_export_publishes(tmp_path):
    """A rollout is handed one case directory and derives the asset directory from it, so the
    two sides agree through ``asset_dir`` rather than through a caller remembering to wire it.
    A disagreement is silent: the export finds no map and the viewer draws over nothing."""
    export(_make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC1, case=6)]), tmp_path / "out")
    out = tmp_path / "out"

    cases = [json.loads(ln) for ln in (out / "cases.jsonl").read_text().splitlines()]
    assert cases, "no cases exported"
    for case in cases:
        assert case["map_asset"] == f"maps/{_MAP_REF}.json.gz"
        assert case["scene_trace"] == f"media/{_SC1}/{case['route']}.scene.jsonl.gz"
        assert (out / case["map_asset"]).is_file()
        assert (out / case["scene_trace"]).is_file()
    # One map serves every case that ran on it.
    assert len(list((out / "maps").iterdir())) == 1
