"""Contract tests for the scenario_sim viewer export.

These assert the shape a consumer outside this repo depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenario_generation.scenario_sim_viewer_export import export, load_submitted

_MAP = "/suite/assets/map/proj/12/12-0001/lanelet2_map.osm"
_SIDECAR = {
    "categories": {"C": "交差点"},
    "scenarios": {},  # _make_run が埋める
}
_MAP_ID = "12"
_SC1 = "5b071057-5e4d-4585-a0c6-2a3e649fb28c"
_SC2 = "8dd3a7fd-6af8-4dcc-9f10-ce46d09b0e88"


def _rel(scenario_id: str, case: int = 5) -> str:
    return f"proj/{scenario_id}/{case}/scenario_0.xosc"


def case_key(rel: str) -> str:
    """The directory name the drivers give a case, mirrored here to build a run."""
    return rel[: -len(".xosc")].replace("/", "_")


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
            [[str(root / case_key(r)), str(suite / "scenarios" / r)] for r in (submitted or rels)]
        )
    )
    trace = [json.dumps({"k": k, "ego": [float(k), 0.0], "speed": 1.0, "clearance_m": 2.0, "collision": False}) for k in range(3)]
    for rel in rels:
        case = root / case_key(rel)
        case.mkdir(parents=True, exist_ok=True)
        (case / "row.json").write_text(json.dumps(_row()))
        (case / "rollout.jsonl").write_text("\n".join(trace) + "\n")
        (case / f"{case.name}.mp4").write_bytes(b"\x00")
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
        assert (media / f"{stem}.mp4").is_file()
        assert (media / f"{stem}.rollout.jsonl").is_file()
        assert (media / f"{stem}.clearance.png").is_file()


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
    osp_out = run / case_key(_rel(_SC1)) / "osp_out"
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


@pytest.mark.parametrize("relative", ["", "inside", ".."])
def test_an_output_path_overlapping_the_run_is_refused(tmp_path, relative):
    """Publishing renames the output aside and deletes it; the run must never be the output."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)])
    before = sorted(str(p.relative_to(run)) for p in run.rglob("*"))

    with pytest.raises(SystemExit, match="overlaps"):
        export(run, run / relative if relative else run)

    assert sorted(str(p.relative_to(run)) for p in run.rglob("*")) == before


@pytest.mark.parametrize("damage", ["truncate", "remove"])
def test_a_run_without_a_usable_submitted_list_fails(tmp_path, damage):
    """The rows cannot supply the denominator: a case that died leaves no row to count."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)], submitted=[_rel(_SC1), _rel(_SC1, case=6)])
    if damage == "truncate":
        (run / "work.json").write_text('[["a", "b"')   # interrupted mid-write
    else:
        (run / "work.json").unlink()

    with pytest.raises(SystemExit, match="unusable submitted-case list"):
        export(run, tmp_path / "out")


def test_a_damaged_verdict_is_not_read_as_a_pass(tmp_path):
    """Pass is the rarest verdict here; a damaged record must never invent one."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)])
    osp_out = run / case_key(_rel(_SC1)) / "osp_out"
    osp_out.mkdir(parents=True)
    (osp_out / "result.junit.xml").write_text("<testsuites></testsuites>")
    export(run, tmp_path / "out")

    verdict = json.loads((tmp_path / "out" / "cases.jsonl").read_text().splitlines()[0])["verdict"]
    assert verdict["kind"] == "Error" and verdict["type"] == "MalformedVerdict"
    entry = json.loads((tmp_path / "out" / "scenarios.json").read_text())[_SC1]
    assert entry["verdicts"]["pass"] == 0


def test_a_run_that_drew_nothing_exports(tmp_path):
    """The driver stamps ``draw_every=off`` when drawing is disabled; it is not a number."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)])
    (run / "run_context.txt").write_text("jobs=4 max_steps=3000 draw_every=off\n")

    export(run, tmp_path / "out")

    assert json.loads((tmp_path / "out" / "run.json").read_text())["draw_every"] == "off"


def test_a_video_that_cannot_be_re_timed_is_not_published(tmp_path):
    """The viewer maps steps onto whatever it finds, so a mis-timed video is worse than none."""
    run = _make_run(tmp_path / "run", [_rel(_SC1)])
    out = tmp_path / "out"
    # 20 fps against draw_every 4 needs a factor of two, and the stub mp4 cannot be re-timed.
    export(run, out, fps=20.0)

    assert not list((out / "media").rglob("*.mp4"))
    assert json.loads((out / "cases.jsonl").read_text().splitlines()[0])["scenario"] == _SC1


def test_a_failed_export_leaves_the_previous_one_untouched(tmp_path, monkeypatch):
    """A tree is published whole or not at all; a half-written one must not replace a good one."""
    import scenario_generation.scenario_sim_viewer_export as module

    out = tmp_path / "out"
    export(_make_run(tmp_path / "run", [_rel(_SC1)]), out)
    before = sorted(str(p.relative_to(out)) for p in out.rglob("*"))

    def explode(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(module, "render_trajectory_colormaps", explode)
    with pytest.raises(RuntimeError):
        export(_make_run(tmp_path / "run2", [_rel(_SC2)]), out)

    assert sorted(str(p.relative_to(out)) for p in out.rglob("*")) == before
    assert not list(tmp_path.glob("out.staging-*"))


def test_a_re_export_leaves_nothing_of_the_previous_one(tmp_path):
    """A viewer must never pair this run's row with the last run's evidence."""
    run = _make_run(tmp_path / "run", [_rel(_SC1), _rel(_SC2)])
    out = tmp_path / "out"
    export(run, out)
    stale = out / "media" / _SC2
    assert stale.is_dir()

    # A second export of fewer scenarios, drawing fewer metrics.
    export(_make_run(tmp_path / "run2", [_rel(_SC1)]), out, colormap_metrics=("speed",))

    assert not stale.exists()
    drawn = sorted(p.name for p in (out / "media").rglob("*.png"))
    assert drawn and all(name.endswith(".speed.png") for name in drawn), drawn


def test_a_run_with_no_scenario_id_fails(tmp_path):
    """Paths carrying no id are a wrong assumption about the layout, not an empty run."""
    with pytest.raises(SystemExit, match="no scenario id"):
        export(_make_run(tmp_path / "run", ["cat/a.xosc"]), tmp_path / "out")


def test_work_list_entries_resolve_against_the_manifest(tmp_path):
    """Entries resolve against the manifest's own directory; absolute ones pass through."""
    run = tmp_path / "run"
    run.mkdir()
    absolute = str(tmp_path / "suite" / _rel(_SC2))
    (run / "work.json").write_text(
        json.dumps([[str(run / "case_a"), _rel(_SC1)], [str(run / "case_b"), absolute]])
    )
    got = {case.name: str(osc) for case, osc in load_submitted(run)}
    assert got["case_a"] == str(run / _rel(_SC1))
    assert got["case_b"] == absolute
