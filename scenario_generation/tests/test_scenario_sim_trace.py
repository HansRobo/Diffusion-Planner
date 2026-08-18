"""The scenario_sim visualization seam: ``rollout.jsonl`` and the readers it feeds.

The colormap images, the video and the HTML gallery are shared with the reproducer path, so what
has to hold is the contract between them: the trace carries the keys those readers actually read,
road-border distance (which only exists after the rollout) reaches every row, and a metric this
path never observed is not offered as a picture of zero.
"""

import json

import numpy as np
import pytest

from scenario_generation.closed_loop_html_report import collect_site_data
from scenario_generation.scenario_sim_rollout import _write_rollout_trace
from scenario_generation.trajectory_colormap import load_step_trace, render_trajectory_colormap
from scenario_generation.wandb_closed_loop import episode_stem

N_STEPS = 6


def _trajectory_log(n=N_STEPS):
    return [
        {
            "step": k,
            "x": float(k),
            "y": 0.5 * k,
            "heading": 0.1 * k,
            "speed": 5.0 - 0.4 * k,
            "goal_d": float(50 - k),
        }
        for k in range(n)
    ]


def _write(tmp_path, *, warm_from=0, n=N_STEPS):
    """Write a trace; ``warm_from`` mimics warmup ticks that were never scored.

    The metric series is short by exactly those ticks, which is the shape the rollout hands over.
    """
    scored = n - warm_from
    _write_rollout_trace(
        tmp_path,
        trajectory_log=_trajectory_log(n),
        clearances=[3.0 - 0.2 * k for k in range(scored)],
        collisions=[False] * scored,
        rb_dists=np.linspace(2.0, 1.0, n),
        terminated_reason="sim_terminated",
    )
    return tmp_path / "rollout.jsonl"


def _site(tmp_path, stem, row, **summary):
    """One site laid out the way ``collect_site_data`` reads it; returns what it read back."""
    site = tmp_path / "siteA"
    (site / stem).mkdir(parents=True)
    _write(site / stem)
    (site / "summary.json").write_text(
        json.dumps({"n_segments": 1, "near_miss_thresh": 0.5, **summary})
    )
    (site / "segments.jsonl").write_text(json.dumps(row) + "\n")
    return collect_site_data(tmp_path, ["siteA"])


def test_trace_carries_every_key_its_readers_read(tmp_path):
    _write(tmp_path)
    rows = load_step_trace(tmp_path)

    assert len(rows) == N_STEPS  # the terminated marker is not a step
    for k, r in enumerate(rows):
        assert r["k"] == k
        assert len(r["ego"]) == 2
        assert r["speed"] is not None
        assert r["collision"] is False
        # Road-border distance is computed over the whole trajectory after the fact; every row
        # must still have it, or the road_border colormap silently draws a capped constant.
        assert r["rb_dist_m"] is not None


def test_unobserved_red_light_is_absent_rather_than_false(tmp_path):
    """A metric with no trace key is skipped, not drawn as a flat "no event"."""
    _write(tmp_path)
    assert all("red_light_violation" not in r for r in load_step_trace(tmp_path))
    assert render_trajectory_colormap(tmp_path, tmp_path / "rl.png", metric="red_light") is None
    # An observed metric on the same trace still renders, so this is not a blanket refusal.
    assert render_trajectory_colormap(tmp_path, tmp_path / "rb.png", metric="road_border")


def test_unscored_warmup_ticks_keep_the_rows_aligned(tmp_path):
    _write(tmp_path, warm_from=2)
    rows = load_step_trace(tmp_path)

    assert len(rows) == N_STEPS
    assert [r["clearance_m"] for r in rows[:2]] == [None, None]
    assert all(r["clearance_m"] is not None for r in rows[2:])
    # The ego pose is what the colormap draws, so it must exist for the warmup steps too.
    assert all(r["ego"] for r in rows)


@pytest.mark.parametrize("row", [{"route": "r"}, {"route": "r", "segment": None}])
def test_episode_stem_tolerates_a_row_without_segments(tmp_path, row):
    assert episode_stem(tmp_path, row) == "r"


def test_episode_stem_still_prefers_the_segment_suffixed_form(tmp_path):
    """Regression: making ``segment`` optional must not change a row that carries one."""
    row = {"route": "r", "segment": [0, 100]}
    # No artifact yet -> the documented fall back to the bare route name.
    assert episode_stem(tmp_path, row) == "r"
    (tmp_path / "r_0_100.mp4").write_bytes(b"")
    assert episode_stem(tmp_path, row) == "r_0_100"
    (tmp_path / "r_0_100.mp4").unlink()
    (tmp_path / "r_0_100").mkdir()
    assert episode_stem(tmp_path, row) == "r_0_100"


def test_report_still_labels_a_segmented_episode(tmp_path):
    """Regression for the reproducer shape: segmented rows keep their ``[start,end]`` label."""
    items, _ = _site(
        tmp_path,
        "route1_0_100",
        {"route": "route1", "segment": [0, 100], "n_steps_run": N_STEPS, "terminated": "goal"},
    )

    assert items[0]["segment"] == "[0,100]"
    assert items[0]["route"] == "route1"
    assert "road_border" in items[0]["colormap_paths"]


def test_report_reads_a_segment_less_site(tmp_path):
    stem = "case0__v1__scenario_0"
    items, summaries = _site(
        tmp_path,
        stem,
        {
            "route": stem,
            "n_steps_run": N_STEPS,
            "terminated": "sim_terminated",
            "result_kind": "Pass",
            "coord_check_ok": True,
            "rb_has_data": True,
            "progress_m": 5.0,
        },
        total_steps=N_STEPS,
    )

    assert len(items) == 1 and len(summaries) == 1
    item = items[0]
    assert item["route"] == stem
    assert item["segment"] == ""  # nothing to sub-divide on this path
    assert item["result_kind"] == "Pass"
    assert item["coord_check_ok"] is True
    # Skipped because the trace never carried it -- no per-site declaration needed.
    assert "red_light" not in item["colormap_paths"]
    assert "road_border" in item["colormap_paths"]
