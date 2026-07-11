import json
from types import SimpleNamespace

import numpy as np
import torch

from scenario_generation.metrics import cl_route_by_area, safety_clearance
from scenario_generation.metrics.centerline_deviation import lateral_offset_from_route_lanes
from scenario_generation.metrics.group_report import aggregate_segment_rows, write_metrics_summary


def test_centerline_offset_uses_segments_not_only_sampled_points():
    route = np.zeros((1, 2, 8), dtype=np.float32)
    route[0, :, :2] = [[-1.0, 0.0], [1.0, 0.0]]
    route[0, :, 2] = 1.0

    assert lateral_offset_from_route_lanes(route) == 0.0
    assert np.isinf(lateral_offset_from_route_lanes(np.zeros_like(route)))


def _timeline(tmp_path):
    paths = [tmp_path / f"{index}.npz" for index in range(10)]
    return SimpleNamespace(npz_paths=paths, frame_indices=np.arange(100, 110))


def test_unreached_episode_is_retained_in_group_metrics(tmp_path):
    row = cl_route_by_area.aggregate_area_metrics(
        [],
        "hard_turn",
        "right_turn",
        "route",
        "bag",
        _timeline(tmp_path),
        labeled_ranges=[[2, 5]],
        video_start_idx=2,
        video_end_idx=5,
        span_index=0,
    )

    assert row["episode_reached"] is False
    assert row["episode_completed"] is False
    assert row["episode_progress_fraction"] == 0.0
    assert row["min_clearance_m"] is None
    summary = aggregate_segment_rows([row])
    assert summary["by_metric_group"]["right_turn"]["episode_reach_rate"] == 0.0


def test_missing_geometry_is_json_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(cl_route_by_area, "is_skipped", lambda _: False)
    step = cl_route_by_area.StepRecord(
        k=0,
        rec_idx=2,
        area="straight_area",
        centerline_m=float("inf"),
        clearance_m=float("inf"),
        rb_dist_m=float("inf"),
    )
    row = cl_route_by_area.aggregate_area_metrics(
        [step],
        "straight_area",
        "straight",
        "route",
        "bag",
        _timeline(tmp_path),
        labeled_ranges=[[2, 3]],
        video_start_idx=2,
        video_end_idx=3,
        span_index=0,
    )

    assert row["episode_completed"] is True
    assert row["centerline_mean_m"] is None
    assert row["min_clearance_m"] is None
    assert row["min_rb_dist_m"] is None
    path = tmp_path / "summary.json"
    write_metrics_summary(aggregate_segment_rows([row]), path)
    assert "Infinity" not in path.read_text(encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


def test_later_route_progress_does_not_mark_a_skipped_episode_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(cl_route_by_area, "is_skipped", lambda _: False)
    steps = [cl_route_by_area.StepRecord(k=0, rec_idx=8, area="later_area")]
    row = cl_route_by_area.aggregate_area_metrics(
        steps,
        "target_area",
        "right_turn",
        "route",
        "bag",
        _timeline(tmp_path),
        labeled_ranges=[[2, 5]],
        video_start_idx=2,
        video_end_idx=5,
        span_index=0,
    )

    assert row["episode_reached"] is False
    assert row["episode_completed"] is False
    assert row["episode_progress_fraction"] == 0.0


def test_near_miss_count_excludes_collision_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(cl_route_by_area, "is_skipped", lambda _: False)
    steps = [
        cl_route_by_area.StepRecord(
            k=0, rec_idx=2, area="target_area", clearance_m=-0.1, collision=True
        ),
        cl_route_by_area.StepRecord(k=1, rec_idx=3, area="target_area", clearance_m=0.2),
        cl_route_by_area.StepRecord(k=2, rec_idx=4, area="target_area", clearance_m=0.5),
    ]
    row = cl_route_by_area.aggregate_area_metrics(
        steps,
        "target_area",
        "straight",
        "route",
        "bag",
        _timeline(tmp_path),
        labeled_ranges=[[2, 5]],
        video_start_idx=2,
        video_end_idx=5,
        span_index=0,
        near_miss_thresh=0.3,
    )

    assert row["n_collision_steps"] == 1
    assert row["n_near_miss_steps"] == 1


def test_safety_metric_does_not_double_batch_line_strings(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        safety_clearance,
        "score_step",
        lambda *args, **kwargs: (float("inf"), False, 0),
    )

    def fake_road_border(trajectory, ego_shape, data, config):
        captured["shape"] = tuple(data["line_strings"].shape)
        return (
            torch.ones(1),
            torch.zeros(1),
            torch.zeros(1),
            [None],
            torch.zeros(1),
            torch.full((1, 1), 2.0),
        )

    monkeypatch.setattr(safety_clearance, "compute_road_border_penalty", fake_road_border)
    result = safety_clearance.score_safety_step(
        np.zeros((2, 11), dtype=np.float32),
        np.array([2.7, 4.8, 1.8], dtype=np.float32),
        {
            "ego_shape": np.array([[2.7, 4.8, 1.8]], dtype=np.float32),
            "line_strings": np.zeros((1, 60, 20, 4), dtype=np.float32),
        },
        "cpu",
    )

    assert captured["shape"] == (1, 60, 20, 4)
    assert result["rb_dist_m"] == 2.0
