import torch

from planner_metrics.temporal_stability import (
    compute_curvature_rate_batch,
    compute_mean_abs_jerk_batch,
    compute_replan_consistency_batch,
    consecutive_frame_pairs,
    inter_frame_transform,
    parse_frame_key,
)


def _straight_plan(length=80, lateral=0.0):
    x = torch.arange(1, length + 1, dtype=torch.float32)
    y = torch.full_like(x, lateral)
    cos = torch.ones_like(x)
    sin = torch.zeros_like(x)
    return torch.stack([x, y, cos, sin], dim=-1)


def _replan_jump(plan_t, plan_next, gap=1):
    rel_pos, rel_heading = inter_frame_transform(plan_t.unsqueeze(0), gap)
    return compute_replan_consistency_batch(
        plan_t.unsqueeze(0), plan_next.unsqueeze(0), gap, rel_pos, rel_heading
    )


def test_frame_key_groups_basic_dataset_scene_and_frame():
    group, frame = parse_frame_key("/data/10-55-28/10-55-28_00000001_00000034.npz")
    assert group == "/data/10-55-28#00000001"
    assert frame == 34


def test_pair_builder_keeps_modal_gap_and_skips_sequence_breaks():
    paths = [
        "/data/a/scene_00000001_00000001.npz",
        "/data/a/scene_00000001_00000004.npz",
        "/data/a/scene_00000001_00000007.npz",
        "/data/a/scene_00000001_00000020.npz",
    ]
    pairs = list(consecutive_frame_pairs(paths))
    assert [(idx_t, idx_next, gap) for idx_t, _, idx_next, _, gap in pairs] == [
        (1, 4, 3),
        (4, 7, 3),
    ]


def test_replan_consistency_is_zero_when_overlapping_plans_match():
    out = _replan_jump(_straight_plan(), _straight_plan(), gap=1)
    assert out["overlap_len"] == 79
    assert torch.allclose(out["position_jump"], torch.zeros(1), atol=1.0e-6)
    assert torch.allclose(out["heading_jump"], torch.zeros(1), atol=1.0e-6)


def test_replan_consistency_reports_lateral_flicker_in_metres():
    out = _replan_jump(_straight_plan(), _straight_plan(lateral=1.0), gap=1)
    assert torch.allclose(out["position_jump"], torch.ones(1), atol=1.0e-6)
    assert torch.allclose(out["heading_jump"], torch.zeros(1), atol=1.0e-6)


def test_single_trajectory_stability_is_zero_for_constant_speed_straight_plan():
    plan = _straight_plan().unsqueeze(0)
    assert torch.allclose(compute_mean_abs_jerk_batch(plan), torch.zeros(1), atol=1.0e-4)
    assert torch.allclose(compute_curvature_rate_batch(plan), torch.zeros(1), atol=1.0e-4)
