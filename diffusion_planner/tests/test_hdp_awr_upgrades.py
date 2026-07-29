"""Tests for the AWR pipeline upgrades ported from the original-DP post-training audits.

Covers the first-waypoint candidate gate (with the mandatory 5 cm tangent floor), the
gate-aware advantage statistics, prefix reward/loss horizons, the gated-product reward
aggregation, the restricted diffusion-time draw, the active-groups-only expert anchor,
and deterministic deployment-aligned checkpoint selection.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.hdp_rl_epoch import validate_hdp_reward_policy
from diffusion_planner.hdp_rl_utils import (
    compute_hdp_reward,
    compute_reward_weighted_loss,
    compute_reward_weights,
    first_waypoint_candidate_gate,
)
from train_hdp_rl_predictor import selection_score_from_reward_metrics

from diffusion_planner import hdp_rl_utils

# ─────────────────────────── first-waypoint gate ────────────────────────────


def _gate_args(**overrides):
    defaults = dict(
        rl_first_waypoint_gate=True,
        rl_first_waypoint_gate_speed_max_mps=1.0,
        rl_first_waypoint_gate_max_step_m=0.25,
        rl_first_waypoint_gate_max_lateral_m=0.20,
        rl_first_waypoint_gate_max_backward_m=0.05,
        rl_first_waypoint_gate_max_tangent_deg=75.0,
        rl_first_waypoint_gate_tangent_min_step_m=0.05,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _candidates(first_steps):
    ego = torch.zeros(len(first_steps), 4, 4)
    for index, (x, y) in enumerate(first_steps):
        ego[index, 0, 0] = x
        ego[index, 0, 1] = y
    ego[..., 2] = 1.0
    return ego


def test_gate_rejects_standstill_jump_and_keeps_small_steps():
    ego = _candidates([(0.05, 0.0), (1.0, 0.0)])
    keep, metrics = first_waypoint_candidate_gate(
        ego, torch.zeros(1), num_scenes=1, n=2, args=_gate_args()
    )
    assert keep.tolist() == [True, False]
    assert metrics["reward_first_waypoint_gated_fraction"].item() == pytest.approx(0.5)
    assert metrics["reward_first_waypoint_gate_active_scene_fraction"].item() == pytest.approx(1.0)


def test_gate_tangent_floor_keeps_millimetre_standstill_steps():
    # 3 mm purely-lateral step: without the floor this reads as 90° off-tangent and the
    # candidate is silently discarded, which was the audited original-DP failure mode.
    ego = _candidates([(0.0, 0.003), (0.0, 0.1)])
    keep, _ = first_waypoint_candidate_gate(
        ego, torch.zeros(1), num_scenes=1, n=2, args=_gate_args()
    )
    assert keep.tolist() == [True, False]


def test_gate_rejects_backward_first_step():
    ego = _candidates([(0.05, 0.0), (-0.1, 0.0)])
    keep, _ = first_waypoint_candidate_gate(
        ego, torch.zeros(1), num_scenes=1, n=2, args=_gate_args()
    )
    assert keep.tolist() == [True, False]


def test_gate_inactive_above_speed_threshold():
    ego = _candidates([(1.0, 0.0), (0.0, 1.0)])
    keep, _ = first_waypoint_candidate_gate(
        ego, torch.tensor([5.0]), num_scenes=1, n=2, args=_gate_args()
    )
    assert keep.all()


def test_gate_disabled_keeps_everything():
    ego = _candidates([(1.0, 0.0), (0.0, 1.0)])
    keep, metrics = first_waypoint_candidate_gate(
        ego,
        torch.zeros(1),
        num_scenes=1,
        n=2,
        args=_gate_args(rl_first_waypoint_gate=False),
    )
    assert keep.all()
    assert metrics["reward_first_waypoint_gated_fraction"].item() == 0.0


# ─────────────────────── gate-aware advantage weights ───────────────────────


def test_reward_weights_mask_all_true_matches_unmasked():
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
    unmasked = compute_reward_weights(reward, 2, 2, "group", 1.0, 1e-6)
    masked = compute_reward_weights(
        reward, 2, 2, "group", 1.0, 1e-6, candidate_valid_mask=torch.ones(4, dtype=torch.bool)
    )
    torch.testing.assert_close(masked[0], unmasked[0])
    torch.testing.assert_close(masked[1], unmasked[1])


def test_reward_weights_mask_excludes_candidate_from_group_stats():
    # The masked candidate must not shift the surviving candidates' advantage: the
    # three-candidate group with one veto must weight exactly like the two-candidate
    # group made of the survivors.
    reward = torch.tensor([1.0, 2.0, 100.0])
    mask = torch.tensor([True, True, False])
    weights, valid = compute_reward_weights(
        reward, 1, 3, "group", 1.0, 1e-6, candidate_valid_mask=mask
    )
    reference, _ = compute_reward_weights(torch.tensor([1.0, 2.0]), 1, 2, "group", 1.0, 1e-6)
    assert valid.tolist() == [True, True, False]
    assert weights[2].item() == 0.0
    torch.testing.assert_close(weights[:2], reference)


def test_reward_weights_group_needs_two_surviving_candidates():
    reward = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, False])
    weights, valid = compute_reward_weights(
        reward, 1, 3, "group", 1.0, 1e-6, candidate_valid_mask=mask
    )
    assert not valid.any()
    assert (weights == 0).all()


def test_reward_weights_mask_shape_and_dtype_are_validated():
    reward = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="candidate_valid_mask"):
        compute_reward_weights(
            reward, 1, 2, "group", 1.0, 1e-6, candidate_valid_mask=torch.ones(3, dtype=torch.bool)
        )
    with pytest.raises(TypeError, match="boolean"):
        compute_reward_weights(reward, 1, 2, "group", 1.0, 1e-6, candidate_valid_mask=torch.ones(2))


# ─────────────────── reward horizon and gated aggregation ───────────────────


def _reward_scene(time_steps=4):
    candidates = torch.zeros(2, time_steps, 4)
    candidates[..., 0] = torch.arange(1.0, time_steps + 1)
    candidates[..., 2] = 1.0
    candidates[1, :, 1] = 0.2
    lanes = torch.zeros(1, 1, time_steps + 1, 8)
    lanes[0, 0, :, 0] = torch.linspace(0.1, 5.0, time_steps + 1)
    lanes[..., 2] = 1.0
    expert = torch.zeros(1, time_steps, 3)
    expert[..., 0] = torch.arange(1.0, time_steps + 1)
    scene_inputs = {
        "ego_current_state": torch.zeros(1, 10),
        "ego_shape": torch.tensor([[2.5, 4.0, 2.0]]),
        "neighbor_agents_past": torch.zeros(1, 1, 2, 11),
        "ego_agent_future": expert,
        "lanes": lanes,
        "route_lanes": lanes.clone(),
        "line_strings": torch.zeros(1, 1, 2, 4),
        "static_objects": torch.zeros(1, 1, 10),
        "turn_indicators": torch.zeros(1, 2, dtype=torch.long),
    }
    neighbors = torch.zeros(1, 1, time_steps, 4)
    return candidates, scene_inputs, neighbors


def _reward_args(**overrides):
    defaults = dict(
        rl_reward_w_risk=1.0,
        rl_reward_w_follow=3.0,
        rl_reward_w_lane=2.5,
        rl_reward_w_progress=3.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_reward_horizon_matches_manually_truncated_inputs(monkeypatch):
    monkeypatch.setattr(hdp_rl_utils, "_RL_GEOMETRY_PAIR_STEP_BUDGET", 1)
    candidates, scene_inputs, neighbors = _reward_scene(time_steps=4)

    prefix_reward, _ = compute_hdp_reward(
        candidates,
        scene_inputs,
        neighbors,
        num_scenes=1,
        n=2,
        args=_reward_args(rl_reward_horizon_steps=2),
    )
    truncated_inputs = dict(scene_inputs)
    truncated_inputs["ego_agent_future"] = scene_inputs["ego_agent_future"][:, :2]
    truncated_reward, _ = compute_hdp_reward(
        candidates[:, :2],
        truncated_inputs,
        neighbors[:, :, :2],
        num_scenes=1,
        n=2,
        args=_reward_args(),
    )
    torch.testing.assert_close(prefix_reward, truncated_reward)


def test_gated_product_reward_is_bounded_and_distinct(monkeypatch):
    monkeypatch.setattr(hdp_rl_utils, "_RL_GEOMETRY_PAIR_STEP_BUDGET", 1)
    candidates, scene_inputs, neighbors = _reward_scene()

    gated_reward, _ = compute_hdp_reward(
        candidates,
        scene_inputs,
        neighbors,
        num_scenes=1,
        n=2,
        args=_reward_args(rl_reward_aggregation="gated_product"),
    )
    additive_reward, _ = compute_hdp_reward(
        candidates, scene_inputs, neighbors, num_scenes=1, n=2, args=_reward_args()
    )
    assert torch.isfinite(gated_reward).all()
    assert ((gated_reward >= 0.0) & (gated_reward <= 1.0)).all()
    # The additive objective with these weights exceeds 1 on a clean scene, so the two
    # aggregations must be genuinely different objectives.
    assert not torch.allclose(gated_reward, additive_reward)


def test_gated_product_requires_positive_quality_weights():
    candidates, scene_inputs, neighbors = _reward_scene()
    with pytest.raises(ValueError, match="gated_product"):
        compute_hdp_reward(
            candidates,
            scene_inputs,
            neighbors,
            num_scenes=1,
            n=2,
            args=_reward_args(
                rl_reward_aggregation="gated_product",
                rl_reward_w_risk=0.0,
                rl_reward_w_follow=0.0,
                rl_reward_w_lane=0.0,
                rl_reward_w_progress=0.0,
            ),
        )


def test_unknown_reward_aggregation_is_rejected():
    candidates, scene_inputs, neighbors = _reward_scene()
    with pytest.raises(ValueError, match="rl_reward_aggregation"):
        compute_hdp_reward(
            candidates,
            scene_inputs,
            neighbors,
            num_scenes=1,
            n=2,
            args=_reward_args(rl_reward_aggregation="pdm"),
        )


# ──────────────── prefix loss horizon and expert anchor scope ───────────────


def _loss_args(**overrides):
    defaults = dict(
        rl_reward_normalize="group",
        rl_reward_beta=1.0,
        advantage_eps=1e-6,
        ddp=False,
        rl_bc_weight=0.0,
        rl_bc_active_groups_only=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_candidate_loss_horizon_follows_reward_horizon(monkeypatch):
    captured_horizons = []

    def fake_policy_loss(_model, _inputs, target, _args, *_pos, loss_horizon=None, **_kwargs):
        captured_horizons.append(loss_horizon)
        per_sample = torch.ones(target.shape[0])
        return {
            "ego_loss_per_sample": per_sample,
            "ego_hdp_diffusion_loss": per_sample.mean(),
            "ego_hdp_waypoint_loss": per_sample.mean(),
        }

    monkeypatch.setattr(hdp_rl_utils, "_compute_policy_ego_loss_per_sample", fake_policy_loss)
    result = compute_reward_weighted_loss(
        model=None,
        norm_inputs={},
        ego_pseudo_gt=torch.zeros(4, 80, 4),
        reward=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        num_scenes=2,
        n=2,
        args=_loss_args(rl_reward_horizon_steps=40),
    )
    assert torch.isfinite(result["loss"])
    assert captured_horizons == [40]


def test_candidate_loss_horizon_explicit_override(monkeypatch):
    captured_horizons = []

    def fake_policy_loss(_model, _inputs, target, _args, *_pos, loss_horizon=None, **_kwargs):
        captured_horizons.append(loss_horizon)
        per_sample = torch.ones(target.shape[0])
        return {
            "ego_loss_per_sample": per_sample,
            "ego_hdp_diffusion_loss": per_sample.mean(),
            "ego_hdp_waypoint_loss": per_sample.mean(),
        }

    monkeypatch.setattr(hdp_rl_utils, "_compute_policy_ego_loss_per_sample", fake_policy_loss)
    compute_reward_weighted_loss(
        model=None,
        norm_inputs={},
        ego_pseudo_gt=torch.zeros(4, 80, 4),
        reward=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        num_scenes=2,
        n=2,
        args=_loss_args(rl_reward_horizon_steps=40, rl_candidate_loss_horizon=20),
    )
    assert captured_horizons == [20]


def test_bc_anchor_active_groups_only(monkeypatch):
    def fake_policy_loss(_model, _inputs, target, _args, *_pos, **_kwargs):
        # Four rows is the candidate pass; anything else is the expert pass, one per scene.
        per_sample = torch.ones(4) if target.shape[0] == 4 else torch.tensor([10.0, 20.0])
        return {
            "ego_loss_per_sample": per_sample,
            "ego_hdp_diffusion_loss": per_sample.mean(),
            "ego_hdp_waypoint_loss": per_sample.mean(),
        }

    monkeypatch.setattr(hdp_rl_utils, "_compute_policy_ego_loss_per_sample", fake_policy_loss)
    # Scene 0 has distinct rewards (active group); scene 1 is tied (discarded group).
    reward = torch.tensor([1.0, 2.0, 5.0, 5.0])
    common = dict(
        model=None,
        norm_inputs={},
        ego_pseudo_gt=torch.zeros(4, 80, 4),
        reward=reward,
        num_scenes=2,
        n=2,
        expert_norm_inputs={},
        expert_ego_gt=torch.zeros(2, 80, 4),
    )

    active_only = compute_reward_weighted_loss(
        args=_loss_args(rl_bc_weight=1.0, rl_bc_active_groups_only=True), **common
    )
    assert active_only["bc_loss"].item() == pytest.approx(10.0)

    broad = compute_reward_weighted_loss(
        args=_loss_args(rl_bc_weight=1.0, rl_bc_active_groups_only=False), **common
    )
    assert broad["bc_loss"].item() == pytest.approx(15.0)


# ───────────────────── restricted diffusion-time sampling ───────────────────


class _StubPolicy:
    """Minimal callable standing in for Diffusion_Planner in the RL loss path."""

    def __init__(self):
        self.captured_times = []

    def __call__(self, merged_inputs):
        self.captured_times.append(merged_inputs["diffusion_time"].clone())
        xT = merged_inputs["sampled_trajectories"]
        return None, {"model_output": torch.zeros_like(xT)}


class _StubNormalizer:
    def ego_velocity_stats_on(self, device, dtype=None):
        return (
            torch.zeros(4, device=device),
            torch.ones(4, device=device),
        )


def test_diffusion_time_draw_respects_configured_range():
    torch.manual_seed(0)
    model = _StubPolicy()
    args = SimpleNamespace(
        use_velocity_representation=True,
        diffusion_model_type="x_start",
        diffusion_supervision_type="x_start",
        diffusion_time_sample_method="uniform",
        state_normalizer=_StubNormalizer(),
        planning_hybrid_loss=0.01,
        hybrid_loss_window=10,
        ego_prediction_horizon=80,
        amp_dtype="off",
        rl_diffusion_t_min=0.001,
        rl_diffusion_t_max=0.2,
    )
    hdp_rl_utils._compute_policy_ego_loss_per_sample(model, {}, torch.zeros(64, 80, 4), args)
    times = model.captured_times[0]
    assert times.shape == (64,)
    assert times.min().item() >= 0.001
    assert times.max().item() <= 0.2


def test_diffusion_time_range_is_validated():
    args = SimpleNamespace(
        use_velocity_representation=True,
        diffusion_model_type="x_start",
        diffusion_supervision_type="x_start",
        diffusion_time_sample_method="uniform",
        state_normalizer=_StubNormalizer(),
        planning_hybrid_loss=0.01,
        hybrid_loss_window=10,
        ego_prediction_horizon=80,
        amp_dtype="off",
        rl_diffusion_t_min=0.5,
        rl_diffusion_t_max=0.5,
    )
    with pytest.raises(ValueError, match="diffusion time range"):
        hdp_rl_utils._compute_policy_ego_loss_per_sample(
            _StubPolicy(), {}, torch.zeros(2, 80, 4), args
        )


def test_loss_horizon_prefix_shortens_reduction():
    model = _StubPolicy()
    args = SimpleNamespace(
        use_velocity_representation=True,
        diffusion_model_type="x_start",
        diffusion_supervision_type="x_start",
        diffusion_time_sample_method="uniform",
        state_normalizer=_StubNormalizer(),
        planning_hybrid_loss=0.0,
        hybrid_loss_window=10,
        ego_prediction_horizon=80,
        amp_dtype="off",
    )
    target = torch.zeros(1, 80, 4)
    target[:, 40:, 0] = 100.0  # error mass only in the tail
    torch.manual_seed(1)
    tail_included = hdp_rl_utils._compute_policy_ego_loss_per_sample(model, {}, target, args)[
        "ego_loss_per_sample"
    ]
    torch.manual_seed(1)
    prefix_only = hdp_rl_utils._compute_policy_ego_loss_per_sample(
        model, {}, target, args, loss_horizon=40
    )["ego_loss_per_sample"]
    assert prefix_only.item() < tail_included.item()


# ─────────────── deterministic deployment metric and selection ──────────────


def test_validation_reports_deterministic_deployment_reward(monkeypatch):
    observed = []

    def fake_sample_group(_model, inputs, noise_scale, *_args, **kwargs):
        observed.append((noise_scale, kwargs["group_size"]))
        batch = inputs["ego_current_state"].shape[0]
        return torch.zeros(batch, 80, 4)

    def fake_reward(_ego, _inputs, _neighbors, num_scenes, n, _args):
        value = 0.5 if n > 1 else 0.9
        return torch.full((num_scenes * n,), value), {
            "reward_risk_score": torch.tensor(value),
        }

    monkeypatch.setattr("diffusion_planner.hdp_rl_epoch.sample_group", fake_sample_group)
    monkeypatch.setattr("diffusion_planner.hdp_rl_epoch.compute_hdp_reward", fake_reward)

    batch = {
        "ego_agent_past": torch.zeros(2, 2, 3),
        "goal_pose": torch.zeros(2, 3),
        "neighbor_agents_future": torch.zeros(2, 1, 80, 3),
        "ego_current_state": torch.zeros(2, 10),
        "route_lanes": torch.zeros(2, 1, 1, 12),
    }
    args = SimpleNamespace(
        num_generations=2,
        device="cpu",
        seed=7,
        observation_normalizer=lambda inputs: inputs,
        rl_eval_noise_scale=0.25,
        rl_eval_num_generations=2,
        amp_dtype="off",
        rl_rollout_steps=6,
        diffusion_sample_steps=5,
        ddp=False,
        rl_eval_deterministic=True,
    )
    metrics = validate_hdp_reward_policy([batch], torch.nn.Linear(1, 1), args)

    assert metrics["mean"].item() == pytest.approx(0.5)
    assert metrics["deterministic_mean"].item() == pytest.approx(0.9)
    assert (0.25, 2) in observed and (0.0, 1) in observed


def test_selection_score_prefers_deterministic_metric():
    metrics = {"mean": torch.tensor(0.5), "deterministic_mean": torch.tensor(0.9)}
    deterministic = selection_score_from_reward_metrics(
        metrics, SimpleNamespace(rl_selection_metric="deterministic")
    )
    legacy = selection_score_from_reward_metrics(
        metrics, SimpleNamespace(rl_selection_metric="mean")
    )
    assert deterministic == pytest.approx(0.9)
    assert legacy == pytest.approx(0.5)
    with pytest.raises(KeyError, match="deterministic_mean"):
        selection_score_from_reward_metrics(
            {"mean": 0.5}, SimpleNamespace(rl_selection_metric="deterministic")
        )


def test_gate_math_uses_radians_conversion():
    # 75° in radians guards against an accidental degree/radian mixup in the port.
    assert math.isclose(math.radians(75.0), 1.3089969389957472)


# ───────────────── velocity-safe rollout-candidate augmentation ──────────────


def _aug_args(**overrides):
    defaults = dict(
        rl_candidate_aug_prob=1.0,
        rl_candidate_aug_std=0.5,
        rl_candidate_aug_eta_scheme="gaussian",
        rl_candidate_aug_beta_concentration=1.6931471805599454,
        rl_candidate_aug_stretch=0.0,
        rl_candidate_aug_ramp_steps=20,
        rl_candidate_aug_keep=1,
        rl_candidate_aug_speed_min_mps=2.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _straight_candidates(num_scenes=1, n=4, future_len=80, step_m=0.5):
    ego = torch.zeros(num_scenes * n, future_len, 4)
    ego[..., 0] = torch.arange(1, future_len + 1, dtype=torch.float32) * step_m
    ego[..., 2] = 1.0
    return ego


def test_quintic_onset_ramp_shape_and_bounds():
    ramp = hdp_rl_utils._quintic_onset_ramp(80, 20, torch.device("cpu"), torch.float32)
    assert ramp.shape == (80,)
    assert ramp[19:].allclose(torch.ones(61))
    assert ramp[0].item() < 0.002  # near-zero onset: first-waypoint gate compatible
    increments = torch.diff(torch.cat([torch.zeros(1), ramp]))
    assert increments.min().item() >= 0.0
    # Minimum-jerk peak slope is 1.875 / ramp_steps.
    assert increments.max().item() <= 1.875 / 20 + 1e-6


def test_candidate_aug_offsets_are_velocity_safe():
    """The core representation claim: ramped offsets never spike per-step deltas.

    Under velocity actions the regression target picks up offset *increments*; the
    onset must bound them, and the first delta (which a constant offset would hit with
    the full 0.5 m ≈ 5 m/s impulse) must stay essentially unchanged.
    """
    from diffusion_planner.loss import waypoints_to_velocity

    torch.manual_seed(0)
    base = _straight_candidates(n=4)
    augmented, metrics = hdp_rl_utils.augment_rollout_candidates(
        base.clone(), torch.tensor([5.0]), 1, 4, _aug_args(rl_candidate_aug_keep=0)
    )
    offsets = (augmented[..., :2] - base[..., :2]).norm(dim=-1)  # [4, 80]
    assert offsets.abs().max().item() > 0.01  # augmentation actually happened
    base_v = waypoints_to_velocity(base)[..., :2]
    aug_v = waypoints_to_velocity(augmented)[..., :2]
    delta_v = (aug_v - base_v).norm(dim=-1)  # per-step increment magnitude [4, 80]
    max_offset = offsets[:, -1]  # saturated total offset per candidate
    # Every per-step increment obeys the quintic slope bound of its own offset.
    assert (delta_v.max(dim=1).values <= max_offset * (1.875 / 20) * math.sqrt(2) + 1e-5).all()
    # The first delta is essentially untouched (a constant offset would put ~100% here).
    assert (delta_v[:, 0] <= max_offset * 0.002 + 1e-6).all()


def test_candidate_aug_stretch_is_exact_velocity_scaling():
    from diffusion_planner.loss import waypoints_to_velocity

    torch.manual_seed(3)
    base = _straight_candidates(n=4)
    augmented, metrics = hdp_rl_utils.augment_rollout_candidates(
        base.clone(),
        torch.tensor([5.0]),
        1,
        4,
        _aug_args(rl_candidate_aug_std=0.0, rl_candidate_aug_stretch=0.25),
    )
    base_v = waypoints_to_velocity(base)[..., :2]
    aug_v = waypoints_to_velocity(augmented)[..., :2]
    ratio = aug_v[..., 0] / base_v[..., 0]  # [4, 80]
    # One scalar per candidate, constant over time, within 1 ± 0.25.
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5)
    assert (ratio >= 0.75 - 1e-5).all() and (ratio <= 1.25 + 1e-5).all()
    # Headings are exact: the path direction is unchanged by a speed-profile scale.
    torch.testing.assert_close(augmented[..., 2:4], base[..., 2:4])
    assert metrics["reward_candidate_aug_stretch_abs_mean"].item() > 0.0


def test_candidate_aug_keeps_anchor_and_skips_low_speed():
    torch.manual_seed(1)
    base = _straight_candidates(num_scenes=2, n=4)
    speeds = torch.tensor([5.0, 0.5])  # scene 1 below the 2.0 m/s guard
    augmented, metrics = hdp_rl_utils.augment_rollout_candidates(
        base.clone(), speeds, 2, 4, _aug_args()
    )
    grouped_base = base.view(2, 4, 80, 4)
    grouped_aug = augmented.view(2, 4, 80, 4)
    torch.testing.assert_close(grouped_aug[0, 0], grouped_base[0, 0])  # anchor
    assert not torch.allclose(grouped_aug[0, 1:], grouped_base[0, 1:])
    torch.testing.assert_close(grouped_aug[1], grouped_base[1])  # low-speed scene
    assert metrics["reward_candidate_aug_scene_fraction"].item() == pytest.approx(0.5)


def test_candidate_aug_disabled_paths_are_noops():
    base = _straight_candidates()
    unchanged, metrics = hdp_rl_utils.augment_rollout_candidates(
        base, torch.tensor([5.0]), 1, 4, _aug_args(rl_candidate_aug_prob=0.0)
    )
    assert unchanged is base
    assert metrics["reward_candidate_aug_scene_fraction"].item() == 0.0
    unchanged, _ = hdp_rl_utils.augment_rollout_candidates(
        base,
        torch.tensor([5.0]),
        1,
        4,
        _aug_args(rl_candidate_aug_std=0.0, rl_candidate_aug_stretch=0.0),
    )
    assert unchanged is base


def test_candidate_aug_rejects_constant_offset():
    base = _straight_candidates()
    with pytest.raises(ValueError, match="first-delta impulse"):
        hdp_rl_utils.augment_rollout_candidates(
            base, torch.tensor([5.0]), 1, 4, _aug_args(rl_candidate_aug_ramp_steps=0)
        )


def test_candidate_aug_is_deterministic_under_generator():
    base = _straight_candidates()
    gen_a = torch.Generator().manual_seed(11)
    gen_b = torch.Generator().manual_seed(11)
    out_a, _ = hdp_rl_utils.augment_rollout_candidates(
        base.clone(), torch.tensor([5.0]), 1, 4, _aug_args(), generator=gen_a
    )
    out_b, _ = hdp_rl_utils.augment_rollout_candidates(
        base.clone(), torch.tensor([5.0]), 1, 4, _aug_args(), generator=gen_b
    )
    torch.testing.assert_close(out_a, out_b)


def test_stratified_beta_covers_every_stratum():
    from scipy import special

    gen = torch.Generator().manual_seed(5)
    eta = hdp_rl_utils._bounded_eta(
        8, 10, "stratified_beta", 1.6931471805599454, torch.device("cpu"), gen
    )
    assert eta.shape == (8, 10)
    assert (eta >= -1.0).all() and (eta <= 1.0).all()
    # Mapping back through the Beta CDF must land one draw per equal-probability
    # stratum in every scene (the PlannerRFT variance-reduction property).
    cdf = special.betainc(1.6931471805599454, 1.6931471805599454, ((eta + 1) / 2).numpy())
    strata = (cdf * 10).astype(int).clip(0, 9)
    for row in strata:
        assert sorted(row.tolist()) == list(range(10))


# ───────────────────────── ported hdp_pdm training reward ────────────────────


def _pdm_reward_scene(time_steps=4):
    """Toy scene on the real 33-channel map schema planner_metrics expects."""
    candidates, scene_inputs, neighbors = _reward_scene(time_steps)
    lanes = torch.zeros(1, 1, time_steps + 1, 33)
    lanes[0, 0, :, 0] = torch.linspace(0.1, 5.0, time_steps + 1)
    lanes[..., 2] = 1.0
    scene_inputs = dict(scene_inputs)
    scene_inputs["lanes"] = lanes
    scene_inputs["route_lanes"] = lanes.clone()
    return candidates, scene_inputs, neighbors


def test_pdm_port_reward_is_bounded_and_shaped(monkeypatch):
    from diffusion_planner.pdm_reward_port import compute_pdm_port_reward

    candidates, scene_inputs, neighbors = _pdm_reward_scene()
    scene_batch = {key: value for key, value in scene_inputs.items()}
    reward, metrics = compute_pdm_port_reward(
        candidates,
        scene_batch,
        neighbors,
        num_scenes=1,
        n=2,
        args=SimpleNamespace(rl_pdm_red_light_gate=True),
    )
    assert reward.shape == (2,)
    assert torch.isfinite(reward).all()
    assert ((reward >= 0.0) & (reward <= 1.0)).all()
    for key in (
        "reward_pdm_ttc_score",
        "reward_pdm_ep_score",
        "reward_pdm_lane_score",
        "reward_pdm_comfort_score",
        "reward_pdm_terminal_gate",
    ):
        assert torch.isfinite(metrics[key]).all()

    ungated, _ = compute_pdm_port_reward(
        candidates,
        scene_batch,
        neighbors,
        num_scenes=1,
        n=2,
        args=SimpleNamespace(rl_pdm_red_light_gate=False),
    )
    # The toy scene has no red signal, so the T4 red-light gate must be neutral.
    torch.testing.assert_close(reward, ungated)


def test_pdm_port_horizon_matches_truncated_inputs():
    from diffusion_planner.pdm_reward_port import compute_pdm_port_reward

    candidates, scene_inputs, neighbors = _pdm_reward_scene(time_steps=4)
    prefix, _ = compute_pdm_port_reward(
        candidates,
        scene_inputs,
        neighbors,
        num_scenes=1,
        n=2,
        args=SimpleNamespace(rl_reward_horizon_steps=2),
    )
    truncated_inputs = dict(scene_inputs)
    truncated_inputs["ego_agent_future"] = scene_inputs["ego_agent_future"][:, :2]
    full_on_truncated, _ = compute_pdm_port_reward(
        candidates[:, :2],
        truncated_inputs,
        neighbors[:, :, :2],
        num_scenes=1,
        n=2,
        args=SimpleNamespace(),
    )
    torch.testing.assert_close(prefix, full_on_truncated)
