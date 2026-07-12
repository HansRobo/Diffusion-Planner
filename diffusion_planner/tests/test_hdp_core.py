import json
import math
import random
from types import SimpleNamespace

import diffusion_planner.hdp_rl_utils as hdp_rl_utils
import diffusion_planner.model.module.decoder as decoder_module
import numpy as np
import pytest
import torch
from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_RED,
)
from diffusion_planner.hdp_rl_epoch import commit_ema_policy_update
from diffusion_planner.hdp_rl_utils import (
    HDPRewardConfig,
    _collision_and_leader_terms,
    _hdp_lane_score,
    _lane_reward_centerlines,
    _occupancy_score,
    _relative_progress_score,
    _road_border_clearance_exact,
    _scene_neighbors,
    _scene_neighbors_batch,
    compute_hdp_reward,
    compute_reward_weighted_loss,
    compute_reward_weights,
    heading_to_cos_sin_if_needed,
)
from diffusion_planner.loss import (
    _detached_integral,
    inverse_normalize_ego_velocity,
    normalize_ego_velocity,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import (
    Decoder,
    GlobalRouteEncoder,
    compute_training_loss,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.train import (
    _finite_validation_metrics,
    assert_checkpoint_compatible,
    load_weights_only,
)
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.train_epoch import prepare_neighbor_supervision
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import (
    BatchAlignedDistributedSampler,
    DiffusionPlannerData,
    DistributedEvalSampler,
    align_legacy_neighbor_futures_on_load,
)
from diffusion_planner.utils.lr_schedule import LinearWarmupConstantLR
from diffusion_planner.utils.masks import neighbor_future_padding_mask
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import (
    build_dummy_inputs,
    build_dynamic_axes,
)
from diffusion_planner.utils.train_utils import (
    ModelEma,
    atomic_torch_save,
    capture_rng_state,
    compile_model_components,
    compute_grad_stats,
    finalize_epoch_loss_sums,
    resume_model,
    update_epoch_loss_sums,
)
from diffusion_planner.validate_model import _multisample_metrics, aggregate_valid_metrics


def test_hdp_representation_and_normalization_round_trip():
    torch.manual_seed(7)
    xy = torch.randn(3, 80, 2).cumsum(dim=-2)
    yaw = torch.randn(3, 80)
    waypoints = torch.cat((xy, yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)), dim=-1)

    velocity = waypoints_to_velocity(waypoints)
    reconstructed = velocity_to_waypoints(velocity)
    torch.testing.assert_close(reconstructed, waypoints, rtol=0, atol=2e-6)

    normalizer = StateNormalizer(
        [[[10.0, 0.0, 0.0, 0.0]]],
        [[[20.0, 20.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
    )
    normalized = normalize_ego_velocity(velocity, normalizer)
    torch.testing.assert_close(
        inverse_normalize_ego_velocity(normalized, normalizer), velocity, rtol=0, atol=0
    )


def test_batched_neighbor_collision_terms_match_scene_loop():
    torch.manual_seed(19)
    batch_size, candidates, neighbor_slots, horizon = 3, 2, 5, 8
    ego = torch.randn(batch_size, candidates, horizon, 4)
    ego[..., 2] = 1.0
    ego[..., 3] = 0.0
    ego_shape = torch.tensor([[2.7, 4.8, 1.9]]).expand(batch_size, -1).clone()
    future = torch.zeros(batch_size, neighbor_slots, horizon, 4)
    past = torch.zeros(batch_size, neighbor_slots, 4, 11)
    for scene, count in enumerate((0, 2, 4)):
        future[scene, :count, :, :2] = torch.randn(count, horizon, 2) + 8.0
        future[scene, :count, :, 2] = 1.0
        past[scene, :count, -1, :2] = future[scene, :count, 0, :2]
        past[scene, :count, -1, 2] = 1.0
        past[scene, :count, -1, 6:8] = torch.tensor([2.0, 4.5])
        past[scene, :count, -1, 8] = 1.0
    reference = torch.randn(batch_size, horizon, 4)
    reference[..., 2] = 1.0
    reference[..., 3] = 0.0
    config = HDPRewardConfig()

    packed = _scene_neighbors_batch(future, past)

    def batched_fn(ego_scene, shape, nf, ns, nv, ni, vehicle, ref):
        return _collision_and_leader_terms(
            ego_scene, shape, nf, ns, nv, ni, vehicle, config, leader_reference=ref
        )

    actual = torch.vmap(batched_fn)(ego, ego_shape, *packed, reference)
    expected = []
    for scene in range(batch_size):
        scene_neighbors = _scene_neighbors(future[scene], past[scene])
        expected.append(
            _collision_and_leader_terms(
                ego[scene],
                ego_shape[scene],
                *scene_neighbors,
                config,
                leader_reference=reference[scene],
            )
        )

    for key, actual_value in actual.items():
        expected_value = torch.stack([scene[key] for scene in expected])
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=1e-6)


def test_batched_lane_and_progress_match_scene_loop():
    torch.manual_seed(23)
    batch_size, candidates, horizon = 4, 3, 8
    ego = torch.randn(batch_size, candidates, horizon, 4)
    ego[..., 2] = 1.0
    ego[..., 3] = 0.0
    expert = torch.zeros(batch_size, horizon, 4)
    expert[..., 0] = torch.linspace(0.5, 8.0, horizon)
    expert[..., 2] = 1.0
    lanes = torch.zeros(batch_size, 3, 6, 12)
    route = torch.zeros(batch_size, 2, 6, 12)
    x = torch.linspace(0.0, 10.0, 6)
    for scene in (0, 1, 3):
        lanes[scene, 0, :, 0] = x
        lanes[scene, 1, :, 0] = x
        lanes[scene, 1, :, 1] = 3.5
    route[0, 0, :, 0] = x
    route[1, 0, :, 0] = x
    route[1, 0, :, 1] = 20.0
    route[3, 0, :, 0] = x
    expert[1, :, 1] = torch.linspace(0.0, 3.5, horizon)
    indicators = torch.zeros(batch_size, 5, dtype=torch.long)
    indicators[1, -1] = 2
    config = HDPRewardConfig()

    actual = hdp_rl_utils._batched_lane_and_progress(ego, expert, lanes, route, indicators, config)
    expected_lane = []
    expected_off_lane = []
    expected_lane_change = []
    expected_route = []
    expected_ratio = []
    expected_progress = []
    expected_available = []
    for scene in range(batch_size):
        use_route = hdp_rl_utils._route_alignment(expert[scene], route[scene], config)
        centerlines, _ = _lane_reward_centerlines(
            expert[scene], lanes[scene], route[scene], config, route_aligned=bool(use_route)
        )
        lane_valid = lanes[scene, ..., :2].abs().sum(dim=-1) > 1e-6
        has_lanes = (lane_valid[..., :-1] & lane_valid[..., 1:]).any()
        lane, off_lane, lane_change = _hdp_lane_score(
            ego[scene],
            expert[scene],
            centerlines,
            indicators[scene],
            config,
            lanes_available=bool(use_route | has_lanes),
        )
        ratio, progress = _relative_progress_score(ego[scene], expert[scene])
        expected_lane.append(lane)
        expected_off_lane.append(off_lane)
        expected_lane_change.append(lane_change)
        expected_route.append(use_route)
        expected_ratio.append(ratio)
        expected_progress.append(progress)
        expected_available.append(has_lanes)

    expected = (
        torch.stack(expected_lane),
        torch.stack(expected_off_lane),
        torch.stack(expected_lane_change),
        torch.stack(expected_route),
        torch.stack(expected_ratio),
        torch.stack(expected_progress),
        torch.stack(expected_available),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=1e-6)


def test_rollout_generator_is_independent_from_global_training_rng():
    class EchoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(_predicted_neighbor_num=0, _sample_steps=6)

        def forward(self, inputs):
            return None, {"prediction": inputs["sampled_trajectories"]}

    model = EchoModel()
    inputs = {"ego_current_state": torch.zeros(2, 10)}
    first_generator = torch.Generator().manual_seed(123)
    first = hdp_rl_utils.sample_group(
        model,
        inputs,
        1.5,
        torch.device("cpu"),
        generator=first_generator,
    )
    torch.randn(10_000)
    second_generator = torch.Generator().manual_seed(123)
    second = hdp_rl_utils.sample_group(
        model,
        inputs,
        1.5,
        torch.device("cpu"),
        generator=second_generator,
    )
    different = hdp_rl_utils.sample_group(
        model,
        inputs,
        1.5,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(124),
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, different)


def test_state_normalizer_caches_stats_by_kind_device_and_dtype():
    normalizer = StateNormalizer(
        [[[10, 0, 0, 0]]],
        [[[20, 20, 1, 1]]],
        [0, 0, 0, 0],
        [0.5, 0.5, 1, 1],
    )

    state_fp32 = normalizer._mean_std_on(torch.device("cpu"), torch.float32)
    velocity_fp32 = normalizer.ego_velocity_stats_on(torch.device("cpu"), torch.float32)
    state_bf16 = normalizer._mean_std_on(torch.device("cpu"), torch.bfloat16)

    assert state_fp32[0] is normalizer._mean_std_on(torch.device("cpu"), torch.float32)[0]
    assert (
        velocity_fp32[0] is normalizer.ego_velocity_stats_on(torch.device("cpu"), torch.float32)[0]
    )
    assert state_fp32[0] is not velocity_fp32[0]
    assert state_fp32[0].dtype == torch.float32
    assert state_bf16[0].dtype == torch.bfloat16
    data = torch.tensor([[[[0.5, -0.25, 1.0, 0.0]]]], dtype=torch.float32)
    expected = (data - torch.tensor([0.0, 0.0, 0.0, 0.0])) / torch.tensor([0.5, 0.5, 1.0, 1.0])
    torch.testing.assert_close(normalize_ego_velocity(data, normalizer), expected)


def test_observation_normalizer_preserves_zero_padding_in_both_directions():
    normalizer = ObservationNormalizer(
        {"feature": {"mean": torch.tensor([2.0, -3.0]), "std": torch.tensor([2.0, 4.0])}}
    )
    data = {"feature": torch.tensor([[[0.0, 0.0], [4.0, 5.0]]])}

    normalized = normalizer(data)
    restored = normalizer.inverse(normalized)

    torch.testing.assert_close(normalized["feature"][0, 0], torch.zeros(2))
    torch.testing.assert_close(normalized["feature"][0, 1], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(restored["feature"], data["feature"])


def test_compile_model_components_preserves_checkpoint_keys():
    class TinyPlanner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)

    model = TinyPlanner()
    keys_before = tuple(model.state_dict())

    compile_model_components(model)

    assert tuple(model.state_dict()) == keys_before
    assert model.encoder._compiled_call_impl is not None
    assert model.decoder._compiled_call_impl is not None


def test_hdp_dit_self_attention_runs_over_future_timesteps():
    model = DiT(depth=1, output_dim=4, hidden_dim=32, heads=4, future_len=80).eval()
    observed = []
    handle = model.blocks[0].attn.register_forward_pre_hook(
        lambda _module, args: observed.append(tuple(args[0].shape))
    )
    with torch.no_grad():
        output = model(
            torch.randn(2, 1, 80, 4),
            torch.rand(2),
            torch.randn(2, 7, 32),
            torch.randn(2, 2),
            torch.randn(2, 32),
        )
    handle.remove()

    assert observed == [(2, 80, 32)]
    assert output.shape == (2, 1, 80, 4)


def test_hdp_global_route_encoder_is_batch_safe_and_zero_for_missing_route():
    encoder = GlobalRouteEncoder(
        route_num=3,
        route_len=5,
        hidden_dim=32,
        drop_path_rate=0.0,
    ).eval()
    routes = torch.zeros(2, 3, 5, 13)
    routes[1, :, :, :4] = torch.randn(3, 5, 4)

    with torch.no_grad():
        condition = encoder(routes)

    assert condition.shape == (2, 32)
    torch.testing.assert_close(condition[0], torch.zeros(32))
    assert torch.count_nonzero(condition[1]).item() > 0


def test_global_route_construction_does_not_shift_policy_initialization(monkeypatch):
    args = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
        hidden_dim=32,
        decoder_depth=1,
    )
    args.state_normalizer = StateNormalizer(
        [[[10.0, 0.0, 0.0, 0.0]]],
        [[[20.0, 20.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer({})

    torch.manual_seed(3407)
    reference = Decoder(args)

    class EntropyConsumingRoute(torch.nn.Module):
        def __init__(self, *unused_args, **unused_kwargs):
            super().__init__()
            self.register_buffer("random_values", torch.randn(10_000))

    monkeypatch.setattr(decoder_module, "GlobalRouteEncoder", EntropyConsumingRoute)
    torch.manual_seed(3407)
    perturbed_route = Decoder(args)

    for key, value in reference.dit.state_dict().items():
        torch.testing.assert_close(value, perturbed_route.dit.state_dict()[key], rtol=0, atol=0)
    for key, value in reference.turn_indicator_predictor.state_dict().items():
        torch.testing.assert_close(
            value, perturbed_route.turn_indicator_predictor.state_dict()[key], rtol=0, atol=0
        )


def test_hdp_decoder_repeats_only_global_route_condition_for_rl_groups():
    args = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
        hidden_dim=32,
        decoder_depth=1,
    )
    args.state_normalizer = StateNormalizer(
        [[[10.0, 0.0, 0.0, 0.0]]],
        [[[20.0, 20.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer({})
    decoder = Decoder(args).train()
    scene_batch, group_size = 2, 3
    candidate_batch = scene_batch * group_size
    encoding = torch.randn(candidate_batch, 7, 32)
    route_lanes = torch.randn(scene_batch, 25, 20, 33)
    inputs = {
        "route_lanes": route_lanes,
        "_global_route_repeat_interleave": group_size,
        "sampled_trajectories": torch.randn(candidate_batch, 1, 80, 4),
        "diffusion_time": torch.rand(candidate_batch),
        "ego_current_state": torch.randn(candidate_batch, 10),
        "_skip_turn_indicator": True,
    }

    output = decoder(encoding, inputs)["model_output"]
    assert output.shape == (candidate_batch, 1, 80, 4)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None for parameter in decoder.global_route_encoder.parameters()
    )


def test_hdp_decoder_skips_frozen_turn_head_without_changing_inference_prediction(monkeypatch):
    args = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
        hidden_dim=32,
        decoder_depth=1,
        diffusion_sample_steps=2,
    )
    args.state_normalizer = StateNormalizer(
        [[[10.0, 0.0, 0.0, 0.0]]],
        [[[20.0, 20.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer({})
    decoder = Decoder(args).eval()
    encoding = torch.randn(2, 7, 32)
    inputs = {
        "route_lanes": torch.randn(2, 25, 20, 33),
        "sampled_trajectories": torch.randn(2, 1, 80, 4),
        "ego_current_state": torch.randn(2, 10),
    }

    with torch.no_grad():
        full = decoder(encoding, inputs)

    def fail_pool(_encoding):
        raise AssertionError("RL-only inference must not pool the expanded scene encoding")

    monkeypatch.setattr(decoder, "_pool_encoding", fail_pool)
    with torch.no_grad():
        skipped = decoder(encoding, {**inputs, "_skip_turn_indicator": True})

    assert "turn_indicator_logit" in full
    assert "turn_indicator_logit" not in skipped
    torch.testing.assert_close(skipped["prediction"], full["prediction"], rtol=0, atol=0)


def test_hdp_temporal_position_initialization_matches_navsim_frequency_base():
    args = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
        hidden_dim=32,
        decoder_depth=1,
    )
    args.state_normalizer = StateNormalizer(
        [[[10.0, 0.0, 0.0, 0.0]]],
        [[[20.0, 20.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer({})
    position = Decoder(args).dit.action_pos_emb.detach()[0]
    expected_frequency = torch.exp(-torch.log(torch.tensor(100.0)) * torch.arange(0, 32, 2) / 32)
    torch.testing.assert_close(position[1, 0::2], torch.sin(expected_frequency))
    torch.testing.assert_close(position[1, 1::2], torch.cos(expected_frequency))


def test_dpm_uses_one_diffusion_time_for_the_full_trajectory():
    observed_times = []

    def noise_model(x, t):
        observed_times.append(t.detach().clone())
        return torch.zeros_like(x)

    solver = DPM_Solver(noise_model, NoiseScheduleVP())
    solver.sample(
        torch.randn(1, 1, 4, 4),
        steps=2,
        skip_type="logSNR",
    )

    assert observed_times
    for time in observed_times:
        assert time.shape == (1,)


def test_supervised_lr_warms_up_then_stays_fixed_for_twenty_epochs():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = LinearWarmupConstantLR(optimizer, total_epochs=20, warm_up_epochs=5)

    used_lrs = []
    for _ in range(20):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert used_lrs[:5] == pytest.approx([2e-5, 6.5e-5, 1.1e-4, 1.55e-4, 2e-4])
    assert used_lrs[5:] == pytest.approx([2e-4] * 15)


def test_detached_integral_preserves_forward_and_limits_gradient_window():
    velocity = torch.arange(1, 7, dtype=torch.float64).view(1, 6, 1).requires_grad_()
    integrated = _detached_integral(velocity, W=3)

    torch.testing.assert_close(integrated, torch.cumsum(velocity, dim=-2), rtol=0, atol=0)
    integrated.sum().backward()
    expected = torch.tensor([3.0, 3.0, 3.0, 3.0, 2.0, 1.0], dtype=torch.float64).view(1, 6, 1)
    torch.testing.assert_close(velocity.grad, expected, rtol=0, atol=0)


def test_reward_weights_discard_identical_and_nonfinite_groups():
    reward = torch.tensor([0.4, 0.4, 0.4, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, float("nan"), 0.3, 0.4])
    weights, valid = compute_reward_weights(
        reward, num_scenes=3, n=4, normalize="group", beta=1.0, eps=1e-6
    )

    assert not valid[:4].any()
    assert valid[4:8].all()
    assert not valid[8:].any()
    torch.testing.assert_close(weights[:4], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(weights[8:], torch.zeros(4), rtol=0, atol=0)
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("normalize", ["batch", "none"])
def test_reward_weight_ablations_still_discard_groups_without_preferences(normalize):
    reward = torch.tensor([0.4, 0.4, 0.1, 0.3])
    weights, valid = compute_reward_weights(
        reward, num_scenes=2, n=2, normalize=normalize, beta=1.0, eps=1e-6
    )

    assert not valid[:2].any()
    assert valid[2:].all()
    torch.testing.assert_close(weights[:2], torch.zeros(2), rtol=0, atol=0)


def test_hdp_behavior_cloning_anchor_uses_one_expert_target_per_scene(monkeypatch):
    def fake_policy_loss(_model, _inputs, target, _args, _encoding=None):
        per_sample = (
            torch.tensor([1.0, 2.0, 3.0, 4.0])
            if target.shape[0] == 4
            else torch.tensor([10.0, 20.0])
        )
        return {
            "ego_loss_per_sample": per_sample,
            "ego_hdp_diffusion_loss": per_sample.mean(),
            "ego_hdp_waypoint_loss": per_sample.mean(),
        }

    monkeypatch.setattr(hdp_rl_utils, "_compute_policy_ego_loss_per_sample", fake_policy_loss)
    args = SimpleNamespace(
        rl_reward_normalize="group",
        rl_reward_beta=1.0,
        advantage_eps=1e-6,
        rl_bc_weight=0.25,
        ddp=False,
    )
    output = compute_reward_weighted_loss(
        None,
        {},
        torch.zeros(4, 80, 4),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        2,
        2,
        args,
        reward_weights=torch.ones(4),
        valid_sample=torch.ones(4, dtype=torch.bool),
        global_valid_count=torch.tensor(4.0),
        ddp_world_size=1,
        expert_norm_inputs={},
        expert_ego_gt=torch.zeros(2, 80, 4),
    )

    torch.testing.assert_close(output["rl_loss"], torch.tensor(2.5))
    torch.testing.assert_close(output["bc_loss"], torch.tensor(15.0))
    torch.testing.assert_close(output["loss"], torch.tensor(6.25))


def test_rl_weight_diagnostics_exclude_discarded_groups(monkeypatch):
    def fake_policy_loss(_model, _inputs, _target, _args, _encoding=None):
        per_sample = torch.ones(4)
        return {
            "ego_loss_per_sample": per_sample,
            "ego_hdp_diffusion_loss": per_sample.mean(),
            "ego_hdp_waypoint_loss": per_sample.mean(),
        }

    monkeypatch.setattr(hdp_rl_utils, "_compute_policy_ego_loss_per_sample", fake_policy_loss)
    args = SimpleNamespace(rl_bc_weight=0.0, ddp=False)
    output = compute_reward_weighted_loss(
        None,
        {},
        torch.zeros(4, 80, 4),
        torch.zeros(4),
        2,
        2,
        args,
        reward_weights=torch.tensor([0.0, 0.0, 1.0, 3.0]),
        valid_sample=torch.tensor([False, False, True, True]),
        global_valid_count=torch.tensor(2.0),
        ddp_world_size=1,
    )

    torch.testing.assert_close(output["reward_weight_mean"], torch.tensor(2.0))
    torch.testing.assert_close(output["reward_weight_max"], torch.tensor(3.0))
    torch.testing.assert_close(output["reward_weight_min"], torch.tensor(1.0))


def test_legacy_short_neighbor_future_alignment_keeps_full_tracks():
    past = np.zeros((2, 3, 11), dtype=np.float32)
    past[:, -1, 2] = 1.0
    past[0, -1, 0] = 5.0
    past[1, -1, 0] = 10.0

    future = np.zeros((2, 4, 3), dtype=np.float32)
    # Legacy short track: current frame is duplicated, followed by t+0.1 and t+0.2.
    future[0, :3, 0] = np.array([5.0, 6.0, 7.0])
    # Full tracks were already correct because the converter deque evicted its seed.
    future[1, :, 0] = np.array([11.0, 12.0, 13.0, 14.0])

    data = {"neighbor_agents_future": future, "neighbor_agents_past": past}
    align_legacy_neighbor_futures_on_load(data)

    np.testing.assert_allclose(data["neighbor_agents_future"][0, :, 0], [6.0, 7.0, 0.0, 0.0])
    np.testing.assert_allclose(data["neighbor_agents_future"][1], future[1])
    # The source array loaded from the shared NPZ remains untouched.
    np.testing.assert_allclose(future[0, :, 0], [5.0, 6.0, 7.0, 0.0])


def test_extra_dataset_weighting_stays_in_memory(tmp_path):
    base_list = tmp_path / "base.json"
    extra_list = tmp_path / "extra.json"
    base_list.write_text(json.dumps(["base-a.npz", "base-b.npz"]), encoding="utf-8")
    extra_list.write_text(json.dumps(["extra.npz"]), encoding="utf-8")

    dataset = DiffusionPlannerData(
        str(base_list),
        extra_data_list=str(extra_list),
        extra_data_repeat=3,
    )
    assert dataset.data_list == [
        "base-a.npz",
        "base-b.npz",
        "extra.npz",
        "extra.npz",
        "extra.npz",
    ]


def test_multiple_extra_dataset_lists_share_the_requested_repeat(tmp_path):
    base_list = tmp_path / "base.json"
    extra_lists = [tmp_path / f"extra-{index}.json" for index in range(3)]
    base_list.write_text(json.dumps(["base.npz"]), encoding="utf-8")
    extra_lists[0].write_text(json.dumps(["x2.npz"]), encoding="utf-8")
    extra_lists[1].write_text(json.dumps(["xx1.npz", "xx1-b.npz"]), encoding="utf-8")
    extra_lists[2].write_text(json.dumps(["psim.npz"]), encoding="utf-8")

    dataset = DiffusionPlannerData(
        str(base_list),
        extra_data_list=[str(path) for path in extra_lists],
        extra_data_repeat=2,
    )
    assert dataset.data_list == [
        "base.npz",
        "x2.npz",
        "xx1.npz",
        "xx1-b.npz",
        "psim.npz",
        "x2.npz",
        "xx1.npz",
        "xx1-b.npz",
        "psim.npz",
    ]


def test_extra_dataset_traffic_light_mask_is_in_memory_and_extra_only(tmp_path):
    lanes = np.zeros((1, 2, 33), dtype=np.float32)
    route_lanes = np.zeros((1, 2, 33), dtype=np.float32)
    lanes[0, :, 0] = [1.0, 2.0]
    route_lanes[0, :, 1] = [1.0, 2.0]
    lanes[..., TRAFFIC_LIGHT_RED] = 1.0
    route_lanes[..., TRAFFIC_LIGHT_GREEN] = 1.0

    base_npz = tmp_path / "base.npz"
    extra_npz = tmp_path / "extra.npz"
    for path in (base_npz, extra_npz):
        np.savez(path, lanes=lanes, route_lanes=route_lanes)

    base_list = tmp_path / "base.json"
    extra_list = tmp_path / "extra.json"
    base_list.write_text(json.dumps([str(base_npz)]), encoding="utf-8")
    extra_list.write_text(json.dumps([str(extra_npz)]), encoding="utf-8")
    dataset = DiffusionPlannerData(
        str(base_list),
        extra_data_list=str(extra_list),
        extra_data_repeat=1,
        extra_data_mask_traffic_lights=True,
    )

    base_sample = dataset[0]
    extra_sample = dataset[1]
    assert np.all(base_sample["lanes"][..., TRAFFIC_LIGHT_RED] == 1.0)
    assert np.all(base_sample["route_lanes"][..., TRAFFIC_LIGHT_GREEN] == 1.0)
    assert np.all(extra_sample["lanes"][..., TRAFFIC_LIGHT_RED] == 0.0)
    assert np.all(extra_sample["route_lanes"][..., TRAFFIC_LIGHT_GREEN] == 0.0)
    assert np.all(extra_sample["lanes"][..., TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT] == 1.0)
    assert np.all(extra_sample["route_lanes"][..., TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT] == 1.0)

    # Loading the shared source again must recover the original traffic-light labels.
    with np.load(extra_npz) as source:
        assert np.all(source["lanes"][..., TRAFFIC_LIGHT_RED] == 1.0)
        assert np.all(source["route_lanes"][..., TRAFFIC_LIGHT_GREEN] == 1.0)


def test_extra_dataset_mask_tracks_occurrence_when_base_and_extra_overlap(tmp_path):
    lanes = np.zeros((1, 2, 33), dtype=np.float32)
    lanes[0, :, 0] = [1.0, 2.0]
    lanes[..., TRAFFIC_LIGHT_RED] = 1.0
    sample = tmp_path / "overlap.npz"
    np.savez(sample, lanes=lanes)
    base_list = tmp_path / "base.json"
    extra_list = tmp_path / "extra.json"
    base_list.write_text(json.dumps([str(sample)]), encoding="utf-8")
    extra_list.write_text(json.dumps([str(sample)]), encoding="utf-8")

    dataset = DiffusionPlannerData(
        str(base_list),
        extra_data_list=str(extra_list),
        extra_data_repeat=2,
        extra_data_mask_traffic_lights=True,
    )
    assert dataset[0]["lanes"][..., TRAFFIC_LIGHT_RED].all()
    assert not dataset[1]["lanes"][..., TRAFFIC_LIGHT_RED].any()
    assert not dataset[2]["lanes"][..., TRAFFIC_LIGHT_RED].any()

    dataset.subsample(2)
    assert dataset.data_list == [str(sample), str(sample)]
    assert dataset[0]["lanes"][..., TRAFFIC_LIGHT_RED].all()
    assert not dataset[1]["lanes"][..., TRAFFIC_LIGHT_RED].any()


def test_ego_only_dataset_skips_unused_neighbor_future_npz_payload(tmp_path):
    sample_path = tmp_path / "sample.npz"
    np.savez(
        sample_path,
        ego_agent_future=np.zeros((80, 3), dtype=np.float32),
        neighbor_agents_future=np.ones((320, 80, 3), dtype=np.float32),
    )
    data_list = tmp_path / "data.json"
    data_list.write_text(json.dumps([str(sample_path)]), encoding="utf-8")

    ego_only = DiffusionPlannerData(str(data_list), include_neighbor_futures=False)[0]
    joint = DiffusionPlannerData(str(data_list), include_neighbor_futures=True)[0]

    assert "neighbor_agents_future" not in ego_only
    assert joint["neighbor_agents_future"].shape == (320, 80, 3)


def test_ego_only_neighbor_supervision_skips_unused_full_future_conversion():
    raw = torch.randn(2, 320, 80, 3)
    action_future, action_mask, collision_futures = prepare_neighbor_supervision(
        raw,
        action_neighbor_num=0,
        include_collision_futures=False,
    )

    assert action_future.shape == (2, 0, 80, 4)
    assert action_mask.shape == (2, 0, 80)
    assert collision_futures is None
    assert action_future.data_ptr() != raw.data_ptr()


def test_distributed_eval_sampler_has_no_duplicates_or_padding():
    shards = [
        list(DistributedEvalSampler(range(10), num_replicas=3, rank=rank)) for rank in range(3)
    ]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_batch_aligned_distributed_sampler_uses_all_data_with_minimal_padding():
    samplers = [
        BatchAlignedDistributedSampler(
            range(10), num_replicas=3, rank=rank, local_batch_size=2, seed=17
        )
        for rank in range(3)
    ]
    shards = [list(sampler) for sampler in samplers]
    flattened = [index for shard in shards for index in shard]

    assert all(len(shard) == 4 and len(shard) % 2 == 0 for shard in shards)
    assert set(flattened) == set(range(10))
    assert len(flattened) == 12
    assert all(sampler.padding_size == 2 for sampler in samplers)


def test_batch_aligned_distributed_sampler_reshuffles_padding_each_epoch():
    sampler = BatchAlignedDistributedSampler(
        range(10), num_replicas=1, rank=0, local_batch_size=4, seed=17
    )
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    assert set(epoch_zero) == set(epoch_one) == set(range(10))
    assert len(epoch_zero) == len(epoch_one) == 12


def test_checkpoint_metadata_omits_non_finite_validation_metrics():
    metrics = _finite_validation_metrics(
        "valid_epdms",
        {"total": torch.tensor(0.8), "missing": float("nan"), "overflow": float("inf")},
    )

    assert metrics == {"valid_epdms/total": pytest.approx(0.8)}
    json.dumps(metrics, allow_nan=False)


def test_augmentation_transforms_goal_pose_with_the_scene():
    augmentor = StatePerturbation(
        augment_prob=0.0,
        num_refine=3,
        device="cpu",
        ego_past_noise_std=0.0,
        use_smoothing_future_trajectory=False,
    )
    inputs = {
        "ego_current_state": torch.tensor([[10.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        "ego_agent_past": torch.zeros(1, 1, 4),
        "neighbor_agents_past": torch.zeros(1, 0, 1, 11),
        "lanes": torch.zeros(1, 1, 2, 8),
        "route_lanes": torch.zeros(1, 1, 2, 8),
        "polygons": torch.zeros(1, 1, 2, 3),
        "line_strings": torch.zeros(1, 1, 2, 4),
        "static_objects": torch.zeros(1, 1, 10),
        "goal_pose": torch.tensor([[10.0, 3.0, 0.0, 1.0]]),
    }
    ego_future = torch.tensor([[[10.0, 3.0, torch.pi / 2]]])
    neighbors_future = torch.zeros(1, 0, 1, 3)

    transformed, _, _ = augmentor.centric_transform(inputs, ego_future, neighbors_future)
    torch.testing.assert_close(
        transformed["goal_pose"], torch.tensor([[1.0, 0.0, 1.0, 0.0]]), atol=1e-6, rtol=0
    )


def _collision_terms_for_neighbor_x(neighbor_x: float):
    T = 4
    ego = torch.zeros(1, T, 4)
    ego[..., 2] = 1.0
    neighbor = torch.zeros(1, T, 4)
    neighbor[..., 0] = neighbor_x
    neighbor[..., 2] = 1.0
    return _collision_and_leader_terms(
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        neighbor,
        torch.tensor([[2.0, 4.0]]),
        torch.ones(1, T, dtype=torch.bool),
        torch.tensor([[neighbor_x, 0.0]]),
        torch.tensor([True]),
        HDPRewardConfig(),
    )


def test_hdp_empty_neighbor_reward_preserves_full_metric_contract():
    time_steps = 4
    ego = torch.zeros(2, time_steps, 4)
    ego[..., 0] = torch.arange(1, time_steps + 1)
    ego[..., 2] = 1.0
    terms = _collision_and_leader_terms(
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        torch.zeros(0, time_steps, 4),
        torch.zeros(0, 2),
        torch.zeros(0, time_steps, dtype=torch.bool),
        torch.zeros(0, 2),
        torch.zeros(0, dtype=torch.bool),
        HDPRewardConfig(),
    )

    assert "comfort" in terms
    assert terms["comfort"].shape == (2,)
    assert torch.isfinite(terms["comfort"]).all()
    torch.testing.assert_close(terms["follow"], torch.ones(2))


def test_hdp_reward_full_contract_reports_finite_component_diagnostics():
    time_steps = 4
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
    reward, metrics = compute_hdp_reward(
        candidates,
        scene_inputs,
        neighbors,
        num_scenes=1,
        n=2,
        args=SimpleNamespace(
            rl_reward_w_risk=1.0,
            rl_reward_w_follow=3.0,
            rl_reward_w_lane=2.5,
            rl_reward_w_progress=3.0,
        ),
    )

    assert reward.shape == (2,)
    assert torch.isfinite(reward).all()
    assert {
        "reward_risk_correlation",
        "reward_follow_correlation",
        "reward_lane_correlation",
        "reward_progress_correlation",
        "reward_winner_risk_advantage",
        "reward_risk_group_std",
        "reward_progress_group_range",
        "reward_leader_group_range",
    } <= metrics.keys()
    assert all(torch.isfinite(value) for value in metrics.values())


def test_hdp_collision_reward_attenuates_rear_end_only():
    active = _collision_terms_for_neighbor_x(3.0)
    rear = _collision_terms_for_neighbor_x(-1.0)

    torch.testing.assert_close(active["safety"], torch.tensor([0.0]))
    torch.testing.assert_close(rear["safety"], torch.tensor([0.7]))
    assert active["collision_active"].item() == 1.0
    assert rear["collision_rear"].item() == 1.0


def test_hdp_stopped_occupancy_preserves_rear_end_attenuation():
    config = HDPRewardConfig()
    ego = torch.zeros(1, 4, 4)
    ego[..., 2] = 1.0
    rear = _collision_terms_for_neighbor_x(-1.0)
    active = _collision_terms_for_neighbor_x(3.0)

    rear_occupancy, _ = _occupancy_score(
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        None,
        None,
        rear["stopped_clearance"],
        rear["stopped_available"],
        config,
        stopped_is_rear=rear["stopped_is_rear"],
    )
    active_occupancy, _ = _occupancy_score(
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        None,
        None,
        active["stopped_clearance"],
        active["stopped_available"],
        config,
        stopped_is_rear=active["stopped_is_rear"],
    )

    torch.testing.assert_close(rear_occupancy, torch.full_like(rear_occupancy, 0.7))
    torch.testing.assert_close(active_occupancy, torch.zeros_like(active_occupancy))


def test_hdp_static_occupancy_is_not_attenuated_by_rear_stopped_vehicle():
    config = HDPRewardConfig()
    ego = torch.zeros(1, 4, 4)
    ego[..., 2] = 1.0
    rear = _collision_terms_for_neighbor_x(-1.0)
    static_object = torch.tensor([[0.0, 0.0, 1.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0]])

    occupancy, sources = _occupancy_score(
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        static_object,
        None,
        rear["stopped_clearance"],
        rear["stopped_available"],
        config,
        stopped_is_rear=rear["stopped_is_rear"],
    )

    assert sources["static"] and sources["stopped"]
    torch.testing.assert_close(occupancy, torch.zeros_like(occupancy))


def test_hdp_neighbor_at_ego_origin_remains_valid():
    future = torch.zeros(1, 3, 4)
    future[..., 2] = 1.0
    past = torch.zeros(1, 2, 11)
    past[:, -1, 2] = 1.0
    past[:, -1, 6:8] = torch.tensor([2.0, 4.0])
    past[:, -1, 8] = 1.0

    parsed, _, valid, _, _ = _scene_neighbors(future, past)
    assert parsed.shape[0] == 1
    assert valid.all()

    raw = torch.zeros(1, 3, 3)
    raw[:, 0, 0] = 1.0
    converted = heading_to_cos_sin_if_needed(raw)
    torch.testing.assert_close(converted[..., :2], raw[..., :2])
    torch.testing.assert_close(converted[..., 2], torch.ones_like(converted[..., 2]))
    torch.testing.assert_close(converted[..., 3], torch.zeros_like(converted[..., 3]))


def test_hdp_stationary_ego_future_is_not_treated_as_padding():
    converted = heading_to_cos_sin_if_needed(torch.zeros(2, 80, 3))

    assert converted[..., 2].eq(1.0).all()
    assert converted[..., 3].eq(0.0).all()


def test_neighbor_future_mask_preserves_internal_zero_pose():
    future = torch.zeros(1, 1, 5, 3)
    future[0, 0, 0, 0] = 1.0
    # t=1 is a real origin pose; t=2 proves the track continues after it.
    future[0, 0, 2, 0] = -1.0
    mask = neighbor_future_padding_mask(future)
    assert mask.tolist() == [[[False, False, False, True, True]]]


def test_hdp_following_ignores_non_vehicle_leader():
    T = 4
    ego = torch.zeros(1, T, 4)
    ego[..., 0] = torch.arange(1, T + 1)
    ego[..., 2] = 1.0
    neighbor = torch.zeros(1, T, 4)
    neighbor[..., 0] = torch.arange(4, T + 4)
    neighbor[..., 2] = 1.0
    terms = _collision_and_leader_terms(
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        neighbor,
        torch.tensor([[0.8, 0.8]]),
        torch.ones(1, T, dtype=torch.bool),
        torch.tensor([[3.0, 0.0]]),
        torch.tensor([False]),
        HDPRewardConfig(),
    )

    torch.testing.assert_close(terms["leader_fraction"], torch.tensor([0.0]))
    torch.testing.assert_close(terms["follow"], torch.tensor([1.0]))


def test_hdp_following_uses_reference_leader_instead_of_candidate_escape():
    time_steps = 4
    ego = torch.zeros(2, time_steps, 4)
    ego[..., 0] = torch.arange(1.0, time_steps + 1)
    ego[..., 2] = 1.0
    ego[1, :, 1] = 4.0
    neighbor = torch.zeros(1, time_steps, 4)
    neighbor[..., 0] = torch.arange(7.0, time_steps + 7.0)
    neighbor[..., 2] = 1.0
    common_args = (
        ego,
        torch.tensor([2.5, 4.0, 2.0]),
        neighbor,
        torch.tensor([[2.0, 4.0]]),
        torch.ones(1, time_steps, dtype=torch.bool),
        torch.tensor([[6.0, 0.0]]),
        torch.tensor([True]),
        HDPRewardConfig(),
    )

    candidate_association = _collision_and_leader_terms(*common_args)
    reference_association = _collision_and_leader_terms(
        *common_args,
        leader_reference=ego[0],
    )

    assert candidate_association["leader_fraction"].tolist() == [1.0, 0.0]
    torch.testing.assert_close(
        reference_association["leader_fraction"],
        torch.ones(2),
    )
    assert candidate_association["follow"][1].item() == pytest.approx(1.0)
    assert reference_association["follow"][1] < candidate_association["follow"][1]


def test_hdp_lane_change_mask_and_neutral_occupancy_fallback():
    lanes = torch.zeros(1, 20, 8)
    lanes[0, :, 0] = torch.linspace(0.1, 20.0, 20)
    lanes[0, :, 2] = 1.0
    prediction = torch.zeros(2, 8, 4)
    prediction[..., 0] = torch.linspace(0.1, 8.0, 8)
    prediction[..., 2] = 1.0
    expert = prediction[0].clone()
    expert[:, 1] = torch.linspace(0.1, 3.0, 8)

    lane_score, _, lane_change = _hdp_lane_score(
        prediction,
        expert,
        lanes,
        torch.tensor([0, 2]),
        HDPRewardConfig(),
    )
    assert lane_change
    torch.testing.assert_close(lane_score, torch.zeros(2))

    occupancy, sources = _occupancy_score(
        prediction,
        torch.tensor([2.5, 4.0, 2.0]),
        None,
        None,
        torch.full((2, 8), float("inf")),
        torch.tensor(False),
        HDPRewardConfig(),
    )
    torch.testing.assert_close(occupancy, torch.ones_like(occupancy))
    assert not any(sources.values())


def test_hdp_road_border_occupancy_fallback_is_speed_independent():
    prediction = torch.zeros(2, 8, 4)
    prediction[..., 2] = 1.0
    prediction[1, :, 0] = torch.linspace(0.0, 5.0, 8)
    line_strings = torch.zeros(1, 2, 4)
    line_strings[0, :, 0] = torch.tensor([-10.0, 10.0])
    line_strings[0, :, 1] = 1.3
    line_strings[0, :, 3] = 1.0

    occupancy, sources = _occupancy_score(
        prediction,
        torch.tensor([2.5, 4.0, 2.0]),
        None,
        line_strings,
        torch.full((2, 8), float("inf")),
        torch.tensor(False),
        HDPRewardConfig(),
    )

    assert sources == {"static": False, "stopped": False, "road_border": True}
    torch.testing.assert_close(occupancy[0], occupancy[1], atol=1e-5, rtol=0.0)
    torch.testing.assert_close(occupancy.mean(), torch.tensor(0.5), atol=1e-4, rtol=0.0)


def test_hdp_road_border_clearance_is_exact_for_gap_crossing_and_containment():
    prediction = torch.zeros(1, 1, 4)
    prediction[..., 2] = 1.0
    # ego_shape=(wheelbase=2, length=4, width=2) occupies x=[-1, 3], y=[-1, 1].
    segment_pairs = (
        ([[5.0, -4.0], [5.0, 4.0]], 2.0),
        ([[2.0, -4.0], [2.0, 4.0]], 0.0),
        ([[0.0, -0.2], [0.0, 0.2]], 0.0),
        ([[0.0, 0.0], [0.0, 4.0]], 0.0),
    )
    for segment_pair, expected in segment_pairs:
        line_strings = torch.zeros(1, 2, 4)
        line_strings[0, :, :2] = torch.tensor(segment_pair)
        line_strings[..., 3] = 1.0
        clearance = _road_border_clearance_exact(
            prediction,
            torch.tensor([2.0, 4.0, 2.0]),
            line_strings,
        )
        torch.testing.assert_close(clearance, torch.tensor([[expected]]))


def test_hdp_invalid_road_border_is_not_reported_as_occupancy_source():
    prediction = torch.zeros(1, 8, 4)
    prediction[..., 2] = 1.0
    line_strings = torch.zeros(1, 2, 4)

    occupancy, sources = _occupancy_score(
        prediction,
        torch.tensor([2.5, 4.0, 2.0]),
        None,
        line_strings,
        torch.full((1, 8), float("inf")),
        torch.tensor(False),
        HDPRewardConfig(),
    )

    torch.testing.assert_close(occupancy, torch.ones_like(occupancy))
    assert not any(sources.values())


def test_hdp_relative_progress_is_capped_and_stopped_expert_is_neutral():
    expert = torch.zeros(4, 4)
    expert[:, 0] = torch.arange(1.0, 5.0)
    expert[:, 2] = 1.0
    candidates = expert.unsqueeze(0).repeat(3, 1, 1)
    candidates[0, :, 0] *= 0.5
    candidates[2, :, 0] *= 1.5

    ratio, score = _relative_progress_score(candidates, expert)

    torch.testing.assert_close(ratio, torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(score, torch.tensor([0.5, 1.0, 1.0]))

    stopped = torch.zeros_like(expert)
    stopped[:, 2] = 1.0
    ratio, score = _relative_progress_score(candidates, stopped)
    torch.testing.assert_close(ratio, torch.ones(3))
    torch.testing.assert_close(score, torch.ones(3))


def test_hdp_lane_reward_does_not_mask_a_curved_centerline():
    theta = torch.linspace(0.0, 0.15, 20)
    radius = 80.0
    lanes = torch.zeros(1, 20, 8)
    lanes[0, :, 0] = radius * torch.sin(theta)
    lanes[0, :, 1] = radius * (1.0 - torch.cos(theta))
    lanes[0, :, 2] = torch.cos(theta)
    lanes[0, :, 3] = torch.sin(theta)
    expert = lanes[0, 1:17, :4].clone()

    lane_score, off_lane, lane_change = _hdp_lane_score(
        expert.unsqueeze(0),
        expert,
        lanes,
        torch.tensor([0, 0]),
        HDPRewardConfig(),
    )

    assert not off_lane
    assert not lane_change
    assert lane_score.item() > 0.99


def test_hdp_lane_reward_masks_centerline_to_centerline_change_without_indicator():
    x = torch.linspace(0.1, 20.0, 20)
    lanes = torch.zeros(2, 20, 8)
    lanes[:, :, 0] = x
    lanes[1, :, 1] = 3.5
    lanes[..., 2] = 1.0
    expert = torch.zeros(20, 4)
    expert[:, 0] = x
    expert[:, 1] = 1.75 * (1.0 - torch.cos(torch.linspace(0.0, torch.pi, 20)))
    expert[:, 2] = 1.0

    lane_score, _, lane_change = _hdp_lane_score(
        expert.unsqueeze(0),
        expert,
        lanes,
        torch.tensor([0, 0]),
        HDPRewardConfig(),
    )

    assert lane_change
    torch.testing.assert_close(lane_score, torch.zeros(1))


def test_hdp_lane_reward_prefers_aligned_route_and_falls_back_when_route_is_wrong():
    x = torch.linspace(0.1, 20.0, 20)
    lanes = torch.zeros(2, 20, 8)
    lanes[:, :, 0] = x
    lanes[1, :, 1] = 3.5
    lanes[..., 2] = 1.0
    expert = lanes[0, :, :4].clone()

    aligned_route = lanes[:1].clone()
    selected, route_used = _lane_reward_centerlines(expert, lanes, aligned_route, HDPRewardConfig())
    assert route_used
    torch.testing.assert_close(selected, aligned_route)

    wrong_route = aligned_route.clone()
    wrong_route[..., 1] = 20.0
    selected, route_used = _lane_reward_centerlines(expert, lanes, wrong_route, HDPRewardConfig())
    assert not route_used
    torch.testing.assert_close(selected, lanes)


def test_seeded_multisample_metrics_compute_minade_and_minfde():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Identity()
            self.decoder._sample_steps = 10
            self.decoder._predicted_neighbor_num = 0
            self.decoder.global_route_encoder = lambda route: route.new_zeros((route.shape[0], 4))

        def forward(self, inputs):
            B = inputs["sampled_trajectories"].shape[0]
            prediction = torch.zeros(B, 1, 80, 4)
            prediction[..., 2] = 1.0
            return inputs["_cached_encoding"], {"prediction": prediction}

    args = type(
        "Args",
        (),
        {
            "multisample_eval_num_samples": 6,
            "multisample_eval_noise_scale": 0.1,
            "multisample_eval_seed": 7,
            "multisample_eval_sample_steps": 6,
        },
    )()
    gt = torch.zeros(1, 80, 4)
    gt[..., 0] = 1.0
    gt[..., 2] = 1.0
    metrics = _multisample_metrics(
        FakeModel(),
        {
            "ego_current_state": torch.zeros(1, 10),
            "route_lanes": torch.zeros(1, 1, 1, 4),
        },
        torch.zeros(1, 2, 4),
        gt,
        args,
        0,
    )

    torch.testing.assert_close(metrics["multisample_minADE"], torch.tensor([1.0]))
    torch.testing.assert_close(metrics["multisample_minFDE"], torch.tensor([1.0]))


def test_multisample_metrics_include_valid_stationary_scenes():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Identity()
            self.decoder._sample_steps = 6
            self.decoder._predicted_neighbor_num = 0
            self.decoder.global_route_encoder = lambda route: route.new_zeros((route.shape[0], 4))

        def forward(self, inputs):
            batch = inputs["sampled_trajectories"].shape[0]
            prediction = torch.zeros(batch, 1, 80, 4)
            prediction[..., 2] = 1.0
            return inputs["_cached_encoding"], {"prediction": prediction}

    args = type(
        "Args",
        (),
        {
            "multisample_eval_num_samples": 2,
            "multisample_eval_noise_scale": 0.1,
            "multisample_eval_seed": 7,
            "multisample_eval_sample_steps": 6,
            "amp_dtype": "off",
        },
    )()
    gt = torch.zeros(2, 80, 4)
    gt[1, :, 0] = 1.0
    gt[1, :, 2] = 1.0
    metrics = _multisample_metrics(
        FakeModel(),
        {
            "ego_current_state": torch.zeros(2, 10),
            "route_lanes": torch.zeros(2, 1, 1, 4),
        },
        torch.zeros(2, 2, 4),
        gt,
        args,
        0,
    )

    assert all(value.shape == (2,) for value in metrics.values())
    torch.testing.assert_close(metrics["multisample_minADE"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(metrics["multisample_minFDE"], torch.tensor([0.0, 1.0]))


def test_turn_indicator_class_metrics_keep_counts_visible():
    metrics = aggregate_valid_metrics(
        {
            "_loss_ego_sum": 0.0,
            "_samples_ego": 1,
            "_loss_neighbor_sum": 0.0,
            "_samples_neighbor": 0,
            "_turn_correct": 3,
            "_turn_total": 5,
            "_turn_change_correct": 2,
            "turn_indicator_change_total": 4,
            "_turn_class_correct": [0, 0, 1, 1, 1],
            "_turn_class_total": [0, 1, 1, 2, 1],
        },
        "cpu",
    )

    assert metrics["turn_indicator_class_accuracy"] == {
        "none": 0.0,
        "disable": 0.0,
        "enable_left": 1.0,
        "enable_right": 0.5,
        "keep": 1.0,
    }
    assert metrics["turn_indicator_class_count"]["none"] == 0
    assert metrics["turn_indicator_class_count"]["enable_right"] == 2


def test_empty_multisample_metrics_aggregate_to_nan_not_false_zero():
    metrics = aggregate_valid_metrics(
        {
            "_loss_ego_sum": 0.0,
            "_samples_ego": 1,
            "_loss_neighbor_sum": 0.0,
            "_samples_neighbor": 0,
            "_turn_correct": 0,
            "_turn_total": 0,
            "_turn_change_correct": 0,
            "turn_indicator_change_total": 0,
            "_turn_class_correct": [0, 0, 0, 0, 0],
            "_turn_class_total": [0, 0, 0, 0, 0],
            "multisample_minADE": torch.empty(0),
        },
        "cpu",
    )

    assert math.isnan(metrics["multisample_means"]["minADE"])


def test_epdms_metric_without_availability_excludes_nonfinite_values():
    metrics = aggregate_valid_metrics(
        {
            "_loss_ego_sum": 0.0,
            "_samples_ego": 1,
            "_loss_neighbor_sum": 0.0,
            "_samples_neighbor": 0,
            "_turn_correct": 0,
            "_turn_total": 0,
            "_turn_change_correct": 0,
            "turn_indicator_change_total": 0,
            "_turn_class_correct": [0, 0, 0, 0, 0],
            "_turn_class_total": [0, 0, 0, 0, 0],
            "epdms_unmasked_metric": torch.tensor([0.25, float("nan"), 0.75]),
        },
        "cpu",
    )

    assert metrics["epdms_means"]["unmasked_metric"] == pytest.approx(0.5)


def test_ego_only_supervised_loss_and_onnx_shapes():
    class FakeObservationNormalizer:
        @staticmethod
        def stats(key):
            assert key == "ego_current_state"
            return torch.zeros(10), torch.ones(10)

        @staticmethod
        def inverse(data):
            assert set(data) == {"line_strings"}
            return data

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sde = VPSDE_linear()

        def forward(self, inputs):
            B = inputs["gt_trajectories"].shape[0]
            expert_logit = torch.zeros(B, 5)
            expert_logit[:, 4] = 10.0
            return {}, {
                "model_output": inputs["gt_trajectories"],
                "turn_indicator_logit": torch.zeros(B, 5),
                "turn_indicator_expert_logit": expert_logit,
            }

    args = type(
        "Args",
        (),
        {
            "state_normalizer": StateNormalizer(
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 1.0, 1.0]]],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ),
            "observation_normalizer": FakeObservationNormalizer(),
            "diffusion_model_type": "x_start",
            "diffusion_supervision_type": "x_start",
            "diffusion_time_sample_method": "uniform",
            "use_velocity_representation": True,
            "hybrid_loss_window": 10,
            "planning_hybrid_loss": 0.01,
            # CPU tests the guard that must keep CUDA autocast disabled even
            # when a production config requests bf16.
            "amp_dtype": "bf16",
            "ego_prediction_horizon": 80,
            "coeff_road_border_loss": 1.0,
            "coeff_neighbor_collision_loss": 0.0,
            "road_border_margin": 0.25,
            "road_border_n_interp": 2,
            "turn_indicator_generated_loss_weight": 1.0,
            "turn_indicator_expert_loss_weight": 1.0,
        },
    )()
    B, T = 2, 80
    ego_future = torch.zeros(B, T, 4)
    ego_future[..., 0] = torch.linspace(0.1, 8.0, T)
    ego_future[..., 2] = 1.0
    neighbors_future = torch.zeros(B, 0, T, 4)
    neighbor_mask = torch.zeros(B, 0, T, dtype=torch.bool)
    inputs = {
        "ego_current_state": torch.zeros(B, 10),
        "neighbor_agents_past": torch.zeros(B, 0, 1, 11),
        "line_strings": torch.zeros(B, 1, 2, 4),
        "ego_shape": torch.tensor([[2.8, 4.8, 1.8]]).expand(B, -1),
        "turn_indicators": torch.ones(B, 31),
    }
    loss = compute_training_loss(
        FakeModel(),
        inputs,
        (ego_future, neighbors_future, neighbor_mask),
        args,
    )
    assert torch.isfinite(loss["ego_planning_loss"])
    assert loss["turn_indicator_accuracy"].item() == 0.0
    assert loss["turn_indicator_generated_accuracy"].item() == 0.0
    assert loss["turn_indicator_expert_accuracy"].item() == 1.0
    torch.testing.assert_close(
        loss["turn_indicator_loss"],
        (loss["turn_indicator_generated_loss"] + loss["turn_indicator_expert_loss"]) / 2,
    )

    dummy = build_dummy_inputs(action_agent_num=1)
    assert dummy["sampled_trajectories"].shape == (1, 1, 80, 4)
    assert build_dynamic_axes(["sampled_trajectories"], ["prediction"])["sampled_trajectories"] == {
        0: "batch"
    }


def test_rl_weights_only_init_prefers_sft_ema(tmp_path):
    model = torch.nn.Linear(2, 1)
    live = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    ema = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "sft.pth"
    torch.save({"model": live, "ema_state_dict": ema}, checkpoint)

    load_weights_only(str(checkpoint), model, "cpu", prefer_ema=True)

    for value in model.state_dict().values():
        torch.testing.assert_close(value, torch.ones_like(value))


def test_rl_ema_rate_005_maps_to_timm_decay_095():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ModelEma(model, decay=0.95, device="cpu")
    assert ema.foreach
    assert ema.ema is ema.module
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    torch.testing.assert_close(ema.ema.weight, torch.full_like(ema.ema.weight, 0.05))


def test_rl_policy_commit_updates_once_syncs_live_and_clears_optimizer_state():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ModelEma(model, decay=0.95, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    assert optimizer.state
    proposal = model.weight.detach().clone()

    relative_l2 = commit_ema_policy_update(model, ema, optimizer, use_ddp=False)

    expected = proposal * 0.05
    torch.testing.assert_close(model.weight, expected)
    torch.testing.assert_close(ema.ema.weight, expected)
    assert relative_l2 > 0
    assert not optimizer.state


def test_rl_policy_commit_drift_ignores_frozen_parameters():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1, bias=False),
        torch.nn.Linear(1, 1, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(1000.0)
        model[1].weight.fill_(2.0)
    model[0].weight.requires_grad_(False)
    ema = ModelEma(model, decay=0.95, device="cpu")
    optimizer = torch.optim.AdamW([model[1].weight], lr=0.1)
    with torch.no_grad():
        model[1].weight.fill_(3.0)

    relative_l2 = commit_ema_policy_update(model, ema, optimizer, use_ddp=False)

    # Only the trainable policy changed: ||3 - 2|| / ||2|| = 0.5. The much larger
    # frozen parameter must not dilute the monitoring signal.
    torch.testing.assert_close(relative_l2, torch.tensor(0.5))


def _checkpoint_compat_config(predicted_neighbor_num: int):
    args = TrainConfig(
        exp_name="test",
        save_dir="unused",
        train_set_list="unused",
        valid_set_list="unused",
        train_subsample_step=1,
        predicted_neighbor_num=predicted_neighbor_num,
    )
    args.state_normalizer = StateNormalizer(
        [[[0.0, 0.0, 0.0, 0.0]]] * (1 + predicted_neighbor_num),
        [[[1.0, 1.0, 1.0, 1.0]]] * (1 + predicted_neighbor_num),
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer(
        {"ego_current_state": {"mean": torch.zeros(10), "std": torch.ones(10)}}
    )
    return args


def test_checkpoint_compatibility_is_strict_for_resume_but_allows_weights_only(tmp_path):
    checkpoint_args = _checkpoint_compat_config(predicted_neighbor_num=1)
    current_args = _checkpoint_compat_config(predicted_neighbor_num=0)
    serializable = {
        key: value.to_dict()
        if isinstance(value, (StateNormalizer, ObservationNormalizer))
        else value
        for key, value in vars(checkpoint_args).items()
    }
    (tmp_path / "args.json").write_text(json.dumps(serializable), encoding="utf-8")
    checkpoint_path = tmp_path / "latest.pth"

    with pytest.raises(RuntimeError, match="action shape mismatch"):
        assert_checkpoint_compatible(str(checkpoint_path), current_args)
    assert_checkpoint_compatible(
        str(checkpoint_path),
        current_args,
        allow_predicted_neighbor_change=True,
        strict_training_config=False,
    )

    same_shape = _checkpoint_compat_config(predicted_neighbor_num=1)
    same_shape.turn_indicator_generated_loss_weight = 0.25
    with pytest.raises(RuntimeError, match="training configuration mismatch"):
        assert_checkpoint_compatible(str(checkpoint_path), same_shape)

    shorter_horizon = _checkpoint_compat_config(predicted_neighbor_num=1)
    shorter_horizon.train_epochs = 20
    assert_checkpoint_compatible(str(checkpoint_path), shorter_horizon)


def test_atomic_checkpoint_save_and_resume_restore_global_step(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint_path = tmp_path / "latest.pth"
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    rng_state = capture_rng_state()
    assert not any(
        torch.is_tensor(value)
        for group in rng_state.values()
        for value in (group.values() if isinstance(group, dict) else (group,))
    )
    expected_random = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(())
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": scheduler.state_dict(),
            "epoch": 3,
            "global_step": 47,
            "wandb_id": "resume-run-id",
            "rng_states": [rng_state],
        },
        checkpoint_path,
    )

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored, _, _, epoch, wandb_id, _ = resume_model(
        str(checkpoint_path),
        restored,
        restored_optimizer,
        restored_scheduler,
        None,
        "cpu",
        strict_training_state=True,
    )

    assert epoch == 3
    assert wandb_id == "resume-run-id"
    assert random.random() == expected_random
    assert np.random.rand() == expected_numpy
    torch.testing.assert_close(torch.rand(()), expected_torch)
    assert restored._resume_global_step == 47
    assert not list(tmp_path.glob(".latest.pth.tmp.*"))


def test_strict_resume_rejects_checkpoint_without_rng_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": scheduler.state_dict(),
            "epoch": 1,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="no per-rank RNG state"):
        resume_model(
            str(checkpoint_path),
            model,
            optimizer,
            scheduler,
            None,
            "cpu",
            strict_training_state=True,
        )


def test_epoch_metric_accumulator_uses_per_metric_counts():
    sums = {}
    counts = {}
    update_epoch_loss_sums(sums, counts, {"loss": torch.tensor(2.0)})
    update_epoch_loss_sums(
        sums,
        counts,
        {"loss": torch.tensor(4.0), "grad/l2_norm": 12.0},
    )

    means = finalize_epoch_loss_sums(sums, counts)

    assert means == {"loss": 3.0, "grad/l2_norm": 12.0}


def test_rear_end_attenuation_does_not_hide_other_risk_sources():
    active = _collision_terms_for_neighbor_x(3.0)
    rear = _collision_terms_for_neighbor_x(-1.0)

    torch.testing.assert_close(active["ttc"].amin(), torch.tensor(0.0))
    torch.testing.assert_close(rear["ttc"].amin(), torch.tensor(0.7))
    occupancy = torch.full_like(rear["ttc"], 0.2)
    risk = torch.stack([rear["ttc"], rear["thw"], occupancy]).amin()
    torch.testing.assert_close(risk, torch.tensor(0.2))


def test_ddp_epoch_reducer_accepts_python_floats(monkeypatch):
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: tensor.mul_(2))

    reduced = ddp.reduce_and_average_losses(
        {"python": 3.5, "tensor": torch.tensor(2.0)}, torch.device("cpu")
    )

    assert reduced == {"python": 3.5, "tensor": 2.0}


def test_streaming_gradient_stats_match_concatenated_reference():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(3))
    first.grad = torch.tensor([1.0, -2.0])
    second.grad = torch.tensor([3.0, -4.0, 5.0])
    reference = torch.cat([first.grad, second.grad])

    stats = compute_grad_stats([first, second])

    assert stats["grad/l1_norm"] == reference.abs().sum().item()
    assert stats["grad/l2_norm"] == pytest.approx(reference.norm().item())
    assert stats["grad/linf_norm"] == reference.abs().max().item()
    assert stats["grad/mean"] == pytest.approx(reference.mean().item())
    assert stats["grad/std"] == pytest.approx(reference.std().item())
