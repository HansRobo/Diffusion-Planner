import json

import numpy as np
import pytest
import torch
from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_RED,
)
from diffusion_planner.hdp_rl_utils import (
    HDPRewardConfig,
    _collision_and_leader_terms,
    _hdp_lane_score,
    _occupancy_score,
    _scene_neighbors,
    compute_reward_weights,
    heading_to_cos_sin_if_needed,
)
from diffusion_planner.loss import (
    _detached_integral,
    clamp_known_prefix,
    inverse_normalize_ego_velocity,
    normalize_ego_velocity,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.train import assert_checkpoint_compatible, load_weights_only
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.train_epoch import prepare_neighbor_supervision
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import (
    DiffusionPlannerData,
    DistributedEvalSampler,
    align_legacy_neighbor_futures_on_load,
)
from diffusion_planner.utils.masks import neighbor_future_padding_mask
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import (
    build_decoder_inputs,
    build_dummy_inputs,
    build_dynamic_axes,
)
from diffusion_planner.utils.train_utils import atomic_torch_save, resume_model
from diffusion_planner.validate_model import _multisample_metrics, aggregate_valid_metrics
from timm.utils import ModelEma


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


def test_detached_integral_preserves_forward_and_limits_gradient_window():
    velocity = torch.arange(1, 7, dtype=torch.float64).view(1, 6, 1).requires_grad_()
    integrated = _detached_integral(velocity, W=3)

    torch.testing.assert_close(integrated, torch.cumsum(velocity, dim=-2), rtol=0, atol=0)
    integrated.sum().backward()
    expected = torch.tensor([3.0, 3.0, 3.0, 3.0, 2.0, 1.0], dtype=torch.float64).view(1, 6, 1)
    torch.testing.assert_close(velocity.grad, expected, rtol=0, atol=0)


def test_known_delay_prefix_is_clamped_and_has_no_prediction_gradient():
    prediction = torch.arange(6.0).reshape(1, 3, 2).requires_grad_()
    target = torch.full_like(prediction, 10.0)
    clamped = clamp_known_prefix(prediction, target, torch.tensor([[True, False, False]]))
    clamped.sum().backward()

    torch.testing.assert_close(clamped[:, 0], target[:, 0])
    torch.testing.assert_close(prediction.grad[:, 0], torch.zeros_like(prediction.grad[:, 0]))
    torch.testing.assert_close(prediction.grad[:, 1:], torch.ones_like(prediction.grad[:, 1:]))


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


def test_augmentation_transforms_goal_pose_with_the_scene():
    augmentor = StatePerturbation(
        augment_prob=0.0,
        num_refine=2,
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


def test_hdp_collision_reward_attenuates_rear_end_only():
    active = _collision_terms_for_neighbor_x(3.0)
    rear = _collision_terms_for_neighbor_x(-1.0)

    torch.testing.assert_close(active["safety"], torch.tensor([0.0]))
    torch.testing.assert_close(rear["safety"], torch.tensor([0.7]))
    assert active["collision_active"].item() == 1.0
    assert rear["collision_rear"].item() == 1.0


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
    torch.testing.assert_close(converted[:, 1:], torch.zeros_like(converted[:, 1:]))


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


def test_seeded_multisample_metrics_compute_minade_and_minfde():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Identity()
            self.decoder._sample_steps = 10
            self.decoder._predicted_neighbor_num = 320

        def forward(self, inputs):
            B = inputs["sampled_trajectories"].shape[0]
            prediction = torch.zeros(B, 321, 80, 4)
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
    metrics = _multisample_metrics(FakeModel(), {}, torch.zeros(1, 2, 4), gt, args, 0)

    torch.testing.assert_close(metrics["multisample_minADE"], torch.tensor([1.0]))
    torch.testing.assert_close(metrics["multisample_minFDE"], torch.tensor([1.0]))


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
            "amp_dtype": "off",
            "coeff_position_lat_loss": 1.0,
            "coeff_position_lon_loss": 1.0,
            "coeff_heading_l2_loss": 1.0,
            "coeff_velocity": 0.05,
            "coeff_timestep": [1.0, 1.0, 1.0, 1.0],
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
    assert loss["neighbor_prediction_loss"].item() == 0.0
    assert loss["turn_indicator_accuracy"].item() == 0.0
    assert loss["turn_indicator_generated_accuracy"].item() == 0.0
    assert loss["turn_indicator_expert_accuracy"].item() == 1.0
    torch.testing.assert_close(
        loss["turn_indicator_loss"],
        (loss["turn_indicator_generated_loss"] + loss["turn_indicator_expert_loss"]) / 2,
    )

    dummy = build_dummy_inputs(action_agent_num=1)
    decoder_inputs = build_decoder_inputs(dummy, torch.zeros(1, 4, 8))
    assert dummy["sampled_trajectories"].shape == (1, 1, 81, 4)
    assert decoder_inputs["diffusion_time"].shape == (1, 1, 81, 1)
    assert build_dynamic_axes(["sampled_trajectories", "delay"], ["prediction"])["delay"] == {
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
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    torch.testing.assert_close(ema.ema.weight, torch.full_like(ema.ema.weight, 0.05))


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

def test_atomic_checkpoint_save_and_resume_restore_global_step(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint_path = tmp_path / "latest.pth"
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": scheduler.state_dict(),
            "epoch": 3,
            "global_step": 47,
        },
        checkpoint_path,
    )

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    restored, _, _, epoch, _, _ = resume_model(
        str(checkpoint_path),
        restored,
        restored_optimizer,
        restored_scheduler,
        None,
        "cpu",
        strict_training_state=True,
    )

    assert epoch == 3
    assert restored._resume_global_step == 47
    assert not list(tmp_path.glob(".latest.pth.tmp.*"))
