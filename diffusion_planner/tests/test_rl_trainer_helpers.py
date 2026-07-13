from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.hdp_rl_epoch import (
    _backward_reward_weighted_update,
    _policy_observation_inputs,
    _reward_eval_scene_chunk_size,
    _rl_update_scene_chunk_size,
    _slice_grouped_policy_inputs,
    _step_metrics_for_logging,
    validate_hdp_reward_policy,
)
from train_hdp_rl_predictor import (
    best_valid_score_from_rows,
    configure_rl_trainable_parameters,
    find_checkpoint_run_artifact,
    finite_scalar_metrics,
    initialize_fresh_rl_policy_and_ema,
    load_resume_train_rows,
    source_policy_selection_guards,
    validate_compiled_candidate_batch,
    write_train_log_atomic,
)


def test_all_scope_does_not_expand_reward_only_futures():
    observations = {
        "ego_current_state": torch.zeros(2, 10),
        "lanes": torch.zeros(2, 4, 3),
        "ego_agent_future": torch.zeros(2, 80, 4),
        "neighbor_agents_future": torch.zeros(2, 320, 80, 3),
    }

    policy_inputs = _policy_observation_inputs(observations)

    assert set(policy_inputs) == {"ego_current_state", "lanes"}


class _TinyPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Module()
        self.decoder.dit = torch.nn.Linear(1, 1)
        self.decoder.global_route_encoder = torch.nn.Linear(1, 1)
        self.decoder.turn_indicator_predictor = torch.nn.Linear(1, 1)


def test_all_scope_freezes_only_the_unused_turn_head():
    model = _TinyPlanner()
    configure_rl_trainable_parameters(model, "all")
    state = {name: param.requires_grad for name, param in model.named_parameters()}

    assert all(value for name, value in state.items() if name.startswith("encoder."))
    assert all(value for name, value in state.items() if name.startswith("decoder.dit."))
    assert all(
        value for name, value in state.items() if name.startswith("decoder.global_route_encoder.")
    )
    assert not any(
        value
        for name, value in state.items()
        if name.startswith("decoder.turn_indicator_predictor.")
    )


def test_decoder_scope_freezes_encoder_and_turn_head():
    model = _TinyPlanner()
    configure_rl_trainable_parameters(model, "decoder")
    state = {name: param.requires_grad for name, param in model.named_parameters()}

    assert not any(value for name, value in state.items() if name.startswith("encoder."))
    assert all(value for name, value in state.items() if name.startswith("decoder.dit."))
    assert all(
        value for name, value in state.items() if name.startswith("decoder.global_route_encoder.")
    )
    assert not any(
        value
        for name, value in state.items()
        if name.startswith("decoder.turn_indicator_predictor.")
    )


def test_fresh_rl_initializes_live_and_ema_from_the_same_checkpoint(tmp_path):
    source = _TinyPlanner()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))
    checkpoint = tmp_path / "sft.pth"
    torch.save({"model": source.state_dict(), "ema_state_dict": source.state_dict()}, checkpoint)

    live = _TinyPlanner()
    policy_ema = initialize_fresh_rl_policy_and_ema(
        str(checkpoint),
        live,
        "cpu",
        use_ddp=False,
        prefer_checkpoint_ema=True,
        ema_update_rate=0.05,
    )

    for expected, live_value, shadow_value in zip(
        source.state_dict().values(),
        live.state_dict().values(),
        policy_ema.ema.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(live_value, expected)
        torch.testing.assert_close(shadow_value, expected)


def test_resume_train_log_uses_checkpoint_epoch_as_commit_boundary(tmp_path):
    path = tmp_path / "train_log.tsv"
    pd_rows = [
        {"epoch": 1, "valid_reward_mean": 1.0},
        {"epoch": 2, "valid_reward_mean": 2.0},
        {"epoch": 3, "valid_reward_mean": 3.0},
    ]
    write_train_log_atomic(pd_rows, str(path))

    rows = load_resume_train_rows(path, init_epoch=2)

    assert [row["epoch"] for row in rows] == [1, 2]
    assert not list(tmp_path.glob(".train_log.tsv.tmp.*"))


def test_resume_train_log_rejects_missing_checkpointed_epoch(tmp_path):
    path = tmp_path / "train_log.tsv"
    write_train_log_atomic([{"epoch": 1}, {"epoch": 3}], str(path))

    with pytest.raises(ValueError, match="one contiguous log row"):
        load_resume_train_rows(path, init_epoch=3)


def test_compiled_rl_candidate_batch_rejects_corrupted_h100_shape():
    assert validate_compiled_candidate_batch(64, 32, True) == 2048
    assert validate_compiled_candidate_batch(128, 32, False) == 4096
    with pytest.raises(ValueError, match="silently corrupted backward gradients"):
        validate_compiled_candidate_batch(128, 32, True)


def test_reward_validation_caps_candidate_batch_without_dropping_scenes():
    assert _reward_eval_scene_chunk_size(64, 32) == 32
    assert _reward_eval_scene_chunk_size(17, 32) == 17
    assert _reward_eval_scene_chunk_size(64, 2048) == 1


def test_step_logging_uses_ddp_global_means_without_mutating_local_metrics(monkeypatch):
    local = {"reward_mean": torch.tensor(1.0), "valid_group_fraction": 0.5}
    observed = []

    def fake_reduce(metrics, device):
        observed.append((dict(metrics), device))
        metrics["reward_mean"] = 2.0
        metrics["valid_group_fraction"] = 0.75
        metrics["reward_stationary_reference_fraction"] = 0.25
        metrics["reward_stationary_progress_weighted"] = 0.125
        metrics["reward_stationary_progress_group_range_weighted"] = 0.2
        return metrics

    monkeypatch.setattr("diffusion_planner.hdp_rl_epoch.ddp.reduce_and_average_losses", fake_reduce)

    logged = _step_metrics_for_logging(local, torch.device("cpu"), use_ddp=True)

    assert observed and observed[0][1] == torch.device("cpu")
    assert logged == {
        "reward_mean": 2.0,
        "valid_group_fraction": 0.75,
        "reward_stationary_reference_fraction": 0.25,
        "reward_stationary_progress_weighted": 0.125,
        "reward_stationary_progress_group_range_weighted": 0.2,
        "reward_stationary_progress_score": 0.5,
        "reward_stationary_progress_group_range": 0.8,
    }
    assert local["reward_mean"].item() == 1.0
    assert local["valid_group_fraction"] == 0.5


def test_rl_update_chunk_preserves_groups_and_uses_one_static_shape():
    assert _rl_update_scene_chunk_size(64, 8, 1024) == 64
    assert _rl_update_scene_chunk_size(64, 32, 1024) == 32
    assert _rl_update_scene_chunk_size(60, 32, 1024) == 30
    assert _rl_update_scene_chunk_size(64, 32, 0) == 64
    with pytest.raises(ValueError, match="non-negative"):
        _rl_update_scene_chunk_size(64, 32, -1)
    with pytest.raises(ValueError, match="complete generation group"):
        _rl_update_scene_chunk_size(64, 32, 16)


def test_rl_update_slice_keeps_scene_and_candidate_alignment():
    inputs = {
        "scene": torch.arange(6).view(3, 2),
        "candidate": torch.arange(24).view(12, 2),
        "flag": 4,
    }

    sliced = _slice_grouped_policy_inputs(inputs, 1, 3, num_scenes=3, group_size=4)

    torch.testing.assert_close(sliced["scene"], inputs["scene"][1:3])
    torch.testing.assert_close(sliced["candidate"], inputs["candidate"][4:12])
    assert sliced["flag"] == 4


def test_rl_update_microbatches_preserve_full_group_objective_and_gradient(monkeypatch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))

    def fake_policy_loss(model, _inputs, target, _args, _encoding=None, _time=None, _noise=None):
        per_sample = model.scale * target[:, 0, 0]
        mean = per_sample.mean()
        return {
            "ego_loss_per_sample": per_sample,
            "ego_hdp_diffusion_loss": mean,
            "ego_hdp_waypoint_loss": mean,
        }

    monkeypatch.setattr(
        "diffusion_planner.hdp_rl_utils._compute_policy_ego_loss_per_sample",
        fake_policy_loss,
    )
    model = Policy()
    num_scenes, group_size = 4, 2
    candidates = num_scenes * group_size
    target = torch.zeros(candidates, 80, 4)
    target[:, 0, 0] = torch.arange(1.0, candidates + 1.0)
    weights = torch.tensor([0.5, 1.5, 0.25, 1.75, 1.0, 2.0, 0.75, 1.25])
    valid = torch.ones(candidates, dtype=torch.bool)
    args = SimpleNamespace(
        rl_update_max_candidates_per_rank=4,
        rl_bc_weight=0.0,
        ddp=False,
    )

    result = _backward_reward_weighted_update(
        model,
        {"scene": torch.zeros(num_scenes, 1), "candidate": torch.zeros(candidates, 1)},
        target,
        torch.zeros(candidates),
        weights,
        valid,
        torch.tensor(float(candidates)),
        1,
        num_scenes,
        group_size,
        args,
        None,
        {},
        torch.zeros(num_scenes, 80, 4),
        None,
    )

    expected_unscaled = (weights * target[:, 0, 0]).sum() / candidates
    torch.testing.assert_close(result["loss"], model.scale.detach() * expected_unscaled)
    torch.testing.assert_close(model.scale.grad, expected_unscaled)
    assert result["update_candidate_chunk_size"].item() == 4
    assert result["update_chunk_count"].item() == 2


def test_resume_best_score_ignores_nan_and_non_full_eval_rows():
    score = best_valid_score_from_rows(
        [
            {"valid_full_eval": float("nan"), "valid_epdms_total": 0.99},
            {"valid_full_eval": "False", "valid_epdms_total": 0.98},
            {"valid_full_eval": True, "valid_epdms_total": float("nan"), "valid_loss_ego": 0.4},
            {"valid_full_eval": "true", "valid_epdms_total": 0.75, "valid_loss_ego": 0.5},
            {
                "valid_full_eval": True,
                "valid_reward_mean": 7.2,
                "valid_epdms_total": 0.99,
                "valid_loss_ego": 0.3,
            },
        ]
    )

    assert score == 7.2
    assert best_valid_score_from_rows(
        [
            {
                "valid_full_eval": True,
                "valid_reward_mean": float("nan"),
                "valid_epdms_total": 0.99,
                "valid_loss_ego": 0.1,
            }
        ]
    ) == -float("inf")


def test_resume_best_score_does_not_promote_guard_rejected_reward():
    assert (
        best_valid_score_from_rows(
            [
                {
                    "valid_full_eval": True,
                    "valid_reward_mean": 7.2,
                    "valid_improves_best": True,
                    "best_valid_score": 7.2,
                },
                {
                    "valid_full_eval": True,
                    "valid_reward_mean": 7.5,
                    "valid_improves_best": False,
                    "valid_within_source_risk_guard": False,
                    "best_valid_score": 7.2,
                },
            ]
        )
        == 7.2
    )


def test_find_checkpoint_run_artifact_handles_latest_and_nested_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    nested = run_dir / "epoch0002"
    nested.mkdir(parents=True)
    artifact = run_dir / "source_baseline_metrics.json"
    artifact.write_text("{}", encoding="utf-8")

    assert find_checkpoint_run_artifact(str(run_dir / "latest.pth"), artifact.name) == artifact
    assert find_checkpoint_run_artifact(str(nested / "best_model.pth"), artifact.name) == artifact
    assert find_checkpoint_run_artifact(str(nested / "best_model.pth"), "missing.json") is None


def test_finite_scalar_metrics_excludes_invalid_json_values():
    metrics = finite_scalar_metrics(
        {"finite": torch.tensor(0.75), "nan": float("nan"), "inf": float("inf")}
    )

    assert metrics == {"finite": 0.75}


def test_source_policy_selection_guards_require_safety_and_epdms_non_regression():
    source = {
        "baseline_available": True,
        "valid_reward_risk": 0.4,
        "valid_reward_safety": 0.98,
        "valid_reward_ttc": 0.5,
        "valid_reward_thw": 0.8,
        "valid_reward_occupancy": 0.9,
        "valid_reward_comfort": 0.99,
        "valid_reward_collision_active": 0.02,
        "valid_reward_collision_rear": 0.03,
        "valid_epdms_total": 0.85,
    }
    improved = {
        "risk": 0.41,
        "safety": 0.99,
        "ttc": 0.51,
        "thw": 0.81,
        "occupancy": 0.91,
        "comfort": 1.0,
        "collision_active": 0.01,
        "collision_rear": 0.02,
    }
    kwargs = {
        "max_safety_regression": 0.0,
        "max_epdms_regression": 0.0,
        "require_epdms": True,
    }

    guards = source_policy_selection_guards(source, improved, 0.86, **kwargs)
    assert guards["available"]
    assert all(value for key, value in guards.items() if key != "available")
    for name, regressed in (
        ("risk", 0.39),
        ("safety", 0.97),
        ("ttc", 0.49),
        ("thw", 0.79),
        ("occupancy", 0.89),
        ("comfort", 0.98),
        ("collision_active", 0.03),
        ("collision_rear", 0.04),
    ):
        current = dict(improved)
        current[name] = regressed
        assert not source_policy_selection_guards(source, current, 0.86, **kwargs)[name]
    assert not source_policy_selection_guards(source, improved, 0.84, **kwargs)["epdms"]


def test_source_policy_selection_guards_honor_tolerance_and_unavailable_baseline():
    source = {
        "valid_reward_risk": 0.4,
        "valid_reward_safety": 0.98,
        "valid_reward_ttc": 0.5,
        "valid_reward_thw": 0.8,
        "valid_reward_occupancy": 0.9,
        "valid_reward_comfort": 0.99,
        "valid_reward_collision_active": 0.02,
        "valid_reward_collision_rear": 0.03,
        "valid_epdms_total": 0.85,
    }
    guards = source_policy_selection_guards(
        source,
        {
            "risk": 0.3995,
            "safety": 0.9795,
            "ttc": 0.4995,
            "thw": 0.7995,
            "occupancy": 0.8995,
            "comfort": 0.9895,
            "collision_active": 0.0205,
            "collision_rear": 0.0305,
        },
        0.8495,
        max_safety_regression=0.001,
        max_epdms_regression=0.001,
        require_epdms=True,
    )
    assert guards["available"]
    assert all(value for key, value in guards.items() if key != "available")
    unavailable = source_policy_selection_guards(
        {"baseline_available": False},
        {},
        float("nan"),
        max_safety_regression=0.0,
        max_epdms_regression=0.0,
        require_epdms=True,
    )
    assert not unavailable["available"]
    assert all(value for key, value in unavailable.items() if key != "available")


def test_reward_validation_weights_tail_batches_by_candidate_count(monkeypatch):
    observed_noise_scales = []
    observed_sample_steps = []
    observed_reward_weights = []
    observed_behavior_gates = []
    observed_road_border_settings = []
    observed_stationary_settings = []
    reward_call_count = 0

    def fake_sample_group(_model, inputs, *_args, **_kwargs):
        observed_noise_scales.append(_args[0])
        observed_sample_steps.append(_kwargs["sample_steps"])
        batch = inputs["ego_current_state"].shape[0]
        return torch.zeros(batch, 80, 4)

    def fake_reward(_ego, _inputs, _neighbors, num_scenes, n, _args):
        nonlocal reward_call_count
        reward_call_count += 1
        observed_behavior_gates.append(_args.rl_behavior_gate)
        observed_road_border_settings.append(_args.rl_occupancy_use_road_border)
        observed_stationary_settings.append(
            (
                _args.rl_stationary_progress_mode,
                _args.rl_stationary_reference_threshold_m,
                _args.rl_stationary_progress_tolerance_m,
            )
        )
        observed_reward_weights.append(
            tuple(
                getattr(_args, f"rl_reward_w_{name}")
                for name in ("safety", "risk", "follow", "lane", "progress")
            )
        )
        value = 2.0 if num_scenes == 2 else 10.0
        reward = torch.full((num_scenes * n,), value)
        stationary_fraction = 0.25 if reward_call_count == 1 else 1.0
        stationary_score_weighted = 0.125 if reward_call_count == 1 else 0.8
        stationary_range_weighted = 0.2 if reward_call_count == 1 else 0.4
        return reward, {
            "reward_risk_score": reward.mean(),
            "reward_stationary_reference_fraction": reward.new_tensor(stationary_fraction),
            "reward_stationary_progress_weighted": reward.new_tensor(stationary_score_weighted),
            "reward_stationary_progress_group_range_weighted": reward.new_tensor(
                stationary_range_weighted
            ),
        }

    monkeypatch.setattr("diffusion_planner.hdp_rl_epoch.sample_group", fake_sample_group)
    monkeypatch.setattr("diffusion_planner.hdp_rl_epoch.compute_hdp_reward", fake_reward)

    def batch(size):
        return {
            "ego_agent_past": torch.zeros(size, 2, 3),
            "goal_pose": torch.zeros(size, 3),
            "neighbor_agents_future": torch.zeros(size, 1, 80, 3),
            "ego_current_state": torch.zeros(size, 10),
            "route_lanes": torch.zeros(size, 1, 1, 12),
        }

    args = SimpleNamespace(
        num_generations=2,
        device="cpu",
        seed=7,
        observation_normalizer=lambda inputs: inputs,
        rl_noise_scale=0.5,
        rl_eval_noise_scale=0.25,
        rl_eval_num_generations=2,
        rl_reward_w_safety=5.0,
        rl_reward_w_risk=9.0,
        rl_reward_w_follow=8.0,
        rl_reward_w_lane=7.0,
        rl_reward_w_progress=6.0,
        rl_eval_reward_w_safety=0.0,
        rl_eval_reward_w_risk=1.0,
        rl_eval_reward_w_follow=3.0,
        rl_eval_reward_w_lane=2.5,
        rl_eval_reward_w_progress=3.0,
        rl_behavior_gate="none",
        rl_eval_behavior_gate="risk",
        rl_occupancy_use_road_border=False,
        rl_eval_occupancy_use_road_border=True,
        rl_stationary_progress_mode="constant",
        rl_stationary_reference_threshold_m=0.5,
        rl_stationary_progress_tolerance_m=4.0,
        rl_eval_stationary_progress_mode="distance",
        rl_eval_stationary_reference_threshold_m=1.0,
        rl_eval_stationary_progress_tolerance_m=2.0,
        amp_dtype="off",
        rl_rollout_steps=6,
        diffusion_sample_steps=5,
        ddp=False,
    )

    metrics = validate_hdp_reward_policy([batch(2), batch(1)], torch.nn.Linear(1, 1), args)

    expected = (2.0 * 4 + 10.0 * 2) / 6
    assert metrics["mean"].item() == pytest.approx(expected)
    assert metrics["group_max"].item() == pytest.approx((2.0 * 2 + 10.0) / 3)
    assert metrics["risk"].item() == pytest.approx(expected)
    assert metrics["stationary_reference_fraction"].item() == pytest.approx(0.5)
    assert metrics["stationary_progress_weighted"].item() == pytest.approx(0.35)
    assert metrics["stationary_progress"].item() == pytest.approx(0.7)
    assert metrics["stationary_progress_group_range_weighted"].item() == pytest.approx(4.0 / 15.0)
    assert metrics["stationary_progress_group_range"].item() == pytest.approx(8.0 / 15.0)
    assert observed_noise_scales == [0.25, 0.25]
    assert observed_sample_steps == [5, 5]
    assert observed_reward_weights == [(0.0, 1.0, 3.0, 2.5, 3.0)] * 2
    assert observed_behavior_gates == ["risk", "risk"]
    assert observed_road_border_settings == [True, True]
    assert observed_stationary_settings == [("distance", 1.0, 2.0)] * 2
    assert args.rl_reward_w_safety == 5.0
    assert args.rl_reward_w_risk == 9.0
    assert args.rl_behavior_gate == "none"
    assert args.rl_occupancy_use_road_border is False
    assert args.rl_stationary_progress_mode == "constant"
    assert args.rl_stationary_reference_threshold_m == 0.5
    assert args.rl_stationary_progress_tolerance_m == 4.0
