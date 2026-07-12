from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.hdp_rl_epoch import (
    _policy_observation_inputs,
    _reward_eval_scene_chunk_size,
    validate_hdp_reward_policy,
)
from train_hdp_rl_predictor import (
    best_valid_score_from_rows,
    configure_rl_trainable_parameters,
    find_checkpoint_run_artifact,
    finite_scalar_metrics,
    validate_compiled_candidate_batch,
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


def test_compiled_rl_candidate_batch_rejects_corrupted_h100_shape():
    assert validate_compiled_candidate_batch(64, 32, True) == 2048
    assert validate_compiled_candidate_batch(128, 32, False) == 4096
    with pytest.raises(ValueError, match="silently corrupted backward gradients"):
        validate_compiled_candidate_batch(128, 32, True)


def test_reward_validation_caps_candidate_batch_without_dropping_scenes():
    assert _reward_eval_scene_chunk_size(64, 32) == 32
    assert _reward_eval_scene_chunk_size(17, 32) == 17
    assert _reward_eval_scene_chunk_size(64, 2048) == 1


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
    assert (
        best_valid_score_from_rows(
            [
                {
                    "valid_full_eval": True,
                    "valid_reward_mean": float("nan"),
                    "valid_epdms_total": 0.99,
                    "valid_loss_ego": 0.1,
                }
            ]
        )
        == -float("inf")
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


def test_reward_validation_weights_tail_batches_by_candidate_count(monkeypatch):
    observed_noise_scales = []
    observed_sample_steps = []
    observed_reward_weights = []

    def fake_sample_group(_model, inputs, *_args, **_kwargs):
        observed_noise_scales.append(_args[0])
        observed_sample_steps.append(_kwargs["sample_steps"])
        batch = inputs["ego_current_state"].shape[0]
        return torch.zeros(batch, 80, 4)

    def fake_reward(_ego, _inputs, _neighbors, num_scenes, n, _args):
        observed_reward_weights.append(
            tuple(
                getattr(_args, f"rl_reward_w_{name}")
                for name in ("safety", "risk", "follow", "lane", "progress")
            )
        )
        value = 2.0 if num_scenes == 2 else 10.0
        reward = torch.full((num_scenes * n,), value)
        return reward, {"reward_risk_score": reward.mean()}

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
    assert observed_noise_scales == [0.25, 0.25]
    assert observed_sample_steps == [5, 5]
    assert observed_reward_weights == [(0.0, 1.0, 3.0, 2.5, 3.0)] * 2
    assert args.rl_reward_w_safety == 5.0
    assert args.rl_reward_w_risk == 9.0
