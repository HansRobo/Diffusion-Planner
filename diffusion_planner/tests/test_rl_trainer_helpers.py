import torch
from diffusion_planner.hdp_rl_epoch import _policy_observation_inputs
from train_hdp_rl_predictor import (
    best_valid_score_from_rows,
    configure_rl_trainable_parameters,
    find_checkpoint_run_artifact,
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


def test_resume_best_score_ignores_nan_and_non_full_eval_rows():
    score = best_valid_score_from_rows(
        [
            {"valid_full_eval": float("nan"), "valid_epdms_total": 0.99},
            {"valid_full_eval": "False", "valid_epdms_total": 0.98},
            {"valid_full_eval": True, "valid_epdms_total": float("nan"), "valid_loss_ego": 0.4},
            {"valid_full_eval": "true", "valid_epdms_total": 0.75, "valid_loss_ego": 0.5},
        ]
    )

    assert score == 0.75


def test_find_checkpoint_run_artifact_handles_latest_and_nested_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    nested = run_dir / "epoch0002"
    nested.mkdir(parents=True)
    artifact = run_dir / "source_baseline_metrics.json"
    artifact.write_text("{}", encoding="utf-8")

    assert find_checkpoint_run_artifact(str(run_dir / "latest.pth"), artifact.name) == artifact
    assert find_checkpoint_run_artifact(str(nested / "best_model.pth"), artifact.name) == artifact
    assert find_checkpoint_run_artifact(str(nested / "best_model.pth"), "missing.json") is None
