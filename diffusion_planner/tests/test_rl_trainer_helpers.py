import torch
from train_hdp_rl_predictor import (
    best_valid_score_from_rows,
    configure_rl_trainable_parameters,
)


class _TinyPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Module()
        self.decoder.dit = torch.nn.Linear(1, 1)
        self.decoder.turn_indicator_predictor = torch.nn.Linear(1, 1)


def test_all_scope_freezes_only_the_unused_turn_head():
    model = _TinyPlanner()
    configure_rl_trainable_parameters(model, "all")
    state = {name: param.requires_grad for name, param in model.named_parameters()}

    assert all(value for name, value in state.items() if name.startswith("encoder."))
    assert all(value for name, value in state.items() if name.startswith("decoder.dit."))
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
