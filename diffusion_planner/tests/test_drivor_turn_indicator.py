"""Turn-indicator head on the DrivoR predictor (``drivor_turn_indicator``).

Covers the wiring only: the network itself is the diffusion Decoder's and the
PDM side of the head has its own suite (``test_drivor_head.py``).
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from diffusion_planner.dimensions import (
    INPUT_T,
    POINTS_PER_LANELET,
    SEGMENT_POINT_DIM,
    TURN_INDICATOR_OUTPUT_DIM,
)
from diffusion_planner.loss import make_turn_indicator_gt, turn_indicator_loss_terms
from diffusion_planner.model.module.drivor_decoder import DrivoRDecoder
from diffusion_planner.model.module.drivor_loss import DrivoRLoss
from diffusion_planner.model.module.drivor_scorer import (
    DRIVOR_HEAD_METRICS,
    HEAD_BY_METRIC,
    aggregate_pdm_score,
)
from diffusion_planner.model.module.turn_indicator import TurnIndicatorNetwork
from diffusion_planner.utils.drivor_metrics import epoch_metrics
from diffusion_planner.utils.drivor_oracle import ORACLE_METRIC_NAMES

HIDDEN = 32
HORIZON = 8
PROPOSALS = 6
TOKENS = 12


@dataclass
class _Config:
    """The subset of ``TrainConfig`` the head reads, shrunk for speed."""

    hidden_dim: int = HIDDEN
    num_heads: int = 4
    encoder_mixer_depth: int = 2
    encoder_fusion_depth: int = 2
    encoder_drop_path_rate: float = 0.0
    drivor_turn_indicator: bool = True
    drivor_num_poses: int = HORIZON
    drivor_pose_dt: float = 0.1
    drivor_proposal_num: int = PROPOSALS
    drivor_ref_num: int = 2
    drivor_scorer_ref_num: int = 2
    drivor_tf_d_ffn: int = 64
    drivor_refiner_num_heads: int = 4
    drivor_refiner_ls_values: float = 1.0
    drivor_trajectory_proj_drop: float = 0.0
    drivor_trajectory_drop_path: float = 0.0
    drivor_scorer_proj_drop: float = 0.0
    drivor_scorer_drop_path: float = 0.0
    drivor_human_teacher_weight: float = 0.2
    drivor_logit_bound: float = 10.0
    drivor_weight_no_at_fault_collisions: float = 1.0
    drivor_weight_drivable_area_compliance: float = 1.0
    drivor_weight_driving_direction_compliance: float = 0.0
    drivor_weight_time_to_collision_within_bound: float = 5.0
    drivor_weight_ego_progress: float = 5.0
    drivor_weight_history_comfort: float = 2.0


def _head(**overrides) -> DrivoRDecoder:
    torch.manual_seed(0)
    return DrivoRDecoder(_Config(**overrides))


def _encoding(batch=3):
    torch.manual_seed(1)
    return torch.randn(batch, TOKENS, HIDDEN), torch.zeros(batch, TOKENS, dtype=torch.bool)


def _inputs(batch=3, lanes=6, routes=3):
    """The raw-input keys the turn network reads; lane counts are free."""
    torch.manual_seed(3)
    ti = torch.randint(0, 4, (batch, INPUT_T + 1))
    ti[0, -1] = ti[0, -2]  # a kept frame
    ti[1, -1] = (ti[1, -2] + 1) % 4  # a change frame
    return {
        "lanes": torch.randn(batch, lanes, POINTS_PER_LANELET, SEGMENT_POINT_DIM),
        "lanes_speed_limit": torch.rand(batch, lanes, 1),
        "lanes_has_speed_limit": torch.ones(batch, lanes, 1, dtype=torch.bool),
        "route_lanes": torch.randn(batch, routes, POINTS_PER_LANELET, SEGMENT_POINT_DIM),
        "route_lanes_speed_limit": torch.rand(batch, routes, 1),
        "route_lanes_has_speed_limit": torch.ones(batch, routes, 1, dtype=torch.bool),
        "turn_indicators": ti,
    }


# ---------------------------------------------------------------- decoder


def test_head_emits_five_class_logit():
    head = _head().eval()
    enc, mask = _encoding()
    out = head(enc, mask, inputs=_inputs())
    assert out["turn_indicator_logit"].shape == (3, TURN_INDICATOR_OUTPUT_DIM)
    assert torch.isfinite(out["turn_indicator_logit"]).all()
    # The trajectory outputs are untouched by the extra head.
    assert out["trajectory"].shape == (3, HORIZON, 4)


def test_flag_off_builds_no_head():
    head = _head(drivor_turn_indicator=False).eval()
    assert head.turn_indicator_predictor is None
    assert not any("turn_indicator" in name for name, _ in head.named_parameters())
    enc, mask = _encoding()
    assert "turn_indicator_logit" not in head(enc, mask)


def test_head_requires_inputs():
    head = _head().eval()
    enc, mask = _encoding()
    with pytest.raises(ValueError):
        head(enc, mask)


def test_training_is_teacher_forced_and_eval_uses_the_selected_proposal():
    head = _head()
    enc, mask = _encoding()
    inputs = _inputs()
    torch.manual_seed(5)
    target_a = torch.randn(3, HORIZON, 4)
    target_b = torch.randn(3, HORIZON, 4)

    head.train()
    with torch.no_grad():
        logit_a = head(enc, mask, inputs=inputs, target_trajectory=target_a)
        logit_b = head(enc, mask, inputs=inputs, target_trajectory=target_b)
    assert not torch.allclose(logit_a["turn_indicator_logit"], logit_b["turn_indicator_logit"])

    head.eval()
    with torch.no_grad():
        with_target = head(enc, mask, inputs=inputs, target_trajectory=target_a)
        without = head(enc, mask, inputs=inputs)
        from_selected = head._compute_turn_indicator(without["trajectory"], inputs)
    assert torch.allclose(with_target["turn_indicator_logit"], without["turn_indicator_logit"])
    assert torch.allclose(without["turn_indicator_logit"], from_selected)


def test_network_is_fixed_to_its_trajectory_len():
    net = TurnIndicatorNetwork(
        hidden_dim=16, num_heads=2, mixer_depth=1, fusion_depth=1, trajectory_len=8
    )
    inputs = _inputs(batch=2)
    assert net(torch.randn(2, 8, 4), inputs).shape == (2, TURN_INDICATOR_OUTPUT_DIM)
    with pytest.raises(ValueError):
        net(torch.randn(2, 80, 4), inputs)


# ---------------------------------------------------------------- loss


def test_loss_terms_match_the_inline_formula():
    """Regression for the diffusion head, which used to inline this."""
    torch.manual_seed(7)
    logit = torch.randn(6, TURN_INDICATOR_OUTPUT_DIM)
    ti = torch.randint(0, 4, (6, INPUT_T + 1))
    ti[:3, -1] = ti[:3, -2]
    gt = make_turn_indicator_gt(ti)
    change = ti[:, -2] != ti[:, -1]
    expected = (
        F.cross_entropy(logit, gt, reduction="none") * torch.where(change, 1.0, 0.05)
    ).mean()

    terms = turn_indicator_loss_terms(logit, ti)
    assert terms["turn_indicator_loss"].item() == pytest.approx(expected.item(), rel=1e-6)
    correct = (logit.argmax(-1) == gt).float()
    assert terms["turn_indicator_accuracy"].item() == pytest.approx(correct.mean().item())
    assert terms["turn_indicator_change_count"].item() == change.sum().item()
    assert terms["turn_indicator_change_correct"].item() == correct[change].sum().item()


def _pred(batch=2):
    torch.manual_seed(2)
    proposals = torch.randn(batch, PROPOSALS, HORIZON, 4)
    logits = {HEAD_BY_METRIC[name]: torch.zeros(batch, PROPOSALS) for name in DRIVOR_HEAD_METRICS}
    logits["human_closeness"] = torch.zeros(batch, PROPOSALS)
    weights = torch.tensor([1.0, 1.0, 0.0, 5.0, 5.0, 2.0]).repeat(batch, 1)
    score, components = aggregate_pdm_score(logits, weights, human_weight=0.2)
    return {
        "proposals": proposals,
        "proposal_list": [proposals],
        "pred_logit": logits,
        "pdm_score": score,
        "score_components": components,
        "chosen_index": score.argmax(dim=1),
        "turn_indicator_logit": torch.randn(batch, TURN_INDICATOR_OUTPUT_DIM, requires_grad=True),
    }


def test_loss_adds_the_weighted_turn_term():
    target = torch.zeros(2, HORIZON, 4)
    oracle = torch.ones(2, PROPOSALS, len(ORACLE_METRIC_NAMES))
    ti = _inputs(batch=2)["turn_indicators"]
    pred = _pred()
    loss_fn = DrivoRLoss(turn_indicator_weight=2.0)

    base = loss_fn(pred, target, oracle)  # no history handed in -> no term
    assert "turn_indicator_loss" not in base
    with_turn = loss_fn(pred, target, oracle, turn_indicators=ti)
    turn = turn_indicator_loss_terms(pred["turn_indicator_logit"], ti)["turn_indicator_loss"]
    assert with_turn["loss"].item() == pytest.approx(
        base["loss"].item() + 2.0 * turn.item(), rel=1e-5
    )
    for key in (
        "turn_indicator_loss",
        "turn_indicator_accuracy",
        "turn_indicator_change_correct",
        "turn_indicator_change_count",
    ):
        assert key in with_turn, key
    with_turn["loss"].backward()
    assert pred["turn_indicator_logit"].grad is not None


# ---------------------------------------------------------------- metrics


def test_epoch_metrics_derive_change_accuracy():
    out = epoch_metrics(
        {
            "loss": 1.0,
            "turn_indicator_loss": 0.5,
            "turn_indicator_accuracy": 0.9,
            "turn_indicator_change_correct": 3.0,
            "turn_indicator_change_count": 4.0,
        },
        "val",
    )
    assert out["val/loss/turn_indicator"] == 0.5
    assert out["val/turn_indicator/accuracy"] == 0.9
    assert out["val/turn_indicator/change_accuracy"] == pytest.approx(0.75)
    assert not any(key.endswith(("change_correct", "change_count")) for key in out)
