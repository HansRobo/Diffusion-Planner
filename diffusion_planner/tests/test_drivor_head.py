"""Unit tests for the DrivoR predictor head: decoder, loss, aggregate, guard.

The PDM oracle has its own parity suite (``test_drivor_oracle.py``); what is
checked here is the wiring around it -- output shapes and layout, the
winner-takes-all objective, the label conventions the six BCE heads rely on, the
selection aggregate, and the divergence guard's verdict.
"""

import math
from dataclasses import dataclass

import pytest
import torch
from diffusion_planner.model.module.drivor_decoder import DrivoRDecoder
from diffusion_planner.model.module.drivor_loss import (
    SCORER_HEAD_WEIGHTS,
    DrivoRLoss,
    three_to_two_classes,
)
from diffusion_planner.model.module.drivor_scorer import (
    DRIVOR_HEAD_METRICS,
    HEAD_BY_METRIC,
    SCORE_WEIGHT_ORDER,
    aggregate_pdm_score,
)
from diffusion_planner.utils.drivor_metrics import selection_metrics, trajectory_metrics
from diffusion_planner.utils.drivor_oracle import ORACLE_METRIC_NAMES, TTC_UNDEFINED
from diffusion_planner.utils.drivor_train import DivergenceGuard, heading_to_cos_sin

HIDDEN = 32
# Shrunk for speed; the head is horizon-agnostic (the shipped default is 40 poses
# at 0.1 s -- see ``diffusion_planner/utils/drivor_sampling.py``).
HORIZON = 8
PROPOSALS = 6
TOKENS = 12


@dataclass
class HeadConfig:
    """The subset of ``TrainConfig`` the head reads."""

    hidden_dim: int = HIDDEN
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
    return DrivoRDecoder(HeadConfig(**overrides)).eval()


def _forward(head, batch=3, mask_tail=True):
    encoding = torch.randn(batch, TOKENS, HIDDEN)
    mask = torch.zeros(batch, TOKENS, dtype=torch.bool)
    if mask_tail:
        mask[:, TOKENS // 2 :] = True  # token 0 (ego) is never padding
    return head(encoding, mask)


# ---------------------------------------------------------------- decoder


def test_output_shapes_and_layout():
    head = _head()
    out = _forward(head)
    assert out["proposals"].shape == (3, PROPOSALS, HORIZON, 4)
    assert out["trajectory"].shape == (3, HORIZON, 4)
    # Ego-only output kept in the repository's [B, P, T, 4] layout.
    assert out["prediction"].shape == (3, 1, HORIZON, 4)
    assert torch.equal(out["prediction"][:, 0], out["trajectory"])
    # One head per refinement stage, plus the pre-refinement decode.
    assert len(out["proposal_list"]) == head.ref_num + 1
    for proposals in out["proposal_list"]:
        assert proposals.shape == (3, PROPOSALS, HORIZON, 4)


def test_selected_trajectory_is_the_argmax_proposal():
    out = _forward(_head())
    chosen = out["chosen_index"]
    assert torch.equal(chosen, out["pdm_score"].argmax(dim=1))
    for index, row in enumerate(chosen.tolist()):
        assert torch.equal(out["trajectory"][index], out["proposals"][index, row])


def test_head_emits_every_metric_logit():
    out = _forward(_head())
    expected = {HEAD_BY_METRIC[name] for name in DRIVOR_HEAD_METRICS} | {"human_closeness"}
    assert set(out["pred_logit"]) == expected
    for name, logit in out["pred_logit"].items():
        assert logit.shape == (3, PROPOSALS), name


def test_logit_bound_is_respected():
    out = _forward(_head(drivor_logit_bound=2.0))
    for name, logit in out["pred_logit"].items():
        if name == "human_closeness":  # the demonstration head is deliberately unbounded
            continue
        assert logit.abs().max().item() <= 2.0 + 1e-5, name


def test_scorer_does_not_backprop_into_the_generator():
    """The scorer sees proposals only through a detached embedding."""
    head = _head()
    head.train()
    encoding = torch.randn(2, TOKENS, HIDDEN, requires_grad=True)
    out = head(encoding, torch.zeros(2, TOKENS, dtype=torch.bool))
    out["pred_logit"]["no_at_fault_collisions"].sum().backward()
    for name, parameter in head.traj_head.named_parameters():
        assert parameter.grad is None or parameter.grad.abs().max() == 0.0, name


def test_padding_mask_changes_the_result():
    head = _head()
    torch.manual_seed(1)
    encoding = torch.randn(2, TOKENS, HIDDEN)
    unmasked = head(encoding, torch.zeros(2, TOKENS, dtype=torch.bool))
    mask = torch.zeros(2, TOKENS, dtype=torch.bool)
    mask[:, 1:] = True
    masked = head(encoding, mask)
    assert not torch.allclose(unmasked["proposals"], masked["proposals"])


# ---------------------------------------------------------------- aggregate


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def _uniform_logits(**probabilities):
    base = {name: 1.0 for name in DRIVOR_HEAD_METRICS}
    base.update(probabilities)
    return {HEAD_BY_METRIC[name]: torch.full((1, 1), _logit(value)) for name, value in base.items()}


def _weights():
    order = (1.0, 1.0, 0.0, 5.0, 5.0, 2.0)
    assert len(order) == len(SCORE_WEIGHT_ORDER)
    return torch.tensor([order])


def test_collision_zeroes_the_aggregate():
    score, _ = aggregate_pdm_score(_uniform_logits(no_at_fault_collisions=0.0), _weights())
    assert score.item() == pytest.approx(0.0, abs=1e-4)


def test_perfect_proposal_scores_one():
    score, components = aggregate_pdm_score(_uniform_logits(), _weights())
    assert score.item() == pytest.approx(1.0, abs=1e-3)
    assert components.shape == (1, 1, len(SCORE_WEIGHT_ORDER))


def test_zero_weight_metric_is_ignored():
    """DDC carries weight 0 in the profile, so it cannot move the score."""
    kept, _ = aggregate_pdm_score(_uniform_logits(driving_direction_compliance=0.0), _weights())
    assert kept.item() == pytest.approx(1.0, abs=1e-3)


def test_weighted_mean_matches_the_profile():
    # EP at 0 removes its 5/(5+5+2) share of the behaviour mean.
    score, _ = aggregate_pdm_score(_uniform_logits(ego_progress=0.0), _weights())
    assert score.item() == pytest.approx((5.0 + 2.0) / 12.0, abs=1e-3)


def test_human_weight_is_additive_and_breaks_ties():
    logits = _uniform_logits()
    logits = {name: value.repeat(1, 2) for name, value in logits.items()}
    human = torch.tensor([[_logit(0.0), _logit(1.0)]])
    tied, _ = aggregate_pdm_score(logits, _weights())
    assert tied[0, 0].item() == pytest.approx(tied[0, 1].item(), abs=1e-6)
    broken, _ = aggregate_pdm_score(
        {**logits, "human_closeness": human}, _weights(), human_weight=0.2
    )
    assert broken[0, 1].item() > broken[0, 0].item()
    assert broken[0, 1].item() - tied[0, 1].item() == pytest.approx(0.2, abs=1e-3)


# ---------------------------------------------------------------- loss


def test_selection_top_hits_are_tie_aware():
    """A tied-best pick is a hit.

    The oracle aggregate ties heavily -- every colliding proposal sits at exactly
    0 and a third of samples have several proposals at the maximum -- so scoring
    ``argmax`` index equality reported chance-level selection for a scorer whose
    picks were in fact tied-best.
    """
    oracle = torch.tensor([[0.8, 0.8, 0.2, 0.0], [0.9, 0.1, 0.1, 0.1]])
    # Sample 0: index 1 is tied-best but is not ``argmax`` (which returns 0).
    # Sample 1: index 0 is the unique best, and is picked.
    chosen = torch.tensor([1, 0])
    out = selection_metrics({"_oracle_total": oracle, "_chosen_index": chosen}, "val")
    assert out["val/selection/top1_hit"].item() == pytest.approx(1.0)
    assert out["val/selection/oracle_gap"].item() == pytest.approx(0.0)
    assert out["val/selection/oracle_rank"].item() == pytest.approx(1.0)

    # A genuine miss still misses.
    missed = selection_metrics(
        {"_oracle_total": oracle, "_chosen_index": torch.tensor([2, 1])}, "val"
    )
    assert missed["val/selection/top1_hit"].item() == pytest.approx(0.0)


def test_scorer_entropy_is_the_floor_of_the_smoothed_bce():
    """``score_kl_loss`` must vanish when a head sits exactly on its BCE floor.

    With label smoothing the minimum of ``BCE(logit, smooth(y))`` is
    ``H(smooth(y))``, not ``H(y)``: for a hard label at smoothing 0.02 that is
    0.056, so subtracting ``H(y) = 0`` reports a converged head as still having
    0.056 of learnable loss per head.
    """
    batch, proposals = 4, PROPOSALS
    smoothing = 0.02
    loss = DrivoRLoss(label_smoothing=smoothing)

    # Constant-0 labels for every metric: the degenerate case the comfort head
    # actually hit in training.
    oracle = torch.zeros(batch, proposals, len(ORACLE_METRIC_NAMES))
    target = torch.zeros(batch, HORIZON, 4)
    smoothed = 0.5 * smoothing
    optimal = math.log(smoothed / (1 - smoothed))
    pred = {
        "proposal_list": [torch.zeros(batch, proposals, HORIZON, 4)],
        "proposals": torch.zeros(batch, proposals, HORIZON, 4),
        "pdm_score": torch.zeros(batch, proposals),
        "pred_logit": {
            HEAD_BY_METRIC[name]: torch.full((batch, proposals), optimal)
            for name in DRIVOR_HEAD_METRICS
        },
    }
    out = loss(pred, target, oracle)

    floor = -(smoothed * math.log(smoothed) + (1 - smoothed) * math.log(1 - smoothed))
    assert float(out["comfort_loss"]) == pytest.approx(floor, abs=1e-5)
    # Every head is at its floor, so nothing learnable is left.
    assert float(out["score_kl_loss"]) == pytest.approx(0.0, abs=1e-5)
    assert float(out["label_entropy"]) == pytest.approx(len(DRIVOR_HEAD_METRICS) * floor, abs=1e-4)


def test_three_to_two_classes_only_maps_the_half():
    values = torch.tensor([0.0, 0.5, 1.0, 0.25])
    assert torch.equal(three_to_two_classes(values), torch.tensor([0.0, 0.0, 1.0, 0.25]))


def test_wta_is_zero_when_a_proposal_matches_the_target():
    target = torch.zeros(1, HORIZON, 4)
    target[..., 2] = 1.0
    proposals = torch.randn(1, PROPOSALS, HORIZON, 4)
    proposals[:, 3] = target
    loss = DrivoRLoss(prev_weight=1.0)
    terms = loss.local_terms(
        {"proposal_list": [proposals], "proposals": proposals, "pred_logit": {}}, target
    )
    assert terms["min_loss_list"][0].item() == pytest.approx(0.0, abs=1e-6)


def test_wta_accumulates_over_refinement_stages():
    target = torch.zeros(1, HORIZON, 4)
    proposals = torch.ones(1, PROPOSALS, HORIZON, 4)
    loss = DrivoRLoss(prev_weight=1.0)
    common = {"proposals": proposals, "pred_logit": {}}
    one_stage = loss.local_terms({"proposal_list": [proposals], **common}, target)
    two_stages = loss.local_terms({"proposal_list": [proposals, proposals], **common}, target)
    # prev_weight=1 makes stage N's total the running sum of the stage losses.
    assert two_stages["trajectory_loss"].item() == pytest.approx(
        2.0 * one_stage["trajectory_loss"].item(), rel=1e-5
    )


def _prediction(batch=2):
    torch.manual_seed(2)
    proposals = torch.randn(batch, PROPOSALS, HORIZON, 4)
    logits = {
        HEAD_BY_METRIC[name]: torch.zeros(batch, PROPOSALS, requires_grad=True)
        for name in DRIVOR_HEAD_METRICS
    }
    logits["human_closeness"] = torch.zeros(batch, PROPOSALS, requires_grad=True)
    score, components = aggregate_pdm_score(logits, _weights().repeat(batch, 1), human_weight=0.2)
    return {
        "proposals": proposals,
        "proposal_list": [proposals],
        "pred_logit": logits,
        "pdm_score": score,
        "score_components": components,
        "chosen_index": score.argmax(dim=1),
    }


def _oracle(batch=2, ttc=1.0):
    labels = torch.ones(batch, PROPOSALS, len(ORACLE_METRIC_NAMES))
    labels[..., ORACLE_METRIC_NAMES.index("time_to_collision_within_bound")] = ttc
    return labels


def test_loss_reports_every_head_and_stays_finite():
    loss = DrivoRLoss()
    target = torch.zeros(2, HORIZON, 4)
    out = loss(_prediction(), target, _oracle())
    for key in ("loss", "trajectory_loss", "final_score_loss", "human_loss"):
        assert torch.isfinite(out[key]).all(), key
    for metric in DRIVOR_HEAD_METRICS:
        assert metric in SCORER_HEAD_WEIGHTS
    out["loss"].backward()


def test_undefined_ttc_is_masked_out():
    """The TTC sentinel must not be learned as a label."""
    loss = DrivoRLoss()
    target = torch.zeros(2, HORIZON, 4)
    masked = loss(_prediction(), target, _oracle(ttc=TTC_UNDEFINED))
    assert masked["ttc_loss"].item() == pytest.approx(0.0, abs=1e-6)
    kept = loss(_prediction(), target, _oracle(ttc=1.0))
    assert kept["ttc_loss"].item() > 0.0


def test_oracle_width_is_validated():
    loss = DrivoRLoss()
    target = torch.zeros(2, HORIZON, 4)
    with pytest.raises((AssertionError, ValueError)):
        loss(_prediction(), target, _oracle()[..., :-2])


# ---------------------------------------------------------------- metrics


def test_trajectory_metrics_have_known_values():
    target = torch.zeros(1, HORIZON, 4)
    target[..., 2] = 1.0  # heading 0 as (cos, sin)
    proposals = target[:, None].repeat(1, PROPOSALS, 1, 1).clone()
    selected = target.clone()
    selected[..., 0] = 3.0  # a constant 3 m longitudinal offset
    selected[..., 4 - 2] = math.cos(0.5)
    selected[..., 4 - 1] = math.sin(0.5)
    pred = {"trajectory": selected, "proposals": proposals, "chosen_index": torch.zeros(1).long()}
    metrics = trajectory_metrics(pred, target)
    named = {key.split("/")[-1]: float(value) for key, value in metrics.items()}
    assert named["selected_ADE"] == pytest.approx(3.0, abs=1e-5)
    assert named["selected_FDE"] == pytest.approx(3.0, abs=1e-5)
    assert named["heading_MAE"] == pytest.approx(0.5, abs=1e-5)


def test_heading_error_wraps_at_pi():
    target = torch.zeros(1, HORIZON, 4)
    target[..., 2] = math.cos(math.pi - 0.1)
    target[..., 3] = math.sin(math.pi - 0.1)
    selected = torch.zeros(1, HORIZON, 4)
    selected[..., 2] = math.cos(-math.pi + 0.1)
    selected[..., 3] = math.sin(-math.pi + 0.1)
    pred = {
        "trajectory": selected,
        "proposals": selected[:, None],
        "chosen_index": torch.zeros(1).long(),
    }
    named = {
        key.split("/")[-1]: float(value) for key, value in trajectory_metrics(pred, target).items()
    }
    # 0.2 rad apart across the +/-pi seam, not 2*pi - 0.2.
    assert named["heading_MAE"] == pytest.approx(0.2, abs=1e-5)


def test_heading_to_cos_sin_is_idempotent():
    three = torch.zeros(2, HORIZON, 3)
    three[..., 2] = 0.3
    four = heading_to_cos_sin(three)
    assert four.shape == (2, HORIZON, 4)
    assert four[0, 0, 2].item() == pytest.approx(math.cos(0.3))
    assert torch.equal(heading_to_cos_sin(four), four)


# ---------------------------------------------------------------- guard


class _Optimizer:
    def __init__(self, lr=1e-4):
        self.param_groups = [{"lr": lr}]


def _loss_dict(loss, absmax=1.0):
    return {
        "loss": torch.tensor(float(loss)),
        "logit_absmax": torch.tensor(float(absmax)),
    }


def test_guard_skips_non_finite_loss():
    guard = DivergenceGuard(True, logit_bound=10.0)
    assert guard.check(_loss_dict(float("nan")), _Optimizer()) is True


def test_guard_skips_saturated_logits():
    guard = DivergenceGuard(True, logit_bound=10.0)
    assert guard.check(_loss_dict(1.0, absmax=1e3), _Optimizer()) is True


def test_guard_passes_a_normal_step():
    guard = DivergenceGuard(True, logit_bound=10.0)
    assert guard.check(_loss_dict(1.0), _Optimizer()) is False


def test_disabled_guard_never_skips():
    guard = DivergenceGuard(False, logit_bound=10.0)
    assert guard.check(_loss_dict(float("inf"), absmax=1e6), _Optimizer()) is False


def test_repeated_breaches_cut_the_learning_rate():
    guard = DivergenceGuard(True, logit_bound=10.0)
    optimizer = _Optimizer(lr=1e-4)
    for _ in range(guard.max_skips):
        guard.check(_loss_dict(float("nan")), optimizer)
    assert optimizer.param_groups[0]["lr"] < 1e-4
