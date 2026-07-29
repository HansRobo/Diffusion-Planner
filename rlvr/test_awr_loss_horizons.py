import math

import pytest
import torch

from rlvr.grpo_loss import (
    _mean_over_ego_loss_horizons,
    compute_batched_trajectory_losses,
)
from rlvr.test_neighbor_reg import _make_model_args, _make_scene_data, _StubDiT
from rlvr.train_awr import (
    _build_awr_target_loss_horizons,
    _build_awr_target_loss_starts,
    _expert_safe_over_consumed_horizon,
    _inject_expert_anchor_batch,
    _low_speed_scene_mask,
    _mask_expert_anchor_to_active_awr_groups,
    validate_expert_anchor_secondary_config,
)


def _plain_mse_loss(
    coeff_timestep: list[float] | None,
    ego_loss_horizons: torch.Tensor | None = None,
) -> torch.Tensor:
    torch.manual_seed(7)
    model = _StubDiT(P=5, T=80)
    model_args = _make_model_args(P=5, T=80)
    # MagicMock.__int__ is 1, so the shared helper would silently clamp the ego
    # horizon to a single step -- under which every timestep profile collapses
    # to the same weighted mean and this test could not fail.
    model_args.ego_prediction_horizon = 80
    if coeff_timestep is not None:
        model_args.coeff_timestep = coeff_timestep
    return compute_batched_trajectory_losses(
        model,
        _make_scene_data(B=1, P=5, T=80),
        torch.randn(4, 80, 4),
        model_args,
        torch.randn(1, 5, 80, 4),
        torch.tensor([0.5]),
        torch.device("cpu"),
        awr_loss_type="plain_mse",
        ego_loss_horizons=ego_loss_horizons,
    )


def test_plain_mse_honours_coeff_timestep() -> None:
    # Regression: the timestep profile used to be applied only to the geometry
    # decomposition, so plain_mse -- what the PlannerRFT campaigns run -- threw
    # it away silently.  A front-loaded profile must move the loss.
    flat = _plain_mse_loss([1.0] * 80)
    front_loaded = _plain_mse_loss([8.0] * 20 + [1.0] * 60)
    assert not torch.allclose(flat, front_loaded)


def test_plain_mse_block_and_per_step_uniform_profiles_agree() -> None:
    # [1]*4 and [1]*80 describe the same uniform profile at different
    # granularity, and both must reproduce the historical unweighted loss.
    assert torch.equal(_plain_mse_loss(None), _plain_mse_loss([1.0] * 4))
    assert torch.equal(_plain_mse_loss([1.0] * 4), _plain_mse_loss([1.0] * 80))


def test_default_ego_loss_horizon_preserves_full_mean() -> None:
    values = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 9.0]])
    actual = _mean_over_ego_loss_horizons(
        values,
        default_horizon=3,
        loss_horizons=None,
    )
    assert torch.equal(actual, values.mean(dim=-1))


def test_per_target_ego_loss_horizon_masks_unscored_tail_and_gradient() -> None:
    values = torch.tensor(
        [[1.0, 3.0, 100.0, 100.0], [2.0, 4.0, 6.0, 8.0]],
        requires_grad=True,
    )
    actual = _mean_over_ego_loss_horizons(
        values,
        default_horizon=4,
        loss_horizons=torch.tensor([2, 4]),
    )
    assert torch.equal(actual, torch.tensor([2.0, 5.0]))
    actual.sum().backward()
    assert torch.equal(
        values.grad,
        torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]]
        ),
    )


@pytest.mark.parametrize("horizon", [0, 5])
def test_per_target_ego_loss_horizon_fails_outside_default(horizon: int) -> None:
    with pytest.raises(ValueError, match="must lie in"):
        _mean_over_ego_loss_horizons(
            torch.ones(1, 4),
            default_horizon=4,
            loss_horizons=torch.tensor([horizon]),
        )


def test_uniform_timestep_weights_reproduce_the_step_count_denominator() -> None:
    # Every historical run has a uniform coeff_timestep, so the weighted path
    # must not perturb them at all.
    values = torch.tensor([[1.0, 3.0, 100.0, 100.0], [2.0, 4.0, 6.0, 8.0]])
    horizons = torch.tensor([2, 4])
    baseline = _mean_over_ego_loss_horizons(
        values, default_horizon=4, loss_horizons=horizons
    )
    assert torch.equal(
        _mean_over_ego_loss_horizons(
            values,
            default_horizon=4,
            loss_horizons=horizons,
            timestep_weights=torch.ones(4),
        ),
        baseline,
    )


def test_weighted_denominator_keeps_short_and_long_targets_comparable() -> None:
    # A front-loaded profile puts almost all its mass in the first half.  A
    # 2-step candidate target and a 4-step anchor target that are equally wrong
    # per step must still score equally; dividing by the step count instead
    # would inflate the candidate by the ratio of the two weight masses.
    weights = torch.tensor([8.0, 4.0, 1.0, 1.0])
    values = torch.ones(2, 4) * weights  # unit per-step error, already weighted
    actual = _mean_over_ego_loss_horizons(
        values,
        default_horizon=4,
        loss_horizons=torch.tensor([2, 4]),
        timestep_weights=weights,
    )
    assert torch.allclose(actual, torch.ones(2))


def test_timestep_weights_scale_the_per_step_gradient() -> None:
    # The prefix-weighted arm exists to move gradient onto the one step the
    # vehicle executes, so the property that has to hold is on the gradient,
    # not the loss value: d(loss)/d(step t) must scale with w_t.  Under the
    # flat profile the executed step receives 0.05% of the ego gradient; a
    # profile with w[0]=35 has to deliver ~35x the per-unit-error pull there.
    weights = torch.tensor([35.0, 4.0, 0.5, 0.05])
    values = torch.ones(1, 4, requires_grad=True)
    weighted = values * weights
    _mean_over_ego_loss_horizons(
        weighted,
        default_horizon=4,
        loss_horizons=None,
        timestep_weights=weights,
    ).sum().backward()
    # Weighted mean: each step contributes w_t/sum(w) of the total, so the
    # gradient ratio between steps is exactly the weight ratio.
    assert torch.allclose(values.grad, weights / weights.sum())
    ratio = float(values.grad[0, 0] / values.grad[0, 3])
    assert math.isclose(ratio, 700.0, rel_tol=1e-5)


def test_target_horizons_keep_deterministic_and_appended_expert_full() -> None:
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=False,
    )
    assert torch.equal(
        horizons.reshape(2, 4),
        torch.tensor([[80, 40, 40, 80], [80, 40, 40, 80]]),
    )


def test_replaced_expert_gets_full_horizon_only_when_safe() -> None:
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([2.0, 0.0, 1.0, 0.0, 3.0, 4.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=False,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=True,
    )
    assert torch.equal(
        horizons.reshape(2, 3),
        torch.tensor([[40, 80, 40], [40, 40, 40]]),
    )


def test_single_step_anchor_horizon_survives_the_real_loss_path() -> None:
    # The unit tests above cover the horizon builder and the weighted mean in
    # isolation; this drives a horizon vector containing 1 through the loss the
    # arm actually calls, because a crash there costs nine unattended GPU-hours.
    # Row 3 stands in for the truncated anchor, rows 0-2 for candidates.
    horizons = torch.tensor([80, 40, 40, 1])
    actual = _plain_mse_loss([1.0] * 80, ego_loss_horizons=horizons)
    assert actual.shape == (4,)
    assert torch.isfinite(actual).all()
    # A 1-step mean is a plain squared error, not an average over 80, so the
    # truncated row must differ from the same row scored full-horizon.
    assert not torch.isclose(
        actual[3], _plain_mse_loss([1.0] * 80, ego_loss_horizons=None)[3]
    )


def test_expert_anchor_horizon_truncates_only_the_appended_anchor() -> None:
    # The executed-waypoint arm: anchor scores step 1 alone, candidates keep
    # their scorer-supported 40, and the deterministic retention row keeps 80.
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=False,
        expert_horizon=1,
    )
    assert torch.equal(
        horizons.reshape(2, 4),
        torch.tensor([[80, 40, 40, 1], [80, 40, 40, 1]]),
    )


def test_the_secondary_config_is_off_and_silent_for_every_historical_run() -> None:
    # Every run so far omits both keys.  Off must resolve without touching any
    # other flag, including the combinations that are refused when it is on.
    for config in (
        {},
        {"expert_anchor_loss_horizon": 1},
        {"expert_anchor_replace_worst": True},
        {"expert_anchor_loss_horizon_max_speed": 1.0},
        {"expert_anchor_secondary_weight": 0.0, "expert_anchor_loss_horizon": 80},
    ):
        assert validate_expert_anchor_secondary_config(config, future_len=80) == (
            0.0,
            80,
        )


def test_the_secondary_config_resolves_zero_to_the_full_model_future() -> None:
    assert validate_expert_anchor_secondary_config(
        {"expert_anchor_secondary_weight": 0.4, "expert_anchor_loss_horizon": 1},
        future_len=80,
    ) == (0.4, 80)
    assert validate_expert_anchor_secondary_config(
        {
            "expert_anchor_secondary_weight": 0.4,
            "expert_anchor_secondary_horizon": 40,
            "expert_anchor_loss_horizon": 1,
        },
        future_len=80,
    ) == (0.4, 40)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"expert_anchor_secondary_weight": -0.1}, "must be non-negative"),
        (
            {"expert_anchor_secondary_weight": 0.4, "expert_anchor_secondary_horizon": 81},
            "lie inside the model",
        ),
        # A second slot at or below the primary horizon supervises nothing new,
        # so reading it as tail coverage would be wrong.
        (
            {
                "expert_anchor_secondary_weight": 0.4,
                "expert_anchor_secondary_horizon": 1,
                "expert_anchor_loss_horizon": 1,
            },
            "must exceed",
        ),
        (
            {
                "expert_anchor_secondary_weight": 0.4,
                "expert_anchor_secondary_horizon": 40,
                "expert_anchor_loss_horizon": 40,
            },
            "must exceed",
        ),
        # Default primary horizon is the full future, so a secondary can never
        # exceed it -- the two slots would be one anchor at double the weight.
        ({"expert_anchor_secondary_weight": 0.4}, "must exceed"),
        # And the speed mask reintroduces exactly that collapse per scene.
        (
            {
                "expert_anchor_secondary_weight": 0.4,
                "expert_anchor_loss_horizon": 1,
                "expert_anchor_loss_horizon_max_speed": 1.0,
            },
            "expert_anchor_loss_horizon_max_speed",
        ),
        (
            {
                "expert_anchor_secondary_weight": 0.4,
                "expert_anchor_loss_horizon": 1,
                "expert_anchor_replace_worst": True,
            },
            "incompatible with expert_anchor_replace_worst",
        ),
    ],
)
def test_the_secondary_config_refuses_at_launch_not_at_the_first_replay_step(
    config: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_expert_anchor_secondary_config(config, future_len=80)


def test_a_secondary_horizon_appends_a_second_untruncated_anchor_column() -> None:
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=False,
        expert_horizon=1,
        secondary_horizon=80,
    )
    assert torch.equal(
        horizons.reshape(2, 5),
        torch.tensor([[80, 40, 40, 1, 80], [80, 40, 40, 1, 80]]),
    )


def test_omitting_the_secondary_horizon_is_byte_identical() -> None:
    kwargs = dict(
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=False,
        expert_horizon=1,
    )
    weights = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
    assert torch.equal(
        _build_awr_target_loss_horizons(weights, **kwargs),
        _build_awr_target_loss_horizons(weights, secondary_horizon=None, **kwargs),
    )


def test_the_dual_anchor_adds_the_step_one_pull_without_removing_the_smoothing() -> None:
    # This is the arithmetic the whole change exists for.  Each term is a mean
    # over its OWN horizon, so two slots at (1, a) and (80, b) give effective
    # per-step weights a/1 + b/80 at step 1 and b/80 at steps 2..80 -- i.e. the
    # untruncated objective is retained EXACTLY at b = expert_weight and the
    # executed-waypoint pull is added on top.  Truncation alone instead deletes
    # the b/80 column, which is what costs +9.73% within-plan jerk.
    T, a, b = 80, 0.4, 0.4
    horizons = _build_awr_target_loss_horizons(
        torch.zeros(3),
        batch_size=1,
        group_size=3,
        future_len=T,
        candidate_horizon=40,
        deterministic_first=False,
        expert_safe=torch.tensor([True]),
        expert_replace_worst=False,
        expert_horizon=1,
        secondary_horizon=T,
    )
    # Only the two anchor columns carry weight, so the gradient each step sees
    # is exactly the effective anchor weight there.
    weights = torch.tensor([0.0, 0.0, 0.0, a, b])
    error = torch.ones(5, T, requires_grad=True)
    (
        _mean_over_ego_loss_horizons(error, default_horizon=T, loss_horizons=horizons)
        * weights
    ).sum().backward()
    per_step = error.grad.sum(0)
    assert float(per_step[0]) == pytest.approx(a / 1 + b / T, rel=1e-6)
    for step in (1, 39, 40, T - 1):
        assert float(per_step[step]) == pytest.approx(b / T, rel=1e-6)

    # And the control's smoothing column is reproduced bit-for-bit: an
    # untruncated single slot at weight b gives b/T at every step, including 1.
    error_ctrl = torch.ones(1, T, requires_grad=True)
    (
        _mean_over_ego_loss_horizons(
            error_ctrl, default_horizon=T, loss_horizons=torch.tensor([T])
        )
        * b
    ).sum().backward()
    assert torch.allclose(error_ctrl.grad[0], per_step - torch.tensor([a] + [0.0] * (T - 1)))


def test_a_secondary_horizon_is_incompatible_with_replace_worst() -> None:
    with pytest.raises(ValueError, match="incompatible with expert_replace_worst"):
        _build_awr_target_loss_horizons(
            torch.tensor([2.0, 0.0, 1.0]),
            batch_size=1,
            group_size=3,
            future_len=80,
            candidate_horizon=40,
            deterministic_first=False,
            expert_safe=torch.tensor([True]),
            expert_replace_worst=True,
            expert_horizon=1,
            secondary_horizon=80,
        )


@pytest.mark.parametrize("horizon", [0, -1, 81])
def test_a_secondary_horizon_must_lie_inside_the_model_future(horizon: int) -> None:
    with pytest.raises(ValueError, match="secondary expert anchor loss horizon"):
        _build_awr_target_loss_horizons(
            torch.tensor([2.0, 0.0, 1.0]),
            batch_size=1,
            group_size=3,
            future_len=80,
            candidate_horizon=40,
            deterministic_first=False,
            expert_safe=torch.tensor([True]),
            expert_replace_worst=False,
            expert_horizon=1,
            secondary_horizon=horizon,
        )


def test_expert_anchor_horizon_truncates_the_replaced_slot_too() -> None:
    # replace_worst puts the anchor in a candidate slot rather than appending
    # it; the truncation has to follow the anchor, not the column index.
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([2.0, 0.0, 1.0, 0.0, 3.0, 4.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=False,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=True,
        expert_horizon=1,
    )
    assert torch.equal(
        horizons.reshape(2, 3),
        torch.tensor([[40, 1, 40], [40, 40, 40]]),
    )


def test_expert_anchor_horizon_defaults_to_the_full_model_future() -> None:
    # Every historical run omits the flag; omitting it must be byte-identical
    # to passing the full horizon, or the A/B is confounded at cycle 1.
    def build(expert_horizon: int | None) -> torch.Tensor:
        return _build_awr_target_loss_horizons(
            torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
            batch_size=2,
            group_size=3,
            future_len=80,
            candidate_horizon=40,
            deterministic_first=True,
            expert_safe=torch.tensor([True, True]),
            expert_replace_worst=False,
            expert_horizon=expert_horizon,
        )

    assert torch.equal(build(None), build(80))


@pytest.mark.parametrize("horizon", [0, 81])
def test_expert_anchor_horizon_fails_outside_the_model_future(horizon: int) -> None:
    with pytest.raises(ValueError, match="expert anchor loss horizon"):
        _build_awr_target_loss_horizons(
            torch.tensor([0.0, 1.0]),
            batch_size=1,
            group_size=2,
            future_len=80,
            candidate_horizon=40,
            deterministic_first=False,
            expert_safe=torch.tensor([True]),
            expert_replace_worst=False,
            expert_horizon=horizon,
        )


def test_short_anchor_horizon_raises_its_effective_weight_at_step_one() -> None:
    # The whole point of the truncation: both terms are weighted means over
    # their own horizon, so an anchor scored over 1 step pulls step 1 with
    # candidate_horizon times the gradient of one scored over 40.  Measured on
    # the cycle-1 mine this is what turns 0.04% of the standstill ego gradient
    # at the executed waypoint into 18%.
    error = torch.ones(1, 40, requires_grad=True)
    short = _mean_over_ego_loss_horizons(
        error, default_horizon=40, loss_horizons=torch.tensor([1])
    )
    short.sum().backward()
    step_one_short = float(error.grad[0, 0])

    error = torch.ones(1, 40, requires_grad=True)
    _mean_over_ego_loss_horizons(
        error, default_horizon=40, loss_horizons=torch.tensor([40])
    ).sum().backward()
    assert math.isclose(step_one_short / float(error.grad[0, 0]), 40.0, rel_tol=1e-6)


def test_expert_anchor_active_group_gate_drops_expert_only_sft_scenes() -> None:
    safe = torch.tensor([True, True, False, True])
    weights = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    actual = _mask_expert_anchor_to_active_awr_groups(
        safe,
        weights.flatten(),
        batch_size=4,
        group_size=3,
    )
    assert torch.equal(actual, torch.tensor([False, True, False, True]))


def test_short_horizon_mask_truncates_only_the_masked_scenes() -> None:
    # The speed-conditional arm: scene 0 is slow enough to need the executed
    # waypoint, scene 1 keeps the expert's full-horizon smoothing.  The mask
    # has to select per scene, not per column.
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, True]),
        expert_replace_worst=False,
        expert_horizon=1,
        expert_short_horizon_mask=torch.tensor([True, False]),
    )
    assert torch.equal(
        horizons.reshape(2, 4),
        torch.tensor([[80, 40, 40, 1], [80, 40, 40, 80]]),
    )


def test_short_horizon_mask_follows_the_anchor_into_a_replaced_slot() -> None:
    # replace_worst puts the anchor in the lowest-weight candidate column, so
    # the mask must land on that column and leave the unmasked scene's anchor
    # at the full horizon.
    horizons = _build_awr_target_loss_horizons(
        torch.tensor([5.0, 0.0, 6.0, 7.0, 0.0, 8.0]),
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=False,
        expert_safe=torch.tensor([True, True]),
        expert_replace_worst=True,
        expert_horizon=1,
        expert_short_horizon_mask=torch.tensor([False, True]),
    )
    assert torch.equal(
        horizons.reshape(2, 3),
        torch.tensor([[40, 80, 40], [40, 1, 40]]),
    )


def test_short_horizon_mask_defaults_to_truncating_every_anchor() -> None:
    common = dict(
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, True]),
        expert_replace_worst=False,
        expert_horizon=1,
    )
    assert torch.equal(
        _build_awr_target_loss_horizons(
            torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]), **common
        ),
        _build_awr_target_loss_horizons(
            torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
            expert_short_horizon_mask=torch.tensor([True, True]),
            **common,
        ),
    )


def test_short_horizon_mask_rejects_a_wrong_shaped_mask() -> None:
    with pytest.raises(ValueError, match="short-horizon mask"):
        _build_awr_target_loss_horizons(
            torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0]),
            batch_size=2,
            group_size=3,
            future_len=80,
            candidate_horizon=40,
            deterministic_first=True,
            expert_safe=torch.tensor([True, True]),
            expert_replace_worst=False,
            expert_horizon=1,
            expert_short_horizon_mask=torch.tensor([True, False, True]),
        )


def test_an_all_false_mask_reproduces_the_untruncated_anchor() -> None:
    # The safety property that lets the flag ship: a mask that selects nothing
    # must be bit-identical to not passing expert_horizon at all, so a config
    # whose speed ceiling never fires cannot silently change the objective.
    common = dict(
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, True]),
        expert_replace_worst=False,
    )
    weights = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
    assert torch.equal(
        _build_awr_target_loss_horizons(
            weights,
            expert_horizon=1,
            expert_short_horizon_mask=torch.zeros(2, dtype=torch.bool),
            **common,
        ),
        _build_awr_target_loss_horizons(weights, expert_horizon=None, **common),
    )


def test_active_group_gate_keep_mask_rearms_the_zero_weight_scenes() -> None:
    # Scene 0 is the case the exemption exists for: expert_safe but the whole
    # group cleared to zero weight, which at standstill is 87.6% of rows.  The
    # gate drops its anchor, so it trains on nothing at all; the keep mask
    # restores it.  Scene 2 shows the mask cannot manufacture an anchor where
    # no safe expert exists, and scene 3 that a moving active scene is
    # unaffected either way.
    safe = torch.tensor([True, True, False, True])
    weights = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    gated = _mask_expert_anchor_to_active_awr_groups(
        safe, weights.flatten(), batch_size=4, group_size=3
    )
    exempted = _mask_expert_anchor_to_active_awr_groups(
        safe,
        weights.flatten(),
        batch_size=4,
        group_size=3,
        keep_mask=torch.tensor([True, True, True, False]),
    )
    assert torch.equal(gated, torch.tensor([False, True, False, True]))
    assert torch.equal(exempted, torch.tensor([True, True, False, True]))


def test_an_all_false_keep_mask_reproduces_the_plain_gate() -> None:
    # The ship-safety property, mirroring the horizon mask's: a ceiling that
    # never fires must leave the gate bit-identical to today's behaviour.
    safe = torch.tensor([True, True, True])
    weights = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    common = dict(batch_size=3, group_size=2)
    assert torch.equal(
        _mask_expert_anchor_to_active_awr_groups(
            safe, weights.flatten(), keep_mask=torch.zeros(3, dtype=torch.bool),
            **common
        ),
        _mask_expert_anchor_to_active_awr_groups(
            safe, weights.flatten(), **common
        ),
    )


def test_active_group_gate_rejects_a_wrong_shaped_keep_mask() -> None:
    with pytest.raises(ValueError, match="keep mask"):
        _mask_expert_anchor_to_active_awr_groups(
            torch.tensor([True, True]),
            torch.tensor([1.0, 0.0, 0.0, 2.0]),
            batch_size=2,
            group_size=2,
            keep_mask=torch.tensor([True, False, True]),
        )


def test_low_speed_scene_mask_reads_the_velocity_column() -> None:
    state = torch.zeros(4, 10)
    state[:, 4] = torch.tensor([0.0, -0.4, 1.2, 8.0])
    assert torch.equal(
        _low_speed_scene_mask(
            {"ego_current_state": state}, threshold=1.0, flag_name="knob"
        ),
        torch.tensor([True, True, False, False]),
    )


def test_low_speed_scene_mask_is_off_at_zero_and_strict_about_the_column() -> None:
    state = torch.zeros(2, 10)
    assert (
        _low_speed_scene_mask(
            {"ego_current_state": state}, threshold=0.0, flag_name="knob"
        )
        is None
    )
    with pytest.raises(ValueError, match="knob needs ego_current_state"):
        _low_speed_scene_mask(
            {"ego_current_state": torch.zeros(2, 3)},
            threshold=1.0,
            flag_name="knob",
        )


def _anchor_with_travel(travels: list[float], T: int = 80) -> torch.Tensor:
    """Expert futures that reach ``travel`` metres by step 1 and hold there.

    Ego-frame, so the position at step t is the displacement by step t; making
    it constant after step 0 keeps the max-over-consumed-steps reading equal to
    the value under test at every horizon.
    """

    anchor = torch.zeros(len(travels), T, 4)
    anchor[:, :, 0] = torch.as_tensor(travels, dtype=torch.float32)[:, None]
    return anchor


def test_a_stationary_expert_is_safe_over_the_steps_the_anchor_scores():
    # Row 1 is the case the whole knob exists for: the human fails its
    # whole-future verdict but does not move over the scored step, so the
    # target is "stay where you are" and cannot collide with anything.
    safe = torch.tensor([True, False, False, False])
    anchor = _anchor_with_travel([0.0, 0.001, 0.400, 0.001])
    rescoped = _expert_safe_over_consumed_horizon(
        safe,
        anchor,
        max_step_m=0.005,
        horizon=1,
        short_horizon_mask=torch.tensor([True, True, True, False]),
    )
    # Row 2 moves 40 cm, far past the floor; row 3 is stationary but its anchor
    # is not truncated, so the stored whole-future verdict is the right scope.
    assert rescoped.tolist() == [True, True, False, False]


def test_rescoping_never_removes_a_stored_safe_verdict():
    safe = torch.tensor([True, True])
    rescoped = _expert_safe_over_consumed_horizon(
        safe,
        _anchor_with_travel([5.0, 9.0]),
        max_step_m=0.005,
        horizon=1,
        short_horizon_mask=None,
    )
    assert rescoped.tolist() == [True, True]


def test_rescoping_is_off_at_zero_and_at_a_full_length_horizon():
    safe = torch.tensor([False, False])
    anchor = _anchor_with_travel([0.0, 0.0], T=8)
    for kwargs in (
        {"max_step_m": 0.0, "horizon": 1},
        # A full-length anchor consumes every step, so the stored verdict
        # already covers exactly what is scored -- nothing to re-scope.
        {"max_step_m": 0.005, "horizon": 8},
    ):
        rescoped = _expert_safe_over_consumed_horizon(
            safe, anchor, short_horizon_mask=None, **kwargs
        )
        assert rescoped.tolist() == [False, False]


def test_rescoping_reads_the_whole_consumed_prefix_not_just_its_last_step():
    # A human that lurches 30 cm at step 0 and returns is NOT stay-put, even
    # though its position at the final consumed step is back near zero.
    anchor = torch.zeros(1, 80, 4)
    anchor[0, 0, 0] = 0.300
    rescoped = _expert_safe_over_consumed_horizon(
        torch.tensor([False]),
        anchor,
        max_step_m=0.005,
        horizon=4,
        short_horizon_mask=None,
    )
    assert rescoped.tolist() == [False]


def test_rescoping_rejects_an_anchor_that_does_not_match_the_scene_batch():
    with pytest.raises(ValueError, match="scene batch"):
        _expert_safe_over_consumed_horizon(
            torch.tensor([False, False]),
            _anchor_with_travel([0.0]),
            max_step_m=0.005,
            horizon=1,
            short_horizon_mask=None,
        )


def test_gate_off_arm_anchors_every_safe_row_including_dead_groups():
    """The shipped arm's exact config, through the real horizon builder.

    The arm is ``--no-expert_anchor_active_groups_only --expert_anchor_loss_
    horizon 1`` with no speed masks, which is a DIFFERENT path from the
    keep-mask exemption tested above: the active-group gate never runs at all,
    so ``expert_safe`` reaches the builder untouched.  What has to hold is that
    a scene whose candidates all carry zero weight -- 79% of standstill rows,
    because the positive-advantage margin cannot rank a group that shares one
    plan -- still gets a live single-step anchor.  That row is the whole point
    of the arm, and if the builder dropped it the run would be a null.
    """

    # Two scenes, three candidates each.  Scene 0 is a live AWR group; scene 1
    # is the dead group the arm exists to reach.
    weights = torch.tensor([2.0, 1.0, 3.0, 0.0, 0.0, 0.0])
    horizons = _build_awr_target_loss_horizons(
        weights,
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=torch.tensor([True, True]),
        expert_replace_worst=False,
        expert_horizon=1,
        expert_short_horizon_mask=None,
    ).reshape(2, 4)
    # Column 3 is the appended anchor: one step on BOTH rows, dead group
    # included.  Columns 0-2 are the candidates, deterministic-first at 80.
    assert torch.equal(
        horizons, torch.tensor([[80, 40, 40, 1], [80, 40, 40, 1]])
    )


def test_a_dead_groups_anchor_is_its_only_target_so_step_one_reaches_the_human():
    """Why the arm's fixed point is the human exactly, not an a_eff=16 blend.

    On a live group the anchor shares step 1 with the candidate mix, and that
    blend is what made the horizon-only arm price NEGATIVE: it shortens the
    step vector while keeping its lateral proportion, and the reward's implied
    steer atan(WB*2|y|/s^2) is quadratic in s, so the penalty inflates.  On a
    dead group there is nothing to blend with -- every candidate weight is
    zero -- so the descent has a single fixed point and it is the human.
    """

    anchor_weight = 0.4
    weights = torch.tensor([0.0, 0.0, 0.0, anchor_weight])
    # The weighted-mean aggregation the loss actually uses: with every
    # candidate at zero, the anchor's share of the step-1 gradient is 1.0
    # regardless of its own weight.
    share = weights / weights.sum()
    assert torch.equal(share, torch.tensor([0.0, 0.0, 0.0, 1.0]))

    policy_step1 = torch.tensor([0.200, 0.0180])   # 18 mm of lateral
    human_step1 = torch.tensor([0.201, 0.0002])
    # Gradient descent on a mean of squared errors whose only live term is the
    # anchor converges to the anchor, whatever the learning rate.
    current = policy_step1.clone()
    for _ in range(400):
        current = current - 0.05 * 2.0 * (current - human_step1)
    assert torch.allclose(current, human_step1, atol=1e-6)


def test_shipped_arm_rescopes_safety_then_weights_the_anchor():
    """The launched composition end to end: re-scope, then inject.

    Safety reaches the objective as the anchor's WEIGHT, not as its horizon --
    ``_build_awr_target_loss_horizons`` deliberately gives every anchor slot the
    same horizon to keep the compiled K+1 graph shape-aligned, so a horizon
    assertion cannot see this at all.  The weight is where a re-scoped verdict
    either resurrects a row or does not.

    Three scenes cover the only three outcomes the arm can produce.  Scene 2 is
    the one the proxy is deliberately conservative about: the human moves 40 cm
    over the consumed step, so no displacement argument makes it safe and it
    stays excluded.
    """

    group_size = 2
    expert_weight = 0.4
    stored_safe = torch.tensor([True, False, False])
    # Ego-frame futures: position at step t IS the displacement by step t.
    anchor = torch.zeros(3, 80, 4)
    anchor[0, :, 0] = 3.0        # safe already; travel is irrelevant to it
    anchor[1, :, 0] = 0.001      # stay-put: 1 mm over the consumed step
    anchor[2, :, 0] = 0.400      # genuinely moving: 40 cm

    rescoped = _expert_safe_over_consumed_horizon(
        stored_safe,
        anchor,
        max_step_m=0.005,
        horizon=1,
        short_horizon_mask=None,   # the arm carries no speed ceiling
    )
    assert torch.equal(rescoped, torch.tensor([True, True, False]))

    # Scene 0 is a live group; scenes 1 and 2 are dead ones, which is what makes
    # the anchor their only possible target.
    weights = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).flatten()
    trajectories = torch.zeros(3 * group_size, 80, 4)
    _, out = _inject_expert_anchor_batch(
        trajectories,
        weights,
        anchor,
        rescoped,
        group_size=group_size,
        expert_weight=expert_weight,
        replace_worst=False,
    )
    out = out.reshape(3, group_size + 1)
    assert torch.equal(out[:, -1], torch.tensor([0.4, 0.4, 0.0]))

    # The resurrected row now has a live target where it previously had none,
    # and that target is its ONLY one -- the whole point of the arm.
    assert out[1].sum() == pytest.approx(0.4)
    assert out[1, -1] / out[1].sum() == pytest.approx(1.0)
    # The row the proxy refuses is left exactly as the control leaves it: no
    # candidate mass, no anchor, no gradient.
    assert out[2].sum() == pytest.approx(0.0)


def test_the_dual_arm_gives_the_rescued_row_the_truncated_slot_only():
    """The next arm end to end, and the one property that is easy to get wrong.

    The re-scope resurrects a row by asking whether the human is safe over the
    steps the *truncated* anchor scores.  That verdict says nothing about steps
    2..80, so admitting the untruncated slot on it would train the tail toward a
    human who collides at 4 s -- on precisely the rows the arm exists to reach.
    So the second slot is admitted on the pre-rescope verdict instead, and a
    rescued row carries the step-1 pull with no tail term.

    Scene 0: safe before the re-scope   -> both slots.
    Scene 1: rescued by the re-scope    -> truncated slot only.
    Scene 2: unsafe either way          -> neither, exactly as the control.
    """

    group_size = 2
    stored_safe = torch.tensor([True, False, False])
    anchor = torch.zeros(3, 80, 4)
    anchor[0, :, 0] = 3.0
    anchor[1, :, 0] = 0.001
    anchor[2, :, 0] = 0.400

    rescoped = _expert_safe_over_consumed_horizon(
        stored_safe, anchor, max_step_m=0.005, horizon=1, short_horizon_mask=None
    )
    assert torch.equal(rescoped, torch.tensor([True, True, False]))

    weights = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).flatten()
    _, out = _inject_expert_anchor_batch(
        torch.zeros(3 * group_size, 80, 4),
        weights,
        anchor,
        rescoped,
        group_size=group_size,
        expert_weight=0.4,
        replace_worst=False,
        secondary_weight=0.4,
        # What the call site computes: rescoped AND the verdict as stored.
        secondary_safe=rescoped & stored_safe,
    )
    out = out.reshape(3, group_size + 2)
    assert torch.equal(out[:, -2], torch.tensor([0.4, 0.4, 0.0]))   # truncated
    assert torch.equal(out[:, -1], torch.tensor([0.4, 0.0, 0.0]))   # untruncated

    horizons = _build_awr_target_loss_horizons(
        weights,
        batch_size=3,
        group_size=group_size,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=False,
        expert_safe=rescoped,
        expert_replace_worst=False,
        expert_horizon=1,
        secondary_horizon=80,
    ).reshape(3, group_size + 2)
    # The horizon vector is uniform by design -- safety rides on the weight --
    # so the zero weight above is the whole mechanism, and the shapes agree.
    assert torch.equal(horizons[:, -2], torch.tensor([1, 1, 1]))
    assert torch.equal(horizons[:, -1], torch.tensor([80, 80, 80]))
    assert horizons.numel() == out.numel()

    # Scene 0 keeps the control's entire smoothing column (0.4/80 per step from
    # step 1 to 80) and adds 0.4/1 at step 1 on top; scene 1 gets only the pull.
    eff = out[:, -2:, None] / torch.tensor([[1.0], [80.0]])
    assert float(eff[0, 0, 0] + eff[0, 1, 0]) == pytest.approx(0.4 + 0.4 / 80)
    assert float(eff[1, 0, 0] + eff[1, 1, 0]) == pytest.approx(0.4)


def test_rescoping_is_bit_identical_to_the_control_when_the_flag_is_off():
    """Default-off must not perturb the arm this is being A/B'd against."""

    stored_safe = torch.tensor([True, False, False])
    anchor = torch.zeros(3, 80, 4)
    anchor[1, :, 0] = 0.001      # would be resurrected if the flag were on

    unchanged = _expert_safe_over_consumed_horizon(
        stored_safe, anchor, max_step_m=0.0, horizon=1, short_horizon_mask=None
    )
    assert torch.equal(unchanged, stored_safe)


def test_step_scoped_verdict_reaches_a_moving_human_the_proxy_cannot():
    """The exact verdict recovers rows the displacement proxy must refuse.

    This is the whole reason the step-resolved verdict is worth a buffer
    sidecar.  Row 1 is a human driving forward in a straight line whose hard
    gate fires at 4 s; a one-step anchor never reaches step 40, so the target is
    safe -- but the human has travelled 380 mm by step 1, so no displacement
    threshold small enough to be safe can accept it.  Measured on 400k overlay
    rows, that describes 84.5% of the rows the anchor should reach.
    """

    stored_safe = torch.tensor([True, False, False, False])
    anchor = torch.zeros(4, 80, 4)
    anchor[0, :, 0] = 3.0
    anchor[1, :, 0] = 0.380      # moving, gate at 4 s -> recoverable
    anchor[2, :, 0] = 0.380      # moving, gate on the executed step -> not
    anchor[3, :, 0] = 0.001      # stationary, gate at 4 s -> both agree
    # future_len is the "never fires" sentinel.
    gate_step = torch.tensor([80.0, 40.0, 0.0, 40.0])

    proxy_only = _expert_safe_over_consumed_horizon(
        stored_safe, anchor, max_step_m=0.005, horizon=1, short_horizon_mask=None
    )
    assert torch.equal(proxy_only, torch.tensor([True, False, False, True]))

    exact = _expert_safe_over_consumed_horizon(
        stored_safe,
        anchor,
        max_step_m=0.0,
        horizon=1,
        short_horizon_mask=None,
        gate_step=gate_step,
    )
    # Row 1 is the population the proxy cannot see; row 2 stays refused because
    # its gate fires at step 0, which IS the step the anchor consumes.
    assert torch.equal(exact, torch.tensor([True, True, False, True]))


def test_step_scoped_verdict_collapses_to_expert_safe_at_the_full_horizon():
    """The sentinel must make the re-scoping exact, not merely permissive.

    ``expert_safe`` is the same verdict at horizon == future_len.  If the
    sentinel did not reconstruct it there, the flag would be silently widening
    the gate on every untruncated anchor rather than re-scoping a truncated one.
    """

    stored_safe = torch.tensor([True, False, False])
    anchor = torch.zeros(3, 80, 4)
    anchor[:, :, 0] = 3.0
    gate_step = torch.tensor([80.0, 79.0, 0.0])

    for horizon in (80, 81):
        collapsed = _expert_safe_over_consumed_horizon(
            stored_safe,
            anchor,
            max_step_m=0.0,
            horizon=horizon,
            short_horizon_mask=None,
            gate_step=gate_step,
        )
        assert torch.equal(collapsed, stored_safe), horizon

    # And a row whose gate fires at the last consumed step is never safe: the
    # comparison is ``>=``, so step 39 does not clear a 40-step anchor.
    boundary = _expert_safe_over_consumed_horizon(
        torch.tensor([False, False]),
        torch.zeros(2, 80, 4),
        max_step_m=0.0,
        horizon=40,
        short_horizon_mask=None,
        gate_step=torch.tensor([39.0, 40.0]),
    )
    assert torch.equal(boundary, torch.tensor([False, True]))


def test_step_scoped_flag_is_inert_without_a_gate_step_array():
    """A buffer mined before the sidecar existed must behave as flag-off."""

    stored_safe = torch.tensor([True, False])
    anchor = torch.zeros(2, 80, 4)
    anchor[1, :, 0] = 0.380

    assert torch.equal(
        _expert_safe_over_consumed_horizon(
            stored_safe, anchor, max_step_m=0.0, horizon=1,
            short_horizon_mask=None, gate_step=None,
        ),
        stored_safe,
    )
    # The reconstruction an older buffer gets (safe -> sentinel, unsafe -> 0)
    # must reproduce expert_safe at EVERY horizon, so the flag cannot recover a
    # row the mine never justified.
    reconstructed = torch.where(
        stored_safe, torch.full((2,), 80.0), torch.zeros(2)
    )
    for horizon in (1, 2, 40, 79):
        assert torch.equal(
            _expert_safe_over_consumed_horizon(
                stored_safe, anchor, max_step_m=0.0, horizon=horizon,
                short_horizon_mask=None, gate_step=reconstructed,
            ),
            stored_safe,
        ), horizon
    # The writer creates the gate-step array unconditionally, so a producer that
    # does not populate it leaves an all-zeros array on disk.  Zero reads as
    # "gates at the executed waypoint", which no horizon can clear, so that
    # buffer also degrades to the un-scoped verdict rather than resurrecting.
    for horizon in (1, 2, 40, 79, 80):
        assert torch.equal(
            _expert_safe_over_consumed_horizon(
                stored_safe, anchor, max_step_m=0.0, horizon=horizon,
                short_horizon_mask=None, gate_step=torch.zeros(2),
            ),
            stored_safe,
        ), horizon


def test_loss_starts_average_only_the_open_window() -> None:
    # A start offset must exclude the leading steps from BOTH the numerator and
    # the denominator.  Step 0 carries a value no other step has, so a wrong
    # denominator (dividing the shortened sum by the full horizon) cannot pass.
    per_timestep = torch.zeros(2, 4)
    per_timestep[:, 0] = 100.0
    per_timestep[:, 1:] = 2.0
    full = _mean_over_ego_loss_horizons(
        per_timestep,
        default_horizon=4,
        loss_horizons=torch.tensor([4, 4]),
    )
    assert torch.allclose(full, torch.full((2,), (100.0 + 6.0) / 4.0))
    offset = _mean_over_ego_loss_horizons(
        per_timestep,
        default_horizon=4,
        loss_horizons=torch.tensor([4, 4]),
        loss_starts=torch.tensor([1, 1]),
    )
    assert torch.allclose(offset, torch.full((2,), 2.0))


def test_loss_starts_are_per_target() -> None:
    # The whole point is a batch that mixes an offset candidate with an anchor
    # that keeps step 0, so the two must not share one window.
    per_timestep = torch.zeros(2, 4)
    per_timestep[:, 0] = 9.0
    per_timestep[:, 1:] = 3.0
    actual = _mean_over_ego_loss_horizons(
        per_timestep,
        default_horizon=4,
        loss_horizons=torch.tensor([4, 4]),
        loss_starts=torch.tensor([1, 0]),
    )
    assert torch.allclose(actual, torch.tensor([3.0, (9.0 + 9.0) / 4.0]))


def test_loss_starts_reweight_a_timestep_profile() -> None:
    # With coeff_timestep the denominator is the weight mass actually applied,
    # so dropping step 0 must drop its weight too.
    per_timestep = torch.ones(1, 4)
    weights = torch.tensor([5.0, 1.0, 1.0, 1.0])
    actual = _mean_over_ego_loss_horizons(
        per_timestep * weights[None, :],
        default_horizon=4,
        loss_horizons=torch.tensor([4]),
        timestep_weights=weights,
        loss_starts=torch.tensor([1]),
    )
    assert torch.allclose(actual, torch.tensor([1.0]))


def test_loss_starts_reject_an_empty_window() -> None:
    # An empty window divides by zero and silently drops the target from the
    # objective; that must be an error, not a NaN.
    with pytest.raises(ValueError, match=r"starts must lie in \[0, horizon\)"):
        _mean_over_ego_loss_horizons(
            torch.ones(1, 4),
            default_horizon=4,
            loss_horizons=torch.tensor([2]),
            loss_starts=torch.tensor([2]),
        )


def test_loss_starts_require_horizons() -> None:
    with pytest.raises(ValueError, match="require per-target horizons"):
        _mean_over_ego_loss_horizons(
            torch.ones(1, 4),
            default_horizon=4,
            loss_horizons=None,
            loss_starts=torch.tensor([1]),
        )


def test_candidate_loss_starts_spare_every_anchor_slot() -> None:
    # Shape-alignment with the horizon vector is the invariant that keeps the
    # two from drifting, and anchors must keep the executed step.
    weights = torch.ones(2 * 3)
    safe = torch.tensor([True, False])
    starts = _build_awr_target_loss_starts(
        weights,
        batch_size=2,
        group_size=3,
        candidate_start=1,
        expert_safe=safe,
        expert_replace_worst=False,
        has_secondary=True,
    )
    horizons = _build_awr_target_loss_horizons(
        weights,
        batch_size=2,
        group_size=3,
        future_len=80,
        candidate_horizon=40,
        deterministic_first=True,
        expert_safe=safe,
        expert_replace_worst=False,
        expert_horizon=1,
        secondary_horizon=80,
    )
    assert starts.shape == horizons.shape
    # [K candidates | anchor | secondary] per scene: candidates offset, both
    # anchor slots at 0.
    assert starts.tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]


def test_candidate_loss_starts_follow_a_replaced_anchor_slot() -> None:
    # expert_replace_worst overwrites a candidate slot in place, so that slot is
    # an anchor now and must keep step 0 even though it sits among candidates.
    weights = torch.tensor([0.9, 0.1, 0.5, 0.4, 0.7, 0.2])
    starts = _build_awr_target_loss_starts(
        weights,
        batch_size=2,
        group_size=3,
        candidate_start=1,
        expert_safe=torch.tensor([True, False]),
        expert_replace_worst=True,
        has_secondary=False,
    )
    # Scene 0 is safe: its lowest-weight slot (index 1) becomes the anchor.
    # Scene 1 is unsafe, so no slot is replaced.
    assert starts.tolist() == [1, 0, 1, 1, 1, 1]
