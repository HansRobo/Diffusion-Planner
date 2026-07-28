"""Fidelity tests for ``--rl_paper_exact``: the published HDP-RL configuration.

Every assertion here is anchored to a source that can be re-read:

- ``reference/papers/hyper_diffusion_planner_paper/src/neurips_2026.tex``
  (Eq. eq:optim / eq:awr / eq:awr_hybrid, Table tab:param, Appendix app:rewards,
  Appendix ap:implementation, Appendix "Hybrid Loss");
- ``reference/papers/.../src/code.tex`` (Algorithm 1) and ``code_rl.tex``
  (Algorithm 2);
- the authors' released code under ``reference/external/Hyper-Diffusion-Planner``
  (``HDP-navsim/.../dp_vla_rl_agent.py``, ``HDP-nuplan/.../traj_kinematics.py``).

Where the sources contradict each other the divergence is recorded in
``docs/hdp_rl_paper_fidelity.md``; the tests lock the value this repository ships.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.hdp_rl_paper_exact import (
    HYBRID_FIELDS,
    PAPER_REWARD_VARIANTS,
    apply_paper_exact_settings,
    assert_paper_exact,
    base_corpus_mismatches,
    paper_exact_fields,
    paper_exact_values,
    paper_hybrid_values,
)
from diffusion_planner.hdp_rl_utils import compute_reward_weighted_loss, compute_reward_weights
from diffusion_planner.loss import _detached_integral
from train_hdp_rl_predictor import get_args

from diffusion_planner import hdp_rl_utils

_NORMALIZATION = Path(__file__).resolve().parents[1] / "normalization.json"


# The frozen IL base this pipeline post-trains. Paper-exact reads omega and W
# from here rather than from Table 3, and checks the corpus against it, so the
# tests need a base run directory that looks like a real one. The values are the
# ones the shipped base80 run recorded (omega=0.01, W=10, StatePerturbation on).
_BASE_ARGS = {
    "planning_hybrid_loss": 0.01,
    "hybrid_loss_window": 10,
    "train_set_list": "train.json",
    "valid_set_list": "valid.json",
    "extra_train_set_list": [],
    "extra_train_set_repeat": 0,
    "filter_skipped": True,
    "train_subsample_step": 1,
    "align_legacy_neighbor_futures": False,
    "use_data_augment": True,
    "augment_type": "quintic",
    "augment_prob": 0.5,
    "num_refine": 20,
    "ego_past_noise_std": 0.1,
    "use_smoothing_future_trajectory": True,
}

_BASE_RUN = Path(tempfile.mkdtemp(prefix="hdp_paper_exact_base_"))
(_BASE_RUN / "args.json").write_text(json.dumps(_BASE_ARGS), encoding="utf-8")


def _write_base(**overrides):
    """A second base run directory whose args.json differs from the default."""
    run = Path(tempfile.mkdtemp(prefix="hdp_paper_exact_base_"))
    (run / "args.json").write_text(json.dumps({**_BASE_ARGS, **overrides}), encoding="utf-8")
    return run


def _required_args(base_run=_BASE_RUN):
    return [
        "--exp_name",
        "paper_exact",
        "--save_dir",
        "out",
        "--train_set_list",
        "train.json",
        "--valid_set_list",
        "valid.json",
        "--init_weights_path",
        str(base_run / "latest.pth"),
        "--normalization_file_path",
        str(_NORMALIZATION),
    ]


def _paper_args(*extra, base_run=_BASE_RUN):
    return get_args(
        _required_args(base_run) + ["--rl_paper_exact", "True", "--rl_replay_dir", "replay", *extra]
    )


# ───────────────────────── Table 3 and the reward appendix ──────────────────


def test_table3_rl_hyperparameters_are_locked():
    """neurips_2026.tex tab:param, the RL block."""
    args = _paper_args()
    assert args.num_generations == 32  # Group size
    assert args.rl_reward_beta == 1.0  # Temperature beta
    assert args.rl_ema_update_rate == 0.05  # EMA
    assert args.rl_reward_w_risk == 1.0  # lambda_risk
    assert args.rl_reward_w_follow == 3.0  # lambda_follow
    assert args.rl_reward_w_lane == 2.5  # lambda_lane
    # omega and W are the one published pair this mode does NOT take from Table 3:
    # see test_hybrid_norm_is_inherited_from_the_frozen_il_base.


def test_total_training_reward_matches_the_appendix_cases():
    """app:rewards, "Total Training Reward"."""
    multi = _paper_args()
    assert multi.rl_reward_w_safety == 0.0
    assert (
        multi.rl_reward_w_risk,
        multi.rl_reward_w_follow,
        multi.rl_reward_w_lane,
    ) == (1.0, 3.0, 2.5)

    single = _paper_args("--rl_paper_reward", "single")
    assert single.rl_reward_w_safety == 1.0
    assert (
        single.rl_reward_w_risk,
        single.rl_reward_w_follow,
        single.rl_reward_w_lane,
    ) == (0.0, 0.0, 0.0)


def test_total_reward_is_a_plain_weighted_sum():
    """The appendix total reward has no gate, no progress term, no road-border term."""
    args = _paper_args()
    assert args.rl_reward_aggregation == "weighted_sum"
    assert args.rl_behavior_gate == "none"
    assert args.rl_reward_w_progress == 0.0
    assert args.rl_reward_w_road_border == 0.0
    assert args.rl_red_light_constraint is False
    assert args.rl_occupancy_use_road_border is False


def test_rewards_cover_the_full_planning_horizon():
    """app:rewards: every reward is "evaluated ... over the planning horizon of L steps"."""
    args = _paper_args()
    assert args.rl_reward_horizon_steps == 0  # 0 == full horizon
    assert args.rl_candidate_loss_horizon == 0


def test_released_rollout_schedule_is_reproduced():
    """HDP-navsim config/agent/dp_vla_rl_agent.yaml, ``rl_config``."""
    args = _paper_args()
    assert args.rl_rollout_steps == 5  # rollout_steps: 5
    assert args.rl_rollout_interval == 10  # replay_buffer_update_epoch: 10
    assert args.rl_updates_per_rollout == 1  # max_rollout_iter: 1


def test_awr_objective_has_no_repo_only_terms():
    """eq:awr_hybrid is exp(beta*r) times the hybrid loss and nothing else."""
    args = _paper_args()
    assert args.rl_bc_weight == 0.0  # no behaviour-cloning anchor in the paper
    assert args.rl_candidate_aug_prob == 0.0  # candidates come from pi^{k-1} only
    assert args.rl_first_waypoint_gate is False  # no candidate rejection in the paper
    assert args.rl_noise_scale == 1.0  # pi^{k-1} sampled at its own temperature
    assert args.rl_diffusion_t_min == 0.0  # expectation over the full t range
    assert args.rl_diffusion_t_max == 1.0
    assert args.rl_reward_source == "native"


def test_group_normalization_and_ema_come_from_the_paper():
    """ap:implementation: group normalization, constant-group discard, EMA."""
    args = _paper_args()
    assert args.rl_reward_normalize == "group"
    assert args.advantage_eps == 1e-6  # the 1e-6 of code_rl.tex Algorithm 2
    assert args.rl_init_use_ema is True


# ─────────────────────────────── the mode itself ────────────────────────────


def test_mode_is_off_by_default_and_changes_nothing():
    args = get_args(_required_args())
    assert args.rl_paper_exact is False
    assert args.rl_paper_exact_changes == []
    # A repo default that the paper contradicts, left untouched.
    assert args.num_generations != 32


def test_applied_changes_are_recorded_for_the_run_log():
    args = _paper_args()
    changed = {line.split(":", 1)[0] for line in args.rl_paper_exact_changes}
    assert "num_generations" in changed
    assert "rl_behavior_gate" in changed
    # Every recorded line carries its citation.
    assert all("  [" in line for line in args.rl_paper_exact_changes)


@pytest.mark.parametrize(
    "option, value",
    [
        ("--num_generations", "8"),
        ("--rl_reward_beta", "0.5"),
        ("--rl_behavior_gate", "safety"),
    ],
)
def test_explicit_flag_contradicting_the_paper_is_rejected(option, value):
    extra = [option] if value is None else [option, value]
    with pytest.raises(ValueError, match="cannot be combined with these explicit overrides"):
        _paper_args(*extra)


def test_explicit_flag_agreeing_with_the_paper_is_accepted():
    args = _paper_args("--num_generations", "32", "--rl_reward_beta", "1.0")
    assert args.num_generations == 32
    assert args.rl_reward_beta == 1.0


def test_replay_directory_is_required():
    with pytest.raises(ValueError, match="requires\n?\\s*--rl_replay_dir"):
        get_args(_required_args() + ["--rl_paper_exact", "True"])


def test_assert_paper_exact_catches_later_mutation():
    args = _paper_args()
    assert_paper_exact(args, "multi")
    args.rl_reward_beta = 0.5
    with pytest.raises(ValueError, match="Run is not paper-exact"):
        assert_paper_exact(args, "multi")


def test_every_controlled_field_exists_on_the_trainer():
    args = get_args(_required_args())
    for field in paper_exact_fields():
        assert hasattr(args, field), field


def test_unknown_reward_variant_is_rejected():
    with pytest.raises(ValueError, match="Unsupported rl_paper_reward"):
        paper_exact_values("dagger", SimpleNamespace(future_len=80))


def test_horizon_relative_values_follow_the_configured_horizon():
    for horizon in (8, 80):
        values = paper_hybrid_values(SimpleNamespace(future_len=horizon))
        assert values["hybrid_loss_window"] == horizon - 1
        assert values["planning_hybrid_loss"] == 0.1


def test_apply_rejects_a_field_the_trainer_does_not_have():
    for variant in PAPER_REWARD_VARIANTS:
        with pytest.raises(AttributeError, match="drifted apart"):
            apply_paper_exact_settings(SimpleNamespace(future_len=80), variant)


# ──────────── the hybrid norm: inherited from the frozen IL base ─────────────
#
# IL is never retrained here, so omega and W are taken from the checkpoint the
# run post-trains rather than from Table 3. Which pair is better *for RL* is an
# experiment, and --rl_hybrid_ablation is how it is asked.


def test_hybrid_norm_is_inherited_from_the_frozen_il_base():
    args = _paper_args()
    assert args.planning_hybrid_loss == _BASE_ARGS["planning_hybrid_loss"]
    assert args.hybrid_loss_window == _BASE_ARGS["hybrid_loss_window"]
    # Not the published pair -- and the run log has to say so, with both values.
    published = paper_hybrid_values(args)
    assert published != {f: getattr(args, f) for f in HYBRID_FIELDS}
    log = "\n".join(args.rl_paper_exact_changes)
    for field in HYBRID_FIELDS:
        assert field in log
        assert repr(published[field]) in log
    assert "frozen IL base" in log


def test_hybrid_norm_follows_whichever_base_the_run_starts_from():
    base = _write_base(planning_hybrid_loss=0.05, hybrid_loss_window=20)
    args = _paper_args(base_run=base)
    assert args.planning_hybrid_loss == 0.05
    assert args.hybrid_loss_window == 20


@pytest.mark.parametrize(
    "option, value",
    [
        ("--planning_hybrid_loss", "0.1"),  # tab:param omega
        ("--hybrid_loss_window", "79"),  # Appendix W = L - 1
    ],
)
def test_explicit_hybrid_flag_contradicting_the_base_is_rejected(option, value):
    with pytest.raises(ValueError, match="does not retrain IL"):
        _paper_args(option, value)


def test_explicit_hybrid_flag_agreeing_with_the_base_is_accepted():
    args = _paper_args("--planning_hybrid_loss", "0.01", "--hybrid_loss_window", "10")
    assert args.planning_hybrid_loss == 0.01
    assert args.hybrid_loss_window == 10


def test_ablation_releases_the_hybrid_norm_and_stamps_the_run():
    args = _paper_args(
        "--rl_hybrid_ablation",
        "True",
        "--planning_hybrid_loss",
        "0.1",
        "--hybrid_loss_window",
        "79",
    )
    assert args.planning_hybrid_loss == 0.1
    assert args.hybrid_loss_window == 79
    log = "\n".join(args.rl_paper_exact_changes)
    assert "released for ablation" in log
    # The arm is only readable if the log carries both comparison points.
    assert "frozen base" in log and "paper" in log


def test_a_base_without_args_json_is_rejected():
    orphan = Path(tempfile.mkdtemp(prefix="hdp_paper_exact_orphan_")) / "run"
    orphan.mkdir()
    with pytest.raises(ValueError, match="no args.json was found beside it"):
        _paper_args(base_run=orphan)


# ─────────── the corpus: identical to the run that produced the base ─────────


@pytest.mark.parametrize(
    "option, value",
    [
        ("--train_subsample_step", "10"),
        ("--extra_train_set_repeat", "10"),
        ("--filter_skipped", "False"),
        ("--align_legacy_neighbor_futures", "True"),
    ],
)
def test_a_different_corpus_than_the_base_is_rejected(option, value):
    with pytest.raises(ValueError, match="corpus and the input perturbation"):
        _paper_args(option, value)


@pytest.mark.parametrize(
    "option, value",
    [
        ("--use_data_augment", "False"),
        ("--augment_prob", "0.25"),
        ("--augment_type", "bridge"),
        ("--num_refine", "5"),
        ("--ego_past_noise_std", "0.0"),
        ("--use_smoothing_future_trajectory", "False"),
    ],
)
def test_a_different_perturbation_than_the_base_is_rejected(option, value):
    with pytest.raises(ValueError, match="corpus and the input perturbation"):
        _paper_args(option, value)


def test_corpus_check_can_be_waived_deliberately():
    args = _paper_args("--rl_base_corpus_check", "False", "--train_subsample_step", "10")
    assert args.train_subsample_step == 10


def test_extra_manifests_are_compared_by_basename():
    # The same artifact has a different absolute path in each checkout, so the
    # check must not fire on the prefix alone.
    base_args = {**_BASE_ARGS, "extra_train_set_list": ["/node02/artifacts/right_turn.json"]}
    args = SimpleNamespace(
        **{k: v for k, v in _BASE_ARGS.items() if k != "extra_train_set_list"},
        extra_train_set_list=["/node01/artifacts/right_turn.json"],
    )
    assert base_corpus_mismatches(args, base_args) == []
    args.extra_train_set_list = ["/node01/artifacts/left_turn.json"]
    assert any("extra_train_set_list" in m for m in base_corpus_mismatches(args, base_args))


def test_a_base_that_does_not_record_a_corpus_field_is_reported():
    base_args = {k: v for k, v in _BASE_ARGS.items() if k != "augment_prob"}
    args = SimpleNamespace(**_BASE_ARGS)
    assert base_corpus_mismatches(args, base_args) == [
        "  augment_prob: the base args.json does not record it"
    ]


# ───────────────────── Algorithm 2, verbatim reimplementation ───────────────


def _algorithm_2_weights(r, beta, eps=1e-6):
    """code_rl.tex Algorithm 2, ``rl_hybrid_loss``.

    The listing reads ``r_n = (r - r.mean() / (r.std() + 1e-6)`` -- an unbalanced
    parenthesis that also divides only the mean. The intended expression, and the
    one the released code implements, is the group-relative normalization of
    deepseekmath cited next to it in ap:implementation.
    """
    r_n = (r - r.mean()) / (r.std() + eps)
    return torch.exp(beta * r_n).detach()


def test_group_weights_reproduce_algorithm_2():
    reward = torch.tensor([0.10, 0.40, 0.35, 0.90])
    weights, valid = compute_reward_weights(
        reward,
        num_scenes=1,
        n=4,
        normalize="group",
        beta=1.0,
        eps=1e-6,
    )
    assert bool(valid.all())
    torch.testing.assert_close(weights, _algorithm_2_weights(reward, beta=1.0))


def test_beta_enters_the_exponent_exactly_once():
    reward = torch.tensor([0.10, 0.40, 0.35, 0.90])
    for beta in (0.5, 1.0, 2.0):
        weights, _ = compute_reward_weights(
            reward, num_scenes=1, n=4, normalize="group", beta=beta, eps=1e-6
        )
        torch.testing.assert_close(weights, _algorithm_2_weights(reward, beta=beta))


def test_constant_reward_group_is_discarded():
    """ap:implementation: "we discard samples in which all actions receive identical rewards"."""
    reward = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.1, 0.9, 0.2, 0.8])
    weights, valid = compute_reward_weights(
        reward, num_scenes=2, n=4, normalize="group", beta=1.0, eps=1e-6
    )
    assert not bool(valid[:4].any())  # constant group dropped, not left at exp(0) == 1
    assert bool(valid[4:].all())
    assert float(weights[:4].abs().sum()) == 0.0


def test_reward_weighted_loss_matches_algorithm_2(monkeypatch):
    """eq:awr: the loss is the reward-weighted mean of the per-sample hybrid distance."""
    per_sample = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reward = torch.tensor([0.10, 0.40, 0.35, 0.90])

    def fake_policy_loss(_model, _inputs, target, _args, *_pos, **_kwargs):
        assert target.shape[0] == per_sample.shape[0]
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
        reward=reward,
        num_scenes=1,
        n=4,
        args=SimpleNamespace(
            rl_reward_normalize="group",
            rl_reward_beta=1.0,
            advantage_eps=1e-6,
            ddp=False,
            rl_bc_weight=0.0,
            rl_bc_active_groups_only=True,
        ),
    )
    expected = (_algorithm_2_weights(reward, beta=1.0) * per_sample).mean()
    torch.testing.assert_close(result["loss"], expected)


# ───────────────── Algorithm 1 / detached_integral, verbatim port ───────────


def _released_detached_integral(u, detach_window_size):
    """Verbatim ``detached_integral`` from HDP-navsim dp_vla_agent.py (correct axis)."""
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[..., :detach_window_size, :] = 0
    sum_recent = cum_normal - shifted

    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[..., :detach_window_size, :] = 0

    return cum_detach_shifted + sum_recent


@pytest.mark.parametrize("window", [1, 3, 7, 8])
def test_detached_integral_matches_the_released_port(window):
    base = torch.randn(2, 8, 2, dtype=torch.float64)

    ours = base.clone().requires_grad_(True)
    theirs = base.clone().requires_grad_(True)

    out_ours = _detached_integral(ours, window)
    out_theirs = _released_detached_integral(theirs, window)
    torch.testing.assert_close(out_ours, out_theirs)

    # The stop-gradient pattern is the point of Algorithm 1, so compare gradients too.
    (out_ours * base).sum().backward()
    (out_theirs * base).sum().backward()
    torch.testing.assert_close(ours.grad, theirs.grad)


def test_detach_window_slices_time_not_the_coordinate_axis():
    """Regression guard for the HDP-nuplan axis bug.

    ``traj_kinematics.py`` writes ``shifted[:, :, :detach_window_size] = 0`` on a
    ``(B, T, D)`` tensor, which zeroes the whole tensor whenever ``W >= D``. The
    window then has no effect at all and the loss reduces to a plain cumsum.
    """
    v = torch.randn(1, 6, 2, dtype=torch.float64, requires_grad=True)
    out = _detached_integral(v, 3)
    # Only the last step's gradient path is exercised, so a W-limited window must
    # differ from the full-gradient cumsum the buggy indexing collapses to.
    grad_windowed = torch.autograd.grad(out[0, -1].sum(), v, retain_graph=True)[0]
    grad_full = torch.autograd.grad(torch.cumsum(v, dim=-2)[0, -1].sum(), v)[0]
    assert not torch.allclose(grad_windowed, grad_full)
    assert int(grad_windowed[0, :, 0].count_nonzero()) == 3  # exactly W time steps
