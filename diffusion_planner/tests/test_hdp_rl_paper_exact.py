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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.hdp_rl_paper_exact import (
    PAPER_REWARD_VARIANTS,
    apply_paper_exact_settings,
    assert_paper_exact,
    paper_exact_fields,
    paper_exact_values,
)
from diffusion_planner.hdp_rl_utils import compute_reward_weighted_loss, compute_reward_weights
from diffusion_planner.loss import _detached_integral
from train_hdp_rl_predictor import get_args

from diffusion_planner import hdp_rl_utils

_NORMALIZATION = Path(__file__).resolve().parents[1] / "normalization.json"


def _required_args():
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
        "init.pth",
        "--normalization_file_path",
        str(_NORMALIZATION),
    ]


def _paper_args(*extra):
    return get_args(
        _required_args() + ["--rl_paper_exact", "True", "--rl_replay_dir", "replay", *extra]
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
    assert args.planning_hybrid_loss == 0.1  # omega, IL block, reused by eq:awr_hybrid
    assert args.hybrid_loss_window == args.future_len - 1  # Appendix: "we set W=L-1"


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
        ("--hybrid_loss_window", "10"),
        ("--planning_hybrid_loss=0.01", None),
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
        values = paper_exact_values("multi", SimpleNamespace(future_len=horizon))
        assert values["hybrid_loss_window"] == horizon - 1


def test_apply_rejects_a_field_the_trainer_does_not_have():
    for variant in PAPER_REWARD_VARIANTS:
        with pytest.raises(AttributeError, match="drifted apart"):
            apply_paper_exact_settings(SimpleNamespace(future_len=80), variant)


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
