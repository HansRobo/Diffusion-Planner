"""Flow-matching (linear path, x0-parameterized) mode for the HDP decoder.

The FM mode must reuse the exact x0-prediction training contract (hybrid loss, turn
head, RL reconstruction) while swapping only the corruption schedule and the sampler.
"""

import json

import pytest
import torch
from diffusion_planner.loss import sample_diffusion_time
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.diffusion_utils.fm_solver import FMEulerSolver
from diffusion_planner.model.diffusion_utils.sde import LinearFlowPath, VPSDE_linear
from diffusion_planner.model.module.decoder import Decoder, compute_training_loss
from diffusion_planner.train import assert_checkpoint_compatible
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def _hdp_config(**overrides):
    args = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
        hidden_dim=32,
        decoder_depth=1,
        **overrides,
    )
    args.state_normalizer = StateNormalizer(
        [[[10.0, 0.0, 0.0, 0.0]]],
        [[[20.0, 20.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer({})
    return args


def test_linear_flow_path_matches_linear_interpolation_schedule():
    path = LinearFlowPath()
    t = torch.tensor([0.0, 0.25, 1.0])
    torch.testing.assert_close(path.marginal_alpha(t), 1.0 - t)
    torch.testing.assert_close(path.marginal_prob_std(t), t)
    assert path.T == 1.0

    x = torch.randn(3, 1, 8, 4)
    mean, std = path.marginal_prob(x, t)
    torch.testing.assert_close(mean, (1.0 - t).view(3, 1, 1, 1) * x)
    torch.testing.assert_close(std, t.view(3, 1, 1, 1))


def test_linear_flow_path_schedule_outputs_do_not_alias_the_time_tensor():
    path = LinearFlowPath()
    t = torch.tensor([0.25, 0.5])
    std = path.marginal_prob_std(t)
    std.mul_(2.0)
    torch.testing.assert_close(t, torch.tensor([0.25, 0.5]))


def test_logit_normal_time_sampling_concentrates_mid_trajectory():
    """SD3 (Esser et al. 2024): t = sigmoid(z), z ~ N(0, 1) beats uniform for RF."""
    torch.manual_seed(0)
    eps = 1e-3
    t = sample_diffusion_time(20_000, torch.device("cpu"), eps, "logit_normal")
    assert t.shape == (20_000,)
    assert float(t.min()) >= eps and float(t.max()) <= 1.0
    assert abs(float(t.mean()) - 0.5) < 0.02
    # Mid-trajectory mass must exceed uniform's 0.5 over (0.25, 0.75).
    assert float(((t > 0.25) & (t < 0.75)).float().mean()) > 0.6


def test_beta_high_noise_time_sampling_emphasizes_the_noise_end():
    """pi0-style shifted Beta(1.5, 1) mass toward high noise (t -> 1 in our convention)."""
    torch.manual_seed(0)
    eps = 1e-3
    t = sample_diffusion_time(20_000, torch.device("cpu"), eps, "beta_high_noise")
    assert t.shape == (20_000,)
    assert float(t.min()) >= eps and float(t.max()) <= 1.0
    # Beta(1.5, 1) mean is 0.6, mapped onto [eps, 1].
    assert abs(float(t.mean()) - 0.6) < 0.02
    assert float((t > 0.5).float().mean()) > 0.55


def test_unknown_time_sampling_method_is_rejected():
    with pytest.raises(ValueError, match="diffusion_time_sample_method"):
        sample_diffusion_time(4, torch.device("cpu"), 1e-3, "bogus")


def test_fm_euler_solver_step_is_interpolation_toward_prediction():
    x0 = torch.full((2, 1, 8, 4), 3.0)
    evaluated_times = []

    def exact_model(x, t):
        evaluated_times.append(round(float(t[0]), 6))
        return x0

    x1 = torch.randn(2, 1, 8, 4)
    result = FMEulerSolver(exact_model).sample(x1, steps=2)

    # t: 1 -> 0.5 gives 0.5*x1 + 0.5*x0; the final step targets t=0 exactly -> x0.
    torch.testing.assert_close(result, x0)
    assert evaluated_times == [1.0, 0.5]


def test_fm_euler_solver_recovers_x0_with_exact_model_for_any_step_count():
    torch.manual_seed(0)
    x0 = torch.randn(3, 1, 80, 4)
    for steps in (1, 2, 6):
        out = FMEulerSolver(lambda x, t: x0).sample(torch.randn_like(x0), steps=steps)
        torch.testing.assert_close(out, x0)


def test_fm_euler_solver_uses_exactly_steps_model_evaluations():
    evaluations = 0

    def counting_model(x, t):
        nonlocal evaluations
        evaluations += 1
        return torch.zeros_like(x)

    FMEulerSolver(counting_model).sample(torch.randn(1, 1, 8, 4), steps=6)
    assert evaluations == 6


def test_fm_euler_solver_rejects_invalid_inputs():
    solver = FMEulerSolver(lambda x, t: x)
    with pytest.raises(ValueError):
        solver.sample(torch.randn(1, 8, 4), steps=2)
    with pytest.raises(ValueError):
        solver.sample(torch.randn(1, 1, 8, 4), steps=0)


def test_decoder_default_diffusion_path_remains_vpsde():
    decoder = Decoder(_hdp_config())
    assert isinstance(decoder.sde, VPSDE_linear)


def test_decoder_rejects_unknown_diffusion_path():
    with pytest.raises(ValueError, match="diffusion_path"):
        Decoder(_hdp_config(diffusion_path="bogus"))


def test_decoder_dispatches_linear_fm_inference():
    torch.manual_seed(0)
    args = _hdp_config(diffusion_path="linear_fm")
    decoder = Decoder(args).eval()
    assert isinstance(decoder.sde, LinearFlowPath)

    batch = 2
    inputs = {
        "route_lanes": torch.randn(batch, 25, 20, 33),
        "sampled_trajectories": torch.randn(batch, 1, 80, 4),
        "ego_current_state": torch.randn(batch, 10),
        "_skip_turn_indicator": True,
    }
    with torch.no_grad():
        output = decoder(torch.randn(batch, 7, 32), inputs)
    assert output["prediction"].shape == (batch, 1, 80, 4)
    assert torch.isfinite(output["prediction"]).all()


def test_diffusion_planner_exposes_configured_flow_path():
    model = Diffusion_Planner(_hdp_config(diffusion_path="linear_fm"))
    assert isinstance(model.sde, LinearFlowPath)


def test_fm_training_loss_end_to_end_backward():
    """compute_training_loss must consume the model's LinearFlowPath via duck typing."""
    torch.manual_seed(0)
    args = _hdp_config(
        diffusion_path="linear_fm",
        agent_num=4,
        static_objects_num=2,
        lane_num=6,
        polygon_num=2,
        polygon_len=8,
        line_string_num=3,
        line_string_len=8,
        encoder_mixer_depth=1,
        encoder_fusion_depth=1,
    )
    model = Diffusion_Planner(args).train()
    batch = 2
    inputs = {
        "ego_agent_past": torch.randn(batch, 31, 4),
        "ego_current_state": torch.randn(batch, 10),
        "neighbor_agents_past": torch.randn(batch, 4, 31, 11),
        "static_objects": torch.randn(batch, 2, 10),
        "lanes": torch.randn(batch, 6, 20, 33),
        "lanes_speed_limit": torch.randn(batch, 6, 1),
        "lanes_has_speed_limit": torch.ones(batch, 6, 1, dtype=torch.bool),
        "route_lanes": torch.randn(batch, 25, 20, 33),
        "route_lanes_speed_limit": torch.randn(batch, 25, 1),
        "route_lanes_has_speed_limit": torch.ones(batch, 25, 1, dtype=torch.bool),
        "polygons": torch.randn(batch, 2, 8, 3),
        "line_strings": torch.randn(batch, 3, 8, 4),
        "goal_pose": torch.randn(batch, 4),
        "ego_shape": torch.tensor([[2.75, 4.34, 1.70]] * batch),
        "turn_indicators": torch.randint(0, 4, (batch, 31)),
    }
    heading = torch.randn(batch, 80, 2)
    heading = heading / heading.norm(dim=-1, keepdim=True)
    ego_future = torch.cat([torch.randn(batch, 80, 2), heading], dim=-1)
    futures = (
        ego_future,
        ego_future.new_zeros((batch, 0, 80, 4)),
        torch.zeros(batch, 0, 80, dtype=torch.bool),
    )

    assert isinstance(model.sde, LinearFlowPath)
    loss = compute_training_loss(model, inputs, futures, args)
    total = loss["ego_planning_loss"] + loss["ego_planning_hybrid_loss"]
    assert torch.isfinite(total)
    total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.decoder.dit.parameters()
    )


def _compat_config():
    args = TrainConfig(
        exp_name="test",
        save_dir="unused",
        train_set_list="unused",
        valid_set_list="unused",
        train_subsample_step=1,
    )
    args.state_normalizer = StateNormalizer(
        [[[0.0, 0.0, 0.0, 0.0]]],
        [[[1.0, 1.0, 1.0, 1.0]]],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    args.observation_normalizer = ObservationNormalizer(
        {"ego_current_state": {"mean": torch.zeros(10), "std": torch.ones(10)}}
    )
    return args


def test_legacy_vpsde_checkpoint_cannot_be_silently_reinterpreted_as_fm(tmp_path):
    checkpoint_args = _compat_config()
    serializable = {
        key: value.to_dict()
        if isinstance(value, (StateNormalizer, ObservationNormalizer))
        else value
        for key, value in vars(checkpoint_args).items()
    }
    # Legacy checkpoints predate the flag entirely; they are implicitly VP-SDE.
    serializable.pop("diffusion_path", None)
    (tmp_path / "args.json").write_text(json.dumps(serializable), encoding="utf-8")
    checkpoint_path = str(tmp_path / "latest.pth")

    current = _compat_config()
    current.diffusion_path = "linear_fm"
    with pytest.raises(RuntimeError, match="diffusion_path"):
        assert_checkpoint_compatible(checkpoint_path, current)

    # Weights-only bootstrap across paths is a deliberate, explicit opt-in: the DiT
    # I/O contract is identical, only the x_t marginal distribution changes.
    assert_checkpoint_compatible(
        checkpoint_path,
        current,
        allow_diffusion_path_change=True,
        strict_training_config=False,
    )

    matching = _compat_config()
    assert_checkpoint_compatible(checkpoint_path, matching)


def test_deployment_config_defaults_legacy_checkpoints_to_vpsde(tmp_path):
    args_json = {
        "decoder_tokenization": "temporal",
        "use_velocity_representation": True,
        "diffusion_model_type": "x_start",
        "diffusion_supervision_type": "x_start",
        "state_normalizer": {
            "mean": [[[0.0, 0.0, 0.0, 0.0]]],
            "std": [[[1.0, 1.0, 1.0, 1.0]]],
            "ego_velocity_mean": [0.0, 0.0, 0.0, 0.0],
            "ego_velocity_std": [1.0, 1.0, 1.0, 1.0],
        },
        "observation_normalizer": {},
    }
    path = tmp_path / "args.json"
    path.write_text(json.dumps(args_json), encoding="utf-8")
    config = Config(str(path))
    assert config.diffusion_path == "vpsde"
