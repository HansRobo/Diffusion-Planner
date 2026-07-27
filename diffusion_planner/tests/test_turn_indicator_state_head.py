"""Three-state turn-intent head and temporal output stabilization."""

from unittest.mock import patch

import pytest
import torch
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import make_turn_indicator_gt
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.module.decoder import (
    Decoder,
    TurnIndicatorHead,
    compute_training_loss,
    compute_turn_indicator_head_training_loss,
)
from diffusion_planner.train import configure_supervised_trainable_parameters, load_weights_only
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import (
    ENCODER_INPUT_NAMES,
    FULL_INPUT_NAMES,
    TURN_INDICATOR_INPUT_NAMES,
    build_dummy_inputs,
)
from diffusion_planner.utils.turn_indicator import (
    TurnIndicatorStateMachine,
    TurnIndicatorStateMachineConfig,
)
from diffusion_planner.validate_model import (
    TURN_INDICATOR_CLASS_NAMES,
    turn_indicator_metrics_from_confusion,
)


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


def _encoder_config(**overrides):
    return _hdp_config(
        agent_num=4,
        static_objects_num=2,
        lane_num=6,
        polygon_num=2,
        polygon_len=8,
        line_string_num=3,
        line_string_len=8,
        encoder_mixer_depth=1,
        encoder_fusion_depth=1,
        encoder_drop_path_rate=0.0,
        ego_history_dropout_rate=0.0,
        **overrides,
    )


def _encoder_inputs(batch):
    torch.manual_seed(7)
    return {
        "ego_agent_past": torch.randn(batch, 31, 4),
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
        "turn_indicators": torch.randint(1, 4, (batch, 31)),
    }


def test_turn_indicator_output_is_three_real_states():
    assert TURN_INDICATOR_OUTPUT_DIM == 3
    assert TURN_INDICATOR_CLASS_NAMES == ("disable", "enable_left", "enable_right")


def test_model_rejects_obsolete_indicator_architecture_modes():
    with pytest.raises(ValueError, match="must not consume"):
        Diffusion_Planner(_encoder_config(policy_uses_turn_indicator_history=True))
    with pytest.raises(ValueError, match="exactly 3 classes"):
        Diffusion_Planner(_encoder_config(turn_indicator_output_dim=4))


@pytest.mark.parametrize(
    ("stage", "policy_trainable", "head_trainable"),
    (("joint", True, True), ("policy", True, False), ("turn_indicator", False, True)),
)
def test_supervised_stage_owns_exact_parameter_subset(stage, policy_trainable, head_trainable):
    model = Diffusion_Planner(_encoder_config())
    configure_supervised_trainable_parameters(model, stage)
    for name, parameter in model.named_parameters():
        expected = (
            head_trainable
            if name.startswith("decoder.turn_indicator_predictor.")
            else policy_trainable
        )
        assert parameter.requires_grad is expected, name


def test_make_turn_indicator_gt_maps_raw_reports_to_dense_classes():
    sequence = torch.stack(
        [
            torch.full((31,), 1),
            torch.full((31,), 2),
            torch.full((31,), 3),
        ]
    )
    assert make_turn_indicator_gt(sequence).tolist() == [0, 1, 2]
    sequence[0, -1] = 0
    with pytest.raises(ValueError, match="Invalid TurnIndicatorsReport"):
        make_turn_indicator_gt(sequence)
    fractional = torch.ones(1, 31)
    fractional[0, -1] = 1.5
    with pytest.raises(ValueError, match="Invalid TurnIndicatorsReport"):
        make_turn_indicator_gt(fractional)


def test_turn_indicator_head_shapes_and_gradient_isolation():
    head = TurnIndicatorHead(hidden_dim=32, trajectory_dim=64)
    trajectory = torch.randn(2, 64, requires_grad=True)
    tokens = torch.randn(2, 7, 32, requires_grad=True)
    with torch.no_grad():
        tokens[:, 5:] = 0.0
    route = torch.randn(2, 32, requires_grad=True)
    proprio = torch.randn(2, 6, requires_grad=True)
    logits = head(trajectory, tokens, route, proprio)
    assert logits.shape == (2, TURN_INDICATOR_OUTPUT_DIM)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert trajectory.grad is None
    assert tokens.grad is None
    assert route.grad is None
    assert proprio.grad is None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_turn_indicator_head_uses_current_dynamics():
    torch.manual_seed(3)
    head = TurnIndicatorHead(hidden_dim=32, trajectory_dim=64).eval()
    trajectory = torch.randn(2, 64)
    tokens = torch.randn(2, 7, 32)
    route = torch.randn(2, 32)
    stationary = torch.zeros(2, 6)
    moving = stationary.clone()
    moving[:, 0] = 2.0
    assert not torch.allclose(
        head(trajectory, tokens, route, stationary),
        head(trajectory, tokens, route, moving),
    )


def test_decoder_turn_features_cover_the_trajectory_endpoint():
    torch.manual_seed(0)
    decoder = Decoder(_hdp_config())
    latent = torch.randn(2, 1, 80, 4)
    features = decoder._turn_indicator_trajectory_from_latent(latent)
    assert features.shape == (2, 64)
    perturbed = latent.clone()
    perturbed[:, :, -1, :] += 1.0
    assert not torch.allclose(decoder._turn_indicator_trajectory_from_latent(perturbed), features)


def test_decoder_training_emits_three_class_logits_without_policy_gradients():
    torch.manual_seed(0)
    decoder = Decoder(_encoder_config()).train()
    batch = 2
    inputs = {
        "route_lanes": torch.randn(batch, 25, 20, 33),
        "sampled_trajectories": torch.randn(batch, 1, 80, 4),
        "gt_trajectories": torch.randn(batch, 1, 80, 4),
        "turn_indicator_trajectories": torch.randn(batch, 1, 80, 4),
        "diffusion_time": torch.rand(batch),
        "ego_current_state": torch.randn(batch, 10),
    }
    output = decoder(torch.randn(batch, 16, 32), inputs)
    assert output["turn_indicator_logit"].shape == (batch, 3)
    assert output["turn_indicator_expert_logit"].shape == (batch, 3)
    (output["turn_indicator_logit"].sum() + output["turn_indicator_expert_logit"].sum()).backward()
    assert all(parameter.grad is None for parameter in decoder.dit.parameters())
    assert all(parameter.grad is None for parameter in decoder.global_route_encoder.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in decoder.turn_indicator_predictor.parameters()
    )


def test_full_policy_inference_needs_no_signal_input_and_is_signal_invariant():
    torch.manual_seed(11)
    args = _encoder_config(diffusion_sample_steps=2)
    model = Diffusion_Planner(args).eval()
    inputs = _encoder_inputs(batch=2)
    inputs["ego_current_state"] = torch.randn(2, 10)
    inputs["sampled_trajectories"] = torch.randn(2, 1, 80, 4)
    without_signal = {key: value for key, value in inputs.items() if key != "turn_indicators"}
    with torch.no_grad():
        _, baseline = model(without_signal)
        _, with_signal = model(inputs)
    torch.testing.assert_close(with_signal["prediction"], baseline["prediction"], rtol=0, atol=0)
    torch.testing.assert_close(
        with_signal["turn_indicator_logit"], baseline["turn_indicator_logit"], rtol=0, atol=0
    )


def test_training_loss_updates_only_the_intent_head_for_indicator_loss():
    torch.manual_seed(0)
    # Joint mode is an explicit ablation; production Base/SFT defaults to policy.
    args = _encoder_config(supervised_training_stage="joint")
    model = Diffusion_Planner(args).train()
    batch = 2
    inputs = _encoder_inputs(batch)
    inputs["ego_current_state"] = torch.randn(batch, 10)
    heading = torch.randn(batch, 80, 2)
    heading = heading / heading.norm(dim=-1, keepdim=True)
    ego_future = torch.cat([torch.randn(batch, 80, 2), heading], dim=-1)
    futures = (
        ego_future,
        ego_future.new_zeros((batch, 0, 80, 4)),
        torch.zeros(batch, 0, 80, dtype=torch.bool),
    )
    loss = compute_training_loss(model, inputs, futures, args)
    assert torch.isfinite(loss["turn_indicator_loss"])
    loss["turn_indicator_loss"].backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.dit.parameters())


def test_policy_stage_skips_intent_head_completely():
    torch.manual_seed(0)
    args = _encoder_config(supervised_training_stage="policy")
    model = Diffusion_Planner(args).train()
    configure_supervised_trainable_parameters(model, "policy")
    batch = 2
    inputs = _encoder_inputs(batch)
    inputs["ego_current_state"] = torch.randn(batch, 10)
    heading = torch.randn(batch, 80, 2)
    heading = heading / heading.norm(dim=-1, keepdim=True)
    ego_future = torch.cat([torch.randn(batch, 80, 2), heading], dim=-1)
    futures = (
        ego_future,
        ego_future.new_zeros((batch, 0, 80, 4)),
        torch.zeros(batch, 0, 80, dtype=torch.bool),
    )
    loss = compute_training_loss(model, inputs, futures, args)
    assert "turn_indicator_loss" not in loss
    loss["ego_planning_loss"].backward()
    assert all(
        parameter.grad is None for parameter in model.decoder.turn_indicator_predictor.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.decoder.dit.parameters())


def test_head_stage_uses_final_dpm_trajectory_and_only_updates_head():
    torch.manual_seed(0)
    args = _encoder_config(
        supervised_training_stage="turn_indicator",
        diffusion_sample_steps=2,
    )
    model = Diffusion_Planner(args)
    configure_supervised_trainable_parameters(model, "turn_indicator")
    model.eval()
    model.decoder.turn_indicator_predictor.train()
    batch = 2
    inputs = _encoder_inputs(batch)
    inputs["ego_current_state"] = torch.randn(batch, 10)
    heading = torch.randn(batch, 80, 2)
    heading = heading / heading.norm(dim=-1, keepdim=True)
    ego_future = torch.cat([torch.randn(batch, 80, 2), heading], dim=-1)

    with (
        patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder_forward,
        patch.object(model.decoder.dit, "forward", wraps=model.decoder.dit.forward) as dit_forward,
    ):
        loss = compute_turn_indicator_head_training_loss(model, inputs, ego_future, args)
    assert encoder_forward.call_count == 1
    # DPM-Solver++ 2M uses one initial denoise plus one evaluation per configured step.
    assert dit_forward.call_count == args.diffusion_sample_steps + 1
    assert torch.isfinite(loss["turn_indicator_loss"])
    loss["turn_indicator_loss"].backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.dit.parameters())
    assert all(
        parameter.grad is None for parameter in model.decoder.global_route_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.decoder.turn_indicator_predictor.parameters()
    )


def test_expert_head_stage_skips_diffusion_and_only_updates_head():
    torch.manual_seed(0)
    args = _encoder_config(
        supervised_training_stage="turn_indicator",
        turn_indicator_head_training_mode="expert",
        diffusion_sample_steps=2,
    )
    model = Diffusion_Planner(args)
    configure_supervised_trainable_parameters(model, "turn_indicator")
    model.eval()
    model.decoder.turn_indicator_predictor.train()
    batch = 2
    inputs = _encoder_inputs(batch)
    inputs["ego_current_state"] = torch.randn(batch, 10)
    heading = torch.randn(batch, 80, 2)
    heading = heading / heading.norm(dim=-1, keepdim=True)
    ego_future = torch.cat([torch.randn(batch, 80, 2), heading], dim=-1)

    with (
        patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder_forward,
        patch.object(
            model.decoder.dit,
            "forward",
            side_effect=AssertionError("expert head mode must not evaluate DiT"),
        ),
    ):
        loss = compute_turn_indicator_head_training_loss(model, inputs, ego_future, args)

    assert encoder_forward.call_count == 1
    assert torch.isfinite(loss["turn_indicator_loss"])
    assert "turn_indicator_generated_loss" not in loss
    assert "turn_indicator_generated_accuracy" not in loss
    loss["turn_indicator_loss"].backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.dit.parameters())
    assert all(
        parameter.grad is None for parameter in model.decoder.global_route_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.decoder.turn_indicator_predictor.parameters()
    )


def test_turn_indicator_metrics_separate_activation_and_direction():
    # Rows are GT OFF/LEFT/RIGHT, columns are predictions.
    metrics = turn_indicator_metrics_from_confusion([8, 1, 1, 1, 7, 2, 0, 1, 9])
    assert metrics["turn_indicator_accuracy"] == pytest.approx(24 / 30)
    assert metrics["turn_indicator_active_recall"] == pytest.approx(19 / 20)
    assert metrics["turn_indicator_direction_accuracy"] == pytest.approx(16 / 20)
    assert metrics["turn_indicator_macro_f1"] < 1.0


def test_onnx_policy_inputs_have_no_signal_feedback_and_head_has_proprioception():
    assert "turn_indicators" not in FULL_INPUT_NAMES
    assert "turn_indicators" not in ENCODER_INPUT_NAMES
    assert TURN_INDICATOR_INPUT_NAMES == [
        "encoding",
        "final_x0",
        "global_route_condition",
        "ego_current_state",
    ]
    assert "turn_indicators" not in build_dummy_inputs()


def test_state_machine_debounces_activation_and_holds_minimum_duration():
    config = TurnIndicatorStateMachineConfig(
        probability_ema_alpha=1.0,
        activation_seconds=0.3,
        deactivation_seconds=0.3,
        minimum_active_seconds=1.0,
    )
    machine = TurnIndicatorStateMachine(config)
    left = torch.tensor([0.05, 0.90, 0.05])
    off = torch.tensor([0.90, 0.05, 0.05])
    assert [machine.update(left, logits=False) for _ in range(2)] == [0, 0]
    assert machine.update(left, logits=False) == 1
    assert machine.raw_report_state == 2
    assert [machine.update(off, logits=False) for _ in range(9)] == [1] * 9
    assert [machine.update(off, logits=False) for _ in range(3)] == [1, 1, 0]


def test_state_machine_rejects_one_frame_direction_chatter():
    config = TurnIndicatorStateMachineConfig(
        probability_ema_alpha=1.0,
        activation_seconds=0.1,
        direction_switch_seconds=0.5,
        minimum_active_seconds=0.0,
    )
    machine = TurnIndicatorStateMachine(config)
    left = torch.tensor([0.05, 0.90, 0.05])
    right = torch.tensor([0.05, 0.05, 0.90])
    assert machine.update(left, logits=False) == 1
    assert machine.update(right, logits=False) == 1
    assert machine.update(left, logits=False) == 1


def test_weights_only_init_migrates_old_signal_encoder_and_four_class_head(tmp_path):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(2, 2)
            self.encoder = torch.nn.Module()
            self.decoder = torch.nn.Module()
            self.decoder.turn_indicator_predictor = torch.nn.Linear(3, 3)

    model = ToyModel()
    original_head = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("decoder.turn_indicator_predictor.")
    }
    old_state = {
        "policy.weight": torch.ones_like(model.policy.weight),
        "policy.bias": torch.ones_like(model.policy.bias),
        "encoder.turn_indicator_encoder.weight": torch.randn(4, 4),
        "decoder.turn_indicator_predictor.weight": torch.randn(4, 3),
        "decoder.turn_indicator_predictor.bias": torch.randn(4),
    }
    checkpoint = tmp_path / "base80.pth"
    torch.save({"model": old_state}, checkpoint)

    load_weights_only(str(checkpoint), model, "cpu")

    torch.testing.assert_close(model.policy.weight, torch.ones_like(model.policy.weight))
    torch.testing.assert_close(model.policy.bias, torch.ones_like(model.policy.bias))
    for key, value in original_head.items():
        torch.testing.assert_close(model.state_dict()[key], value)
