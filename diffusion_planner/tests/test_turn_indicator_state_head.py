"""Three-state turn-intent head and temporal output stabilization."""

import math
from unittest.mock import patch

import pytest
import torch
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    make_turn_indicator_gt,
    turn_indicator_geometry_evidence,
    turn_indicator_objective,
    turn_indicator_soft_targets,
)
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
    fit_probability_temperature,
)
from diffusion_planner.validate_model import (
    TURN_INDICATOR_CALIBRATION_BINS,
    TURN_INDICATOR_CLASS_NAMES,
    turn_indicator_calibration_counts,
    turn_indicator_calibration_metrics,
    turn_indicator_metrics_from_confusion,
)

# The audit tool's activation-confirmation bar and its next stronger bucket.
_EVIDENCE_BARS = {
    "min_yaw_rad": math.radians(5.0),
    "full_yaw_rad": math.radians(20.0),
    "min_lateral_m": 0.5,
    "full_lateral_m": 2.0,
}


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
    assert torch.isfinite(loss["turn_indicator_opposite_probability"])
    assert torch.isfinite(loss["turn_indicator_implied_intent_mass"])
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


def _turning_future(batch: int, *, yaw_rad: float, lateral_m: float, steps: int = 80):
    """Ego-frame future that ramps linearly to ``yaw_rad`` and ``lateral_m``."""
    ramp = torch.linspace(0.0, 1.0, steps)
    yaw = yaw_rad * ramp
    return torch.stack(
        [
            torch.stack([ramp * 10.0, lateral_m * ramp, yaw.cos(), yaw.sin()], dim=-1)
            for _ in range(batch)
        ]
    )


def test_geometry_evidence_reads_the_committed_turn_direction():
    left = turn_indicator_geometry_evidence(
        _turning_future(1, yaw_rad=0.7, lateral_m=3.0), **_EVIDENCE_BARS
    )
    assert left[0].tolist() == [1]
    assert left[1].tolist() == pytest.approx([1.0])

    right = turn_indicator_geometry_evidence(
        _turning_future(1, yaw_rad=-0.7, lateral_m=-3.0), **_EVIDENCE_BARS
    )
    assert right[0].tolist() == [2]
    assert right[1].tolist() == pytest.approx([1.0])

    straight = turn_indicator_geometry_evidence(
        _turning_future(1, yaw_rad=0.0, lateral_m=0.0), **_EVIDENCE_BARS
    )
    assert straight[0].tolist() == [0]
    assert straight[1].tolist() == [0.0]


def test_geometry_evidence_discounts_a_future_that_swerves_and_returns():
    steps = 80
    ramp = torch.linspace(0.0, 1.0, steps)
    # Out-and-back: peak lateral offset of 1.5 m with equal left and right heading.
    lateral = 1.5 * torch.sin(math.pi * ramp)
    yaw = 0.2 * torch.sin(2.0 * math.pi * ramp)
    swerve = torch.stack([ramp * 10.0, lateral, yaw.cos(), yaw.sin()], dim=-1)[None]
    direction, strength = turn_indicator_geometry_evidence(swerve, **_EVIDENCE_BARS)
    committed = turn_indicator_geometry_evidence(
        _turning_future(1, yaw_rad=0.2, lateral_m=1.5), **_EVIDENCE_BARS
    )
    assert direction.tolist() == [1]
    # Same peak excursion, but the opposite-side evidence removes most of the strength.
    assert 0.0 < float(strength) < 0.5 * float(committed[1])


def test_soft_targets_relax_late_off_labels_without_mixing_directions():
    target = torch.tensor([0, 0, 1])
    implied = torch.tensor([1, 2, 2])
    strength = torch.tensor([1.0, 0.5, 1.0])
    soft, moved = turn_indicator_soft_targets(target, implied, strength, smoothing=0.2)
    assert soft[0].tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert soft[1].tolist() == pytest.approx([0.9, 0.0, 0.1])
    # An active label is never relaxed, so no mass reaches the opposite direction.
    assert soft[2].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert moved.tolist() == pytest.approx([0.2, 0.1, 0.0])
    assert soft.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_objective_reproduces_plain_cross_entropy_at_legacy_settings():
    torch.manual_seed(3)
    args = _hdp_config(
        turn_indicator_opposite_direction_weight=0.0,
        turn_indicator_implied_intent_smoothing=0.0,
    )
    logit = torch.randn(6, 3)
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    loss, diagnostics = turn_indicator_objective(
        logit, target, _turning_future(6, yaw_rad=0.7, lateral_m=3.0), args
    )
    torch.testing.assert_close(
        loss, torch.nn.functional.cross_entropy(logit, target, reduction="none")
    )
    assert diagnostics["turn_indicator_implied_intent_mass"] == 0.0
    # The diagnostic still reports opposite-direction mass even when it is not priced.
    assert diagnostics["turn_indicator_opposite_probability"] > 0.0


def test_objective_prices_opposite_direction_mass_only_on_active_labels():
    args = _hdp_config(
        turn_indicator_opposite_direction_weight=2.0,
        turn_indicator_implied_intent_smoothing=0.0,
    )
    # Row 0 is OFF ground truth, row 1 is LEFT ground truth; both put half their mass on
    # RIGHT, which is an opposite-direction error only for row 1.
    logit = torch.tensor([[0.0, 0.0, math.log(2.0)], [0.0, 0.0, math.log(2.0)]])
    target = torch.tensor([0, 1])
    loss, diagnostics = turn_indicator_objective(logit, target, None, args)
    baseline = torch.nn.functional.cross_entropy(logit, target, reduction="none")
    torch.testing.assert_close(loss[0], baseline[0])
    torch.testing.assert_close(loss[1], baseline[1] + 2.0 * 0.5)
    assert float(diagnostics["turn_indicator_opposite_probability"]) == pytest.approx(0.5)


def test_objective_stops_punishing_an_early_signal_on_a_turning_future():
    args = _hdp_config(
        turn_indicator_opposite_direction_weight=0.0,
        turn_indicator_implied_intent_smoothing=0.2,
    )
    # OFF ground truth on a future that clearly turns left: the head predicts LEFT.
    logit = torch.tensor([[0.0, 2.0, 0.0]])
    target = torch.tensor([0])
    turning, _ = turn_indicator_objective(
        logit, target, _turning_future(1, yaw_rad=0.7, lateral_m=3.0), args
    )
    straight, _ = turn_indicator_objective(
        logit, target, _turning_future(1, yaw_rad=0.0, lateral_m=0.0), args
    )
    baseline = torch.nn.functional.cross_entropy(logit, target, reduction="none")
    torch.testing.assert_close(straight, baseline)
    assert float(turning) < float(straight)


def test_calibration_metrics_match_a_hand_built_reliability_diagram():
    logit = torch.tensor(
        [
            [math.log(0.7), math.log(0.2), math.log(0.1)],
            [math.log(0.4), math.log(0.5), math.log(0.1)],
        ]
    )
    target = torch.tensor([0, 0])
    metrics = turn_indicator_calibration_metrics(turn_indicator_calibration_counts(logit, target))
    assert metrics["turn_indicator_nll"] == pytest.approx(
        -(math.log(0.7) + math.log(0.4)) / 2, rel=1e-6
    )
    # Bin 0.7: one sample, confidence 0.7, accuracy 1.0. Bin 0.5: confidence 0.5, accuracy 0.
    assert metrics["turn_indicator_ece"] == pytest.approx(0.3 / 2 + 0.5 / 2, rel=1e-6)


def test_empty_calibration_accumulator_is_reported_as_unevaluated_zeros():
    metrics = turn_indicator_calibration_metrics([0.0] * (2 + 3 * TURN_INDICATOR_CALIBRATION_BINS))
    assert metrics == {"turn_indicator_nll": 0.0, "turn_indicator_ece": 0.0}


def test_fit_probability_temperature_recovers_a_known_over_confidence():
    probabilities = [0.6, 0.25, 0.15]
    labels = torch.tensor([0] * 60 + [1] * 25 + [2] * 15)
    calibrated = torch.tensor([[math.log(value) for value in probabilities]]).repeat(100, 1)
    assert fit_probability_temperature(calibrated, labels) == pytest.approx(1.0, rel=1e-4)
    # Sharpening the same logits by 3x is exactly a temperature-3 miscalibration.
    assert fit_probability_temperature(3.0 * calibrated, labels) == pytest.approx(3.0, rel=1e-4)


def test_deployment_temperature_moves_the_gate_without_moving_the_argmax():
    logit = torch.tensor([0.3, 0.5, 0.2]).log()
    uncalibrated = TurnIndicatorStateMachine(
        TurnIndicatorStateMachineConfig(probability_ema_alpha=1.0, activation_seconds=0.3)
    )
    # The head is right about the direction but never reaches the 0.60 activation gate.
    assert [uncalibrated.update(logit) for _ in range(10)] == [0] * 10
    sharpened = TurnIndicatorStateMachine(
        TurnIndicatorStateMachineConfig(
            probability_ema_alpha=1.0, activation_seconds=0.3, probability_temperature=0.5
        )
    )
    assert [sharpened.update(logit) for _ in range(3)] == [0, 0, 1]
    # Feeding probabilities instead of logits must scale identically.
    equivalent = TurnIndicatorStateMachine(
        TurnIndicatorStateMachineConfig(
            probability_ema_alpha=1.0, activation_seconds=0.3, probability_temperature=0.5
        )
    )
    assert [equivalent.update(logit.exp(), logits=False) for _ in range(3)] == [0, 0, 1]


def test_state_machine_rejects_an_unusable_temperature():
    with pytest.raises(ValueError, match="probability_temperature"):
        TurnIndicatorStateMachineConfig(probability_temperature=0.0)


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
