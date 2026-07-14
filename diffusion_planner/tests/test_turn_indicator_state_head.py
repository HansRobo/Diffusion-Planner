"""Stateless turn-indicator state head.

Design contract:
- Labels are the raw 4-class TurnIndicatorsReport state at the current frame
  (dense state classification), not transition events. Keep/hysteresis is
  deployment logic, not model memory.
- The head reads planned-trajectory features (16 points incl. the endpoint, with
  headings), an attention probe over the scene tokens, and the global route
  condition — all with zero gradient into the policy.
- The generated branch is weighted by (1 - t) so near-mean high-noise x0
  predictions stop polluting the head.
- The encoder's turn-indicator INPUT gets per-sample dropout against the
  signal->maneuver causal shortcut (deployment feeds back the stack's own signal).
"""

import pytest
import torch
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import make_turn_indicator_gt, turn_indicator_loss_weights
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.module.decoder import Decoder, TurnIndicatorHead, compute_training_loss
from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.validate_model import TURN_INDICATOR_CLASS_NAMES


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


def test_turn_indicator_input_dropout_defaults_off():
    # Mainline discipline: the paper recipe is the default; the causal-shortcut
    # insurance dropout is an explicit opt-in (recommended arm value: 0.5).
    assert TrainConfig.__dataclass_fields__["turn_indicator_dropout_rate"].default == 0.0


def test_turn_indicator_output_is_four_state_classes():
    assert TURN_INDICATOR_OUTPUT_DIM == 4
    assert TURN_INDICATOR_CLASS_NAMES == ("none", "disable", "enable_left", "enable_right")


def test_make_turn_indicator_gt_returns_current_state():
    sequence = torch.full((3, 31), 1, dtype=torch.long)
    sequence[0, -1] = 2  # switched to left on the current frame
    sequence[1, :] = 3  # right the whole window
    sequence[2, -5:] = 0  # unset for the last half second
    labels = make_turn_indicator_gt(sequence)
    assert labels.tolist() == [2, 3, 0]


def test_turn_indicator_loss_weights_combine_class_and_onset_terms():
    steady_off = torch.full((1, 31), 1, dtype=torch.long)
    steady_left = torch.full((1, 31), 2, dtype=torch.long)
    fresh_onset = torch.full((1, 31), 1, dtype=torch.long)
    fresh_onset[0, -3:] = 2  # switched to left 0.3 s ago
    stale_onset = torch.full((1, 31), 1, dtype=torch.long)
    stale_onset[0, 15:] = 2  # switched to left 1.6 s ago (outside the 1 s window)

    weights = turn_indicator_loss_weights(
        torch.cat([steady_off, steady_left, fresh_onset, stale_onset])
    )
    torch.testing.assert_close(weights, torch.tensor([0.1, 1.0, 5.0, 1.0]))


def test_turn_indicator_head_shapes_and_padding_mask():
    head = TurnIndicatorHead(hidden_dim=32, trajectory_dim=64)
    trajectory = torch.randn(2, 64)
    tokens = torch.randn(2, 7, 32)
    tokens[:, 5:] = 0.0  # padded scene tokens must be excluded from the probe
    route = torch.randn(2, 32)
    logits = head(trajectory, tokens, route)
    assert logits.shape == (2, TURN_INDICATOR_OUTPUT_DIM)
    assert torch.isfinite(logits).all()

    # Changing only padded token values must not change the readout.
    perturbed = tokens.clone()
    perturbed[:, 5:] = 123.0
    perturbed[perturbed.abs() > 100] = 0.0  # restore exact zeros
    torch.testing.assert_close(head(trajectory, perturbed, route), logits)


def test_turn_indicator_head_is_gradient_isolated_from_policy_inputs():
    head = TurnIndicatorHead(hidden_dim=32, trajectory_dim=64)
    trajectory = torch.randn(2, 64, requires_grad=True)
    tokens = torch.randn(2, 7, 32, requires_grad=True)
    route = torch.randn(2, 32, requires_grad=True)
    head(trajectory, tokens, route).sum().backward()
    assert tokens.grad is None
    assert route.grad is None
    assert all(p.grad is not None for p in head.parameters() if p.requires_grad)


def test_decoder_turn_features_cover_the_trajectory_endpoint():
    torch.manual_seed(0)
    decoder = Decoder(_hdp_config())
    latent = torch.randn(2, 1, 80, 4)
    features = decoder._turn_indicator_trajectory_from_latent(latent)
    assert features.shape == (2, 64)

    # Perturbing only the final future step must change the features: the old
    # [::10] slicing stopped at 7.1 s and was blind to the endpoint.
    perturbed = latent.clone()
    perturbed[:, :, -1, :] += 1.0
    assert not torch.allclose(decoder._turn_indicator_trajectory_from_latent(perturbed), features)


def test_decoder_training_emits_four_class_logits_without_policy_gradients():
    torch.manual_seed(0)
    decoder = Decoder(
        _hdp_config(agent_num=4, static_objects_num=2, lane_num=6, polygon_num=2, line_string_num=3)
    ).train()
    batch = 2
    inputs = {
        "route_lanes": torch.randn(batch, 25, 20, 33),
        "sampled_trajectories": torch.randn(batch, 1, 80, 4),
        "gt_trajectories": torch.randn(batch, 1, 80, 4),
        "turn_indicator_trajectories": torch.randn(batch, 1, 80, 4),
        "diffusion_time": torch.rand(batch),
        "ego_current_state": torch.randn(batch, 10),
    }
    output = decoder(torch.randn(batch, decoder._turn_indicator_token_index + 1, 32), inputs)
    assert output["turn_indicator_logit"].shape == (batch, TURN_INDICATOR_OUTPUT_DIM)
    assert output["turn_indicator_expert_logit"].shape == (batch, TURN_INDICATOR_OUTPUT_DIM)

    turn_loss = output["turn_indicator_logit"].sum() + output["turn_indicator_expert_logit"].sum()
    turn_loss.backward()
    assert all(p.grad is None for p in decoder.dit.parameters())
    assert all(p.grad is None for p in decoder.global_route_encoder.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.turn_indicator_predictor.parameters()
    )


def test_decoder_inference_emits_state_logits():
    torch.manual_seed(0)
    decoder = Decoder(
        _hdp_config(agent_num=4, static_objects_num=2, lane_num=6, polygon_num=2, line_string_num=3)
    ).eval()
    batch = 2
    inputs = {
        "route_lanes": torch.randn(batch, 25, 20, 33),
        "sampled_trajectories": torch.randn(batch, 1, 80, 4),
        "ego_current_state": torch.randn(batch, 10),
    }
    with torch.no_grad():
        output = decoder(torch.randn(batch, decoder._turn_indicator_token_index + 1, 32), inputs)
    assert output["turn_indicator_logit"].shape == (batch, TURN_INDICATOR_OUTPUT_DIM)


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


def test_encoder_turn_indicator_dropout_matches_disabled_input():
    torch.manual_seed(0)
    config = _encoder_config(turn_indicator_dropout_rate=1.0)
    encoder = Encoder(config)
    inputs = _encoder_inputs(batch=2)

    encoder.train()
    dropped = encoder(inputs)

    encoder.eval()
    encoder.use_turn_indicators = False
    disabled = encoder(inputs)
    torch.testing.assert_close(dropped, disabled)


def test_encoder_turn_indicator_input_is_used_when_not_dropped():
    torch.manual_seed(0)
    config = _encoder_config(turn_indicator_dropout_rate=0.0)
    encoder = Encoder(config).eval()
    inputs = _encoder_inputs(batch=2)
    baseline = encoder(inputs)

    flipped = dict(inputs)
    flipped["turn_indicators"] = torch.where(inputs["turn_indicators"] == 2, 3, 2)
    assert not torch.allclose(encoder(flipped), baseline)


def test_training_loss_turn_metrics_are_four_class_and_finite():
    torch.manual_seed(0)
    args = _encoder_config(turn_indicator_dropout_rate=0.5)
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
    assert 0.0 <= float(loss["turn_indicator_accuracy"]) <= 1.0
    loss["turn_indicator_loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.dit.parameters())


def test_turn_indicator_loss_weights_reject_bad_shapes():
    with pytest.raises(ValueError):
        turn_indicator_loss_weights(torch.zeros(31, dtype=torch.long))


def test_turn_head_cannot_read_the_signal_history_scene_token():
    """The scene tokens include the signal-history token; the head must be blind to
    it (true statelessness), while the policy keeps seeing it."""
    torch.manual_seed(0)
    args = _hdp_config(
        decoder_drop_path_rate=0.0,
        # The route encoder's MLP dropout follows the ENCODER rate; leaving it on
        # makes the two train-mode forwards stochastic and the comparison meaningless.
        encoder_drop_path_rate=0.0,
        agent_num=4,
        static_objects_num=2,
        lane_num=6,
        polygon_num=2,
        line_string_num=3,
    )
    decoder = Decoder(args).train()
    batch = 2
    token_count = decoder._turn_indicator_token_index + 1
    encoding = torch.randn(batch, token_count, 32)
    flipped = encoding.clone()
    flipped[:, decoder._turn_indicator_token_index] += 5.0
    inputs = {
        "route_lanes": torch.randn(batch, 25, 20, 33),
        "sampled_trajectories": torch.randn(batch, 1, 80, 4),
        "gt_trajectories": torch.randn(batch, 1, 80, 4),
        "turn_indicator_trajectories": torch.randn(batch, 1, 80, 4),
        "diffusion_time": torch.rand(batch),
        "ego_current_state": torch.randn(batch, 10),
    }

    base = decoder(encoding, inputs)
    changed = decoder(flipped, inputs)

    # The head must be blind to the signal-history token...
    torch.testing.assert_close(
        base["turn_indicator_expert_logit"], changed["turn_indicator_expert_logit"]
    )

    # ...while its attention probe demonstrably reads other scene tokens (the
    # blindness above is masking, not a dead probe). The DiT's zero-initialized
    # adaLN gates make the untrained POLICY input-insensitive, so probe liveness
    # is the meaningful counter-check here.
    other = encoding.clone()
    other[:, 2] += 5.0
    assert not torch.allclose(
        decoder(other, inputs)["turn_indicator_expert_logit"],
        base["turn_indicator_expert_logit"],
    )
