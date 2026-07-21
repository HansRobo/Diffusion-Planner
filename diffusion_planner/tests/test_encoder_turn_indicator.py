import pytest
import torch
from diffusion_planner.dimensions import (
    EGO_SHAPE_SHAPE,
    GOAL_POSE_SHAPE,
    INPUT_T,
    LANES_HAS_SPEED_LIMIT_SHAPE,
    LANES_SHAPE,
    LANES_SPEED_LIMIT_SHAPE,
    LINE_STRING_TYPE_NUM,
    NEIGHBOR_SHAPE,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    POLYGON_TYPE_NUM,
    POSE_DIM,
    ROUTE_LANES_HAS_SPEED_LIMIT_SHAPE,
    ROUTE_LANES_SHAPE,
    ROUTE_LANES_SPEED_LIMIT_SHAPE,
    STATIC_OBJECTS_SHAPE,
)
from diffusion_planner.model.module.encoder import (
    Encoder,
    GoalPoseEncoder,
    _keep_recent_history,
)
from diffusion_planner.train_config import TrainConfig


def test_keep_recent_history_uses_current_plus_twenty_previous_frames():
    history = torch.arange(31 * 4, dtype=torch.float32).reshape(1, 31, 4)
    selected = _keep_recent_history(history, 21)

    assert selected.shape == history.shape
    torch.testing.assert_close(selected[:, :10], torch.zeros(1, 10, 4))
    torch.testing.assert_close(selected[:, -21:], history[:, -21:])


def test_keep_recent_history_rejects_longer_window_than_input():
    history = torch.zeros(1, 6, 4)
    with pytest.raises(ValueError, match="history_frames must be"):
        _keep_recent_history(history, 7)


def test_goal_pose_encoder_masks_missing_goal_without_masking_origin_heading():
    encoder = GoalPoseEncoder(drop_path_rate=0.0, hidden_dim=16).eval()
    goals = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        encoded, mask, _ = encoder(goals)
    assert mask.tolist() == [[True], [False]]
    torch.testing.assert_close(encoded[0], torch.zeros_like(encoded[0]))
    assert torch.count_nonzero(encoded[1]) > 0


def _encoder_inputs():
    return {
        "ego_agent_past": torch.randn(1, INPUT_T + 1, POSE_DIM),
        "neighbor_agents_past": torch.randn(*NEIGHBOR_SHAPE),
        "static_objects": torch.randn(*STATIC_OBJECTS_SHAPE),
        "lanes": torch.randn(*LANES_SHAPE),
        "lanes_speed_limit": torch.randn(*LANES_SPEED_LIMIT_SHAPE),
        "lanes_has_speed_limit": torch.ones(*LANES_HAS_SPEED_LIMIT_SHAPE, dtype=torch.bool),
        "route_lanes": torch.randn(*ROUTE_LANES_SHAPE),
        "route_lanes_speed_limit": torch.randn(*ROUTE_LANES_SPEED_LIMIT_SHAPE),
        "route_lanes_has_speed_limit": torch.ones(
            *ROUTE_LANES_HAS_SPEED_LIMIT_SHAPE, dtype=torch.bool
        ),
        "polygons": torch.randn(1, NUM_POLYGONS, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM),
        "line_strings": torch.randn(
            1, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM
        ),
        "goal_pose": torch.randn(*GOAL_POSE_SHAPE),
        "ego_shape": torch.randn(*EGO_SHAPE_SHAPE),
        "turn_indicators": torch.randint(0, 4, (1, INPUT_T + 1), dtype=torch.int32),
    }


def test_encoder_is_structurally_independent_of_turn_indicator_history():
    config = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
    )
    encoder = Encoder(config).eval()
    inputs = _encoder_inputs()

    without_signal = {key: value for key, value in inputs.items() if key != "turn_indicators"}
    flipped_signal = {**inputs, "turn_indicators": 4 - inputs["turn_indicators"]}
    with torch.no_grad():
        baseline = encoder(without_signal)
        flipped = encoder(flipped_signal)
    assert not hasattr(encoder, "turn_indicator_encoder")
    assert baseline.shape == (1, encoder.token_num, config.hidden_dim)
    torch.testing.assert_close(flipped, baseline, rtol=0, atol=0)
