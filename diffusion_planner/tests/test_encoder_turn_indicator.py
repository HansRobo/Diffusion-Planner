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
    TURN_INDICATOR_INPUT_ONE_HOT_DIM,
)
from diffusion_planner.model.module.encoder import Encoder, GoalPoseEncoder, one_hot_turn_indicators
from diffusion_planner.train_config import TrainConfig


def test_turn_indicator_one_hot_maps_report_codes():
    output = one_hot_turn_indicators(torch.tensor([[0, 1, 2, 3]], dtype=torch.float32))
    one_hot = output.view(1, 4, TURN_INDICATOR_INPUT_ONE_HOT_DIM)
    assert one_hot[0].tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_turn_indicator_left_and_right_are_orthogonal():
    left = one_hot_turn_indicators(torch.tensor([[2.0]]))
    right = one_hot_turn_indicators(torch.tensor([[3.0]]))
    assert torch.dot(left.flatten(), right.flatten()).item() == 0.0


def test_turn_indicator_out_of_range_codes_become_zero():
    output = one_hot_turn_indicators(torch.tensor([[4, 7, -1]], dtype=torch.float32))
    assert torch.all(output == 0)


def test_turn_indicator_one_hot_accepts_integer_input():
    output = one_hot_turn_indicators(torch.tensor([[1, 2]], dtype=torch.int32))
    one_hot = output.view(1, 2, TURN_INDICATOR_INPUT_ONE_HOT_DIM)
    assert one_hot[0, 0].tolist() == [0, 1, 0, 0]
    assert one_hot[0, 1].tolist() == [0, 0, 1, 0]


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


def test_encoder_consumes_one_hot_turn_indicators_when_enabled_or_disabled():
    config = TrainConfig(
        exp_name="test",
        save_dir="/tmp",
        train_set_list="",
        valid_set_list="",
        train_subsample_step=1,
    )
    encoder = Encoder(config).eval()
    inputs = _encoder_inputs()

    with torch.no_grad():
        enabled = encoder(inputs)
    assert enabled.shape == (1, encoder.token_num, config.hidden_dim)

    encoder.use_turn_indicators = False
    with torch.no_grad():
        disabled = encoder(inputs)
    assert disabled.shape == enabled.shape
