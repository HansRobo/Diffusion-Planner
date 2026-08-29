from dataclasses import dataclass
from typing import Literal

from diffusion_planner.dimensions import (
    INPUT_T,
    MAX_NUM_NEIGHBORS,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    OUTPUT_T,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
)


@dataclass
class ModelConfig:
    """Model configuration including data dimensions and architecture."""

    # ---------------------------------------------------------
    # Data Dimensions
    # ---------------------------------------------------------
    future_len: int = OUTPUT_T
    time_len: int = INPUT_T + 1
    ego_prediction_horizon: int = OUTPUT_T

    agent_state_dim: int = 11
    agent_num: int = MAX_NUM_NEIGHBORS

    static_objects_state_dim: int = 10
    static_objects_num: int = 5

    lane_num: int = NUM_SEGMENTS_IN_LANE
    lane_len: int = POINTS_PER_LANELET

    route_num: int = NUM_SEGMENTS_IN_ROUTE
    route_len: int = POINTS_PER_LANELET

    polygon_num: int = NUM_POLYGONS
    polygon_len: int = POINTS_PER_POLYGON

    line_string_num: int = NUM_LINE_STRINGS
    line_string_len: int = POINTS_PER_LINE_STRING

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    # Per-element MLP-Mixer size. These blocks run on every element (~560 per sample)
    # at every point (20-31), so their activations dominate encoder time and memory:
    # cost scales as depth * mixer dims, not as hidden_dim. 128/64/6 is what ran before.
    encoder_mixer_hidden_dim: int = 128
    encoder_mixer_token_dim: int = 64
    encoder_mixer_depth: int = 2
    encoder_fusion_depth: int = 6
    decoder_depth: int = 3
    num_heads: int = 8
    hidden_dim: int = 256
    diffusion_model_type: Literal["x_start", "flow_matching"] = "x_start"

    # Ego history representation. With use_ego_stop_history the raw pose history is replaced by a
    # per-step stopped/moving flag plus the current speed, which removes the shortcut of
    # extrapolating the ego's own past waypoints.
    use_ego_stop_history: bool = True
    ego_stop_velocity_threshold_mps: float = 0.1
    ego_history_dt_s: float = 0.1

    # Diffusion / flow time sampling. "logit_normal" concentrates the sampled times around the
    # middle of the schedule, where the denoiser actually decides the trajectory shape; uniform
    # spends most of its budget on the trivially easy ends.
    time_sampling: Literal["uniform", "logit_normal"] = "logit_normal"
    time_sampling_mean: float = 0.0
    time_sampling_std: float = 1.0
    predicted_neighbor_num: int = MAX_NUM_NEIGHBORS
