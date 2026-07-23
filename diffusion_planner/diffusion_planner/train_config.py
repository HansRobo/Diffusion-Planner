from dataclasses import dataclass, field
from typing import Literal, Optional

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
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


@dataclass
class TrainConfig:
    # ---------------------------------------------------------
    # Required Arguments (Fields without default values must be declared first)
    # ---------------------------------------------------------
    exp_name: str
    save_dir: str
    train_set_list: str
    valid_set_list: str
    train_subsample_step: int

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
    # DataLoader Parameters
    # ---------------------------------------------------------
    use_data_augment: bool = True
    augment_prob: float = 0.5
    augment_type: Literal["quintic", "bridge"] = "quintic"
    num_refine: int = 20
    ego_past_noise_std: float = 0.1
    use_smoothing_future_trajectory: bool = True
    normalization_file_path: str = "normalization.json"
    num_workers: int = 8
    pin_mem: bool = True

    # ---------------------------------------------------------
    # Training Parameters
    # ---------------------------------------------------------
    seed: int = 3407
    train_epochs: int = 100
    batch_size: int = 512
    save_utd: int = 10
    learning_rate: float = 1e-4
    warm_up_epoch: int = 5
    encoder_drop_path_rate: float = 0.1
    decoder_drop_path_rate: float = 0.1
    use_ego_history: bool = True
    ego_history_dropout_rate: float = 0.6
    use_turn_indicators: bool = True

    # Loss Coefficients
    coeff_position_lat_loss: float = 1.0
    coeff_position_lon_loss: float = 1.0
    coeff_heading_l2_loss: float = 1.0
    coeff_velocity: float = 1.0
    # Use default_factory for mutable default values like lists
    coeff_timestep: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    coeff_road_border_loss: float = 1.0
    road_border_margin: float = 0.25
    road_border_n_interp: int = 2

    coeff_neighbor_collision_loss: float = 0.0
    neighbor_collision_margin_vehicle: float = 0.25
    neighbor_collision_margin_pedestrian: float = 1.0
    neighbor_collision_margin_bicycle: float = 0.5

    # Validation-only Autoware-aligned EPDMS metrics. train_predictor.py reads
    # these defaults when constructing argparse, so this remains the single
    # default source while keeping existing behavior unchanged unless explicitly enabled.
    enable_epdms_eval: bool = False
    # Backward-compatible alias for local scripts that used PDMS naming.
    enable_pdms_eval: bool = False
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True

    alpha_planning_loss: float = 1.0
    alpha_neighbor_loss: float = 0.1

    # Velocity Representation & Hybrid Loss
    use_velocity_representation: bool = False
    hybrid_loss_omega: float = 0.1
    hybrid_loss_window: int = 10

    guidance_scale: float = 0.5
    device: str = "cuda"
    use_ema: bool = True

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    encoder_mixer_depth: int = 6
    encoder_fusion_depth: int = 6
    decoder_depth: int = 3
    num_heads: int = 8
    hidden_dim: int = 256
    diffusion_model_type: Literal["x_start", "flow_matching"] = "x_start"
    predicted_neighbor_num: int = MAX_NUM_NEIGHBORS
    resume_model_path: Optional[str] = None

    # ---------------------------------------------------------
    # Logging & Distributed Setup
    # ---------------------------------------------------------
    use_wandb: bool = False
    wandb_run_id: Optional[str] = None
    wandb_project_name: str = "Diffusion-Planner"
    notes: str = ""
    ddp: bool = True
    port: str = "22323"

    # Validation-only temporal stability metrics. Replan consistency requires full-sequence
    # Step-1 NPZ frames in valid_set_list; the default gap=1 avoids treating skip-N lists
    # as true frame-to-frame replanning data.
    enable_temporal_stability_eval: bool = True
    enable_replan_consistency_eval: bool = True
    replan_consistency_expected_gap: int = 1

    # ---------------------------------------------------------
    # Closed-loop validation (rendered rollout + wandb video), run on the checkpoint-save cadence
    # (``save_utd``). Disabled unless ``closed_loop_npz_root`` is set (dir tree of route NPZ frames,
    # one route).
    #
    # ``closed_loop_sites_root`` is an alternative to ``closed_loop_npz_root`` for multi-site
    # validation: each direct subdirectory of the root (e.g. one per {project}/{map}/) is
    # evaluated as its own independent npz_root, and results are wandb-logged under
    # "closed_loop/<site_name>/...". Sites are never merged into one npz_root/rglob, since route
    # grouping is filename-only and ignores directory structure — mixing sites risks silently
    # merging unrelated routes that happen to share a bag-name prefix. When both are set,
    # ``closed_loop_sites_root`` takes precedence.
    # ---------------------------------------------------------
    closed_loop_npz_root: str = ""
    closed_loop_sites_root: str = ""
    closed_loop_seg_len: int = 100000  # large -> one route = one segment = one trial
    closed_loop_draw_every: int = 4
    closed_loop_fps: int = 10
    closed_loop_near_miss_thresh: float = 0.5
    closed_loop_search_radius: float = 1.5
    closed_loop_warmup_steps: int = 0
    closed_loop_unstick_after: int = 300
    closed_loop_unstick_advance_m: float = 2.5
    closed_loop_replan_interval: int = 10
    # Grouped closed-loop (scenario_classification_json episodes). When a JSON is resolved,
    # training logs per-area metrics + videos to wandb; otherwise falls back to full-route CL.
    closed_loop_classification_json: str = ""
    closed_loop_classification_json_root: str = ""
    closed_loop_scenario_dataset_name: str = ""
    closed_loop_profile: bool = False
    closed_loop_profile_sync_gpu: bool = False
    # Early-abort a badly-diverged segment instead of burning the full step budget (see
    # RolloutParams / reproducer_rollout.render_segment for the exact trigger condition).
    # 0 = disabled for either knob. Defaults mirror RolloutParams's.
    closed_loop_abort_deviation_m: float = 8.0
    closed_loop_abort_after: int = 30
    closed_loop_abort_max_snaps: int = 0
    # wandb payload shaping (see scenario_generation.wandb_closed_loop): only ONE
    # representative episode's video + trajectory-colormap image is uploaded per site per
    # checkpoint (not all routes — those stay in the local/server report), picked by
    # "worst" (default, most collisions) / "first" / "longest". The full report (all videos +
    # HTML gallery) stays at the run's out_dir; closed_loop_report_base_url turns that into a
    # clickable URL in wandb (e.g. when out_dir is served over HTTP on a training server) —
    # leave empty to just record the local path for a human to open by hand.
    closed_loop_wandb_video_pick: str = "worst"
    # Trajectory-colormap metrics rendered for that one representative episode — a small,
    # cheap-to-upload set of images (not videos), so unlike the video/episode pick above there's
    # no need to choose just one: every metric here gets its own wandb key/local PNG, and the
    # local HTML report additionally lets a human switch between them per-card via a dropdown.
    closed_loop_colormap_metrics: list[str] = field(
        default_factory=lambda: ["clearance", "collision", "near_miss", "speed", "road_border"]
    )
    closed_loop_report_base_url: str = ""

    # ---------------------------------------------------------
    # Normalizers (Placeholders to be initialized and set during training execution)
    # ---------------------------------------------------------
    state_normalizer: Optional[StateNormalizer] = None
    observation_normalizer: Optional[ObservationNormalizer] = None

    # ---------------------------------------------------------
    # Deterministic
    # ---------------------------------------------------------
    deterministic: bool = True
