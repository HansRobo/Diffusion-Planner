from dataclasses import dataclass
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
    extra_train_set_list: Optional[str | list[str]] = None
    extra_train_set_repeat: int = 0
    extra_train_set_mask_traffic_lights: bool = False

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
    # The 2026-06 Tier IV corpus predates converter commit 55eff4f and duplicates t=0 in
    # short neighbor futures. Keep this on for that corpus; disable it for regenerated data.
    align_legacy_neighbor_futures: bool = True

    # ---------------------------------------------------------
    # Training Parameters
    # ---------------------------------------------------------
    seed: int = 3407
    train_epochs: int = 100
    batch_size: int = 512
    save_utd: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warm_up_epoch: int = 5
    encoder_drop_path_rate: float = 0.1
    decoder_drop_path_rate: float = 0.1
    use_ego_history: bool = True
    ego_history_dropout_rate: float = 0.4
    use_turn_indicators: bool = True
    # The turn head sees generated trajectories at inference. Train it on both the detached
    # model x-start trajectory and the expert trajectory; the normalized combination keeps the
    # historical loss scale while removing pure teacher-forcing exposure bias.
    turn_indicator_generated_loss_weight: float = 1.0
    turn_indicator_expert_loss_weight: float = 1.0

    # Keep Base diffusion training on the unbiased HDP hybrid objective. Road-border
    # compliance remains an evaluation signal and is optimized explicitly during RL.
    coeff_road_border_loss: float = 0.0
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
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True
    # HDP open-loop protocol: six stochastic trajectories, minADE/minFDE, six DPM steps.
    # Set num_samples=0 for a fast deterministic-only validation pass.
    multisample_eval_num_samples: int = 6
    multisample_eval_noise_scale: float = 0.1
    multisample_eval_sample_steps: int = 6
    multisample_eval_seed: int = 3407

    # HDP ego velocity representation & hybrid loss
    use_velocity_representation: bool = True
    planning_hybrid_loss: float = 0.01
    hybrid_loss_window: int = 10
    diffusion_supervision_type: Literal["x_start"] = "x_start"
    diffusion_time_sample_method: Literal["uniform"] = "uniform"
    # HDP real-vehicle setup reports six DPM-Solver integration steps. With the
    # final denoise-to-zero prediction this executes seven decoder forwards.
    diffusion_sample_steps: int = 6

    # HDP RL objective. The branch intentionally keeps only the official-style
    # reward-weighted RL-Hybrid path.
    rl_reward_normalize: Literal["group", "batch", "none"] = "group"
    rl_reward_beta: float = 0.5
    rl_noise_scale: float = 0.5
    # Keep the RL rollout budget independent from validation/export so it can be profiled
    # explicitly. Six integration steps plus denoise-to-zero are seven decoder forwards.
    rl_rollout_steps: int = 6
    rl_init_use_ema: bool = True
    rl_reward_w_risk: float = 1.0
    rl_reward_w_follow: float = 3.0
    rl_reward_w_lane: float = 2.5
    # Anti-stopping reward, measured as signed endpoint progress relative to the logged expert.
    # Kept independently tunable because the paper reward omits explicit progress.
    rl_reward_w_progress: float = 3.0
    # One expert target per scene costs only 1 / group_size of the candidate update and prevents
    # safety reward optimization from erasing the SFT driving behavior.
    rl_bc_weight: float = 1.0
    # The paper reports beta=1 and EMA update=0.05. Real-data stability audits use a lower
    # exponential temperature and slower previous-policy update to prevent SFT policy collapse.
    rl_ema_update_rate: float = 0.01
    # The paper does not publish the numerical speed-adaptive shaping functions.
    # These checkpointed values are tunable local defaults.
    rl_reward_dt: float = 0.1
    rl_ttc_critical_s: float = 0.5
    rl_ttc_safe_s: float = 3.0
    rl_thw_critical_s: float = 0.5
    rl_thw_safe_s: float = 2.0
    rl_occupancy_critical_m: float = 0.25
    rl_occupancy_safe_m: float = 2.0
    rl_occupancy_speed_gain_s: float = 0.10
    rl_lane_half_width_m: float = 1.75
    rl_leader_lateral_margin_m: float = 0.75
    rl_full_eval_utd: int = 5
    rl_validate_before_training: bool = True
    rl_max_valid_loss_regression: float = 0.25
    rl_best_score_min_delta: float = 0.001

    # ---------------------------------------------------------
    # Throughput knobs. Defaults ON after live verification on 2026-07-07
    # (8xA100 bs512: +50% samples/s, 98% GPU util, loss trajectory matches fp32/TF32).
    # amp_dtype="bf16" autocasts ONLY the model forward — noising, SDE schedule math
    # and all losses stay fp32. Set --amp_dtype off / --fused_optimizer false /
    # --ddp_static_graph false to reproduce the exact legacy numerics.
    amp_dtype: Literal["off", "bf16"] = "bf16"
    fused_optimizer: bool = True
    ddp_static_graph: bool = True
    compile_model: bool = True
    export_onnx_on_save: bool = False

    device: str = "cuda"
    tf32: bool = True
    use_ema: bool = True

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    encoder_mixer_depth: int = 6
    encoder_fusion_depth: int = 6
    # The HDP real-vehicle decoder uses six temporal DiT blocks.
    decoder_depth: int = 6
    num_heads: int = 8
    hidden_dim: int = 256
    decoder_tokenization: Literal["temporal"] = "temporal"
    diffusion_model_type: Literal["x_start"] = "x_start"
    predicted_neighbor_num: int = 0
    resume_model_path: Optional[str] = None
    init_weights_path: Optional[str] = None

    # ---------------------------------------------------------
    # Logging & Distributed Setup
    # ---------------------------------------------------------
    use_wandb: bool = False
    wandb_run_id: Optional[str] = None
    wandb_project_name: str = "Diffusion-Planner-Temporal"
    wandb_step_log_interval: int = 0
    notes: str = ""
    ddp: bool = True
    find_unused_parameters: bool = False
    port: str = "22323"

    # ---------------------------------------------------------
    # Closed-loop validation (rendered rollout + wandb video), run on the checkpoint-save cadence
    # (``save_utd``). Disabled unless ``closed_loop_npz_root`` is set (dir tree of route NPZ frames,
    # one route).
    # ---------------------------------------------------------
    closed_loop_npz_root: str = ""
    closed_loop_seg_len: int = 100000  # large -> one route = one segment = one trial
    # Re-plan every N steps: replan=1 is a model forward EVERY step (~minutes/epoch over a full
    # route); 40 keeps per-epoch cost to ~tens of seconds. Lower it for higher-fidelity validation.
    closed_loop_replan_interval: int = 40
    closed_loop_draw_every: int = 4  # render 1 of every N steps (matplotlib is the dominant cost)
    closed_loop_fps: int = 10
    closed_loop_near_miss_thresh: float = 0.5
    closed_loop_search_radius: float = 1.5
    closed_loop_warmup_steps: int = 0
    closed_loop_unstick_after: int = 300
    closed_loop_unstick_advance_m: float = 2.5
    closed_loop_classification_json: str = ""
    closed_loop_scenario_dataset_name: str = ""
    closed_loop_grouped_wandb_max_videos: int = 24

    # ---------------------------------------------------------
    # Normalizers (Placeholders to be initialized and set during training execution)
    # ---------------------------------------------------------
    state_normalizer: Optional[StateNormalizer] = None
    observation_normalizer: Optional[ObservationNormalizer] = None
