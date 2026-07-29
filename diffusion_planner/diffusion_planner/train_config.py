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
    # AdamW weight decay. Applied only to matmul weights; biases and
    # LayerNorm/BatchNorm/Embedding params are excluded (standard transformer
    # practice, matches original planTF). Previously torch AdamW's default 0.01
    # was applied to ALL params including norms/biases, over-regularizing.
    weight_decay: float = 1e-4
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
    # Mode classification loss weight (decoder_type="plantf" only)
    alpha_mode_cls_loss: float = 1.0
    # planTF loss shaping (docs/plantf_dead_mode_improvement.md). Both default
    # to the diffusion head's original loss behavior. The "dango" collapse
    # turned out to be driven by modes=1 (single-mode L2 = mean regression) and
    # was already resolved by modes>=2 with WTA; the earlier planTF-specific
    # overrides (endpoint loss on, lon down-weighting off) made results WORSE in
    # A/B and are reverted. Both remain as opt-in experiment knobs.
    coeff_endpoint_fde_loss: float = 0.0
    plantf_use_lon_velocity_weight: bool = True
    # Smoothness penalty on the xy second difference of the best planTF mode,
    # to suppress the "comb" jitter of the per-timestep absolute regression.
    # Off by default; try 0.1-1.0. See docs/plantf_original_comparison_and_roadmap.md.
    coeff_smoothness_loss: float = 0.0
    # Feed the current ego motion state (vx, vy, ax, ay, steering, yaw_rate) into
    # the planTF trajectory head so its absolute-waypoint regression is anchored
    # to the current motion (the diffusion decoder gets this via its pinned
    # current state; the planTF head otherwise does not). See
    # docs/plantf_dead_mode_improvement.md.
    plantf_use_ego_state_in_head: bool = True
    # A2 (docs/plantf_original_comparison_and_roadmap.md): replace the ego encoder
    # token with an embedding of the current ego motion state (vx,vy,ax,ay,steer,
    # yaw_rate) instead of the position-history token, matching original planTF's
    # use_ego_history=false path. state_dropout randomly zeroes those channels
    # during training (original uses 0.75). Anchors the whole prediction to the
    # current motion the way the diffusion decoder's pinned state does.
    plantf_ego_state_token: bool = False
    plantf_ego_state_dropout: float = 0.75
    # C1: feed agent (ego + neighbor) history as consecutive-frame xy deltas
    # instead of absolute positions, giving the encoder temporal-difference
    # inputs (original planTF vectorizes history this way).
    plantf_input_delta: bool = False
    # Zero the goal_pose input. The mini goal_pose is global-frame (data bug) while
    # Autoware feeds ego-frame -> train/deploy mismatch. Masking makes the model
    # plan from the route (route_lanes carries the goal). Bakes into ONNX.
    plantf_mask_goal_pose: bool = False
    # --- planTF combinable ablation toggles (docs/plantf_head_development_notes.md §9) ---
    # Trajectory head architecture. "mlp" (default): reshape the single ego token
    # into K modes. "cross_attn": K mode queries cross-attend to ALL encoder
    # tokens (map/agents/route), giving the head scene context the single-token
    # bottleneck lacks (candidate fix for tail divergence). "basis": the mlp head
    # but the trajectory is a Bezier curve of plantf_basis_control_points control
    # points expanded over time — a temporal inductive bias (structural C∞
    # smoothness) the flat per-step head lacks, while keeping mlp's deploy
    # robustness and ONNX-triviality (fixed basis matmul, no recurrence).
    # "gru": recurrent head that unrolls the waypoints with a GRU (temporal
    # recurrence in the architecture). Experimental — RNN heads are more
    # deploy-fragile than mlp/basis; not the deploy default.
    plantf_head_type: Literal["mlp", "cross_attn", "basis", "gru"] = "mlp"
    # Number of Bezier control points for plantf_head_type="basis" (ignored
    # otherwise). Fewer = smoother/stiffer, more = more expressive. 8 is a good
    # default for an 8s / 80-step horizon.
    plantf_basis_control_points: int = 8
    # Inference mode selection. When True, pick the ego mode by route adherence
    # among the top-k pi modes instead of argmax(pi). Recovers oracle-ish modes
    # for the smooth (velocity-rep) head at zero training cost. Applies to the
    # validation forward path only, not the ONNX deploy graph.
    plantf_route_rerank: bool = False
    plantf_route_rerank_topk: int = 3
    # Loss improvements (docs/plantf_head_development_notes.md §9), all opt-in and
    # independently combinable:
    #  - plantf_tail_weight (>0): weight the per-timestep ego regression toward the
    #    tail (w_t = 1 + tail_weight * t/(T-1)).
    #  - plantf_smoothness_tail_weight (>0, #6): same tail weighting on the curvature
    #    (second-difference) penalty.
    #  - plantf_use_laplace_nll (#5): replace the ego smooth-L1 with a Laplace NLL
    #    using a head-predicted per-point log-scale (planTF's probabilistic
    #    regression; calibrates tail uncertainty and mode confidence).
    plantf_tail_weight: float = 0.0
    plantf_smoothness_tail_weight: float = 0.0
    plantf_use_laplace_nll: bool = False

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
    # "plantf" replaces the diffusion decoder with a one-shot multi-modal
    # regression head (planTF, Cheng et al. ICRA 2024) on the same encoder.
    decoder_type: Literal["diffusion", "plantf"] = "diffusion"
    # Number of ego trajectory modes (decoder_type="plantf" only)
    num_modes: int = 6
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
    # Portability toggles: per-checkpoint trajectory rendering (matplotlib +
    # per-checkpoint forward pass) and ONNX export. Both default on to preserve
    # behavior, but can be disabled for lightweight / dependency-free training
    # runs on other environments.
    enable_checkpoint_viz: bool = True
    enable_onnx_export: bool = True

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
    # ---------------------------------------------------------
    closed_loop_npz_root: str = ""
    closed_loop_seg_len: int = 100000  # large -> one route = one segment = one trial
    # Re-plan every N steps: replan=1 is a model forward EVERY step (~minutes/epoch over a full
    # route); 40 keeps per-epoch cost to ~tens of seconds. Lower it for higher-fidelity validation.
    closed_loop_replan_interval: int = 4
    closed_loop_draw_every: int = 4  # render 1 of every N steps (matplotlib is the dominant cost)
    closed_loop_fps: int = 10
    closed_loop_near_miss_thresh: float = 0.5
    closed_loop_search_radius: float = 1.5
    closed_loop_warmup_steps: int = 0
    closed_loop_unstick_after: int = 300
    closed_loop_unstick_advance_m: float = 2.5

    # ---------------------------------------------------------
    # Normalizers (Placeholders to be initialized and set during training execution)
    # ---------------------------------------------------------
    state_normalizer: Optional[StateNormalizer] = None
    observation_normalizer: Optional[ObservationNormalizer] = None

    # ---------------------------------------------------------
    # Deterministic
    # ---------------------------------------------------------
    deterministic: bool = True
