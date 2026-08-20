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
    # Stride into the validation list. 1 = the whole list (unchanged behaviour);
    # >1 trades validation-mean precision for epoch wall-clock, which matters here
    # because the DrivoR head runs the PDM oracle over every validation batch.
    valid_subsample_step: int = 1

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
    # 40 epochs at global batch 512 is 425,480 optimizer steps on the full train
    # list.  The DrivoR schedule derives its cosine ``T_max`` from this, so it
    # has to be what will actually be run -- an aspirational number means the LR
    # never comes down.  NOTE the warmup is specified as a *ratio* of total
    # steps, so raising the epoch count without re-deriving
    # ``--drivor_warmup_ratio`` silently lengthens the ramp by the same factor:
    # the ramp is meant to stay ~2,000 steps regardless of the epoch count.
    train_epochs: int = 40
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
    compile_model: bool = False
    # torch.compile mode.  "default" fuses kernels but still launches each one
    # from Python; "reduce-overhead" additionally captures CUDA graphs, which is
    # the fix for a launch-bound step -- it requires static shapes (so
    # ``drop_last=True``) and no host sync inside the captured region.
    # Measured on 8xH100 at global batch 512:
    # "default" 2.131 batch/s, "reduce-overhead" 1.170 -- CUDA graphs cost 45 %
    # here, because after the EMA fix the step is GPU-bound rather than
    # launch-bound, so graph capture/replay buys nothing and adds overhead.
    # "max-autotune-no-cudagraphs" is the one worth trying next: better kernel
    # selection without the graph machinery.
    compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] = "default"
    use_amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    # DataLoader throughput knobs (worker start-up dominates otherwise: the NPZ
    # shards are DEFLATE-compressed, so decode is the bottleneck and the workers
    # must stay alive across epochs with a filled prefetch queue).
    persistent_workers: bool = True
    prefetch_factor: int = 4
    # Fused AdamW folds the whole optimizer step into one kernel; foreach is the
    # fallback when the fused implementation is unavailable for the device.
    fused_optimizer: bool = True

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
    # Predictor head. ``diffusion`` is the DiT/DPM-Solver decoder; ``drivor`` is the
    # proposal-generate-then-score head (64 proposals -> refinement -> PDM scorer ->
    # argmax), which outputs ONLY an ego trajectory. The DrivoR path uses its own
    # loss, metric taxonomy, training epoch and validation; every ``drivor_*`` field
    # below is inert under ``diffusion``.
    # ---------------------------------------------------------
    predictor_head: Literal["diffusion", "drivor"] = "diffusion"

    # Architecture (DrivoR's own defaults)
    drivor_proposal_num: int = 64
    drivor_ref_num: int = 4
    drivor_scorer_ref_num: int = 4
    drivor_tf_d_ffn: int = 1024
    drivor_refiner_num_heads: int = 8
    drivor_refiner_ls_values: float = 1.0
    drivor_trajectory_proj_drop: float = 0.0
    drivor_trajectory_drop_path: float = 0.0
    drivor_scorer_proj_drop: float = 0.0
    drivor_scorer_drop_path: float = 0.0
    # Demonstration head weight in the selection aggregate. Roughly half the
    # proposals tie at the maximum PDMS, and argmax inside a tie set is arbitrary;
    # this additive term orders them without crossing the PDMS ordering.  Raised
    # from 0.2 to bias selection further towards the human demonstration: the
    # term is additive and bounded by this weight, so it only re-orders
    # proposals whose PDMS gap is smaller than it.
    drivor_human_teacher_weight: float = 0.3
    # Hard bound on the six PDM logits: cap * tanh(raw / cap). 0 disables it.
    drivor_logit_bound: float = 10.0

    # Model-selection profile, DrivoR's PDMS weights (noc, dac multiplicative;
    # ddc/ttc/ep/comfort a weighted mean).
    drivor_weight_no_at_fault_collisions: float = 1.0
    drivor_weight_drivable_area_compliance: float = 1.0
    drivor_weight_driving_direction_compliance: float = 0.0
    drivor_weight_time_to_collision_within_bound: float = 5.0
    drivor_weight_ego_progress: float = 5.0
    drivor_weight_history_comfort: float = 2.0

    # Objective
    drivor_trajectory_weight: float = 1.0
    drivor_final_score_weight: float = 1.0
    drivor_prev_weight: float = 1.0
    drivor_label_smoothing: float = 0.02
    drivor_grad_clip: float = 1.0

    # Trajectory sampling -- NOT Diffusion-Planner's OUTPUT_T (80 poses / 8 s).
    #
    # The *horizon* is the part that is not a free choice: navsim scores 4 s
    # (``pdm_scoring/default_scoring_parameters.yaml``), so anything longer makes
    # PDMS incomparable to it.  The *density* inside those 4 s is free -- the head
    # is one-shot, so the pose count only changes one linear layer's width.
    # Measured on an H100 at batch 64: model forward+backward is 365 ms whether
    # the head emits 8 poses or 80, and the whole step is 393 ms at 40 poses vs
    # 393 ms at 8 (1.00x).
    #
    # Hence 40 @ 0.1 s: navsim's horizon at the 10 Hz the dataset and the
    # downstream controller already use, so no interpolation is needed anywhere.
    # DrivoR upstream instead emits ``num_poses: 8`` (``drivoR.yaml``) at
    # ``t4_trajectory_dt_s: 0.5`` (``t4_training.yaml``) and lets navsim's
    # ``transform_trajectory`` interpolate; set ``--drivor_num_poses 8
    # --drivor_pose_dt 0.5`` for that and everything still lines up.  Measured on
    # 512 validation scenes, the two representations of the *expert* trajectory
    # score identically (PDMS 0.9680 both ways, on all six sub-scores), so this
    # is a control-side choice, not an accuracy one.  See
    # ``diffusion_planner/utils/drivor_sampling.py``.
    drivor_num_poses: int = 40
    drivor_pose_dt: float = 0.1

    # PDM oracle (batched GPU labels for the model's own proposals).
    # ``default_scoring_parameters.yaml``'s ``proposal_sampling`` is
    # ``num_poses: 40, interval_length: 0.1``.  ``drivor_oracle_dt`` is the
    # *scoring* step, which is what the LQR rollout and every finite difference
    # in the metrics run at; at the defaults above the head already emits on this
    # grid, so proposals reach the oracle without interpolation.
    drivor_oracle_dt: float = 0.1
    drivor_scoring_num_poses: int = 40
    # navsim's PDMScorer evaluates every simulated step; a stride > 1 is an
    # approximation.  With the horizon at navsim's 40 steps instead of the
    # dataset's 80, scoring every step is affordable (27 ms of a 393 ms step), so
    # these default to navsim's behaviour.
    drivor_oracle_collision_stride: int = 1
    drivor_oracle_ttc_stride: int = 1
    drivor_oracle_border_stride: int = 1
    drivor_oracle_route_stride: int = 1
    drivor_oracle_max_neighbours: int = 32
    drivor_oracle_max_border_segments: int = 96
    drivor_oracle_max_route_segments: int = 128

    # Training loop
    drivor_log_every_n_steps: int = 25
    drivor_divergence_guard: bool = True
    drivor_ddp_find_unused: bool = False
    # How often the guard reads its verdict back to the host.  1 = exact
    # per-step skip, but ``.item()`` is an implicit sync that stops the CPU from
    # running ahead of the GPU.  >1 keeps the per-step decision on device (the
    # gradients are multiplied by a 0/1 device scalar) and only branches on the
    # host every N steps.  See DivergenceGuard for the NaN caveat.
    drivor_guard_sync_every: int = 1
    # EMA via fused multi-tensor ops instead of timm's deprecated per-tensor
    # Python loop, which issued several hundred kernel launches per step.
    drivor_fused_ema: bool = True

    # Optimizer schedule -- DrivoR's, not this repo's.  See utils/drivor_lr.py:
    # DrivoR advances a linear-ramp + cosine schedule once per OPTIMIZER STEP
    # (drivor_agent.py:483), whereas ``lr_schedule.py`` is advanced per epoch.
    # On the full train list (21,274 steps/epoch) the per-epoch form leaves the
    # first epoch entirely at 0.1x the peak, so "drivor" is the default here.
    drivor_lr_schedule: str = "drivor"
    drivor_warmup_ratio: float = 0.1
    # 0 = --learning_rate is the peak.  Set 64 to reproduce DrivoR's
    # sqrt(global_batch / 64) scaling of base_lr (drivor_agent.py:415).
    drivor_lr_base_batch_size: int = 0
    drivor_lr_probe_max: float = 3e-3
    drivor_lr_probe_steps: int = 900

    # ---------------------------------------------------------
    # Logging & Distributed Setup
    # ---------------------------------------------------------
    use_wandb: bool = False
    wandb_run_id: Optional[str] = None
    wandb_project_name: str = "Diffusion-Planner"
    # Team the run belongs to. May also be given inline as "entity/project" in
    # ``wandb_project_name``, which is how the wandb UI spells a project path.
    wandb_entity: str = ""
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
