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
    TURN_INDICATOR_OUTPUT_DIM,
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
    # Converter sidecars mark frames that were written only for gap-free replay.
    # Training filters those entries once into a run-local manifest cache.
    filter_skipped: bool = True
    skip_filter_sidecar_root: Optional[str] = None
    skip_filter_workers: int = 32
    skip_filter_processes: int = 16

    # ---------------------------------------------------------
    # Data Dimensions
    # ---------------------------------------------------------
    future_len: int = OUTPUT_T
    # Raw temporal tensor width. Feature encoders may select shorter effective
    # windows and left-pad them back to this width.
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
    # New corpora generated after converter commit 55eff4f already start neighbor futures
    # at t+0.1 s.  The legacy +1 correction is opt-in for old manifests only; keeping it
    # enabled by default silently moves valid stationary/short tracks in the new corpus.
    align_legacy_neighbor_futures: bool = False

    # ---------------------------------------------------------
    # Training Parameters
    # ---------------------------------------------------------
    seed: int = 3407
    train_epochs: int = 100
    batch_size: int = 512
    save_utd: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    # Exclude normalization/embedding/bias parameters from AdamW decay. This is an
    # optimizer-policy choice and is protected by strict resume compatibility.
    adamw_no_decay: bool = True
    warm_up_epoch: int = 5
    encoder_drop_path_rate: float = 0.1
    decoder_drop_path_rate: float = 0.1
    use_ego_history: bool = True
    # Current plus the preceding 20 frames (2 seconds at 10 Hz). Neighbor
    # history remains a separate, shorter six-frame perception window.
    ego_history_frames: int = 21
    ego_history_dropout_rate: float = 0.4
    # The turn head sees generated trajectories at inference. In the production staged
    # contract these weights are consumed only by the later head stage (or an explicit
    # joint ablation), never by policy-only Base/SFT.
    turn_indicator_generated_loss_weight: float = 1.0
    turn_indicator_expert_loss_weight: float = 1.0
    # The production SFT contract is deliberately staged: ``policy`` adapts the
    # trajectory planner without evaluating or updating the auxiliary head, then
    # ``turn_indicator`` freezes the planner and trains the detached head below.
    # ``joint`` remains an explicit ablation only; it is never the safe default.
    supervised_training_stage: Literal["joint", "policy", "turn_indicator"] = "policy"
    # Expert pretraining avoids the expensive DPM rollout while the randomly initialized
    # head learns clean intent features. Deployment fine-tuning then exposes it to the
    # exact final DPM trajectory used online.
    turn_indicator_head_training_mode: Literal["expert", "deployment"] = "deployment"
    # Architecture provenance, persisted in args.json. These are deliberately
    # not user-tunable modes: the policy never consumes indicator history and
    # the detached intent head predicts the three valid Autoware driving states.
    policy_uses_turn_indicator_history: bool = False
    turn_indicator_output_dim: int = TURN_INDICATOR_OUTPUT_DIM

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
    #
    # ``rl_paper_exact`` pins every field listed in
    # ``diffusion_planner.hdp_rl_paper_exact`` to its published value (paper Table 3,
    # Eq. (14)-(17), Appendix "Reward Function Details", and the authors' released
    # dp_vla_rl_agent). It turns off every real-vehicle extension of this repository and
    # fails on any explicit flag that contradicts the paper, so a run either reproduces
    # HDP-RL exactly or refuses to start. See docs/hdp_rl_paper_fidelity.md.
    rl_paper_exact: bool = False
    # ``multi`` = lambda_risk r_risk + lambda_follow r_follow + lambda_lane r_lane (HDP-RL);
    # ``single`` = r_safety alone (HDP-RL dagger, the paper's single-reward baseline).
    rl_paper_reward: Literal["multi", "single"] = "multi"
    # The one pair paper-exact does NOT take from the paper. ``planning_hybrid_loss``
    # and ``hybrid_loss_window`` are the geometry of the norm the IL base was fitted
    # in, and this pipeline never retrains IL, so paper-exact inherits them from the
    # checkpoint it post-trains. Setting this True releases them and marks the run as
    # an omega/W ablation arm -- the only way to reach the published pair.
    rl_hybrid_ablation: bool = False
    # RL moves the same policy on the same distribution, so paper-exact compares the
    # corpus and the input perturbation against the IL base's args.json instead of
    # trusting the launcher to match by convention.
    rl_base_corpus_check: bool = True
    num_generations: int = 8
    rl_reward_normalize: Literal["group", "batch", "none"] = "group"
    rl_reward_beta: float = 0.5
    # Real Tier IV sweeps found that 1.5 supplies useful group diversity with G=8 while the
    # held-out/deployment comparison remains fixed at the public-policy temperature 0.5.
    rl_noise_scale: float = 1.5
    # Keep policy selection on one fixed distribution while rollout temperature is swept.
    # The public HDP NAVSIM policy samples with temperature 0.5 by default.
    rl_eval_noise_scale: float = 0.5
    rl_eval_num_generations: int = 32
    # Keep checkpoint selection comparable when the optimization reward weights are swept.
    # This is the current real-vehicle objective: paper multi-reward plus anti-stopping progress.
    rl_eval_reward_w_safety: float = 0.0
    rl_eval_reward_w_risk: float = 1.0
    rl_eval_reward_w_follow: float = 3.0
    rl_eval_reward_w_lane: float = 2.5
    rl_eval_reward_w_progress: float = 3.0
    rl_eval_reward_w_road_border: float = 0.0
    # Keep held-out policy selection on one safety-aware objective even when the
    # optimization gate is ablated against the exact published reward sum.
    rl_eval_behavior_gate: Literal["none", "safety", "risk"] = "safety"
    rl_eval_occupancy_use_road_border: bool = True
    # A stopped logged expert is usually waiting at a red light or behind traffic. The legacy
    # progress proxy assigned every candidate the same full score in those scenes. Keep held-out
    # evaluation on the distance-aware objective even when the training behavior is ablated.
    rl_eval_stationary_progress_mode: Literal["constant", "distance"] = "distance"
    rl_eval_stationary_reference_threshold_m: float = 1.0
    rl_eval_stationary_progress_tolerance_m: float = 2.0
    rl_eval_stopped_neighbor_vel_thresh: float = 0.1
    rl_eval_stopped_neighbor_disp_thresh: float = 0.5
    rl_eval_red_light_constraint: bool = True
    rl_eval_red_light_lane_tolerance_m: float = 2.0
    # Held-out selection stays on one frozen objective while the training aggregation,
    # scoring horizon, or gate are swept.
    rl_eval_reward_aggregation: Literal["weighted_sum", "gated_product"] = "weighted_sum"
    rl_eval_reward_horizon_steps: int = 0
    # The deployed planner executes one zero-noise plan. Selection measures that exact
    # trajectory under the run's OWN training objective (one reward per run, the
    # source repository's discipline). The frozen rl_eval_* stochastic metrics are
    # report-only cross-arm diagnostics; acceptance safety comes from the independent
    # EPDMS/DAC/safety source guards, never from a second reward.
    rl_eval_deterministic: bool = True
    rl_selection_metric: Literal["deterministic", "mean"] = "deterministic"
    # Keep the RL rollout budget independent from validation/export so it can be profiled
    # explicitly. Six integration steps plus denoise-to-zero are seven decoder forwards.
    rl_rollout_steps: int = 6
    # The released NAVSIM trainer performs one noising per replay draw, but each mined
    # group is drawn ~9x across its replay epochs — so ~9 updates per rollout is the
    # released schedule's effective yield. Online, values > 1 reproduce that
    # amortization (fresh diffusion noise per update; reward/weights/encoding reused).
    # The production launcher sets 9; 1 remains the strict one-update-per-draw form.
    rl_updates_per_rollout: int = 1
    # Keep full reward groups but cap differentiable candidate batches. With 64 scenes/rank,
    # G=32 remains one 2,048-candidate update; G=64 becomes two accumulated 2,048-candidate
    # forwards feeding one exact optimizer update.
    rl_update_max_candidates_per_rank: int = 2048
    # 100-epoch relay state machine (the source repository's schedule): epochs
    # 1, 1+interval, ... are rollout-only mining passes that freeze a per-rank
    # disk replay cache; the remaining epochs train exclusively from it with
    # weights/validity frozen at mine time. 0 keeps the online loop.
    rl_rollout_interval: int = 0
    rl_replay_dir: Optional[str] = None
    rl_init_use_ema: bool = True
    # Optional direct collision term. Zero omits the standalone safety component; positive values
    # test whether active=1/rear=0.3 safety prevents progress-risk Pareto regressions.
    rl_reward_w_safety: float = 0.0
    rl_reward_w_risk: float = 1.0
    rl_reward_w_follow: float = 3.0
    rl_reward_w_lane: float = 2.5
    # Anti-stopping reward, measured as signed endpoint progress relative to the logged expert.
    # Kept independently tunable because the paper reward omits explicit progress.
    rl_reward_w_progress: float = 3.0
    # Optional real-vehicle extension: direct ego-footprint clearance to HD-map borders.
    # Zero preserves the paper-style reward without this extra term.
    rl_reward_w_road_border: float = 0.0
    # The published multi-reward formula uses no gate. ``safety`` is our real-vehicle
    # extension that prevents follow/lane/progress from compensating for a collision;
    # ``risk`` also suppresses those terms for near misses.
    rl_behavior_gate: Literal["none", "safety", "risk"] = "safety"
    # ``pdm_port`` swaps the whole training reward for the verbatim port of the audited
    # original-DP ``hdp_pdm`` objective (Col x DAC binary gates x (5*TTC + 5*EP +
    # 2*lane + 2*comfort)/14 on the shared planner_metrics layer). Held-out selection
    # always stays on the frozen native objective.
    rl_reward_source: Literal["native", "pdm_port"] = "native"
    # T4 scenes carry traffic-light semantics the source data lacked; the ported
    # reward gates red-light violations by default (ablatable to exact-faithful).
    rl_pdm_red_light_gate: bool = True
    # ``gated_product`` is the PDM-style bounded objective behind the audited original-DP
    # AWR gains: multiplicative collision/red-light/border gates times a normalized
    # risk/follow/lane/progress quality mix. ``weighted_sum`` preserves the historical
    # additive objective exactly.
    rl_reward_aggregation: Literal["weighted_sum", "gated_product"] = "weighted_sum"
    # Score only the first N steps of each candidate (0 = full horizon). The original-DP
    # AWR profile scored a 4 s prefix (40 steps) of the 8 s plan.
    rl_reward_horizon_steps: int = 0
    # Candidates regressed on the scored prefix only; 0 follows rl_reward_horizon_steps.
    # Regressing the unscored tail was significantly negative in the original-DP audits.
    rl_candidate_loss_horizon: int = 0
    # Restrict the reweighted regression's diffusion-time draw. The original-DP ablation
    # preferred [0.001, 0.2] over the full range; defaults preserve the historical
    # [eps, 1] distribution exactly.
    rl_diffusion_t_min: float = 0.0
    rl_diffusion_t_max: float = 1.0
    # HDP rollout-candidate augmentation (the exploration mechanism of HDP's own RL,
    # `augment_trajectory_batch`): perturb sampled candidates before reward/regression
    # so AWR can rank and internalize behavior beyond the policy's own support. Applied
    # in the velocity-safe form: route-frame offsets with a mandatory minimum-jerk
    # onset ramp (a constant offset is a first-delta impulse under velocity actions and
    # is rejected), plus an optional PlannerRFT candidate stretch that scales per-step
    # displacements — natively smooth for velocity actions. Off by default; the
    # experiment ladder enables it explicitly.
    rl_candidate_aug_prob: float = 0.0
    # Gaussian scheme: route-frame offset std in metres (released HDP uses 0.5).
    # stratified_beta scheme: support half-width (PlannerRFT lambda; upstream used 1.0).
    rl_candidate_aug_std: float = 0.5
    rl_candidate_aug_eta_scheme: Literal["gaussian", "stratified_beta"] = "gaussian"
    # softplus(0) + 1: the PlannerRFT zero-init exploration-head concentration.
    rl_candidate_aug_beta_concentration: float = 1.6931471805599454
    # PlannerRFT candidate stretch half-width; per-step displacements scale by
    # 1 + stretch * eta (upstream lambda_lon = 0.25). Zero disables the stretch.
    rl_candidate_aug_stretch: float = 0.0
    # ~2 s minimum-jerk onset: bounds every per-step velocity increment to
    # ~1.875/ramp_steps of the sampled offset (0.5 m over 20 steps ≈ 0.47 m/s peak).
    rl_candidate_aug_ramp_steps: int = 20
    # Unaugmented on-policy anchors kept at the front of every group.
    rl_candidate_aug_keep: int = 1
    # Skip near-stationary scenes: offsetting a standstill trajectory manufactures the
    # exact jump failure the first-waypoint gate exists to catch (upstream guard: 2.0).
    rl_candidate_aug_speed_min_mps: float = 2.0
    # Reject low-speed candidates whose first waypoint jumps away from the current pose
    # (audited original-DP failure: reward-blind standstill jumps winning the advantage).
    # The 5 cm tangent floor is mandatory — without it a numerically-zero standstill step
    # reads as 90 degrees off-tangent and entire low-speed groups are silently discarded.
    rl_first_waypoint_gate: bool = True
    rl_first_waypoint_gate_speed_max_mps: float = 1.0
    rl_first_waypoint_gate_max_step_m: float = 0.25
    rl_first_waypoint_gate_max_lateral_m: float = 0.20
    rl_first_waypoint_gate_max_backward_m: float = 0.05
    rl_first_waypoint_gate_max_tangent_deg: float = 75.0
    rl_first_waypoint_gate_tangent_min_step_m: float = 0.05
    # Tier IV data has no populated static-object tensor, so OCC falls back to
    # stopped actors and, when none exist, HD-map road borders. Keep this ablatable
    # because road borders are a practical proxy rather than literal occupancy.
    rl_occupancy_use_road_border: bool = True
    rl_stationary_progress_mode: Literal["constant", "distance"] = "distance"
    rl_red_light_constraint: bool = True
    # The EMA policy boundary already limits drift. Real-data and recovery-set ablations found
    # that an additional expert anchor slows reward learning without improving robustness.
    rl_bc_weight: float = 0.0
    # When the anchor is enabled, restrict it to scenes with an active reward group; a
    # broad every-scene anchor was significantly negative in the original-DP AWR audits.
    rl_bc_active_groups_only: bool = True
    # Commit the live proposal into the frozen rollout policy once per policy iteration.
    rl_ema_update_rate: float = 0.05
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
    # Logged neighbors below both thresholds are treated as stationary occupancy
    # references. Keep these values in the checkpoint so reward geometry is reproducible.
    rl_stopped_neighbor_vel_thresh: float = 0.1
    rl_stopped_neighbor_disp_thresh: float = 0.5
    rl_lane_half_width_m: float = 1.75
    rl_leader_lateral_margin_m: float = 0.75
    rl_stationary_reference_threshold_m: float = 1.0
    rl_stationary_progress_tolerance_m: float = 2.0
    rl_red_light_lane_tolerance_m: float = 2.0
    rl_road_border_critical_m: float = 0.20
    rl_road_border_safe_m: float = 0.60
    rl_full_eval_utd: int = 5
    rl_validate_before_training: bool = True
    rl_max_valid_loss_regression: float = 0.25
    # Best-policy selection must not buy behavior reward by degrading held-out safety primitives
    # or the independent EPDMS proxy. These are absolute tolerances because all scores are [0, 1].
    # Held-out policy metrics are finite-sample estimates. A 0.1 percentage-point
    # tolerance avoids rejecting an otherwise improved checkpoint on numerical/sample noise.
    rl_max_valid_safety_regression: float = 0.001
    rl_max_valid_epdms_regression: float = 0.001
    rl_best_score_min_delta: float = 0.0001
    rl_early_stop_patience: int = 0

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
