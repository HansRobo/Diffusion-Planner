import argparse
import math
from functools import partial

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
from diffusion_planner.train import model_training
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.cli import boolean, dataclass_default
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

_train_config_default = partial(dataclass_default, TrainConfig)


def get_args(args_list=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="save path for model ckpt", required=True)

    # Data
    parser.add_argument("--train_set_list", type=str, required=True)
    parser.add_argument(
        "--extra_train_set_list",
        type=str,
        action="append",
        default=None,
        help="repeatable extra datalist; all listed paths are concatenated in memory",
    )
    parser.add_argument(
        "--extra_train_set_repeat",
        type=int,
        default=0,
        help="append the extra list N times in memory; no combined JSON or NPZ is written",
    )
    parser.add_argument("--valid_set_list", type=str, required=True)
    parser.add_argument(
        "--filter_skipped",
        type=boolean,
        default=_train_config_default("filter_skipped"),
        help="filter sidecar entries with is_skipped=true into a run-local manifest cache",
    )
    parser.add_argument(
        "--skip_filter_sidecar_root",
        type=str,
        default=_train_config_default("skip_filter_sidecar_root"),
        help="optional sidecar root when JSON sidecars are not next to each NPZ",
    )
    parser.add_argument(
        "--skip_filter_workers",
        type=int,
        default=_train_config_default("skip_filter_workers"),
        help="parallel sidecar readers used once while preparing filtered manifests",
    )
    parser.add_argument(
        "--skip_filter_processes",
        type=int,
        default=_train_config_default("skip_filter_processes"),
        help="processes for RDMA sidecar reads; 0 uses the in-process thread pool",
    )
    parser.add_argument("--train_subsample_step", type=int, default=1)
    parser.add_argument(
        "--align_legacy_neighbor_futures",
        type=boolean,
        default=_train_config_default("align_legacy_neighbor_futures"),
        help="shift duplicated t=0 short neighbor tracks in pre-55eff4f Tier IV NPZs",
    )

    parser.add_argument("--future_len", type=int, default=OUTPUT_T)
    parser.add_argument("--time_len", type=int, default=INPUT_T + 1)
    parser.add_argument("--ego_prediction_horizon", type=int, default=OUTPUT_T)

    parser.add_argument("--agent_state_dim", type=int, help="past state dim for agents", default=11)
    parser.add_argument("--agent_num", type=int, default=MAX_NUM_NEIGHBORS)

    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_num", type=int, default=NUM_SEGMENTS_IN_LANE)
    parser.add_argument("--lane_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--route_num", type=int, default=NUM_SEGMENTS_IN_ROUTE)
    parser.add_argument("--route_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--polygon_num", type=int, default=NUM_POLYGONS)
    parser.add_argument("--polygon_len", type=int, default=POINTS_PER_POLYGON)

    parser.add_argument("--line_string_num", type=int, default=NUM_LINE_STRINGS)
    parser.add_argument("--line_string_len", type=int, default=POINTS_PER_LINE_STRING)

    # DataLoader parameters
    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--augment_prob", type=float, help="augmentation probability", default=0.5)
    parser.add_argument(
        "--augment_type", type=str, choices=["quintic", "bridge"], default="quintic"
    )
    parser.add_argument(
        "--num_refine", type=int, default=20, help="number of refinement steps for augmentation"
    )
    parser.add_argument(
        "--ego_past_noise_std",
        type=float,
        default=0.1,
        help="std of noise applied to ego past trajectory during augmentation",
    )
    parser.add_argument(
        "--use_smoothing_future_trajectory",
        default=True,
        type=boolean,
        help="whether to apply smoothing to future trajectory",
    )
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true", help="Pin CPU memory in DataLoader")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--save_utd", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=_train_config_default("weight_decay"),
        help="AdamW weight decay; the HDP paper uses 0.01",
    )
    parser.add_argument(
        "--adamw_no_decay",
        type=boolean,
        default=_train_config_default("adamw_no_decay"),
        help="exclude norm, bias, embedding, and positional embedding parameters from decay",
    )
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_ego_history", type=boolean, default=True)
    parser.add_argument(
        "--ego_history_frames",
        type=int,
        default=_train_config_default("ego_history_frames"),
        help="number of most-recent ego frames encoded, including current (default 21)",
    )
    parser.add_argument(
        "--ego_history_dropout_rate",
        type=float,
        default=_train_config_default("ego_history_dropout_rate"),
    )
    parser.add_argument(
        "--turn_indicator_generated_loss_weight",
        type=float,
        default=_train_config_default("turn_indicator_generated_loss_weight"),
    )
    parser.add_argument(
        "--turn_indicator_expert_loss_weight",
        type=float,
        default=_train_config_default("turn_indicator_expert_loss_weight"),
    )
    parser.add_argument(
        "--turn_indicator_head_num_queries",
        type=int,
        default=_train_config_default("turn_indicator_head_num_queries"),
        help="learned cross-attention queries the intent head uses to read the scene",
    )
    parser.add_argument(
        "--turn_indicator_head_num_layers",
        type=int,
        default=_train_config_default("turn_indicator_head_num_layers"),
        help="cross-attention blocks in the intent head; its only capacity knob, since the policy is frozen",
    )
    parser.add_argument(
        "--turn_indicator_head_dropout",
        type=float,
        default=_train_config_default("turn_indicator_head_dropout"),
    )
    parser.add_argument(
        "--turn_indicator_label_smoothing",
        type=float,
        default=_train_config_default("turn_indicator_label_smoothing"),
        help="uniform label smoothing for the head's cross-entropy; 0.0 is plain cross_entropy",
    )
    parser.add_argument(
        "--supervised_training_stage",
        choices=("joint", "policy", "turn_indicator"),
        default=_train_config_default("supervised_training_stage"),
        help="train trajectory policy only (default), an explicit joint ablation, or the detached intent head only",
    )
    parser.add_argument(
        "--turn_indicator_head_training_mode",
        choices=("expert", "deployment"),
        default=_train_config_default("turn_indicator_head_training_mode"),
        help="use cheap expert trajectories or exact final-DPM trajectories for head-only training",
    )

    parser.add_argument(
        "--coeff_road_border_loss",
        type=float,
        default=_train_config_default("coeff_road_border_loss"),
    )
    parser.add_argument("--road_border_margin", type=float, default=0.25)
    parser.add_argument("--road_border_n_interp", type=int, default=2)

    parser.add_argument("--coeff_neighbor_collision_loss", type=float, default=0.0)
    parser.add_argument(
        "--neighbor_collision_margin_vehicle",
        type=float,
        default=0.25,
        help="per-side neighbor box inflation [m] for vehicles",
    )
    parser.add_argument(
        "--neighbor_collision_margin_pedestrian",
        type=float,
        default=1.0,
        help="per-side neighbor box inflation [m] for pedestrians",
    )
    parser.add_argument(
        "--neighbor_collision_margin_bicycle",
        type=float,
        default=0.5,
        help="per-side neighbor box inflation [m] for bicycles",
    )

    parser.add_argument(
        "--enable_epdms_eval",
        default=_train_config_default("enable_epdms_eval"),
        type=boolean,
    )
    parser.add_argument(
        "--epdms_eval_use_agent_boxes",
        default=_train_config_default("epdms_eval_use_agent_boxes"),
        type=boolean,
    )
    parser.add_argument(
        "--epdms_eval_use_road_border",
        default=_train_config_default("epdms_eval_use_road_border"),
        type=boolean,
    )
    parser.add_argument(
        "--multisample_eval_num_samples",
        type=int,
        default=_train_config_default("multisample_eval_num_samples"),
        help="stochastic trajectories for HDP minADE/minFDE; 0 disables the extra pass",
    )
    parser.add_argument(
        "--multisample_eval_noise_scale",
        type=float,
        default=_train_config_default("multisample_eval_noise_scale"),
    )
    parser.add_argument(
        "--multisample_eval_sample_steps",
        type=int,
        default=_train_config_default("multisample_eval_sample_steps"),
    )
    parser.add_argument(
        "--multisample_eval_seed",
        type=int,
        default=_train_config_default("multisample_eval_seed"),
    )

    # Velocity representation & hybrid loss (HDP paper, Section IV-B)
    parser.add_argument(
        "--use_velocity_representation",
        type=boolean,
        default=_train_config_default("use_velocity_representation"),
        help="Represent ego future xy as per-step displacement and train it with HDP hybrid loss",
    )
    parser.add_argument(
        "--planning_hybrid_loss",
        type=float,
        default=_train_config_default("planning_hybrid_loss"),
        help="Weight for waypoint loss term in the official HDP hybrid loss",
    )
    parser.add_argument(
        "--hybrid_loss_window",
        type=int,
        default=_train_config_default("hybrid_loss_window"),
        help="Gradient detach window size W for the waypoint loss term",
    )
    parser.add_argument(
        "--diffusion_supervision_type",
        type=str,
        choices=["x_start"],
        default=_train_config_default("diffusion_supervision_type"),
    )
    parser.add_argument(
        "--diffusion_time_sample_method",
        type=str,
        choices=["uniform"],
        default="uniform",
    )
    parser.add_argument(
        "--diffusion_sample_steps",
        type=int,
        default=_train_config_default("diffusion_sample_steps"),
    )

    parser.add_argument("--device", type=str, help="run on which device", default="cuda")
    parser.add_argument(
        "--tf32",
        default=_train_config_default("tf32"),
        type=boolean,
        help="enable TensorFloat-32 on CUDA",
    )

    parser.add_argument("--use_ema", default=True, type=boolean)

    # Model
    parser.add_argument("--encoder_mixer_depth", type=int, default=6)
    parser.add_argument("--encoder_fusion_depth", type=int, default=6)
    parser.add_argument(
        "--decoder_depth",
        type=int,
        help="number of temporal decoding layers",
        default=_train_config_default("decoder_depth"),
    )
    parser.add_argument("--num_heads", type=int, help="number of multi-head", default=8)
    parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=256)
    parser.set_defaults(decoder_tokenization="temporal")
    parser.add_argument(
        "--diffusion_model_type",
        type=str,
        choices=["x_start"],
        default=_train_config_default("diffusion_model_type"),
    )
    parser.add_argument(
        "--predicted_neighbor_num",
        type=int,
        default=_train_config_default("predicted_neighbor_num"),
        help="must be 0; HDP predicts only the ego trajectory",
    )
    parser.add_argument("--resume_model_path", type=str, help="path to resume model", default=None)
    parser.add_argument(
        "--init_weights_path",
        type=str,
        default=None,
        help="weights-only warm start; loads compatible model weights with fresh optimizer/schedule",
    )

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument(
        "--wandb_run_id", type=str, default=None, help="Existing wandb run ID to attach to"
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=_train_config_default("wandb_project_name"),
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_step_log_interval",
        type=int,
        default=_train_config_default("wandb_step_log_interval"),
        help="log train metrics every N optimizer steps (0 = epoch only)",
    )
    parser.add_argument("--notes", default="", type=str)

    # distributed training parameters
    parser.add_argument("--ddp", default=True, type=boolean, help="use ddp or not")
    parser.add_argument(
        "--find_unused_parameters",
        default=_train_config_default("find_unused_parameters"),
        type=boolean,
        help="set DDP find_unused_parameters=True only when a model has conditional unused parameters",
    )
    parser.add_argument("--port", default="22323", type=str, help="port")

    # throughput knobs (defaults preserve exact fp32/TF32 numerics)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        choices=["off", "bf16"],
        default=_train_config_default("amp_dtype"),
        help="bf16 autocasts ONLY the model forward; noising/SDE/losses stay fp32. "
        "Validate with a closed-loop A/B before adopting.",
    )
    parser.add_argument(
        "--fused_optimizer",
        type=boolean,
        default=_train_config_default("fused_optimizer"),
        help="fused AdamW CUDA kernel (not bitwise-identical to the default foreach path)",
    )
    parser.add_argument(
        "--ddp_static_graph",
        type=boolean,
        default=_train_config_default("ddp_static_graph"),
        help="enable DDP static_graph optimizations (graph must stay identical every step)",
    )
    parser.add_argument(
        "--compile_model",
        type=boolean,
        default=_train_config_default("compile_model"),
        help="compile the encoder and decoder with TorchInductor",
    )
    parser.add_argument(
        "--export_onnx_on_save",
        type=boolean,
        default=_train_config_default("export_onnx_on_save"),
        help="export ONNX synchronously at checkpoint cadence; disabled by default to avoid DDP idle time",
    )

    # per-epoch closed-loop validation (rendered rollout + wandb video).
    # Disabled unless --closed_loop_npz_root is given (dir tree of one route's NPZ frames).
    parser.add_argument(
        "--closed_loop_npz_root",
        type=str,
        default="",
        help="dir tree of route NPZ frames for closed-loop validation, run on the checkpoint-save "
        "cadence (save_utd). Empty = disabled. One route per trial.",
    )
    parser.add_argument(
        "--closed_loop_seg_len",
        type=int,
        default=100000,
        help="frames per segment; large => one route = one segment = one trial",
    )
    parser.add_argument(
        "--closed_loop_replan_interval",
        type=int,
        default=40,
        help="re-plan every N steps; 1 = forward every step (slow, ~minutes/epoch). 40 default",
    )
    parser.add_argument(
        "--closed_loop_draw_every",
        type=int,
        default=4,
        help="render 1 of every N steps (matplotlib render is the dominant cost)",
    )
    parser.add_argument("--closed_loop_fps", type=int, default=10)
    parser.add_argument("--closed_loop_near_miss_thresh", type=float, default=0.5)
    parser.add_argument("--closed_loop_search_radius", type=float, default=1.5)
    parser.add_argument("--closed_loop_warmup_steps", type=int, default=0)
    parser.add_argument("--closed_loop_unstick_after", type=int, default=300)
    parser.add_argument(
        "--closed_loop_unstick_advance_m",
        type=float,
        default=_train_config_default("closed_loop_unstick_advance_m"),
    )
    parser.add_argument(
        "--closed_loop_classification_json",
        type=str,
        default="",
        help="scenario classification JSON for grouped closed-loop evaluation",
    )
    parser.add_argument(
        "--closed_loop_scenario_dataset_name",
        type=str,
        default="",
        help="dataset-relative name used to auto-resolve a scenario classification JSON",
    )
    parser.add_argument(
        "--closed_loop_grouped_wandb_max_videos",
        type=int,
        default=_train_config_default("closed_loop_grouped_wandb_max_videos"),
        help="maximum grouped episode videos uploaded to W&B per evaluation",
    )

    args = parser.parse_args(args_list)
    # Persist the fixed indicator architecture in args.json so an old four-class,
    # signal-conditioned checkpoint cannot be mistaken for an exact resume.
    args.policy_uses_turn_indicator_history = False
    args.turn_indicator_output_dim = TURN_INDICATOR_OUTPUT_DIM
    if args.train_subsample_step < 1:
        raise ValueError("--train_subsample_step must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.save_utd < 1:
        raise ValueError("--save_utd must be >= 1")
    if args.closed_loop_grouped_wandb_max_videos < 0:
        raise ValueError("--closed_loop_grouped_wandb_max_videos must be >= 0")
    finite_fields = (
        "augment_prob",
        "ego_past_noise_std",
        "learning_rate",
        "weight_decay",
        "encoder_drop_path_rate",
        "decoder_drop_path_rate",
        "ego_history_dropout_rate",
        "turn_indicator_generated_loss_weight",
        "turn_indicator_expert_loss_weight",
        "turn_indicator_head_dropout",
        "turn_indicator_label_smoothing",
        "coeff_road_border_loss",
        "road_border_margin",
        "coeff_neighbor_collision_loss",
        "neighbor_collision_margin_vehicle",
        "neighbor_collision_margin_pedestrian",
        "neighbor_collision_margin_bicycle",
        "multisample_eval_noise_scale",
        "planning_hybrid_loss",
        "closed_loop_near_miss_thresh",
        "closed_loop_search_radius",
        "closed_loop_unstick_advance_m",
    )
    non_finite = [name for name in finite_fields if not math.isfinite(getattr(args, name))]
    if non_finite:
        raise ValueError(f"Training fields must be finite: {non_finite}")
    if args.train_epochs < 1:
        raise ValueError("--train_epochs must be >= 1")
    if not 0 <= args.warm_up_epoch <= args.train_epochs:
        raise ValueError("--warm_up_epoch must be between 0 and --train_epochs")
    if not 0.0 <= args.augment_prob <= 1.0:
        raise ValueError("--augment_prob must be in [0, 1]")
    if args.ego_past_noise_std < 0.0:
        raise ValueError("--ego_past_noise_std must be >= 0")
    if not 0.0 <= args.encoder_drop_path_rate < 1.0:
        raise ValueError("--encoder_drop_path_rate must be in [0, 1)")
    if not 0.0 <= args.decoder_drop_path_rate < 1.0:
        raise ValueError("--decoder_drop_path_rate must be in [0, 1)")
    if not 0.0 <= args.ego_history_dropout_rate < 1.0:
        raise ValueError("--ego_history_dropout_rate must be in [0, 1)")
    if not 1 <= args.ego_history_frames <= args.time_len:
        raise ValueError("--ego_history_frames must be in [1, time_len]")
    if not 1 <= args.ego_prediction_horizon <= args.future_len:
        raise ValueError("--ego_prediction_horizon must be in [1, future_len]")
    if not 1 <= args.hybrid_loss_window <= args.future_len:
        raise ValueError("--hybrid_loss_window must be in [1, future_len]")
    if args.planning_hybrid_loss < 0.0:
        raise ValueError("--planning_hybrid_loss must be >= 0")
    if args.coeff_road_border_loss < 0.0 or args.coeff_neighbor_collision_loss < 0.0:
        raise ValueError("loss coefficients must be >= 0")
    if args.road_border_margin < 0.0 or args.road_border_n_interp < 0:
        raise ValueError("road-border margin/interpolation must be non-negative")
    if (
        min(
            args.neighbor_collision_margin_vehicle,
            args.neighbor_collision_margin_pedestrian,
            args.neighbor_collision_margin_bicycle,
        )
        < 0.0
    ):
        raise ValueError("neighbor collision margins must be >= 0")
    if args.diffusion_sample_steps < 2:
        raise ValueError("--diffusion_sample_steps must be >= 2 for the second-order DPM solver")
    if args.multisample_eval_num_samples > 0 and args.multisample_eval_sample_steps < 2:
        raise ValueError(
            "--multisample_eval_sample_steps must be >= 2 for the second-order DPM solver"
        )
    if args.multisample_eval_num_samples < 0:
        raise ValueError("--multisample_eval_num_samples must be >= 0")
    if args.multisample_eval_noise_scale < 0.0:
        raise ValueError("--multisample_eval_noise_scale must be >= 0")
    if args.turn_indicator_head_num_queries < 1:
        raise ValueError("--turn_indicator_head_num_queries must be >= 1")
    if args.turn_indicator_head_num_layers < 1:
        raise ValueError("--turn_indicator_head_num_layers must be >= 1")
    if not 0.0 <= args.turn_indicator_head_dropout < 1.0:
        raise ValueError("--turn_indicator_head_dropout must be in [0, 1)")
    if not 0.0 <= args.turn_indicator_label_smoothing < 1.0:
        raise ValueError("--turn_indicator_label_smoothing must be in [0, 1)")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be > 0")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay must be >= 0")
    if args.wandb_step_log_interval < 0:
        raise ValueError("--wandb_step_log_interval must be >= 0")
    if args.turn_indicator_generated_loss_weight < 0.0:
        raise ValueError("--turn_indicator_generated_loss_weight must be >= 0")
    if args.turn_indicator_expert_loss_weight < 0.0:
        raise ValueError("--turn_indicator_expert_loss_weight must be >= 0")
    if args.turn_indicator_generated_loss_weight + args.turn_indicator_expert_loss_weight <= 0.0:
        raise ValueError("at least one turn-indicator loss weight must be positive")
    if args.extra_train_set_repeat < 0:
        raise ValueError("--extra_train_set_repeat must be >= 0")
    if args.extra_train_set_repeat > 0 and not args.extra_train_set_list:
        raise ValueError("--extra_train_set_list is required when repeat is positive")
    if args.closed_loop_search_radius <= 0.0:
        raise ValueError("--closed_loop_search_radius must be > 0")
    if args.closed_loop_near_miss_thresh < 0.0:
        raise ValueError("--closed_loop_near_miss_thresh must be >= 0")
    if args.closed_loop_unstick_advance_m < 0.0:
        raise ValueError("--closed_loop_unstick_advance_m must be >= 0")
    if args.predicted_neighbor_num != 0:
        raise ValueError("HDP is ego-only; --predicted_neighbor_num must be 0")
    if not args.use_velocity_representation:
        raise ValueError("HDP requires --use_velocity_representation true")
    if args.diffusion_model_type != "x_start" or args.diffusion_supervision_type != "x_start":
        raise ValueError("HDP requires x_start prediction and x_start supervision")
    return args


def main():
    args = get_args()
    args_dict = vars(args)
    train_config = TrainConfig(**args_dict)

    train_config.state_normalizer = StateNormalizer.from_json(train_config)
    train_config.observation_normalizer = ObservationNormalizer.from_json(train_config)

    model_training(train_config)


if __name__ == "__main__":
    main()
