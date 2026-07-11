"""HDP reinforcement-learning fine-tuning entrypoint, placed alongside ``train_predictor.py``.

This mirrors the supervised trainer (same DDP setup, optimizer/scheduler, EMA, checkpointing
and wandb logging) but swaps the per-epoch training step for the HDP reward-weighted
RL-Hybrid objective.
"""

import argparse
import json
import math
import os

import pandas as pd
import torch
import wandb
from diffusion_planner.dimensions import *
from diffusion_planner.hdp_rl_epoch import train_hdp_rl_epoch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train import (
    assert_checkpoint_compatible,
    closed_loop_validate,
    load_weights_only,
)
from diffusion_planner.train_config import TrainConfig, parse_float_list
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.dataset import DiffusionPlannerData, DistributedEvalSampler
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import export_checkpoint_onnx_guarded
from diffusion_planner.utils.train_utils import atomic_torch_save, resume_model, set_seed
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from valid_predictor import aggregate_valid_metrics, validate_model


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _train_config_default(name):
    return TrainConfig.__dataclass_fields__[name].default


def configure_rl_trainable_parameters(model: torch.nn.Module, scope: str) -> None:
    if scope not in {"decoder", "all"}:
        raise ValueError(f"Unsupported RL train scope: {scope!r}")
    for name, param in model.named_parameters():
        if scope == "decoder":
            trainable = name.startswith("decoder.dit.")
        else:
            trainable = not name.startswith("decoder.turn_indicator_predictor.")
        param.requires_grad_(trainable)


def best_valid_score_from_rows(rows: list[dict]) -> float:
    best = -float("inf")
    for row in rows:
        full_eval = row.get("valid_full_eval", False)
        if isinstance(full_eval, str):
            full_eval = full_eval.strip().lower() in {"1", "true", "yes"}
        if not pd.notna(full_eval) or not bool(full_eval):
            continue
        raw_epdms = row.get("valid_epdms_total", 0.0)
        raw_ego_loss = row.get("valid_loss_ego", float("inf"))
        epdms = float(raw_epdms) if pd.notna(raw_epdms) else 0.0
        ego_loss = float(raw_ego_loss) if pd.notna(raw_ego_loss) else float("inf")
        score = epdms if epdms > 0.0 else -ego_loss
        if math.isfinite(score):
            best = max(best, score)
    return best


def get_args():
    parser = argparse.ArgumentParser(description="HDP RL training")
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
    parser.add_argument(
        "--extra_train_set_mask_traffic_lights",
        type=boolean,
        default=False,
    )
    parser.add_argument("--valid_set_list", type=str, required=True)
    parser.add_argument(
        "--train_subsample_step",
        type=int,
        default=1,
        help="keep every Nth training sample (data_list[::N]); 1 = use all, "
        "10 = use 1/10 for faster iteration",
    )
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

    # DataLoader
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
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

    # Data augmentation (StatePerturbation, shared with the supervised trainer).
    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--augment_prob", type=float, default=0.5, help="augmentation probability")
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
        help="whether to apply smoothing to future trajectory during augmentation",
    )

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64, help="number of scenes per step")
    parser.add_argument("--save_utd", type=int, default=5)
    parser.add_argument(
        "--rl_full_eval_utd",
        type=int,
        default=_train_config_default("rl_full_eval_utd"),
        help="run stochastic and EPDMS validation every N epochs; deterministic proxy runs each epoch",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warm_up_epoch", type=int, default=2)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_ego_history", type=boolean, default=True)
    parser.add_argument(
        "--ego_history_dropout_rate",
        type=float,
        default=_train_config_default("ego_history_dropout_rate"),
    )
    parser.add_argument("--use_turn_indicators", type=boolean, default=True)

    # ----- HDP-RL-specific -----
    parser.add_argument(
        "--num_generations",
        type=int,
        default=32,
        help="N: trajectories sampled per scene (the paper reports 32)",
    )
    parser.add_argument(
        "--rl_noise_scale",
        type=float,
        default=_train_config_default("rl_noise_scale"),
        help="rollout sampling temperature used by the HDP RL path",
    )
    parser.add_argument(
        "--rl_rollout_steps",
        type=int,
        default=_train_config_default("rl_rollout_steps"),
        help="DPM-Solver steps used only for RL rollout generation",
    )
    parser.add_argument(
        "--rl_init_use_ema",
        type=boolean,
        default=_train_config_default("rl_init_use_ema"),
        help="initialize a fresh RL run from the SFT checkpoint EMA weights when available",
    )
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument(
        "--rl_reward_w_risk",
        type=float,
        default=_train_config_default("rl_reward_w_risk"),
        help="paper HDP multi-reward weight for risk/safety",
    )
    parser.add_argument(
        "--rl_reward_w_follow",
        type=float,
        default=_train_config_default("rl_reward_w_follow"),
        help="paper HDP multi-reward weight for leader-conditioned following",
    )
    parser.add_argument(
        "--rl_reward_w_lane",
        type=float,
        default=_train_config_default("rl_reward_w_lane"),
        help="paper HDP multi-reward weight for lane keeping",
    )
    parser.add_argument(
        "--rl_reward_normalize",
        "--official_reward_normalize",
        dest="rl_reward_normalize",
        type=str,
        choices=["group", "batch", "none"],
        default=_train_config_default("rl_reward_normalize"),
        help="reward normalization before HDP exponential weighting",
    )
    parser.add_argument(
        "--rl_reward_beta",
        "--official_reward_beta",
        dest="rl_reward_beta",
        type=float,
        default=_train_config_default("rl_reward_beta"),
        help="temperature beta in exp(beta * normalized_reward)",
    )
    parser.add_argument(
        "--rl_ema_update_rate",
        type=float,
        default=_train_config_default("rl_ema_update_rate"),
        help="EMA previous-policy update rate; 0.05 corresponds to timm decay 0.95",
    )
    for name, help_text in (
        ("rl_reward_dt", "reward trajectory timestep in seconds"),
        ("rl_ttc_critical_s", "TTC score-zero threshold"),
        ("rl_ttc_safe_s", "TTC score-one threshold"),
        ("rl_thw_critical_s", "THW score-zero threshold"),
        ("rl_thw_safe_s", "THW score-one threshold"),
        ("rl_occupancy_critical_m", "occupancy clearance score-zero base threshold"),
        ("rl_occupancy_safe_m", "occupancy clearance score-one base threshold"),
        ("rl_occupancy_speed_gain_s", "speed gain applied to occupancy thresholds"),
        ("rl_lane_half_width_m", "lane-center reward zero-distance threshold"),
        ("rl_leader_lateral_margin_m", "extra lateral margin for leader association"),
    ):
        parser.add_argument(
            f"--{name}",
            type=float,
            default=_train_config_default(name),
            help=help_text,
        )

    # Loss coefficients (shared with the supervised trainer / loss machinery)
    parser.add_argument("--coeff_position_lat_loss", type=float, default=1.0)
    parser.add_argument("--coeff_position_lon_loss", type=float, default=1.0)
    parser.add_argument("--coeff_heading_l2_loss", type=float, default=1.0)
    parser.add_argument(
        "--coeff_velocity",
        type=float,
        default=_train_config_default("coeff_velocity"),
        help="per-(m/s) weight for high-speed lon-loss attenuation; 0.05 = legacy behavior",
    )
    parser.add_argument("--coeff_timestep", type=parse_float_list, default=[1.0, 1.0, 1.0, 1.0])

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

    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--alpha_neighbor_loss", type=float, default=0.1)

    parser.add_argument(
        "--use_velocity_representation",
        type=boolean,
        default=_train_config_default("use_velocity_representation"),
    )
    parser.add_argument(
        "--planning_hybrid_loss",
        type=float,
        default=_train_config_default("planning_hybrid_loss"),
    )
    parser.add_argument(
        "--hybrid_loss_window",
        type=int,
        default=_train_config_default("hybrid_loss_window"),
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

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_ema", default=True, type=boolean)

    # Model
    parser.add_argument("--encoder_mixer_depth", type=int, default=6)
    parser.add_argument("--encoder_fusion_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=_train_config_default("decoder_depth"))
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.set_defaults(decoder_tokenization="temporal")
    parser.add_argument(
        "--diffusion_model_type",
        type=str,
        choices=["x_start"],
        default="x_start",
    )
    parser.add_argument(
        "--predicted_neighbor_num",
        type=int,
        default=0,
        help="must be 0; HDP-RL predicts only the ego trajectory",
    )
    parser.add_argument(
        "--resume_model_path",
        type=str,
        default=None,
        help="checkpoint for strict training resume, including optimizer/scheduler state",
    )
    parser.add_argument(
        "--init_weights_path",
        type=str,
        default=None,
        help="weights-only checkpoint for starting a fresh RL run from an SFT model",
    )
    parser.add_argument(
        "--rl_train_scope",
        type=str,
        choices=["decoder", "all"],
        default="decoder",
        help="parameters optimized during RL; the released implementation fine-tunes the decoder",
    )

    parser.add_argument("--use_wandb", default=True, type=boolean)
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="Diffusion-Planner-Temporal",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_step_log_interval",
        type=int,
        default=50,
        help="log rank-0 RL batch metrics every N optimizer steps; 0 disables step logging",
    )
    parser.add_argument("--notes", default="", type=str)

    parser.add_argument("--ddp", default=True, type=boolean)
    parser.add_argument("--tf32", default=_train_config_default("tf32"), type=boolean)
    parser.add_argument(
        "--fused_optimizer",
        default=_train_config_default("fused_optimizer"),
        type=boolean,
        help="use fused AdamW when CUDA supports it",
    )
    parser.add_argument(
        "--ddp_static_graph",
        default=_train_config_default("ddp_static_graph"),
        type=boolean,
        help="enable DDP static_graph for lower reducer overhead",
    )
    parser.add_argument("--port", default="22323", type=str)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        choices=["off", "bf16"],
        default=_train_config_default("amp_dtype"),
        help="bf16 autocasts ONLY the model forward; noising/SDE/losses stay fp32",
    )
    parser.add_argument(
        "--export_onnx_on_save",
        type=boolean,
        default=False,
        help="export ONNX synchronously at checkpoint cadence; disabled by default to avoid DDP idle time",
    )

    # Closed-loop validation (rendered rollout + wandb video), run on the checkpoint-save cadence
    # (save_utd). Disabled unless --closed_loop_npz_root is given (dir tree of one route's NPZ).
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
    parser.add_argument("--closed_loop_unstick_advance_m", type=float, default=2.5)

    args = parser.parse_args()
    if args.resume_model_path is not None and args.init_weights_path is not None:
        raise ValueError("--resume_model_path and --init_weights_path are mutually exclusive")
    if args.train_subsample_step < 1:
        raise ValueError("--train_subsample_step must be >= 1")
    if args.extra_train_set_repeat < 0:
        raise ValueError("--extra_train_set_repeat must be >= 0")
    if args.extra_train_set_repeat > 0 and not args.extra_train_set_list:
        raise ValueError("--extra_train_set_list is required when repeat is positive")
    if args.save_utd < 1:
        raise ValueError("--save_utd must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be > 0")
    if args.wandb_step_log_interval < 0:
        raise ValueError("--wandb_step_log_interval must be >= 0")
    if args.num_generations < 2:
        raise ValueError("--num_generations must be >= 2 for HDP-RL group reward normalization")
    if args.rl_rollout_steps < 3:
        raise ValueError("--rl_rollout_steps must be >= 3 for the third-order DPM solver")
    if args.diffusion_sample_steps < 2:
        raise ValueError("--diffusion_sample_steps must be >= 2 for the second-order DPM solver")
    if args.multisample_eval_num_samples > 0 and args.multisample_eval_sample_steps < 3:
        raise ValueError(
            "--multisample_eval_sample_steps must be >= 3 for the third-order DPM solver"
        )
    if args.rl_noise_scale < 0.0:
        raise ValueError("--rl_noise_scale must be >= 0")
    if args.rl_reward_beta <= 0.0:
        raise ValueError("--rl_reward_beta must be > 0")
    if args.predicted_neighbor_num != 0:
        raise ValueError("HDP-RL is ego-only; --predicted_neighbor_num must be 0")
    if args.rl_full_eval_utd < 1:
        raise ValueError("--rl_full_eval_utd must be >= 1")
    if not 0.0 < args.rl_ema_update_rate <= 1.0:
        raise ValueError("--rl_ema_update_rate must be in (0, 1]")
    if args.rl_ttc_safe_s <= args.rl_ttc_critical_s:
        raise ValueError("--rl_ttc_safe_s must be greater than --rl_ttc_critical_s")
    if args.rl_thw_safe_s <= args.rl_thw_critical_s:
        raise ValueError("--rl_thw_safe_s must be greater than --rl_thw_critical_s")
    if args.rl_occupancy_safe_m <= args.rl_occupancy_critical_m:
        raise ValueError("--rl_occupancy_safe_m must be greater than --rl_occupancy_critical_m")
    if not args.use_velocity_representation:
        raise ValueError("HDP-RL requires --use_velocity_representation true")
    if args.diffusion_model_type != "x_start" or args.diffusion_supervision_type != "x_start":
        raise ValueError("HDP velocity RL requires x_start prediction and x_start supervision")

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args


def mean_ego_loss(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("ego_"):
            result[f"valid_loss/{key}"] = val.mean().item()
    return result


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def model_training(args):
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")
    world_size = ddp.get_world_size()
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must be divisible by DDP world size ({world_size})"
        )

    # Validate the checkpoint sidecar before an in-place resume overwrites args.json.
    if args.resume_model_path is not None:
        assert_checkpoint_compatible(args.resume_model_path, args)
    if args.init_weights_path is not None:
        assert_checkpoint_compatible(
            args.init_weights_path,
            args,
            allow_predicted_neighbor_change=True,
            strict_training_config=False,
        )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if global_rank == 0:
        print("------------- {} -------------".format(args.exp_name))
        print("Scenes per step (batch_size): {}".format(args.batch_size))
        print("Group size (num_generations): {}".format(args.num_generations))
        print("RL objective: HDP reward-weighted hybrid loss")
        print("RL reward: HDP risk/follow/lane with Tier IV occupancy proxies")
        print("RL EMA update rate: {}".format(args.rl_ema_update_rate))
        print("RL train scope: {}".format(args.rl_train_scope))
        print("RL init uses SFT EMA: {}".format(args.rl_init_use_ema))
        print("Rollout sampling temperature: {}".format(args.rl_noise_scale))
        print("RL rollout DPM steps: {}".format(args.rl_rollout_steps))
        print("Learning rate: {}".format(args.learning_rate))
        print("TF32: {}".format(args.tf32))
        print("Fused optimizer: {}".format(args.fused_optimizer))
        print("DDP static graph: {}".format(args.ddp_static_graph))
        if not args.use_ema:
            print(
                "WARNING: --use_ema false removes the stable previous-policy rollout used by "
                "the HDP RL update. This mode is intended only as an ablation."
            )
        if args.resume_model_path is None and args.init_weights_path is None:
            print("WARNING: RL is starting without an imitation-pretrained checkpoint")

        save_path = args.save_dir
        os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 6

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)
    else:
        save_path = None

    set_seed(args.seed + global_rank)

    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    # StatePerturbation data augmentation (same as the supervised trainer).
    if args.use_data_augment:
        if args.augment_type == "bridge":
            aug = BridgeStatePerturbation(augment_prob=args.augment_prob, device=args.device)
        else:
            aug = StatePerturbation(
                augment_prob=args.augment_prob,
                num_refine=args.num_refine,
                device=args.device,
                ego_past_noise_std=args.ego_past_noise_std,
                use_smoothing_future_trajectory=args.use_smoothing_future_trajectory,
            )
        if global_rank == 0:
            print(f"Data augmentation enabled: type={args.augment_type} prob={args.augment_prob}")
    else:
        aug = None

    train_set = DiffusionPlannerData(
        args.train_set_list,
        align_legacy_neighbor_futures=args.align_legacy_neighbor_futures,
        extra_data_list=args.extra_train_set_list,
        extra_data_repeat=args.extra_train_set_repeat,
        extra_data_mask_traffic_lights=args.extra_train_set_mask_traffic_lights,
    )
    valid_set = DiffusionPlannerData(
        args.valid_set_list,
        align_legacy_neighbor_futures=args.align_legacy_neighbor_futures,
    )

    train_set.subsample(args.train_subsample_step)
    if len(train_set) == 0:
        raise ValueError("Training data list is empty after subsampling")
    if len(valid_set) == 0:
        raise ValueError("Validation data list is empty")
    if len(valid_set) < world_size:
        raise ValueError("Validation set must contain at least one sample per DDP rank")

    train_sampler = DistributedSampler(
        train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size // world_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training loader has zero batches; dataset after subsampling must contain at least "
            f"one global batch ({batch_size} scenes) when drop_last=True"
        )

    # Validation is sharded without duplicate padding and the per-rank
    # metrics are all-reduced via aggregate_valid_metrics.
    valid_sampler = DistributedEvalSampler(
        valid_set, num_replicas=ddp.get_world_size(), rank=global_rank
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=max(128 // ddp.get_world_size(), 1),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    # The policy objective does not consume the auxiliary turn classifier. It must stay frozen
    # in both scopes so DDP's find_unused_parameters=False contract remains valid.
    configure_rl_trainable_parameters(diffusion_planner, args.rl_train_scope)

    if args.ddp:
        diffusion_planner = DDP(
            diffusion_planner,
            device_ids=[rank],
            find_unused_parameters=False,
            static_graph=args.ddp_static_graph,
        )

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())
            )
        )

    trainable_params = [
        p for p in ddp.get_model(diffusion_planner, args.ddp).parameters() if p.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for RL training")
    params = [{"params": trainable_params, "lr": args.learning_rate}]
    if args.fused_optimizer and args.device == "cuda":
        try:
            optimizer = optim.AdamW(params, fused=True)
        except TypeError:
            optimizer = optim.AdamW(params)
            args.fused_optimizer = False
    else:
        optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        model_ema = (
            ModelEma(
                diffusion_planner,
                decay=1.0 - args.rl_ema_update_rate,
                device=args.device,
            )
            if args.use_ema
            else None
        )
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
            strict_training_state=True,
        )
        print(
            f"Strict resume at epoch {init_epoch} with optimizer LR "
            f"{optimizer.param_groups[0]['lr']}"
        )
    elif args.init_weights_path is not None:
        print(f"Initializing RL weights from {args.init_weights_path}")
        load_weights_only(
            args.init_weights_path,
            diffusion_planner,
            args.device,
            prefer_ema=args.rl_init_use_ema,
        )
        model_ema = (
            ModelEma(
                diffusion_planner,
                decay=1.0 - args.rl_ema_update_rate,
                device=args.device,
            )
            if args.use_ema
            else None
        )
        init_epoch = 0
        wandb_id = None
    else:
        model_ema = (
            ModelEma(
                diffusion_planner,
                decay=1.0 - args.rl_ema_update_rate,
                device=args.device,
            )
            if args.use_ema
            else None
        )
        init_epoch = 0
        wandb_id = None

    if global_rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=wandb_id,
            dir=f"{save_path}",
        )
        # Strict checkpoint compatibility runs before W&B initialization. An in-place
        # recovery legitimately changes resume_model_path from None to latest.pth.
        wandb.config.update(
            args_dict,
            allow_val_change=args.resume_model_path is not None,
        )
        wandb_id = wandb.run.id

    if args.ddp:
        torch.distributed.barrier()

    args._wandb_global_step = int(
        getattr(diffusion_planner, "_resume_global_step", init_epoch * len(train_loader))
    )
    train_log_path = os.path.join(save_path, "train_log.tsv") if global_rank == 0 else None
    data_list = []
    best_valid_score = -float("inf")
    if global_rank == 0 and args.resume_model_path is not None and os.path.exists(train_log_path):
        previous_log = pd.read_csv(train_log_path, sep="\t")
        data_list = previous_log.to_dict("records")
        best_valid_score = best_valid_score_from_rows(data_list)
    configured_multisample_count = args.multisample_eval_num_samples
    configured_epdms = args.enable_epdms_eval

    for epoch in range(init_epoch, train_epochs):
        train_sampler.set_epoch(epoch)
        if args.ddp:
            torch.distributed.barrier()

        train_loss, train_total_loss = train_hdp_rl_epoch(
            train_loader,
            diffusion_planner,
            optimizer,
            trainable_params,
            args,
            model_ema,
            aug,
        )

        eval_model = model_ema.ema if model_ema is not None else diffusion_planner
        run_full_eval = (epoch + 1) % args.rl_full_eval_utd == 0 or epoch + 1 == train_epochs
        args.multisample_eval_num_samples = configured_multisample_count if run_full_eval else 0
        args.enable_epdms_eval = configured_epdms and run_full_eval
        try:
            valid_dict = validate_model(eval_model, valid_loader, args)
        finally:
            args.multisample_eval_num_samples = configured_multisample_count
            args.enable_epdms_eval = configured_epdms
        agg = aggregate_valid_metrics(valid_dict, args.device)
        # Save the scheduler/optimizer state for the *next* epoch. Previously checkpoints were
        # written first and scheduler.step() ran afterwards, so strict resume repeated one LR.
        train_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if global_rank == 0:
            valid_loss_ego = scalar(agg["avg_loss_ego"])
            valid_neighbor_margin = scalar(agg["ego_means"]["ego_neighbor_margin_loss"])
            valid_road_border = scalar(agg["ego_means"]["ego_road_border_loss"])
            valid_epdms_total = scalar(agg["epdms_means"].get("total", 0.0))
            valid_multisample = {
                key: scalar(value) for key, value in agg["multisample_means"].items()
            }
            valid_turn_metrics = {
                "accuracy": scalar(agg["turn_indicator_accuracy"]),
                "change_accuracy": scalar(agg["turn_indicator_change_accuracy"]),
                "change_total": int(agg["turn_indicator_change_total"]),
                **{
                    f"{name}_accuracy": scalar(value)
                    for name, value in agg["turn_indicator_class_accuracy"].items()
                },
                **{
                    f"{name}_count": int(value)
                    for name, value in agg["turn_indicator_class_count"].items()
                },
            }
            has_reward = "reward_mean" in train_loss
            train_reward = scalar(train_loss["reward_mean"]) if has_reward else float("nan")
            print(
                f"Epoch {epoch + 1}/{train_epochs}\n"
                f"{train_reward=:.4f}\n"
                f"{valid_loss_ego=:.4f}\n"
                f"{valid_neighbor_margin=:.4f}\n"
                f"{valid_road_border=:.4f}\n"
                f"{valid_epdms_total=:.4f}\n"
                f"valid_multisample_minADE="
                f"{valid_multisample.get('minADE', float('nan')):.4f}\n"
                f"valid_multisample_minFDE="
                f"{valid_multisample.get('minFDE', float('nan')):.4f}"
            )

            if args.use_wandb:
                wandb.log(
                    {
                        **{f"train/{k}": v for k, v in train_loss.items()},
                        "lr": train_lr,
                        "valid/ego": valid_loss_ego,
                        "valid/neighbor_margin": valid_neighbor_margin,
                        "valid/road_border": valid_road_border,
                        "valid/epdms_total": valid_epdms_total,
                        "valid/full_eval": float(run_full_eval),
                        **{
                            f"valid_turn_indicator/{key}": value
                            for key, value in valid_turn_metrics.items()
                        },
                        **{
                            f"valid_multisample/{key}": value
                            for key, value in valid_multisample.items()
                        },
                    }
                )

            curr_data = {
                "epoch": epoch + 1,
                "train_reward_mean": train_reward if has_reward else None,
                "train_loss": scalar(train_total_loss),
                "valid_loss_ego": valid_loss_ego,
                "valid_neighbor_margin": valid_neighbor_margin,
                "valid_road_border": valid_road_border,
                "valid_epdms_total": valid_epdms_total,
                "valid_full_eval": run_full_eval,
                **{
                    f"valid_turn_indicator_{key}": value
                    for key, value in valid_turn_metrics.items()
                },
                **{f"valid_multisample_{key}": value for key, value in valid_multisample.items()},
            }
            data_list.append(curr_data)
            pd.DataFrame(data_list).to_csv(
                os.path.join(save_path, "train_log.tsv"), index=False, sep="\t"
            )

            model_dict = {
                "epoch": epoch + 1,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict() if model_ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_loss_ego,
                "wandb_id": wandb_id,
                "global_step": args._wandb_global_step,
            }
            atomic_torch_save(model_dict, f"{save_path}/latest.pth")

            if (epoch + 1 - init_epoch) % save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                if args.export_onnx_on_save:
                    export_checkpoint_onnx_guarded(
                        config_json_path=os.path.join(save_path, "args.json"),
                        ckpt_path=f"{curr_dir}/best_model.pth",
                        output_dir=curr_dir,
                        output_prefix="diffusion_planner",
                        use_ema=model_ema is not None,
                        use_simplify=False,
                        opset_version=20,
                        external_data=False,
                    )
                # Closed-loop validation on the checkpoint-save cadence; videos + metrics land next
                # to the saved weights and are logged to wandb at step=epoch+1.
                closed_loop_validate(eval_model, args, epoch, os.path.join(curr_dir, "closed_loop"))

            selection_score = valid_epdms_total if valid_epdms_total > 0.0 else -valid_loss_ego
            if run_full_eval and selection_score > best_valid_score:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                best_valid_score = selection_score
                curr_data["best_valid_score"] = best_valid_score
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                if args.export_onnx_on_save:
                    export_checkpoint_onnx_guarded(
                        config_json_path=os.path.join(save_path, "args.json"),
                        ckpt_path=f"{curr_dir}/best_model.pth",
                        output_dir=curr_dir,
                        output_prefix="diffusion_planner",
                        use_ema=model_ema is not None,
                        use_simplify=False,
                        opset_version=20,
                        external_data=False,
                    )

    if global_rank == 0 and wandb.run is not None:
        wandb.finish()
    ddp.cleanup()


if __name__ == "__main__":
    args = get_args()

    assert len(args.coeff_timestep) == 4

    model_training(args)
