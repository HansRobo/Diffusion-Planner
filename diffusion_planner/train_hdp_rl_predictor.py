"""HDP reinforcement-learning fine-tuning entrypoint, placed alongside ``train_predictor.py``.

This mirrors the supervised trainer (same DDP setup, optimizer/scheduler, EMA, checkpointing
and wandb logging) but swaps the per-epoch training step for the HDP reward-weighted
RL-Hybrid objective.
"""

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import wandb
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
from diffusion_planner.hdp_rl_epoch import (
    RL_COMPILED_MAX_CANDIDATES_PER_FORWARD,
    _rl_update_scene_chunk_size,
    train_hdp_rl_epoch,
    validate_hdp_reward_policy,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train import (
    assert_checkpoint_compatible,
    closed_loop_validate,
    load_weights_only,
)
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils import ddp
from diffusion_planner.utils.cli import boolean, dataclass_default
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.dataset import (
    BatchAlignedDistributedSampler,
    DiffusionPlannerData,
    DistributedEvalSampler,
)
from diffusion_planner.utils.lr_schedule import LinearWarmupConstantLR
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import export_checkpoint_onnx_guarded
from diffusion_planner.utils.train_utils import (
    ModelEma,
    atomic_torch_save,
    build_adamw_optimizer,
    compile_model_components,
    gather_rng_states,
    resume_model,
    set_seed,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from valid_predictor import aggregate_valid_metrics, validate_model

_train_config_default = partial(dataclass_default, TrainConfig)


def configure_rl_trainable_parameters(model: torch.nn.Module, scope: str) -> None:
    if scope not in {"decoder", "all"}:
        raise ValueError(f"Unsupported RL train scope: {scope!r}")
    for name, param in model.named_parameters():
        if scope == "decoder":
            trainable = name.startswith(("decoder.dit.", "decoder.global_route_encoder."))
        else:
            trainable = not name.startswith("decoder.turn_indicator_predictor.")
        param.requires_grad_(trainable)


def initialize_fresh_rl_policy_and_ema(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: str,
    *,
    use_ddp: bool,
    prefer_checkpoint_ema: bool,
    ema_update_rate: float,
) -> ModelEma:
    """Load the SFT policy before copying it into the previous-policy EMA shadow."""
    load_weights_only(
        checkpoint_path,
        model,
        device,
        prefer_ema=prefer_checkpoint_ema,
    )
    return ModelEma(
        ddp.get_model(model, use_ddp),
        decay=1.0 - ema_update_rate,
        device=device,
    )


def validate_compiled_candidate_batch(
    local_batch_size: int,
    num_generations: int,
    compile_model: bool,
    update_max_candidates_per_rank: int,
) -> int:
    candidates_per_rank = local_batch_size * num_generations
    update_scene_chunk = _rl_update_scene_chunk_size(
        local_batch_size,
        num_generations,
        update_max_candidates_per_rank,
    )
    update_candidates = update_scene_chunk * num_generations
    if compile_model and update_candidates > RL_COMPILED_MAX_CANDIDATES_PER_FORWARD:
        raise ValueError(
            f"Unsafe compiled RL update chunk {update_candidates} exceeds the verified H100 "
            f"limit {RL_COMPILED_MAX_CANDIDATES_PER_FORWARD}. This shape produced silently "
            "corrupted backward gradients with TorchInductor; lower "
            "--rl_update_max_candidates_per_rank or disable --compile_model."
        )
    return candidates_per_rank


def best_valid_score_from_rows(rows: list[dict]) -> float:
    # New logs persist the accepted cumulative best. This must take precedence over raw epoch
    # rewards because a high-reward epoch may have been rejected by a source-policy guard.
    for row in reversed(rows):
        raw_best = row.get("best_valid_score", float("nan"))
        accepted_best = float(raw_best) if pd.notna(raw_best) else float("nan")
        if math.isfinite(accepted_best):
            return accepted_best

    # Legacy logs predate the cumulative field and accepted every finite full-eval reward.
    best = -float("inf")
    for row in rows:
        full_eval = row.get("valid_full_eval", False)
        if isinstance(full_eval, str):
            full_eval = full_eval.strip().lower() in {"1", "true", "yes"}
        if not pd.notna(full_eval) or not bool(full_eval):
            continue
        raw_reward = row.get("valid_reward_mean", float("nan"))
        reward = float(raw_reward) if pd.notna(raw_reward) else float("nan")
        if math.isfinite(reward):
            best = max(best, reward)
    return best


def load_resume_train_rows(path: Path, init_epoch: int) -> list[dict]:
    """Load only rows committed by the checkpoint and reject gaps in epoch history."""
    if not path.is_file():
        if init_epoch == 0:
            return []
        raise FileNotFoundError(f"Strict RL resume is missing train_log.tsv: {path}")
    frame = pd.read_csv(path, sep="\t")
    if "epoch" not in frame:
        raise ValueError(f"RL train log has no epoch column: {path}")
    numeric_epoch = pd.to_numeric(frame["epoch"], errors="coerce")
    if numeric_epoch.isna().any():
        raise ValueError(f"RL train log contains a non-numeric epoch: {path}")
    committed = frame[numeric_epoch <= init_epoch]
    committed_epochs = pd.to_numeric(committed["epoch"], errors="coerce").tolist()
    expected_epochs = list(range(1, init_epoch + 1))
    if committed_epochs != expected_epochs:
        raise ValueError(
            "Strict RL resume needs one contiguous log row per checkpointed epoch: "
            f"expected={expected_epochs}, found={committed_epochs} in {path}"
        )
    return committed.to_dict("records")


def write_train_log_atomic(rows: list[dict], path: str) -> None:
    """Atomically replace the small TSV so interruption cannot leave a partial log."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            pd.DataFrame(rows).to_csv(stream, index=False, sep="\t")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(value: dict, path: str | Path) -> None:
    """Atomically replace checkpoint metadata used by strict resume."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def find_checkpoint_run_artifact(checkpoint_path: str, filename: str) -> Path | None:
    """Find a run-level artifact next to a latest or nested epoch/best checkpoint."""
    checkpoint = Path(checkpoint_path)
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _checkpoint_bool(path: str, key: str, default: bool) -> bool:
    """Read a boolean option from the source run metadata when available."""
    args_path = find_checkpoint_run_artifact(path, "args.json")
    if args_path is None:
        return default
    try:
        with args_path.open(encoding="utf-8") as stream:
            value = json.load(stream).get(key, default)
    except (OSError, TypeError, ValueError):
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


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
    parser.add_argument(
        "--valid_batch_size",
        default=512,
        type=int,
        help="global validation batch size; independent from RL candidate-group batch size",
    )
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
    parser.add_argument(
        "--rl_validate_before_training",
        type=boolean,
        default=_train_config_default("rl_validate_before_training"),
        help="validate and preserve the source SFT policy before the first RL update",
    )
    parser.add_argument(
        "--rl_max_valid_loss_regression",
        type=float,
        default=_train_config_default("rl_max_valid_loss_regression"),
        help="maximum relative ego validation-loss increase allowed for best-model selection",
    )
    parser.add_argument(
        "--rl_max_valid_safety_regression",
        type=float,
        default=_train_config_default("rl_max_valid_safety_regression"),
        help=(
            "maximum absolute decrease in source-policy risk and collision-safety scores "
            "allowed for best-model selection"
        ),
    )
    parser.add_argument(
        "--rl_max_valid_epdms_regression",
        type=float,
        default=_train_config_default("rl_max_valid_epdms_regression"),
        help=("maximum absolute decrease in source-policy EPDMS allowed for best-model selection"),
    )
    parser.add_argument(
        "--rl_best_score_min_delta",
        type=float,
        default=_train_config_default("rl_best_score_min_delta"),
        help="minimum validation-score improvement required to replace the best model",
    )
    parser.add_argument(
        "--rl_early_stop_patience",
        type=int,
        default=_train_config_default("rl_early_stop_patience"),
        help="stop after this many full evaluations without a best-score improvement; 0 disables",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay; disabled by default for conservative SFT-to-RL fine-tuning",
    )
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
        default=_train_config_default("num_generations"),
        help="training trajectories per scene; Tier IV sweeps favor 8 (paper reports 32)",
    )
    parser.add_argument(
        "--rl_noise_scale",
        type=float,
        default=_train_config_default("rl_noise_scale"),
        help="rollout sampling temperature used by the HDP RL path",
    )
    parser.add_argument(
        "--rl_eval_noise_scale",
        type=float,
        default=_train_config_default("rl_eval_noise_scale"),
        help="fixed held-out policy sampling temperature, independent of rollout exploration",
    )
    parser.add_argument(
        "--rl_eval_num_generations",
        type=int,
        default=_train_config_default("rl_eval_num_generations"),
        help="fixed held-out candidate count, independent of the training group size",
    )
    for reward_name, help_name in (
        ("safety", "direct active/rear collision safety"),
        ("risk", "risk/safety"),
        ("follow", "leader-conditioned following"),
        ("lane", "lane keeping"),
        ("progress", "anti-stopping progress"),
        ("road_border", "direct HD-map road-border clearance"),
    ):
        parser.add_argument(
            f"--rl_eval_reward_w_{reward_name}",
            type=float,
            default=_train_config_default(f"rl_eval_reward_w_{reward_name}"),
            help=f"fixed held-out {help_name} weight, independent of optimization weights",
        )
    parser.add_argument(
        "--rl_eval_behavior_gate",
        choices=["none", "safety", "risk"],
        default=_train_config_default("rl_eval_behavior_gate"),
        help="behavior gate used only by the fixed held-out reward objective",
    )
    parser.add_argument(
        "--rl_eval_occupancy_use_road_border",
        type=boolean,
        default=_train_config_default("rl_eval_occupancy_use_road_border"),
        help="use HD-map road borders as OCC fallback in the fixed held-out objective",
    )
    parser.add_argument(
        "--rl_eval_stationary_progress_mode",
        choices=["constant", "distance"],
        default=_train_config_default("rl_eval_stationary_progress_mode"),
        help="stopped-expert progress scoring used only by the fixed held-out objective",
    )
    parser.add_argument(
        "--rl_eval_stationary_reference_threshold_m",
        type=float,
        default=_train_config_default("rl_eval_stationary_reference_threshold_m"),
        help="held-out maximum expert displacement treated as a stopped reference",
    )
    parser.add_argument(
        "--rl_eval_stationary_progress_tolerance_m",
        type=float,
        default=_train_config_default("rl_eval_stationary_progress_tolerance_m"),
        help="held-out endpoint error where stopped-reference progress reaches zero",
    )
    parser.add_argument(
        "--rl_eval_red_light_constraint",
        type=boolean,
        default=_train_config_default("rl_eval_red_light_constraint"),
        help="apply the expert-gated red stop-line constraint in the fixed held-out objective",
    )
    parser.add_argument(
        "--rl_eval_red_light_lane_tolerance_m",
        type=float,
        default=_train_config_default("rl_eval_red_light_lane_tolerance_m"),
        help="held-out maximum distance used to associate a red route lane with a stop line",
    )
    parser.add_argument(
        "--rl_rollout_steps",
        type=int,
        default=_train_config_default("rl_rollout_steps"),
        help="DPM-Solver steps used only for RL rollout generation",
    )
    parser.add_argument(
        "--rl_updates_per_rollout",
        type=int,
        default=_train_config_default("rl_updates_per_rollout"),
        help="optimizer updates that reuse each sampled and scored candidate group",
    )
    parser.add_argument(
        "--rl_update_max_candidates_per_rank",
        type=int,
        default=_train_config_default("rl_update_max_candidates_per_rank"),
        help=(
            "maximum differentiable candidates per rank; complete scene groups are gradient-"
            "accumulated into one exact optimizer update (0 disables the cap)"
        ),
    )
    parser.add_argument(
        "--rl_init_use_ema",
        type=boolean,
        default=_train_config_default("rl_init_use_ema"),
        help="initialize a fresh RL run from the SFT checkpoint EMA weights when available",
    )
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument(
        "--rl_reward_w_safety",
        type=float,
        default=_train_config_default("rl_reward_w_safety"),
        help="optional direct paper safety-reward weight; zero omits the standalone component",
    )
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
        "--rl_reward_w_progress",
        type=float,
        default=_train_config_default("rl_reward_w_progress"),
        help="anti-stopping reward weight for signed progress relative to the logged expert",
    )
    parser.add_argument(
        "--rl_behavior_gate",
        choices=["none", "safety", "risk"],
        default=_train_config_default("rl_behavior_gate"),
        help="gate follow/lane/progress by none (paper), collision safety, or continuous risk",
    )
    parser.add_argument(
        "--rl_occupancy_use_road_border",
        type=boolean,
        default=_train_config_default("rl_occupancy_use_road_border"),
        help="use HD-map road borders as OCC fallback when no static/stopped obstacle exists",
    )
    parser.add_argument(
        "--rl_reward_w_road_border",
        type=float,
        default=_train_config_default("rl_reward_w_road_border"),
        help="direct road-border clearance reward weight; zero disables this real-vehicle extension",
    )
    parser.add_argument(
        "--rl_stationary_progress_mode",
        choices=["constant", "distance"],
        default=_train_config_default("rl_stationary_progress_mode"),
        help=(
            "score a stopped expert as constant (legacy) or by candidate-to-expert endpoint "
            "distance"
        ),
    )
    parser.add_argument(
        "--rl_red_light_constraint",
        type=boolean,
        default=_train_config_default("rl_red_light_constraint"),
        help="reject red stop-line crossings when the logged expert does not cross",
    )
    parser.add_argument(
        "--rl_bc_weight",
        type=float,
        default=_train_config_default("rl_bc_weight"),
        help="hybrid imitation-loss weight on one logged expert target per scene",
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
        help="epoch-boundary previous-policy update rate; 0.05 corresponds to timm decay 0.95",
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
        (
            "rl_stationary_reference_threshold_m",
            "maximum expert endpoint displacement treated as a stopped reference",
        ),
        (
            "rl_stationary_progress_tolerance_m",
            "candidate endpoint error at which stopped-reference progress reaches zero",
        ),
        (
            "rl_red_light_lane_tolerance_m",
            "maximum distance used to associate a red route lane with a stop line",
        ),
        ("rl_road_border_critical_m", "road-border clearance below which the score is zero"),
        ("rl_road_border_safe_m", "road-border clearance at which the score reaches one"),
    ):
        parser.add_argument(
            f"--{name}",
            type=float,
            default=_train_config_default(name),
            help=help_text,
        )

    # Loss coefficients (shared with the supervised trainer / loss machinery)
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
        "--wandb_run_id",
        type=str,
        default=None,
        help="existing W&B run ID to resume, or a stable ID for a fresh RL run",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="hdp-rl",
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
    parser.add_argument(
        "--compile_model",
        type=boolean,
        default=_train_config_default("compile_model"),
        help="compile the encoder and decoder with TorchInductor",
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
    if args.resume_model_path is None and args.init_weights_path is None:
        raise ValueError("HDP-RL requires --init_weights_path or --resume_model_path")
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
    if args.train_epochs < 1:
        raise ValueError("--train_epochs must be >= 1")
    if not 0 <= args.warm_up_epoch <= args.train_epochs:
        raise ValueError("--warm_up_epoch must be between 0 and --train_epochs")
    if not 0.0 <= args.augment_prob <= 1.0:
        raise ValueError("--augment_prob must be in [0, 1]")
    if not 0.0 <= args.encoder_drop_path_rate < 1.0:
        raise ValueError("--encoder_drop_path_rate must be in [0, 1)")
    if not 0.0 <= args.decoder_drop_path_rate < 1.0:
        raise ValueError("--decoder_drop_path_rate must be in [0, 1)")
    if not 1 <= args.ego_prediction_horizon <= args.future_len:
        raise ValueError("--ego_prediction_horizon must be in [1, future_len]")
    if not 1 <= args.hybrid_loss_window <= args.future_len:
        raise ValueError("--hybrid_loss_window must be in [1, future_len]")
    if args.planning_hybrid_loss < 0.0:
        raise ValueError("--planning_hybrid_loss must be >= 0")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be > 0")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay must be >= 0")
    if args.wandb_step_log_interval < 0:
        raise ValueError("--wandb_step_log_interval must be >= 0")
    if args.num_generations < 2:
        raise ValueError("--num_generations must be >= 2 for HDP-RL group reward normalization")
    if args.rl_eval_num_generations < 1:
        raise ValueError("--rl_eval_num_generations must be >= 1")
    if args.rl_rollout_steps < 2:
        raise ValueError("--rl_rollout_steps must be >= 2 for the second-order DPM solver")
    if args.rl_updates_per_rollout < 1:
        raise ValueError("--rl_updates_per_rollout must be >= 1")
    if args.rl_update_max_candidates_per_rank < 0:
        raise ValueError("--rl_update_max_candidates_per_rank must be >= 0")
    if (
        args.rl_update_max_candidates_per_rank > 0
        and args.rl_update_max_candidates_per_rank < args.num_generations
    ):
        raise ValueError(
            "--rl_update_max_candidates_per_rank must fit one complete generation group"
        )
    if args.diffusion_sample_steps < 2:
        raise ValueError("--diffusion_sample_steps must be >= 2 for the second-order DPM solver")
    if args.multisample_eval_num_samples > 0 and args.multisample_eval_sample_steps < 2:
        raise ValueError(
            "--multisample_eval_sample_steps must be >= 2 for the second-order DPM solver"
        )
    if args.rl_noise_scale < 0.0:
        raise ValueError("--rl_noise_scale must be >= 0")
    if args.rl_eval_noise_scale < 0.0:
        raise ValueError("--rl_eval_noise_scale must be >= 0")
    if args.advantage_eps <= 0.0:
        raise ValueError("--advantage_eps must be > 0")
    if args.rl_reward_beta <= 0.0:
        raise ValueError("--rl_reward_beta must be > 0")
    reward_weights = (
        args.rl_reward_w_safety,
        args.rl_reward_w_risk,
        args.rl_reward_w_follow,
        args.rl_reward_w_lane,
        args.rl_reward_w_progress,
        args.rl_reward_w_road_border,
    )
    if any(weight < 0.0 for weight in reward_weights):
        raise ValueError("RL reward weights must be non-negative")
    if sum(reward_weights) <= 0.0:
        raise ValueError("At least one RL reward weight must be positive")
    if args.rl_road_border_critical_m < 0.0:
        raise ValueError("--rl_road_border_critical_m must be >= 0")
    if args.rl_road_border_safe_m <= args.rl_road_border_critical_m:
        raise ValueError("--rl_road_border_safe_m must exceed the critical threshold")
    eval_reward_weights = (
        args.rl_eval_reward_w_safety,
        args.rl_eval_reward_w_risk,
        args.rl_eval_reward_w_follow,
        args.rl_eval_reward_w_lane,
        args.rl_eval_reward_w_progress,
        args.rl_eval_reward_w_road_border,
    )
    if any(weight < 0.0 for weight in eval_reward_weights):
        raise ValueError("RL held-out reward weights must be non-negative")
    if sum(eval_reward_weights) <= 0.0:
        raise ValueError("At least one RL held-out reward weight must be positive")
    if args.rl_bc_weight < 0.0:
        raise ValueError("--rl_bc_weight must be non-negative")
    if args.predicted_neighbor_num != 0:
        raise ValueError("HDP-RL is ego-only; --predicted_neighbor_num must be 0")
    if args.rl_full_eval_utd < 1:
        raise ValueError("--rl_full_eval_utd must be >= 1")
    if args.rl_max_valid_loss_regression < 0.0:
        raise ValueError("--rl_max_valid_loss_regression must be non-negative")
    if args.rl_max_valid_safety_regression < 0.0:
        raise ValueError("--rl_max_valid_safety_regression must be non-negative")
    if args.rl_max_valid_epdms_regression < 0.0:
        raise ValueError("--rl_max_valid_epdms_regression must be non-negative")
    if args.rl_best_score_min_delta < 0.0:
        raise ValueError("--rl_best_score_min_delta must be non-negative")
    if args.rl_early_stop_patience < 0:
        raise ValueError("--rl_early_stop_patience must be non-negative")
    if not args.use_ema:
        raise ValueError("HDP-RL requires --use_ema True for the frozen previous-policy snapshot")
    if not 0.0 < args.rl_ema_update_rate <= 1.0:
        raise ValueError("--rl_ema_update_rate must be in (0, 1]")
    if args.rl_ttc_critical_s < 0.0:
        raise ValueError("--rl_ttc_critical_s must be >= 0")
    if args.rl_ttc_safe_s <= args.rl_ttc_critical_s:
        raise ValueError("--rl_ttc_safe_s must be greater than --rl_ttc_critical_s")
    if args.rl_thw_critical_s < 0.0:
        raise ValueError("--rl_thw_critical_s must be >= 0")
    if args.rl_thw_safe_s <= args.rl_thw_critical_s:
        raise ValueError("--rl_thw_safe_s must be greater than --rl_thw_critical_s")
    if args.rl_occupancy_critical_m < 0.0:
        raise ValueError("--rl_occupancy_critical_m must be >= 0")
    if args.rl_occupancy_safe_m <= args.rl_occupancy_critical_m:
        raise ValueError("--rl_occupancy_safe_m must be greater than --rl_occupancy_critical_m")
    if args.rl_reward_dt <= 0.0:
        raise ValueError("--rl_reward_dt must be > 0")
    if args.rl_occupancy_speed_gain_s < 0.0:
        raise ValueError("--rl_occupancy_speed_gain_s must be >= 0")
    if args.rl_lane_half_width_m <= 0.0:
        raise ValueError("--rl_lane_half_width_m must be > 0")
    if args.rl_leader_lateral_margin_m < 0.0:
        raise ValueError("--rl_leader_lateral_margin_m must be >= 0")
    if args.rl_stationary_reference_threshold_m <= 0.0:
        raise ValueError("--rl_stationary_reference_threshold_m must be > 0")
    if args.rl_stationary_progress_tolerance_m <= 0.0:
        raise ValueError("--rl_stationary_progress_tolerance_m must be > 0")
    if args.rl_eval_stationary_reference_threshold_m <= 0.0:
        raise ValueError("--rl_eval_stationary_reference_threshold_m must be > 0")
    if args.rl_eval_stationary_progress_tolerance_m <= 0.0:
        raise ValueError("--rl_eval_stationary_progress_tolerance_m must be > 0")
    if args.rl_red_light_lane_tolerance_m <= 0.0:
        raise ValueError("--rl_red_light_lane_tolerance_m must be > 0")
    if args.rl_eval_red_light_lane_tolerance_m <= 0.0:
        raise ValueError("--rl_eval_red_light_lane_tolerance_m must be > 0")
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


def finite_scalar_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        number = scalar(value)
        if math.isfinite(number):
            result[key] = number
    return result


def source_policy_selection_guards(
    source_metrics,
    valid_reward_metrics,
    valid_epdms_total,
    *,
    valid_epdms_metrics=None,
    max_safety_regression,
    max_epdms_regression,
    require_epdms,
    require_road_border=False,
    require_dac=False,
):
    """Return source-relative gates used before accepting an RL best checkpoint."""
    higher_is_better = (
        "risk",
        "safety",
        "collision_safety",
        "red_light",
        "ttc",
        "thw",
        "occupancy",
        "comfort",
    )
    lower_is_better = ("collision_active", "collision_rear", "red_light_violation_fraction")
    if require_road_border:
        higher_is_better = higher_is_better + ("road_border",)
    guard_names = higher_is_better + lower_is_better
    if not source_metrics or not bool(source_metrics.get("baseline_available", True)):
        return {"available": False, **dict.fromkeys((*guard_names, "epdms"), True)}

    required_source = tuple(f"valid_reward_{name}" for name in guard_names)
    if require_epdms:
        required_source += ("valid_epdms_total",)
    missing = [key for key in required_source if key not in source_metrics]
    if missing:
        raise ValueError(f"Source-policy metrics are missing required selection guards: {missing}")

    def finite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    guards = {"available": True}
    for name in higher_is_better:
        source = source_metrics[f"valid_reward_{name}"]
        current = valid_reward_metrics.get(name)
        guards[name] = (
            finite(source)
            and finite(current)
            and float(current) >= float(source) - max_safety_regression
        )
    for name in lower_is_better:
        source = source_metrics[f"valid_reward_{name}"]
        current = valid_reward_metrics.get(name)
        guards[name] = (
            finite(source)
            and finite(current)
            and float(current) <= float(source) + max_safety_regression
        )
    guards["epdms"] = True
    if require_epdms:
        source_epdms = source_metrics["valid_epdms_total"]
        guards["epdms"] = (
            finite(source_epdms)
            and finite(valid_epdms_total)
            and float(valid_epdms_total) >= float(source_epdms) - max_epdms_regression
        )
    if require_dac:
        source_dac = source_metrics.get("valid_epdms_dac")
        current_dac = (valid_epdms_metrics or {}).get("dac")
        guards["dac"] = (
            finite(source_dac)
            and finite(current_dac)
            and float(current_dac) >= float(source_dac) - max_epdms_regression
        )
    return guards


def turn_indicator_metrics(agg):
    return {
        "accuracy": scalar(agg["turn_indicator_accuracy"]),
        "change_accuracy": scalar(agg["turn_indicator_change_accuracy"]),
        "change_total": int(agg["turn_indicator_change_total"]),
        **{
            f"{name}_accuracy": scalar(value)
            for name, value in agg["turn_indicator_class_accuracy"].items()
        },
        **{
            f"{name}_count": int(value) for name, value in agg["turn_indicator_class_count"].items()
        },
    }


def model_training(args):
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")
    world_size = ddp.get_world_size()
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must be divisible by DDP world size ({world_size})"
        )
    if args.valid_batch_size < world_size or args.valid_batch_size % world_size != 0:
        raise ValueError(
            f"--valid_batch_size ({args.valid_batch_size}) must be at least and divisible by "
            f"DDP world size ({world_size})"
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
        print("Validation batch size: {}".format(args.valid_batch_size))
        print("Group size (num_generations): {}".format(args.num_generations))
        print("RL objective: HDP reward-weighted hybrid loss")
        print("RL reward: HDP safety/risk/follow/lane with Tier IV occupancy proxies")
        print("RL direct safety reward weight: {}".format(args.rl_reward_w_safety))
        print("RL progress reward weight: {}".format(args.rl_reward_w_progress))
        print("RL behavior reward gate: {}".format(args.rl_behavior_gate))
        print("RL road-border OCC fallback: {}".format(args.rl_occupancy_use_road_border))
        print(
            "RL direct road-border reward (weight/critical/safe): "
            f"{args.rl_reward_w_road_border}/{args.rl_road_border_critical_m}/"
            f"{args.rl_road_border_safe_m}"
        )
        print(
            "RL stopped-reference progress (mode/threshold/tolerance): "
            f"{args.rl_stationary_progress_mode}/"
            f"{args.rl_stationary_reference_threshold_m}/"
            f"{args.rl_stationary_progress_tolerance_m}"
        )
        print(
            "RL red stop-line constraint (enabled/lane tolerance): "
            f"{args.rl_red_light_constraint}/{args.rl_red_light_lane_tolerance_m}"
        )
        print("RL behavior-cloning anchor weight: {}".format(args.rl_bc_weight))
        print("RL EMA update rate: {}".format(args.rl_ema_update_rate))
        print("RL train scope: {}".format(args.rl_train_scope))
        print("RL init uses SFT EMA: {}".format(args.rl_init_use_ema))
        print("Rollout sampling temperature: {}".format(args.rl_noise_scale))
        print("Held-out policy sampling temperature: {}".format(args.rl_eval_noise_scale))
        print("Held-out policy candidate count: {}".format(args.rl_eval_num_generations))
        print(
            "Held-out reward weights (safety/risk/follow/lane/progress/road-border): "
            f"{args.rl_eval_reward_w_safety}/{args.rl_eval_reward_w_risk}/"
            f"{args.rl_eval_reward_w_follow}/"
            f"{args.rl_eval_reward_w_lane}/{args.rl_eval_reward_w_progress}/"
            f"{args.rl_eval_reward_w_road_border}"
        )
        print("Held-out behavior reward gate: {}".format(args.rl_eval_behavior_gate))
        print(
            "Held-out road-border OCC fallback: {}".format(args.rl_eval_occupancy_use_road_border)
        )
        print(
            "Held-out stopped-reference progress (mode/threshold/tolerance): "
            f"{args.rl_eval_stationary_progress_mode}/"
            f"{args.rl_eval_stationary_reference_threshold_m}/"
            f"{args.rl_eval_stationary_progress_tolerance_m}"
        )
        print(
            "Held-out red stop-line constraint (enabled/lane tolerance): "
            f"{args.rl_eval_red_light_constraint}/"
            f"{args.rl_eval_red_light_lane_tolerance_m}"
        )
        print("RL rollout DPM steps: {}".format(args.rl_rollout_steps))
        print("RL updates per rollout: {}".format(args.rl_updates_per_rollout))
        print(
            "RL differentiable candidates/rank cap: {}".format(
                args.rl_update_max_candidates_per_rank or "disabled"
            )
        )
        print("Learning rate: {}".format(args.learning_rate))
        print("Weight decay: {}".format(args.weight_decay))
        print("TF32: {}".format(args.tf32))
        print("Fused optimizer: {}".format(args.fused_optimizer))
        print("DDP static graph: {}".format(args.ddp_static_graph))
        print("TorchInductor model compile: {}".format(args.compile_model))
        save_path = args.save_dir
        os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 6
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

    local_batch_size = batch_size // world_size
    validate_compiled_candidate_batch(
        local_batch_size,
        args.num_generations,
        args.compile_model,
        args.rl_update_max_candidates_per_rank,
    )
    train_sampler = BatchAlignedDistributedSampler(
        train_set,
        num_replicas=ddp.get_world_size(),
        rank=global_rank,
        local_batch_size=local_batch_size,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=local_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has zero batches")

    # Validation is sharded without duplicate padding and the per-rank
    # metrics are all-reduced via aggregate_valid_metrics.
    valid_sampler = DistributedEvalSampler(
        valid_set, num_replicas=ddp.get_world_size(), rank=global_rank
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=args.valid_batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))
        print(
            "Batch-aligned sampling: "
            f"{train_sampler.total_size} samples/epoch "
            f"({train_sampler.padding_size} shuffled repeats for fixed batch shapes)"
        )

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
            gradient_as_bucket_view=True,
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
    optimizer, effective_fused_optimizer = build_adamw_optimizer(
        trainable_params,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fused_requested=args.fused_optimizer,
        device=args.device,
    )
    if args.fused_optimizer and not effective_fused_optimizer:
        print("WARNING: fused AdamW is unavailable; falling back to standard AdamW")
        args.fused_optimizer = False
    if global_rank == 0:
        args_dict["fused_optimizer"] = effective_fused_optimizer
        write_json_atomic(args_dict, os.path.join(save_path, "args.json"))
    scheduler = LinearWarmupConstantLR(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        model_ema = ModelEma(
            ddp.get_model(diffusion_planner, args.ddp),
            decay=1.0 - args.rl_ema_update_rate,
            device=args.device,
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
    else:
        print(f"Initializing RL weights from {args.init_weights_path}")
        model_ema = initialize_fresh_rl_policy_and_ema(
            args.init_weights_path,
            diffusion_planner,
            args.device,
            use_ddp=args.ddp,
            prefer_checkpoint_ema=args.rl_init_use_ema,
            ema_update_rate=args.rl_ema_update_rate,
        )
        init_epoch = 0
        wandb_id = None

    if args.compile_model:
        compile_model_components(diffusion_planner)
        if model_ema is not None:
            compile_model_components(model_ema.ema)

    requested_wandb_id = args.wandb_run_id or wandb_id
    if global_rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=requested_wandb_id,
            dir=f"{save_path}",
        )
        # Strict checkpoint compatibility runs before W&B initialization. An in-place
        # recovery legitimately changes resume_model_path from None to latest.pth.
        wandb.config.update(
            args_dict,
            allow_val_change=(args.resume_model_path is not None or args.wandb_run_id is not None),
        )
        wandb.define_metric("optimizer_step")
        for metric_prefix in (
            "train_step/*",
            "train/*",
            "valid/*",
            "valid_epdms/*",
            "valid_reward/*",
            "valid_turn_indicator/*",
            "valid_multisample/*",
            "closed_loop/*",
        ):
            wandb.define_metric(metric_prefix, step_metric="optimizer_step")
        wandb.define_metric("lr", step_metric="optimizer_step")
        wandb_id = wandb.run.id

    if args.ddp:
        torch.distributed.barrier()

    args._wandb_global_step = int(
        getattr(
            diffusion_planner,
            "_resume_global_step",
            init_epoch * len(train_loader) * args.rl_updates_per_rollout,
        )
    )
    train_log_path = os.path.join(save_path, "train_log.tsv") if global_rank == 0 else None
    baseline_metrics_path = os.path.join(args.save_dir, "source_baseline_metrics.json")
    data_list = []
    best_valid_score = -float("inf")
    full_evals_without_improvement = 0
    baseline_valid_loss = float("inf")
    baseline_metrics = {"baseline_available": False}
    configured_multisample_count = args.multisample_eval_num_samples
    configured_epdms = args.enable_epdms_eval
    resume_baseline_metrics = None
    resume_baseline_data = None
    if args.resume_model_path is not None:
        resume_baseline_metrics = (
            Path(baseline_metrics_path)
            if os.path.exists(baseline_metrics_path)
            else find_checkpoint_run_artifact(
                args.resume_model_path, "source_baseline_metrics.json"
            )
        )
        if resume_baseline_metrics is None:
            # Use the source run's recorded policy rather than the current CLI default.
            # This keeps automatic recovery valid for runs intentionally created without
            # a baseline preflight.
            if _checkpoint_bool(
                args.resume_model_path,
                "rl_validate_before_training",
                args.rl_validate_before_training,
            ):
                raise FileNotFoundError(
                    "Strict RL resume requires source_baseline_metrics.json beside the source run"
                )
            resume_baseline_data = {"baseline_available": False}
        else:
            with open(resume_baseline_metrics, encoding="utf-8") as f:
                resume_baseline_data = json.load(f)
        baseline_available = bool(resume_baseline_data.get("baseline_available", True))
        if baseline_available:
            for key in ("valid_loss_ego", "selection_score"):
                if key not in resume_baseline_data or not math.isfinite(
                    float(resume_baseline_data[key])
                ):
                    raise ValueError(f"Invalid {key} in {resume_baseline_metrics}")
    if global_rank == 0 and args.resume_model_path is not None:
        resume_train_log = (
            Path(train_log_path)
            if os.path.exists(train_log_path)
            else find_checkpoint_run_artifact(args.resume_model_path, "train_log.tsv")
        )
        if resume_train_log is None:
            raise FileNotFoundError(
                f"Strict RL resume is missing train_log.tsv beside {args.resume_model_path}"
            )
        data_list = load_resume_train_rows(resume_train_log, init_epoch)
        best_valid_score = best_valid_score_from_rows(data_list)
        checkpoint_best = float(
            getattr(diffusion_planner, "_resume_best_valid_score", -float("inf"))
        )
        if math.isfinite(checkpoint_best):
            best_valid_score = max(best_valid_score, checkpoint_best)
        best_checkpoint_path = Path(save_path) / "best_model" / "best_model.pth"
        if best_checkpoint_path.is_file():
            best_checkpoint = torch.load(
                best_checkpoint_path, map_location="cpu", weights_only=True
            )
            artifact_checkpoint_best = best_checkpoint.get("best_valid_score", float("nan"))
            if artifact_checkpoint_best is not None:
                artifact_checkpoint_best = float(artifact_checkpoint_best)
                if math.isfinite(artifact_checkpoint_best):
                    best_valid_score = max(best_valid_score, artifact_checkpoint_best)
            del best_checkpoint
        best_info_path = Path(save_path) / "best_model" / "best_model_info.json"
        if best_info_path.is_file():
            with best_info_path.open(encoding="utf-8") as f:
                best_info = json.load(f)
            artifact_best = best_info.get(
                "valid_selection_score", best_info.get("selection_score", float("nan"))
            )
            if artifact_best is not None:
                artifact_best = float(artifact_best)
                if math.isfinite(artifact_best):
                    best_valid_score = max(best_valid_score, artifact_best)
        if data_list:
            raw_patience = data_list[-1].get("full_evals_without_improvement", 0)
            if pd.notna(raw_patience):
                full_evals_without_improvement = int(raw_patience)
        baseline_metrics = resume_baseline_data
        if bool(baseline_metrics.get("baseline_available", True)):
            baseline_valid_loss = float(baseline_metrics["valid_loss_ego"])
            best_valid_score = max(
                best_valid_score,
                float(baseline_metrics["selection_score"]),
            )
        if resume_baseline_metrics is None or resume_baseline_metrics != Path(
            baseline_metrics_path
        ):
            write_json_atomic(baseline_metrics, baseline_metrics_path)

    if args.resume_model_path is None and args.rl_validate_before_training:
        eval_model = model_ema.ema if model_ema is not None else diffusion_planner
        baseline_dict = validate_model(eval_model, valid_loader, args)
        baseline_agg = aggregate_valid_metrics(baseline_dict, args.device)
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        baseline_reward_metrics = validate_hdp_reward_policy(valid_loader, eval_model, args)
        baseline_valid_loss = scalar(baseline_agg["avg_loss_ego"])
        baseline_epdms_metrics = finite_scalar_metrics(baseline_agg["epdms_means"])
        baseline_epdms = baseline_epdms_metrics.get("total", 0.0)
        baseline_turn_metrics = turn_indicator_metrics(baseline_agg)
        baseline_reward_mean = scalar(baseline_reward_metrics["mean"])
        baseline_selection_score = baseline_reward_mean
        baseline_rng_states = gather_rng_states()
        if global_rank == 0:
            best_valid_score = baseline_selection_score
            baseline_metrics = {
                "epoch": 0,
                "baseline_available": True,
                "valid_loss_ego": baseline_valid_loss,
                **{f"valid_epdms_{key}": value for key, value in baseline_epdms_metrics.items()},
                **{
                    f"valid_reward_{key}": scalar(value)
                    for key, value in baseline_reward_metrics.items()
                },
                **{
                    f"valid_turn_indicator_{key}": value
                    for key, value in baseline_turn_metrics.items()
                },
                "selection_score": baseline_selection_score,
                "source_checkpoint": args.init_weights_path,
            }
            write_json_atomic(baseline_metrics, baseline_metrics_path)
            baseline_checkpoint = {
                "epoch": 0,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict() if model_ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": baseline_valid_loss,
                "wandb_id": wandb_id,
                "global_step": args._wandb_global_step,
                "best_valid_score": baseline_selection_score,
                "rng_states": baseline_rng_states,
            }
            best_dir = os.path.join(save_path, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            atomic_torch_save(baseline_checkpoint, os.path.join(best_dir, "best_model.pth"))
            write_json_atomic(args_dict, os.path.join(best_dir, "args.json"))
            write_json_atomic(baseline_metrics, os.path.join(best_dir, "best_model_info.json"))
            if args.use_wandb:
                wandb.run.summary["source/valid_loss_ego"] = baseline_valid_loss
                for key, value in baseline_epdms_metrics.items():
                    wandb.run.summary[f"source/valid_epdms/{key}"] = value
                for key, value in baseline_reward_metrics.items():
                    wandb.run.summary[f"source/valid_reward/{key}"] = scalar(value)
                for key, value in baseline_turn_metrics.items():
                    wandb.run.summary[f"source/valid_turn_indicator/{key}"] = value
            print(
                "Source SFT baseline: "
                f"valid_loss_ego={baseline_valid_loss:.4f}, "
                f"valid_epdms_total={baseline_epdms:.4f}, "
                f"valid_reward_mean={baseline_reward_mean:.4f}"
            )

    if global_rank == 0 and args.resume_model_path is None and not args.rl_validate_before_training:
        baseline_metrics = {
            "epoch": 0,
            "baseline_available": False,
            "source_checkpoint": args.init_weights_path,
        }
        write_json_atomic(baseline_metrics, baseline_metrics_path)

    for epoch in range(init_epoch, train_epochs):
        stop_requested = False
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
            epoch,
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
        if run_full_eval and torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        valid_reward_raw = (
            validate_hdp_reward_policy(valid_loader, eval_model, args) if run_full_eval else {}
        )
        # Save the scheduler/optimizer state for the *next* epoch. Previously checkpoints were
        # written first and scheduler.step() ran afterwards, so strict resume repeated one LR.
        train_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        rng_states = gather_rng_states()
        if global_rank == 0:
            valid_loss_ego = scalar(agg["avg_loss_ego"])
            valid_neighbor_margin = scalar(agg["ego_means"]["ego_neighbor_margin_loss"])
            valid_road_border = scalar(agg["ego_means"]["ego_road_border_loss"])
            valid_epdms_metrics = finite_scalar_metrics(agg["epdms_means"])
            valid_epdms_total = valid_epdms_metrics.get("total", 0.0)
            valid_reward_metrics = {key: scalar(value) for key, value in valid_reward_raw.items()}
            valid_reward_mean = valid_reward_metrics.get("mean", float("nan"))
            valid_multisample = {
                key: scalar(value) for key, value in agg["multisample_means"].items()
            }
            valid_turn_metrics = turn_indicator_metrics(agg)
            has_reward = "reward_mean" in train_loss
            train_reward = scalar(train_loss["reward_mean"]) if has_reward else float("nan")
            epoch_summary = [
                f"Epoch {epoch + 1}/{train_epochs}",
                f"{train_reward=:.4f}",
                f"{valid_loss_ego=:.4f}",
                f"{valid_neighbor_margin=:.4f}",
                f"{valid_road_border=:.4f}",
                f"{valid_epdms_total=:.4f}",
            ]
            if math.isfinite(valid_reward_mean):
                epoch_summary.append(f"{valid_reward_mean=:.4f}")
            if "minADE" in valid_multisample:
                epoch_summary.append(f"valid_multisample_minADE={valid_multisample['minADE']:.4f}")
            if "minFDE" in valid_multisample:
                epoch_summary.append(f"valid_multisample_minFDE={valid_multisample['minFDE']:.4f}")
            print("\n".join(epoch_summary))

            loss_within_guard = valid_loss_ego <= baseline_valid_loss * (
                1.0 + args.rl_max_valid_loss_regression
            )
            source_guards = source_policy_selection_guards(
                baseline_metrics,
                valid_reward_metrics,
                valid_epdms_total,
                valid_epdms_metrics=valid_epdms_metrics,
                max_safety_regression=args.rl_max_valid_safety_regression,
                max_epdms_regression=args.rl_max_valid_epdms_regression,
                require_epdms=configured_epdms,
                require_road_border=args.rl_eval_reward_w_road_border > 0.0,
                require_dac=args.rl_eval_reward_w_road_border > 0.0 and configured_epdms,
            )
            source_policy_within_guard = all(
                value for key, value in source_guards.items() if key != "available"
            )
            selection_score = (
                valid_reward_mean if math.isfinite(valid_reward_mean) else float("nan")
            )
            improves_best = (
                run_full_eval
                and loss_within_guard
                and source_policy_within_guard
                and selection_score > best_valid_score + args.rl_best_score_min_delta
            )
            if run_full_eval:
                full_evals_without_improvement = (
                    0 if improves_best else full_evals_without_improvement + 1
                )
                stop_requested = (
                    args.rl_early_stop_patience > 0
                    and full_evals_without_improvement >= args.rl_early_stop_patience
                )
            if improves_best:
                best_valid_score = selection_score
            logged_best_valid_score = best_valid_score if math.isfinite(best_valid_score) else None

            if args.use_wandb:
                wandb.log(
                    {
                        **{f"train/{k}": v for k, v in train_loss.items()},
                        "optimizer_step": args._wandb_global_step,
                        "lr": train_lr,
                        "valid/ego": valid_loss_ego,
                        "valid/neighbor_margin": valid_neighbor_margin,
                        "valid/road_border": valid_road_border,
                        **{
                            f"valid_epdms/{key}": value
                            for key, value in valid_epdms_metrics.items()
                        },
                        **{
                            f"valid_reward/{key}": value
                            for key, value in valid_reward_metrics.items()
                        },
                        "valid/full_eval": float(run_full_eval),
                        "valid/within_source_loss_guard": float(loss_within_guard),
                        "valid/source_policy_guard_available": float(source_guards["available"]),
                        **{
                            f"valid/within_source_{key}_guard": float(value)
                            for key, value in source_guards.items()
                            if key != "available"
                        },
                        "valid/within_source_policy_guard": float(source_policy_within_guard),
                        "valid/selection_score": selection_score,
                        "valid/best_selection_score": logged_best_valid_score,
                        "valid/improves_best": float(improves_best),
                        "valid/full_evals_without_improvement": full_evals_without_improvement,
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
                **{
                    f"train_{key}": scalar(value)
                    for key, value in train_loss.items()
                    if key not in {"loss", "reward_mean"}
                },
                "valid_loss_ego": valid_loss_ego,
                "valid_neighbor_margin": valid_neighbor_margin,
                "valid_road_border": valid_road_border,
                **{f"valid_epdms_{key}": value for key, value in valid_epdms_metrics.items()},
                **{f"valid_reward_{key}": value for key, value in valid_reward_metrics.items()},
                "valid_full_eval": run_full_eval,
                "valid_within_source_loss_guard": loss_within_guard,
                "valid_source_policy_guard_available": source_guards["available"],
                **{
                    f"valid_within_source_{key}_guard": value
                    for key, value in source_guards.items()
                    if key != "available"
                },
                "valid_within_source_policy_guard": source_policy_within_guard,
                "valid_selection_score": selection_score,
                "valid_improves_best": improves_best,
                "best_valid_score": best_valid_score,
                "full_evals_without_improvement": full_evals_without_improvement,
                **{
                    f"valid_turn_indicator_{key}": value
                    for key, value in valid_turn_metrics.items()
                },
                **{f"valid_multisample_{key}": value for key, value in valid_multisample.items()},
            }
            data_list.append(curr_data)

            model_dict = {
                "epoch": epoch + 1,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict() if model_ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_loss_ego,
                "wandb_id": wandb_id,
                "global_step": args._wandb_global_step,
                "best_valid_score": logged_best_valid_score,
                "rng_states": rng_states,
            }

            # Save a newly accepted policy before committing the epoch through latest.pth. If
            # interruption occurs, resume can never observe a committed best score without its
            # corresponding atomic best checkpoint already being present.
            if improves_best:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                write_json_atomic(args_dict, os.path.join(curr_dir, "args.json"))
                write_json_atomic(curr_data, os.path.join(curr_dir, "best_model_info.json"))
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

            write_train_log_atomic(data_list, os.path.join(save_path, "train_log.tsv"))
            atomic_torch_save(model_dict, f"{save_path}/latest.pth")

            # Resume must not shift the configured checkpoint/closed-loop cadence.
            if (epoch + 1) % save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                write_json_atomic(args_dict, os.path.join(curr_dir, "args.json"))
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

        stop_tensor = torch.tensor(int(stop_requested), device=args.device)
        if args.ddp:
            torch.distributed.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            if global_rank == 0:
                print(
                    "RL early stopping: no meaningful full-validation improvement for "
                    f"{full_evals_without_improvement} evaluations"
                )
            break

    if global_rank == 0 and wandb.run is not None:
        wandb.finish()
    ddp.cleanup()


if __name__ == "__main__":
    model_training(get_args())
