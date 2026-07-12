import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.train_epoch import train_epoch
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.dataset import (
    BatchAlignedDistributedSampler,
    DiffusionPlannerData,
    DistributedEvalSampler,
)
from diffusion_planner.utils.hdp_compat import require_velocity_normalizer
from diffusion_planner.utils.lr_schedule import LinearWarmupConstantLR
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import export_checkpoint_onnx_guarded
from diffusion_planner.utils.train_utils import (
    ModelEma,
    atomic_torch_save,
    compile_model_components,
    gather_rng_states,
    resume_model,
    set_seed,
)
from diffusion_planner.validate_model import aggregate_valid_metrics, validate_model


def load_weights_only(path: str, model, device, *, prefer_ema: bool = False):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    using_ema = bool(
        prefer_ema and isinstance(ckpt, dict) and ckpt.get("ema_state_dict") is not None
    )
    if using_ema:
        state = ckpt["ema_state_dict"]
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if prefer_ema:
            print(f"WARNING: {path} has no EMA weights; falling back to live model weights")

    model_keys = list(model.state_dict().keys())
    state_keys = list(state.keys())
    model_has_module = bool(model_keys and model_keys[0].startswith("module."))
    state_has_module = bool(state_keys and state_keys[0].startswith("module."))
    if model_has_module and not state_has_module:
        state = {f"module.{key}": value for key, value in state.items()}
    elif state_has_module and not model_has_module:
        state = {key.removeprefix("module."): value for key, value in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    disallowed_missing = list(incompatible.missing_keys)
    disallowed_unexpected = list(incompatible.unexpected_keys)

    print(f"Weights-only init loaded from {path} ({'EMA' if using_ema else 'live'} weights)")
    print(
        f"missing_keys={len(incompatible.missing_keys)} "
        f"unexpected_keys={len(incompatible.unexpected_keys)}"
    )
    if incompatible.missing_keys:
        print("missing_keys_preview=" + ", ".join(incompatible.missing_keys[:20]))
    if incompatible.unexpected_keys:
        print("unexpected_keys_preview=" + ", ".join(incompatible.unexpected_keys[:20]))
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Init checkpoint is not compatible with the model. "
            f"disallowed_missing={disallowed_missing[:20]}, "
            f"disallowed_unexpected={disallowed_unexpected[:20]}"
        )


def _checkpoint_args_path(path: str) -> Path | None:
    checkpoint = Path(path)
    candidates = [
        checkpoint.with_name("args.json"),
        checkpoint.parent / "args.json",
        checkpoint.parent.parent / "args.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _assert_numeric_config_equal(label: str, checkpoint_value, current_value) -> None:
    checkpoint_array = np.asarray(checkpoint_value)
    current_array = np.asarray(current_value)
    if checkpoint_array.shape != current_array.shape or not np.allclose(
        checkpoint_array,
        current_array,
        rtol=0.0,
        atol=1e-7,
        equal_nan=True,
    ):
        raise RuntimeError(
            f"Checkpoint {label} mismatch: checkpoint shape/value "
            f"{checkpoint_array.shape}, current {current_array.shape}."
        )


def _assert_normalizers_compatible(
    ckpt_args: dict,
    args,
    *,
    allow_predicted_neighbor_change: bool,
    waypoint_to_velocity_init: bool,
) -> None:
    checkpoint_state = ckpt_args.get("state_normalizer")
    checkpoint_observation = ckpt_args.get("observation_normalizer")
    if checkpoint_state is None or checkpoint_observation is None:
        raise RuntimeError("Checkpoint args.json is missing saved normalization statistics")

    current_state = args.state_normalizer.to_dict()
    current_observation = args.observation_normalizer.to_dict()
    for statistic in ("mean", "std"):
        _assert_numeric_config_equal(
            f"state_normalizer.{statistic}.ego",
            checkpoint_state[statistic][0],
            current_state[statistic][0],
        )
        if not allow_predicted_neighbor_change:
            _assert_numeric_config_equal(
                f"state_normalizer.{statistic}",
                checkpoint_state[statistic],
                current_state[statistic],
            )

    if bool(getattr(args, "use_velocity_representation", False)) and not waypoint_to_velocity_init:
        for statistic in ("ego_velocity_mean", "ego_velocity_std"):
            if statistic not in checkpoint_state or statistic not in current_state:
                raise RuntimeError(f"Checkpoint is missing state_normalizer.{statistic}")
            _assert_numeric_config_equal(
                f"state_normalizer.{statistic}",
                checkpoint_state[statistic],
                current_state[statistic],
            )

    if checkpoint_observation.keys() != current_observation.keys():
        raise RuntimeError(
            "Checkpoint observation_normalizer keys mismatch: "
            f"checkpoint={sorted(checkpoint_observation)}, current={sorted(current_observation)}"
        )
    for key in checkpoint_observation:
        for statistic in ("mean", "std"):
            _assert_numeric_config_equal(
                f"observation_normalizer.{key}.{statistic}",
                checkpoint_observation[key][statistic],
                current_observation[key][statistic],
            )


def assert_checkpoint_compatible(
    path: str,
    args,
    *,
    allow_waypoint_to_velocity_init: bool = False,
    allow_predicted_neighbor_change: bool = False,
    strict_training_config: bool = True,
) -> None:
    args_path = _checkpoint_args_path(path)
    if args_path is None:
        if (
            getattr(args, "use_velocity_representation", False)
            and not allow_waypoint_to_velocity_init
        ):
            raise RuntimeError(
                "Cannot verify representation compatibility for a checkpoint without args.json: "
                f"{path}. HDP velocity checkpoints must be loaded from a directory that contains args.json."
            )
        if getattr(args, "use_velocity_representation", False):
            print(
                "WARNING: init checkpoint has no args.json; proceeding with waypoint-to-velocity "
                f"bootstrap assumption for init_weights_path={path}."
            )
        return

    with open(args_path, "r", encoding="utf-8") as f:
        ckpt_args = json.load(f)

    ckpt_velocity = bool(ckpt_args.get("use_velocity_representation", False))
    current_velocity = bool(getattr(args, "use_velocity_representation", False))
    waypoint_to_velocity_init = False
    if ckpt_velocity != current_velocity:
        if allow_waypoint_to_velocity_init and not ckpt_velocity and current_velocity:
            waypoint_to_velocity_init = True
            print(
                "WARNING: loading waypoint checkpoint as HDP velocity init weights. "
                f"checkpoint_args={args_path}. This is allowed only for weights-only bootstrap; "
                "resume_model_path remains strictly representation-compatible."
            )
        else:
            raise RuntimeError(
                "Checkpoint representation mismatch: "
                f"{args_path} has use_velocity_representation={ckpt_velocity}, "
                f"current run has use_velocity_representation={current_velocity}. "
                "Refusing to reinterpret waypoint and velocity latents silently."
            )
    if ckpt_velocity:
        require_velocity_normalizer(ckpt_args.get("state_normalizer"), str(args_path))

    ckpt_model_type = ckpt_args.get("diffusion_model_type", "x_start")
    current_model_type = getattr(args, "diffusion_model_type", "x_start")
    if ckpt_model_type != current_model_type:
        raise RuntimeError(
            "Checkpoint diffusion_model_type mismatch: "
            f"{args_path} has {ckpt_model_type!r}, current run has {current_model_type!r}."
        )

    architecture_fields = (
        "future_len",
        "time_len",
        "agent_state_dim",
        "agent_num",
        "static_objects_state_dim",
        "static_objects_num",
        "lane_num",
        "lane_len",
        "route_num",
        "route_len",
        "polygon_num",
        "polygon_len",
        "line_string_num",
        "line_string_len",
        "use_ego_history",
        "use_turn_indicators",
        "encoder_mixer_depth",
        "encoder_fusion_depth",
        "decoder_depth",
        "decoder_tokenization",
        "num_heads",
        "hidden_dim",
    )
    missing_fields = [field for field in architecture_fields if field not in ckpt_args]
    if missing_fields:
        raise RuntimeError(f"Checkpoint args.json is missing architecture fields: {missing_fields}")
    mismatches = [
        (field, ckpt_args[field], getattr(args, field))
        for field in architecture_fields
        if ckpt_args[field] != getattr(args, field)
    ]
    if mismatches:
        raise RuntimeError(f"Checkpoint architecture mismatch: {mismatches}")

    if strict_training_config:
        training_fields = (
            "train_set_list",
            "valid_set_list",
            "train_subsample_step",
            "extra_train_set_list",
            "extra_train_set_repeat",
            "extra_train_set_mask_traffic_lights",
            "align_legacy_neighbor_futures",
            "use_data_augment",
            "augment_prob",
            "augment_type",
            "num_refine",
            "ego_past_noise_std",
            "use_smoothing_future_trajectory",
            "seed",
            "batch_size",
            "valid_batch_size",
            "learning_rate",
            "weight_decay",
            "warm_up_epoch",
            "ego_prediction_horizon",
            "planning_hybrid_loss",
            "hybrid_loss_window",
            "diffusion_supervision_type",
            "diffusion_time_sample_method",
            "coeff_road_border_loss",
            "road_border_margin",
            "road_border_n_interp",
            "coeff_neighbor_collision_loss",
            "neighbor_collision_margin_vehicle",
            "neighbor_collision_margin_pedestrian",
            "neighbor_collision_margin_bicycle",
            "turn_indicator_generated_loss_weight",
            "turn_indicator_expert_loss_weight",
            "use_ema",
            "amp_dtype",
            "tf32",
            "num_generations",
            "rl_train_scope",
            "rl_reward_normalize",
            "rl_reward_beta",
            "rl_noise_scale",
            "rl_rollout_steps",
            "rl_updates_per_rollout",
            "rl_reward_w_risk",
            "rl_reward_w_follow",
            "rl_reward_w_lane",
            "rl_reward_w_progress",
            "rl_bc_weight",
            "rl_ema_update_rate",
            "rl_validate_before_training",
            "rl_max_valid_loss_regression",
            "rl_best_score_min_delta",
            "rl_early_stop_patience",
            "rl_reward_dt",
            "rl_ttc_critical_s",
            "rl_ttc_safe_s",
            "rl_thw_critical_s",
            "rl_thw_safe_s",
            "rl_occupancy_critical_m",
            "rl_occupancy_safe_m",
            "rl_occupancy_speed_gain_s",
            "rl_lane_half_width_m",
            "rl_leader_lateral_margin_m",
            "compile_reward_kernels",
        )
        missing_training_fields = [
            field for field in training_fields if hasattr(args, field) and field not in ckpt_args
        ]
        if missing_training_fields:
            raise RuntimeError(
                f"Checkpoint args.json is missing strict training fields: {missing_training_fields}"
            )
        training_mismatches = [
            (field, ckpt_args[field], getattr(args, field))
            for field in training_fields
            if hasattr(args, field)
            and field in ckpt_args
            and ckpt_args[field] != getattr(args, field)
        ]
        if training_mismatches:
            raise RuntimeError(f"Checkpoint training configuration mismatch: {training_mismatches}")

        checkpoint_train_epochs = int(ckpt_args.get("train_epochs", args.train_epochs))
        if checkpoint_train_epochs != int(args.train_epochs):
            print(
                "WARNING: changing the total training horizon on strict resume: "
                f"checkpoint={checkpoint_train_epochs}, current={args.train_epochs}. "
                "Model, optimizer, scheduler, EMA, and completed epoch state remain strict."
            )

    checkpoint_neighbors = int(ckpt_args.get("predicted_neighbor_num", MAX_NUM_NEIGHBORS))
    current_neighbors = int(getattr(args, "predicted_neighbor_num"))
    if checkpoint_neighbors != current_neighbors and not allow_predicted_neighbor_change:
        raise RuntimeError(
            "Checkpoint action shape mismatch: "
            f"predicted_neighbor_num={checkpoint_neighbors}, current={current_neighbors}. "
            "The checkpoint and current run must use the same ego action-head shape."
        )

    _assert_normalizers_compatible(
        ckpt_args,
        args,
        allow_predicted_neighbor_change=allow_predicted_neighbor_change,
        waypoint_to_velocity_init=waypoint_to_velocity_init,
    )


def find_upward(start_file: str, target_name: str) -> Path:
    directory = Path(start_file).resolve().parent
    for candidate_dir in [directory, *directory.parents]:
        candidate = candidate_dir / target_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{target_name} up {directory}")


def log_dataset_artifact(
    run: wandb.sdk.wandb_run.Run,
    exp_name: str,
    train_set_list: str,
    valid_set_list: str,
    extra_train_set_list: str | list[str] | None = None,
    extra_train_set_repeat: int = 0,
) -> None:
    artifact = wandb.Artifact(
        name=f"dataset_{exp_name}",
        type="dataset",
        metadata={
            "train_set_list": train_set_list,
            "valid_set_list": valid_set_list,
            "extra_train_set_list": extra_train_set_list,
            "extra_train_set_repeat": extra_train_set_repeat,
        },
    )
    train_path = Path(train_set_list)
    valid_path = Path(valid_set_list)
    artifact.add_file(str(train_path), name=train_path.name)
    artifact.add_file(str(valid_path), name=valid_path.name)
    if extra_train_set_list:
        extra_lists = (
            [extra_train_set_list]
            if isinstance(extra_train_set_list, str)
            else extra_train_set_list
        )
        for index, list_path in enumerate(extra_lists):
            extra_path = Path(list_path)
            artifact.add_file(str(extra_path), name=f"extra_{index}_{extra_path.name}")
    try:
        summary_csv = find_upward(train_set_list, "summary.csv")
        artifact.add_file(str(summary_csv), name="summary.csv")
    except FileNotFoundError:
        print("summary.csv not found, skipping.")
    try:
        rosbag_summary_csv = find_upward(train_set_list, "rosbag_summary.csv")
        artifact.add_file(str(rosbag_summary_csv), name="rosbag_summary.csv")
    except FileNotFoundError:
        print("rosbag_summary.csv not found, skipping.")
    run.use_artifact(artifact)


def mean_ego_loss(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("ego_"):
            result[f"valid_loss/{key}"] = val.mean().item()
    return result


def mean_epdms_metric(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if not key.startswith("epdms_"):
            continue
        metric = key.removeprefix("epdms_")
        tensor = val.float()
        if metric.endswith("_available"):
            result[f"valid_epdms/{metric}"] = tensor.mean().item()
            continue
        available = loss_dict.get(f"{key}_available")
        if available is None:
            result[f"valid_epdms/{metric}"] = tensor.mean().item()
            continue
        mask = available.float() > 0.5
        result[f"valid_epdms/{metric}_coverage"] = mask.float().mean().item()
        result[f"valid_epdms/{metric}"] = tensor[mask].mean().item() if mask.any() else float("nan")
    return result


def closed_loop_validate(model, args, epoch: int, out_dir: str) -> None:
    """Run grouped or full-route closed-loop evaluation on checkpoint-save cadence."""
    if not args.closed_loop_npz_root:
        return

    from scenario_generation.scenario_classification import resolve_classification_json
    from scenario_generation.wandb_closed_loop import (
        build_full_closed_loop_wandb_log,
        build_grouped_closed_loop_wandb_log,
    )

    explicit_classification = getattr(args, "closed_loop_classification_json", "")
    classification_json = resolve_classification_json(
        args.closed_loop_npz_root,
        explicit_classification or None,
        dataset_name=getattr(args, "closed_loop_scenario_dataset_name", "") or None,
    )
    if explicit_classification and classification_json is None:
        raise FileNotFoundError(
            f"closed-loop classification JSON does not exist: {explicit_classification}"
        )

    net = ddp.get_model(model, args.ddp)
    was_training = net.training
    net.eval()
    try:
        if classification_json is not None:
            from scenario_generation.grouped_closed_loop_eval import run_grouped_closed_loop_eval

            summary = run_grouped_closed_loop_eval(
                net,
                args,
                args.closed_loop_npz_root,
                classification_json,
                os.path.join(out_dir, "grouped"),
                device=args.device,
                near_miss_thresh=args.closed_loop_near_miss_thresh,
                search_radius=args.closed_loop_search_radius,
                warmup_steps=args.closed_loop_warmup_steps,
                unstick_after=args.closed_loop_unstick_after,
                unstick_advance_m=args.closed_loop_unstick_advance_m,
                draw_every=args.closed_loop_draw_every,
                fps=float(args.closed_loop_fps),
                verbose=False,
            )
            log = build_grouped_closed_loop_wandb_log(
                summary,
                max_videos=getattr(args, "closed_loop_grouped_wandb_max_videos", 24),
            )
        else:
            from scenario_generation.closed_loop_eval import run_closed_loop_eval

            summary = run_closed_loop_eval(
                net,
                args,
                args.closed_loop_npz_root,
                out_dir,
                seg_len=args.closed_loop_seg_len,
                device=args.device,
                near_miss_thresh=args.closed_loop_near_miss_thresh,
                search_radius=args.closed_loop_search_radius,
                warmup_steps=args.closed_loop_warmup_steps,
                unstick_after=args.closed_loop_unstick_after,
                unstick_advance_m=args.closed_loop_unstick_advance_m,
                fps=args.closed_loop_fps,
                replan_interval=args.closed_loop_replan_interval,
                draw_every=args.closed_loop_draw_every,
                neighbor_history_mode="recorded",
                verbose=False,
            )
            log = build_full_closed_loop_wandb_log(summary)
    finally:
        net.train(was_training)

    if getattr(args, "use_wandb", False):
        wandb.log(log)
    if classification_json is not None:
        grouped = summary["grouped_summary"]
        print(
            f"grouped closed-loop @epoch {epoch + 1}: "
            f"{summary['n_episodes']}/{summary['n_episodes_expected']} episodes, "
            f"{summary['n_bags']}/{summary['n_bags_expected']} bags in "
            f"{summary['elapsed_sec']:.1f}s, "
            f"reported={grouped['n_segments_total']}"
        )
    else:
        print(
            f"closed-loop @epoch {epoch + 1}: {summary['n_segments']} seg in "
            f"{summary['elapsed_sec']:.1f}s  coll_seg_rate={summary['collision_segment_rate']:.3f}  "
            f"min_clr={summary['global_min_clearance']:.2f}"
        )


def model_training(args: TrainConfig):
    # Reduce CUDA allocator fragmentation for the many small per-step tensors this
    # training loop produces. setdefault so operators can override from the launcher.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")
    world_size = ddp.get_world_size()
    if args.batch_size < world_size or args.batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be a positive multiple of "
            f"DDP world size ({world_size})"
        )

    # Check before writing the new run's args.json. On an in-place resume, writing first would
    # overwrite the only sidecar that describes the checkpoint we are about to validate.
    if args.resume_model_path is not None:
        assert_checkpoint_compatible(args.resume_model_path, args)
    if args.resume_model_path is None and args.init_weights_path is not None:
        assert_checkpoint_compatible(
            args.init_weights_path,
            args,
            allow_waypoint_to_velocity_init=True,
            allow_predicted_neighbor_change=True,
            strict_training_config=False,
        )

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Weight decay: {}".format(args.weight_decay))
        print("Use device: {}".format(args.device))
        print("TF32: {}".format(args.tf32))
        print("TorchInductor model compile: {}".format(args.compile_model))

        save_path = args.save_dir
        os.makedirs(save_path, exist_ok=True)

        # Save args
        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 5

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)
    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    # set up data loaders
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
    else:
        aug = None

    # prepare dataset
    align_legacy_futures = bool(getattr(args, "align_legacy_neighbor_futures", True))
    train_needs_neighbor_futures = args.coeff_neighbor_collision_loss > 0
    train_set = DiffusionPlannerData(
        args.train_set_list,
        align_legacy_neighbor_futures=align_legacy_futures,
        extra_data_list=getattr(args, "extra_train_set_list", None),
        extra_data_repeat=getattr(args, "extra_train_set_repeat", 0),
        extra_data_mask_traffic_lights=getattr(args, "extra_train_set_mask_traffic_lights", False),
        include_neighbor_futures=train_needs_neighbor_futures,
    )
    valid_set = DiffusionPlannerData(
        args.valid_set_list, align_legacy_neighbor_futures=align_legacy_futures
    )

    train_set.subsample(args.train_subsample_step)
    if len(train_set) == 0:
        raise ValueError("Training data list is empty after subsampling")
    if len(valid_set) == 0:
        raise ValueError("Validation data list is empty")
    if len(valid_set) < world_size:
        raise ValueError("Validation set must contain at least one sample per DDP rank")

    local_batch_size = batch_size // world_size
    train_sampler = BatchAlignedDistributedSampler(
        train_set,
        num_replicas=world_size,
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

    # Validation is sharded without duplicate padding; each rank computes
    # metrics on its shard and they are all-reduced via aggregate_valid_metrics.
    valid_sampler = DistributedEvalSampler(valid_set, num_replicas=world_size, rank=global_rank)
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=batch_size // world_size,
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
        if not train_needs_neighbor_futures:
            print("Ego-only training: skipping unused neighbor_agents_future NPZ payloads")

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(
            diffusion_planner,
            device_ids=[rank],
            find_unused_parameters=getattr(args, "find_unused_parameters", False),
            gradient_as_bucket_view=True,
            static_graph=getattr(args, "ddp_static_graph", False),
        )

    if args.resume_model_path is None and args.init_weights_path is not None:
        load_weights_only(args.init_weights_path, diffusion_planner, args.device)

    model_ema = (
        ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )
        if args.use_ema
        else None
    )

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())
            )
        )

    # optimizer
    params = [
        {
            "params": ddp.get_model(diffusion_planner, args.ddp).parameters(),
            "lr": args.learning_rate,
        }
    ]

    optimizer = optim.AdamW(
        params,
        fused=getattr(args, "fused_optimizer", False) or None,
        weight_decay=args.weight_decay,
    )
    scheduler = LinearWarmupConstantLR(optimizer, train_epochs, args.warm_up_epoch)

    checkpoint_wandb_id = None
    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        (
            diffusion_planner,
            optimizer,
            scheduler,
            init_epoch,
            checkpoint_wandb_id,
            model_ema,
        ) = resume_model(
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
        if init_epoch > train_epochs:
            raise RuntimeError(f"Cannot resume epoch {init_epoch} with train_epochs={train_epochs}")

    else:
        init_epoch = 0
    args._wandb_global_step = int(
        getattr(diffusion_planner, "_resume_global_step", init_epoch * len(train_loader))
    )
    if args.compile_model:
        compile_model_components(diffusion_planner)
        if model_ema is not None:
            compile_model_components(model_ema.ema)
    requested_wandb_id = args.wandb_run_id or checkpoint_wandb_id
    active_wandb_id = requested_wandb_id
    # logger
    if global_rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=requested_wandb_id,
            dir=f"{save_path}",
        )
        active_wandb_id = wandb.run.id

        # Strict checkpoint compatibility has already validated every training field.
        # W&B still sees resume_model_path change from None to latest.pth on an in-place
        # recovery, so allow that runtime-state update instead of aborting the resumed job.
        wandb.config.update(
            args_dict,
            allow_val_change=args.resume_model_path is not None,
        )

        # this function creates dataset artifacts and associate them with wandb run
        # if wandb_run_id is given, the input artifact is assumed to be created externally and will not be executed
        if (
            args.use_wandb
            and requested_wandb_id is None
            and os.environ.get("WANDB_LOG_DATASET_ARTIFACT", "0") == "1"
        ):
            log_dataset_artifact(
                wandb.run,
                args.exp_name,
                args.train_set_list,
                args.valid_set_list,
                getattr(args, "extra_train_set_list", None),
                getattr(args, "extra_train_set_repeat", 0),
            )

    if args.ddp:
        torch.distributed.barrier()

    train_log_path = os.path.join(save_path, "train_log.tsv") if global_rank == 0 else None
    data_list = []
    best_loss = float("inf")
    best_epdms = -float("inf")
    if global_rank == 0 and args.resume_model_path is not None and os.path.exists(train_log_path):
        with open(train_log_path, newline="", encoding="utf-8") as file:
            data_list = list(csv.DictReader(file, delimiter="\t"))
        previous_losses = [
            float(row["valid_loss_ego_position_lat_loss"])
            for row in data_list
            if row.get("valid_loss_ego_position_lat_loss") not in (None, "")
        ]
        if previous_losses:
            best_loss = min(previous_losses)
        previous_epdms = [
            float(row["valid_epdms_total"])
            for row in data_list
            if row.get("valid_epdms_total") not in (None, "")
            and math.isfinite(float(row["valid_epdms_total"]))
        ]
        if previous_epdms:
            best_epdms = max(previous_epdms)

    eval_model = model_ema.ema if model_ema is not None else diffusion_planner
    valid_dict = validate_model(eval_model, valid_loader, args)
    agg = aggregate_valid_metrics(valid_dict, args.device)
    if global_rank == 0:
        valid_loss_ego = agg["avg_loss_ego"]
        valid_loss_neighbor = agg["avg_loss_neighbor"]
        mean_ego_loss_dict = {f"valid_loss/{k}": v for k, v in agg["ego_means"].items()}
        mean_epdms_dict = {f"valid_epdms/{k}": v for k, v in agg["epdms_means"].items()}
        mean_multisample_dict = {
            f"valid_multisample/{k}": v for k, v in agg["multisample_means"].items()
        }
        valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lat_loss", 0.0
        )
        valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lon_loss", 0.0
        )
        turn_indicator_accuracy = agg["turn_indicator_accuracy"]
        turn_indicator_change_accuracy = agg["turn_indicator_change_accuracy"]
        turn_indicator_change_total = agg["turn_indicator_change_total"]
        turn_indicator_class_metrics = {
            f"valid_loss/turn_indicator_{name}_accuracy": value
            for name, value in agg["turn_indicator_class_accuracy"].items()
        } | {
            f"valid_loss/turn_indicator_{name}_count": value
            for name, value in agg["turn_indicator_class_count"].items()
        }
        print(
            f"{valid_loss_ego=:.3f}\n"
            f"{valid_loss_neighbor=:.3f}\n"
            f"{valid_loss_ego_position_lat_loss=:.3f}\n"
            f"{valid_loss_ego_position_lon_loss=:.3f}\n"
            f"{turn_indicator_accuracy=:.3f}\n"
            f"{turn_indicator_change_accuracy=:.3f}\n"
            f"{turn_indicator_change_total=:.3f}"
        )
        if args.use_wandb:
            wandb.log(
                {
                    "epoch": init_epoch,
                    "global_step": args._wandb_global_step,
                    "lr/lr": optimizer.param_groups[0]["lr"],
                    "valid_loss/ego": valid_loss_ego,
                    "valid_loss/neighbors": valid_loss_neighbor,
                    "valid_loss/turn_indicator_accuracy": turn_indicator_accuracy,
                    "valid_loss/turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                    **turn_indicator_class_metrics,
                    **mean_ego_loss_dict,
                    **mean_epdms_dict,
                    **mean_multisample_dict,
                }
            )

    # begin training
    for epoch in range(init_epoch, train_epochs):
        train_sampler.set_epoch(epoch)
        # Synchronize all processes before training
        if args.ddp:
            torch.distributed.barrier()

        # training step
        args._current_epoch = epoch + 1
        args._train_epochs = train_epochs
        train_loss, train_total_loss = train_epoch(
            train_loader, diffusion_planner, optimizer, args, model_ema, aug
        )

        eval_model = model_ema.ema if model_ema is not None else diffusion_planner
        valid_dict = validate_model(eval_model, valid_loader, args)
        agg = aggregate_valid_metrics(valid_dict, args.device)
        # Checkpoints contain the scheduler state for the next epoch. Capture the LR used for
        # this epoch before stepping so metrics and resume behavior both remain exact.
        train_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        rng_states = gather_rng_states()
        if global_rank == 0:
            valid_loss_ego = agg["avg_loss_ego"]
            valid_loss_neighbor = agg["avg_loss_neighbor"]
            mean_ego_loss_dict = {f"valid_loss/{k}": v for k, v in agg["ego_means"].items()}
            mean_epdms_dict = {f"valid_epdms/{k}": v for k, v in agg["epdms_means"].items()}
            mean_multisample_dict = {
                f"valid_multisample/{k}": v for k, v in agg["multisample_means"].items()
            }
            valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lat_loss", 0.0
            )
            valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lon_loss", 0.0
            )
            turn_indicator_accuracy = agg["turn_indicator_accuracy"]
            turn_indicator_change_accuracy = agg["turn_indicator_change_accuracy"]
            turn_indicator_change_total = agg["turn_indicator_change_total"]
            turn_indicator_class_metrics = {
                f"valid_loss/turn_indicator_{name}_accuracy": value
                for name, value in agg["turn_indicator_class_accuracy"].items()
            } | {
                f"valid_loss/turn_indicator_{name}_count": value
                for name, value in agg["turn_indicator_class_count"].items()
            }
            print(
                f"Epoch {epoch + 1}/{train_epochs}\n"
                f"{valid_loss_ego=:.3f}\n"
                f"{valid_loss_neighbor=:.3f}\n"
                f"{valid_loss_ego_position_lat_loss=:.3f}\n"
                f"{valid_loss_ego_position_lon_loss=:.3f}\n"
                f"{turn_indicator_accuracy=:.3f}\n"
                f"{turn_indicator_change_accuracy=:.3f}\n"
                f"{turn_indicator_change_total=:.3f}"
            )

            lr_dict = {"lr": train_lr}
            if args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "global_step": getattr(args, "_wandb_global_step", epoch + 1),
                        **{f"train_loss/{k}": v for k, v in train_loss.items()},
                        **{f"lr/{k}": v for k, v in lr_dict.items()},
                        "valid_loss/ego": valid_loss_ego,
                        "valid_loss/neighbors": valid_loss_neighbor,
                        "valid_loss/turn_indicator_accuracy": turn_indicator_accuracy,
                        "valid_loss/turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                        **turn_indicator_class_metrics,
                        **mean_ego_loss_dict,
                        **mean_epdms_dict,
                        **mean_multisample_dict,
                    }
                )

            curr_data = {
                "epoch": epoch + 1,
                "train_loss": train_total_loss,
                "valid_loss_ego": valid_loss_ego,
                "valid_loss_neighbor": valid_loss_neighbor,
                "valid_loss_ego_position_lat_loss": valid_loss_ego_position_lat_loss,
                "valid_loss_ego_position_lon_loss": valid_loss_ego_position_lon_loss,
                "valid_turn_indicator_accuracy": turn_indicator_accuracy,
                "valid_turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                "valid_turn_indicator_change_total": turn_indicator_change_total,
                **{
                    key.replace("valid_loss/", "valid_"): value
                    for key, value in turn_indicator_class_metrics.items()
                },
                **{k.replace("/", "_"): v for k, v in mean_epdms_dict.items()},
                **{k.replace("/", "_"): v for k, v in mean_multisample_dict.items()},
            }
            data_list.append(curr_data)
            fieldnames = list(dict.fromkeys(key for row in data_list for key in row))
            with open(train_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(data_list)

            model_dict = {
                "epoch": epoch + 1,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict() if model_ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_loss_ego,
                "global_step": args._wandb_global_step,
                "wandb_id": active_wandb_id,
                "rng_states": rng_states,
            }
            atomic_torch_save(model_dict, f"{save_path}/latest.pth")

            # Keep checkpoint and closed-loop cadence anchored to the absolute epoch.
            # A resume from (for example) epoch 7 must still save at epochs 10 and 20.
            if (epoch + 1) % save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                if args.export_onnx_on_save:
                    export_checkpoint_onnx_guarded(
                        config_json_path=os.path.join(curr_dir, "args.json"),
                        ckpt_path=f"{curr_dir}/best_model.pth",
                        output_dir=Path(curr_dir),
                        output_prefix="diffusion_planner",
                        use_ema=model_ema is not None,
                        use_simplify=False,
                        opset_version=20,
                        external_data=False,
                    )
                # Closed-loop validation runs on the same cadence as the checkpoint save; outputs
                # (videos + metrics) land next to the saved weights they correspond to.
                closed_loop_validate(eval_model, args, epoch, os.path.join(curr_dir, "closed_loop"))

            valid_epdms_total = float(mean_epdms_dict.get("valid_epdms/total", float("nan")))
            if math.isfinite(valid_epdms_total) and valid_epdms_total > best_epdms:
                curr_dir = os.path.join(save_path, "best_epdms_model")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                best_epdms = valid_epdms_total
                epdms_data = {**curr_data, "best_valid_epdms_total": best_epdms}
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(epdms_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                print(f"New best EPDMS checkpoint: {best_epdms:.6f} at epoch {epoch + 1}")

            if valid_loss_ego_position_lat_loss < best_loss:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                atomic_torch_save(model_dict, f"{curr_dir}/best_model.pth")
                best_loss = valid_loss_ego_position_lat_loss
                curr_data["best_loss"] = best_loss
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                if args.export_onnx_on_save:
                    export_checkpoint_onnx_guarded(
                        config_json_path=os.path.join(curr_dir, "args.json"),
                        ckpt_path=f"{curr_dir}/best_model.pth",
                        output_dir=Path(curr_dir),
                        output_prefix="diffusion_planner",
                        use_ema=model_ema is not None,
                        use_simplify=False,
                        opset_version=20,
                        external_data=False,
                    )

    if global_rank == 0 and wandb.run is not None:
        wandb.finish()
    ddp.cleanup()
