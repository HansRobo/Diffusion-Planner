import json
import os
from pathlib import Path

import pandas as pd
import torch
import wandb
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.train_epoch import train_one_step
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import export_checkpoint_onnx_guarded
from diffusion_planner.utils.train_utils import get_epoch_mean_loss, resume_model, set_seed
from diffusion_planner.validate_model import aggregate_valid_metrics, validate_model


def find_upward(start_file: str, target_name: str) -> Path:
    directory = Path(start_file).resolve().parent
    for candidate_dir in [directory, *directory.parents]:
        candidate = candidate_dir / target_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{target_name} up {directory}")


def log_dataset_artifact(
    run: wandb.sdk.wandb_run.Run, exp_name: str, train_set_list: str, valid_set_list: str
) -> None:
    artifact = wandb.Artifact(
        name=f"dataset_{exp_name}",
        type="dataset",
        metadata={"train_set_list": train_set_list, "valid_set_list": valid_set_list},
    )
    train_path = Path(train_set_list)
    valid_path = Path(valid_set_list)
    artifact.add_file(str(train_path), name=train_path.name)
    artifact.add_file(str(valid_path), name=valid_path.name)
    summary_csv = find_upward(train_set_list, "summary.csv")
    artifact.add_file(str(summary_csv), name="summary.csv")
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


def closed_loop_validate(model, args, step: int, out_dir: str) -> None:
    """Closed-loop rendered rollout; logs metrics + the rollout video to wandb.

    Drives the ego in CLOSED LOOP over the route NPZ frames under ``args.closed_loop_npz_root``
    (one route = one trial), renders an MP4 into ``out_dir``, aggregates collision/clearance
    metrics, and logs both to wandb at ``step``. Called on the checkpoint-save cadence.
    No-op when ``closed_loop_npz_root`` is unset. Rank-0 only: pass the unwrapped model; it is
    switched to eval for the rollout (so the diffusion sampler runs and produces ``prediction``)
    and restored afterwards.
    """
    if not args.closed_loop_npz_root:
        return
    import math

    from scenario_generation.closed_loop_eval import run_closed_loop_eval

    net = ddp.get_model(model, args.ddp)
    was_training = net.training
    net.eval()
    try:
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
    finally:
        net.train(was_training)

    # Scalar metrics (drop non-finite clearances: a segment with no neighbor reports +inf).
    scalar_keys = [
        "collision_segment_rate",
        "collision_step_rate",
        "near_miss_segment_rate",
        "near_miss_step_rate",
        "global_min_clearance",
        "mean_segment_min_clearance",
        "mean_segment_mean_clearance",
        "total_collision_steps",
        "total_near_miss_steps",
        "total_snaps",
        "total_steps",
    ]
    log = {
        f"closed_loop/{k}": summary[k]
        for k in scalar_keys
        if isinstance(summary[k], (int,)) or math.isfinite(summary[k])
    }
    for mp4 in summary["video_mp4s"]:
        log[f"closed_loop/video/{Path(mp4).stem}"] = wandb.Video(str(mp4), format="mp4")
    wandb.log(log, step=step)
    print(
        f"closed-loop @step {step}: {summary['n_segments']} seg in "
        f"{summary['elapsed_sec']:.1f}s  coll_seg_rate={summary['collision_segment_rate']:.3f}  "
        f"min_clr={summary['global_min_clearance']:.2f}  -> {len(summary['video_mp4s'])} video(s)"
    )


def run_validation(model, valid_loader, args):
    """Validate on all ranks (metrics are all-reduced inside ``aggregate_valid_metrics``) and
    return the rank-0 scalar summary used for logging and checkpointing."""
    valid_dict = validate_model(model, valid_loader, args)
    agg = aggregate_valid_metrics(valid_dict, args.device)
    mean_ego_loss_dict = {f"valid_loss/{k}": v for k, v in agg["ego_means"].items()}
    mean_epdms_dict = {f"valid_epdms/{k}": v for k, v in agg["epdms_means"].items()}
    return {
        "valid_loss_ego": agg["avg_loss_ego"],
        "valid_loss_neighbor": agg["avg_loss_neighbor"],
        "valid_loss_ego_position_lat_loss": mean_ego_loss_dict.get(
            "valid_loss/ego_position_lat_loss", 0.0
        ),
        "valid_loss_ego_position_lon_loss": mean_ego_loss_dict.get(
            "valid_loss/ego_position_lon_loss", 0.0
        ),
        "turn_indicator_accuracy": agg["turn_indicator_accuracy"],
        "turn_indicator_change_accuracy": agg["turn_indicator_change_accuracy"],
        "turn_indicator_change_total": agg["turn_indicator_change_total"],
        "mean_ego_loss_dict": mean_ego_loss_dict,
        "mean_epdms_dict": mean_epdms_dict,
    }


def print_validation_summary(summary, header):
    print(
        f"{header}\n"
        f"valid_loss_ego={summary['valid_loss_ego']:.3f}\n"
        f"valid_loss_neighbor={summary['valid_loss_neighbor']:.3f}\n"
        f"valid_loss_ego_position_lat_loss={summary['valid_loss_ego_position_lat_loss']:.3f}\n"
        f"valid_loss_ego_position_lon_loss={summary['valid_loss_ego_position_lon_loss']:.3f}\n"
        f"turn_indicator_accuracy={summary['turn_indicator_accuracy']:.3f}\n"
        f"turn_indicator_change_accuracy={summary['turn_indicator_change_accuracy']:.3f}\n"
        f"turn_indicator_change_total={summary['turn_indicator_change_total']:.3f}"
    )


def model_training(args: TrainConfig):
    assert len(args.coeff_timestep) == 4, "coeff_timestep must be a list of 4 elements"

    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

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

    # training parameters (step-based). All cadences / schedule lengths are resolved from
    # train_steps at runtime (defaults: valid 2%, save 10%, warm-up 1%, final-phase 10%).
    train_steps = args.train_steps
    batch_size = args.batch_size
    valid_interval_steps = int(train_steps * args.valid_interval_ratio)
    save_interval_steps = int(train_steps * args.save_interval_ratio)
    warm_up_steps = int(train_steps * args.warm_up_ratio)
    final_phase_steps = int(train_steps * args.final_phase_ratio)
    if global_rank == 0:
        print(
            f"{train_steps=}, {valid_interval_steps=}, {save_interval_steps=}, "
            f"{warm_up_steps=}, {final_phase_steps=}"
        )

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
    train_set = DiffusionPlannerData(args.train_set_list)
    valid_set = DiffusionPlannerData(args.valid_set_list)

    train_set.data_list = train_set.data_list[:: args.train_subsample_step]

    train_sampler = DistributedSampler(
        train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Validation is sharded across all ranks (DistributedSampler); each rank computes
    # metrics on its shard and they are all-reduced via aggregate_valid_metrics.
    valid_sampler = DistributedSampler(
        valid_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=False
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
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

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_steps, warm_up_steps)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        # We always use new wandb run for each training session, so we don't need to load the wandb_id from the model_dict.
        diffusion_planner, optimizer, scheduler, init_step, _, model_ema = resume_model(
            args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device
        )

        # Override learning rate with the new value
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(f"Learning rate reset to {args.learning_rate}")

    else:
        init_step = 0
    # logger
    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"

        # if wandb_run_id is given, the training will be logged to the existing run instead of creating a new one.
        wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=args.wandb_run_id,
            dir=f"{save_path}",
        )

        wandb.config.update(args_dict)

        # this function creates dataset artifacts and associate them with wandb run
        # if wandb_run_id is given, the input artifact is assumed to be created externally and will not be executed
        if args.use_wandb and args.wandb_run_id is None:
            log_dataset_artifact(wandb.run, args.exp_name, args.train_set_list, args.valid_set_list)

    if args.ddp:
        torch.distributed.barrier()

    data_list = []
    best_loss = float("inf")

    summary = run_validation(diffusion_planner, valid_loader, args)
    if global_rank == 0:
        print_validation_summary(summary, header="Initial validation")

    # begin training (step-based). One optimizer step == one batch; the dataset loader is
    # cycled as many times as needed to reach ``train_steps`` total optimizer steps.
    base_lr = args.learning_rate
    global_step = init_step
    sampler_epoch = 0
    train_sampler.set_epoch(sampler_epoch)
    train_iter = iter(train_loader)
    interval_losses = []

    pbar = (
        tqdm(total=train_steps, initial=global_step, desc="Training", unit="step")
        if global_rank == 0
        else None
    )

    while global_step < train_steps:
        # Restore train mode each iteration: validate_model() below switches the model to eval()
        # and does not restore it.
        diffusion_planner.train()

        # Fetch the next batch, starting a fresh pass over the (reshuffled) loader when exhausted.
        try:
            inputs = next(train_iter)
        except StopIteration:
            sampler_epoch += 1
            train_sampler.set_epoch(sampler_epoch)
            train_iter = iter(train_loader)
            inputs = next(train_iter)

        # Final-phase LR decay over the last ``final_phase_steps`` steps: the last half runs at
        # base_lr * 0.01, the half before it at base_lr * 0.1. Set before the optimizer step; the
        # post-warmup scheduler multiplies by 1.0 so this override persists across steps.
        steps_into_final = global_step - (train_steps - final_phase_steps)
        if steps_into_final >= 0:
            if steps_into_final >= final_phase_steps - final_phase_steps // 2:
                adjusted_lr = base_lr * 0.01
            else:
                adjusted_lr = base_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr

        loss = train_one_step(inputs, diffusion_planner, optimizer, args, model_ema, aug)
        interval_losses.append(loss)
        global_step += 1
        scheduler.step()

        if pbar is not None:
            pbar.update(1)

        is_valid_step = global_step % valid_interval_steps == 0 or global_step == train_steps
        is_save_step = global_step % save_interval_steps == 0 or global_step == train_steps
        # Validation runs on either cadence: a checkpoint save needs fresh validation metrics, so a
        # save step that is not also a valid step still validates here (the two cadences are
        # independent and need not be multiples of each other).
        if not is_valid_step and not is_save_step:
            continue

        # Mean train loss over this interval, reduced across ranks (matches per-key reduction the
        # old per-epoch path used).
        train_loss = get_epoch_mean_loss(interval_losses)
        if args.ddp:
            train_loss = ddp.reduce_and_average_losses(train_loss, torch.device(args.device))
        interval_losses = []

        summary = run_validation(diffusion_planner, valid_loader, args)

        if global_rank == 0:
            print_validation_summary(summary, header=f"Step {global_step}/{train_steps}")
            print(f"train_loss={train_loss['loss']:.4f}")

            mean_ego_loss_dict = summary["mean_ego_loss_dict"]
            mean_epdms_dict = summary["mean_epdms_dict"]
            lr_dict = {"lr": optimizer.param_groups[0]["lr"]}
            wandb.log(
                {
                    **{f"train_loss/{k}": v for k, v in train_loss.items()},
                    **{f"lr/{k}": v for k, v in lr_dict.items()},
                    "valid_loss/ego": summary["valid_loss_ego"],
                    "valid_loss/neighbors": summary["valid_loss_neighbor"],
                    "valid_loss/turn_indicator_accuracy": summary["turn_indicator_accuracy"],
                    "valid_loss/turn_indicator_change_accuracy": summary[
                        "turn_indicator_change_accuracy"
                    ],
                    **mean_ego_loss_dict,
                    **mean_epdms_dict,
                },
                step=global_step,
            )

            curr_data = {
                "step": global_step,
                "train_loss": train_loss["loss"],
                "valid_loss_ego": summary["valid_loss_ego"],
                "valid_loss_neighbor": summary["valid_loss_neighbor"],
                "valid_loss_ego_position_lat_loss": summary["valid_loss_ego_position_lat_loss"],
                "valid_loss_ego_position_lon_loss": summary["valid_loss_ego_position_lon_loss"],
                **{k.replace("/", "_"): v for k, v in mean_epdms_dict.items()},
            }
            data_list.append(curr_data)
            df = pd.DataFrame(data_list)
            df.to_csv(os.path.join(save_path, "train_log.tsv"), index=False, sep="\t")

            model_dict = {
                "step": global_step,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": summary["valid_loss_ego"],
                # We always use new wandb run for each training session, so we don't need to save the wandb_id in the model_dict.
                "wandb_id": None,
            }
            torch.save(model_dict, f"{save_path}/latest.pth")

            if is_save_step:
                curr_dir = os.path.join(save_path, f"step{global_step:08d}")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                # Export ONNX next to the checkpoint (regular weights, ORT validation skipped).
                export_checkpoint_onnx_guarded(
                    config_json_path=os.path.join(curr_dir, "args.json"),
                    ckpt_path=f"{curr_dir}/best_model.pth",
                    output_dir=Path(curr_dir),
                    output_prefix="diffusion_planner",
                    use_ema=False,
                    use_simplify=False,
                    opset_version=20,
                    external_data=False,
                )
                # Closed-loop validation runs on the same cadence as the checkpoint save; outputs
                # (videos + metrics) land next to the saved weights they correspond to.
                closed_loop_validate(
                    diffusion_planner, args, global_step, os.path.join(curr_dir, "closed_loop")
                )

            if summary["valid_loss_ego_position_lat_loss"] < best_loss:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_loss = summary["valid_loss_ego_position_lat_loss"]
                curr_data["best_loss"] = best_loss
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)
                # Export ONNX next to the checkpoint (regular weights, ORT validation skipped).
                export_checkpoint_onnx_guarded(
                    config_json_path=os.path.join(curr_dir, "args.json"),
                    ckpt_path=f"{curr_dir}/best_model.pth",
                    output_dir=Path(curr_dir),
                    output_prefix="diffusion_planner",
                    use_ema=False,
                    use_simplify=False,
                    opset_version=20,
                    external_data=False,
                )

        # Resync all ranks after rank-0's (potentially long) save / ONNX export / closed-loop
        # work so non-zero ranks don't race ahead into the next backward's allreduce.
        if args.ddp:
            torch.distributed.barrier()

    if pbar is not None:
        pbar.close()

    if global_rank == 0 and wandb.run is not None:
        wandb.finish()
