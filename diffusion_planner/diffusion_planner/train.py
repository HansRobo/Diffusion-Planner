import json
import os
import csv
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
import torch
import wandb
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.train_epoch import train_epoch
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.train_utils import resume_model, set_seed
from diffusion_planner.validate_model import validate_model


def load_weights_only(path: str, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_keys = list(model.state_dict().keys())
    state_keys = list(state.keys())
    model_has_module = bool(model_keys and model_keys[0].startswith("module."))
    state_has_module = bool(state_keys and state_keys[0].startswith("module."))
    if model_has_module and not state_has_module:
        state = {f"module.{k}": v for k, v in state.items()}
    elif state_has_module and not model_has_module:
        state = {k.removeprefix("module."): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "decoder.dfp",
        "decoder.joint_temporal",
        "module.decoder.dfp",
        "module.decoder.joint_temporal",
    )
    disallowed_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    disallowed_unexpected = list(incompatible.unexpected_keys)
    print(f"Weights-only init loaded from {path}")
    print(f"missing_keys={len(incompatible.missing_keys)} unexpected_keys={len(incompatible.unexpected_keys)}")
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


def mean_pdms_metric(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("pdms_"):
            result[f"valid_pdms/{key.removeprefix('pdms_')}"] = val.float().mean().item()
    return result


def mean_gate_metric(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("gate_"):
            result[f"valid_gate/{key.removeprefix('gate_')}"] = val.float().mean().item()
    return result


def model_training(args: TrainConfig):
    if getattr(args, "tf32", True):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

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
        args_dict["major_version"] = 4

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)

    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    accum_steps = max(1, args.gradient_accumulation_steps)
    assert batch_size % accum_steps == 0, "batch_size must be divisible by gradient_accumulation_steps"
    micro_batch_size = batch_size // accum_steps
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
    train_set = DiffusionPlannerData(args.train_set_list, skip_filter=args.skip_filter)
    valid_set = DiffusionPlannerData(args.valid_set_list, skip_filter=args.skip_filter)

    train_sampler = DistributedSampler(
        train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=micro_batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Validation is only performed on rank 0 with full dataset
    # Other ranks will get a dummy loader (not used)
    if global_rank == 0:
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size // 4,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )
    else:
        # Dummy loader for non-main processes (won't be used)
        valid_loader = None

    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        find_unused_parameters = getattr(args, "dfp_decoder_mode", "") != "joint_temporal_agent"
        if global_rank == 0:
            print(f"DDP find_unused_parameters={find_unused_parameters}")
        diffusion_planner = DDP(
            diffusion_planner,
            device_ids=[rank],
            find_unused_parameters=find_unused_parameters,
        )

    if args.resume_model_path is None and args.init_weights_path is not None:
        load_weights_only(args.init_weights_path, diffusion_planner, args.device)

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
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device
        )

        # Override learning rate with the new value
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(f"Learning rate reset to {args.learning_rate}")

    else:
        init_epoch = 0
        wandb_id = None

    # logger
    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        # Explicit wandb_run_id always wins; checkpoint wandb_id is only a fallback.
        if args.wandb_run_id is not None:
            wandb_id = args.wandb_run_id

        wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=wandb_id,
            dir=f"{save_path}",
        )
        wandb_id = wandb.run.id
        wandb.config.update(args_dict, allow_val_change=bool(args.resume_model_path))

        # this function creates dataset artifacts and associate them with wandb run
        # if wandb_run_id is given, the input artifact is assumed to be created externally and will not be executed
        if args.use_wandb and args.wandb_run_id is None:
            log_dataset_artifact(wandb.run, args.exp_name, args.train_set_list, args.valid_set_list)

    if args.ddp:
        torch.distributed.barrier()

    data_list = []
    best_loss = float("inf")
    best_pdms_total = -float("inf")

    if global_rank == 0 and args.resume_model_path is not None:
        train_log_path = os.path.join(save_path, "train_log.tsv")
        if os.path.exists(train_log_path):
            if pd is not None:
                data_list = pd.read_csv(train_log_path, sep="\t").to_dict("records")
            else:
                with open(train_log_path, newline="") as f:
                    data_list = list(csv.DictReader(f, delimiter="\t"))
            previous_best = []
            for row in data_list:
                try:
                    previous_best.append(float(row.get("valid_loss_ego_position_lat_loss", "nan")))
                except (TypeError, ValueError):
                    pass
            previous_best = [v for v in previous_best if v == v]
            if previous_best:
                best_loss = min(previous_best)

    if global_rank == 0:
        valid_dict = validate_model(diffusion_planner, valid_loader, args)
        valid_loss_ego = valid_dict["avg_loss_ego"]
        valid_loss_neighbor = valid_dict["avg_loss_neighbor"]
        mean_ego_loss_dict = mean_ego_loss(valid_dict)
        mean_pdms_dict = mean_pdms_metric(valid_dict)
        mean_gate_dict = mean_gate_metric(valid_dict)
        valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lat_loss", 0.0
        )
        valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lon_loss", 0.0
        )
        valid_loss_ego_trajectory_consistency = mean_ego_loss_dict.get(
            "valid_loss/ego_trajectory_consistency_error", 0.0
        )
        valid_pdms_total = mean_pdms_dict.get("valid_pdms/total", 0.0)
        valid_gate_dfp_weight = mean_gate_dict.get("valid_gate/dfp_weight", 0.0)
        turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
        turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
        turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
        print(
            f"{valid_loss_ego=:.3f}\n"
            f"{valid_loss_neighbor=:.3f}\n"
            f"{valid_loss_ego_position_lat_loss=:.3f}\n"
            f"{valid_loss_ego_position_lon_loss=:.3f}\n"
            f"{valid_loss_ego_trajectory_consistency=:.3f}\n"
                f"{valid_pdms_total=:.3f}\n"
            f"{valid_gate_dfp_weight=:.3f}\n"
            f"{turn_indicator_accuracy=:.3f}\n"
            f"{turn_indicator_change_accuracy=:.3f}\n"
            f"{turn_indicator_change_total=:.3f}"
        )
        if args.use_wandb:
            wandb.log(
                {
                    "epoch": 0,
                    "global_step": 0,
                    "valid_loss/ego": valid_loss_ego,
                    "valid_loss/neighbors": valid_loss_neighbor,
                    "valid_loss/turn_indicator_accuracy": turn_indicator_accuracy,
                    "valid_loss/turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                    **mean_ego_loss_dict,
                    **mean_pdms_dict,
                    **mean_gate_dict,
                }
            )

    # begin training
    for epoch in range(init_epoch, train_epochs):
        train_sampler.set_epoch(epoch)

        # Synchronize all processes before training
        if args.ddp:
            torch.distributed.barrier()

        # Adjust learning rate for final 10 epochs. Only apply to runs longer than
        # the final-phase window; otherwise short fine-tunes would start with LR
        # already reduced by 10x/100x and become non-equivalent to epoch1 of SFT20.
        final_epoch_count = 10
        if train_epochs > final_epoch_count and epoch >= train_epochs - final_epoch_count:
            base_lr = args.learning_rate
            if epoch >= train_epochs - final_epoch_count // 2:  # Last 5 epochs: LR * 1/100
                adjusted_lr = base_lr * 0.01
            else:  # First 5 of final 10 epochs: LR * 1/10
                adjusted_lr = base_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr
            if global_rank == 0:
                print(f"Final phase: Epoch {epoch + 1}, LR adjusted to {adjusted_lr}")

        # training step
        train_loss, train_total_loss = train_epoch(
            train_loader, diffusion_planner, optimizer, args, model_ema, aug
        )

        if global_rank == 0:
            valid_dict = validate_model(diffusion_planner, valid_loader, args)
            valid_loss_ego = valid_dict["avg_loss_ego"]
            valid_loss_neighbor = valid_dict["avg_loss_neighbor"]
            mean_ego_loss_dict = mean_ego_loss(valid_dict)
            mean_pdms_dict = mean_pdms_metric(valid_dict)
            mean_gate_dict = mean_gate_metric(valid_dict)
            valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lat_loss", 0.0
            )
            valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lon_loss", 0.0
            )
            valid_loss_ego_trajectory_consistency = mean_ego_loss_dict.get(
                "valid_loss/ego_trajectory_consistency_error", 0.0
            )
            valid_pdms_total = mean_pdms_dict.get("valid_pdms/total", 0.0)
            valid_gate_dfp_weight = mean_gate_dict.get("valid_gate/dfp_weight", 0.0)
            turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
            turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
            turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
            print(
                f"Epoch {epoch + 1}/{train_epochs}\n"
                f"{valid_loss_ego=:.3f}\n"
                f"{valid_loss_neighbor=:.3f}\n"
                f"{valid_loss_ego_position_lat_loss=:.3f}\n"
                f"{valid_loss_ego_position_lon_loss=:.3f}\n"
                f"{valid_loss_ego_trajectory_consistency=:.3f}\n"
                f"{valid_pdms_total=:.3f}\n"
                f"{valid_gate_dfp_weight=:.3f}\n"
                f"{turn_indicator_accuracy=:.3f}\n"
                f"{turn_indicator_change_accuracy=:.3f}\n"
                f"{turn_indicator_change_total=:.3f}"
            )

            lr_dict = {"lr": optimizer.param_groups[0]["lr"]}
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
                    **mean_ego_loss_dict,
                    **mean_pdms_dict,
                    **mean_gate_dict,
                }
            )

            curr_data = {
                "epoch": epoch + 1,
                "train_loss": train_total_loss,
                "valid_loss_ego": valid_loss_ego,
                "valid_loss_neighbor": valid_loss_neighbor,
                "valid_loss_ego_position_lat_loss": valid_loss_ego_position_lat_loss,
                "valid_loss_ego_position_lon_loss": valid_loss_ego_position_lon_loss,
                "valid_loss_ego_trajectory_consistency": valid_loss_ego_trajectory_consistency,
                **{k.replace("/", "_"): v for k, v in mean_pdms_dict.items()},
                **{k.replace("/", "_"): v for k, v in mean_gate_dict.items()},
            }
            data_list.append(curr_data)
            train_log_path = os.path.join(save_path, "train_log.tsv")
            if pd is not None:
                df = pd.DataFrame(data_list)
                df.to_csv(train_log_path, index=False, sep="\t")
            else:
                import csv

                with open(train_log_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(data_list[0].keys()), delimiter="\t")
                    writer.writeheader()
                    writer.writerows(data_list)

            model_dict = {
                "epoch": epoch + 1,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_loss_ego,
                "wandb_id": wandb_id,
            }
            torch.save(model_dict, f"{save_path}/latest.pth")

            if (epoch + 1 - init_epoch) % save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

            if valid_loss_ego_position_lat_loss < best_loss:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_loss = valid_loss_ego_position_lat_loss
                best_loss_data = dict(curr_data)
                best_loss_data["best_loss"] = best_loss
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(best_loss_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

            if getattr(args, "enable_pdms_eval", False) and valid_pdms_total > best_pdms_total:
                curr_dir = os.path.join(save_path, "pdms_best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_pdms_total = valid_pdms_total
                best_pdms_data = dict(curr_data)
                best_pdms_data["best_pdms_total"] = best_pdms_total
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(best_pdms_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

        scheduler.step()
