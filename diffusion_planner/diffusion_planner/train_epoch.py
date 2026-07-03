from contextlib import nullcontext

import torch
import wandb
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def _get_named_parameter(model, use_ddp, name):
    base_model = ddp.get_model(model, use_ddp)
    for param_name, param in base_model.named_parameters():
        if param_name == name or param_name.endswith(name):
            return param
    return None


def _sanitize_and_clip_grad_norm(parameters, max_norm):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return {
            "grad_norm_total_before_clip": 0.0,
            "grad_clip_coef": 1.0,
            "grad_max_abs_before_clip": 0.0,
            "grad_nonfinite_param_count": 0.0,
            "grad_nonfinite_element_count": 0.0,
            "grad_total_norm_nonfinite_after_sanitize": 0.0,
        }

    device = params[0].grad.device
    total_sq = torch.zeros((), device=device, dtype=torch.float64)
    max_abs = torch.zeros((), device=device, dtype=torch.float64)
    nonfinite_param_count = 0
    nonfinite_element_count = 0

    for param in params:
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not bool(finite.all().item()):
            nonfinite_param_count += 1
            nonfinite_element_count += int((~finite).sum().item())
            grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if grad.numel() > 0:
            grad_abs_max = grad.abs().max().double()
            max_abs = torch.maximum(max_abs, grad_abs_max)
            grad64 = grad.double()
            total_sq = total_sq + torch.sum(grad64 * grad64)

    total_norm = torch.sqrt(total_sq)
    total_norm_nonfinite = not bool(torch.isfinite(total_norm).item())
    clip_coef = torch.ones((), device=device, dtype=torch.float64)
    if total_norm_nonfinite:
        for param in params:
            param.grad.detach().zero_()
        clip_coef = torch.zeros((), device=device, dtype=torch.float64)
    elif total_norm > max_norm:
        clip_coef = torch.as_tensor(float(max_norm), device=device, dtype=torch.float64) / (
            total_norm + 1.0e-6
        )
        for param in params:
            param.grad.detach().mul_(clip_coef.to(device=param.grad.device, dtype=param.grad.dtype))

    return {
        "grad_norm_total_before_clip": float(total_norm.detach().cpu()),
        "grad_clip_coef": float(clip_coef.detach().cpu()),
        "grad_max_abs_before_clip": float(max_abs.detach().cpu()),
        "grad_nonfinite_param_count": float(nonfinite_param_count),
        "grad_nonfinite_element_count": float(nonfinite_element_count),
        "grad_total_norm_nonfinite_after_sanitize": float(total_norm_nonfinite),
    }


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))

    Idempotent: a [..., 4] input that is already (x, y, cos, sin) is returned
    unchanged. This guards against double-conversion (cos(cos)) now that scene-gen
    emits 4-col futures — callers can hand it either layout safely.
    """
    if x.shape[-1] == 4:
        return x
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation = None):
    epoch_loss = []

    model.train()
    step_log = getattr(args, "wandb_step_log_interval", 0)
    log_step = args.use_wandb and step_log > 0 and ddp.get_rank() == 0
    if not hasattr(args, "_wandb_global_step"):
        args._wandb_global_step = 0

    if args.ddp:
        torch.cuda.synchronize()

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    accum_steps = max(1, getattr(args, "gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)

    for step_idx, inputs in enumerate(data_loader):
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]
        # Normalize to ego-centric
        if aug is not None:
            inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

        # heading to cos sin
        ego_future = heading_to_cos_sin(ego_future)

        mask = torch.sum(torch.ne(neighbors_future[..., :4], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[mask] = 0.0
        inputs = args.observation_normalizer(inputs)

        # call the model
        loss = compute_training_loss(model, inputs, (ego_future, neighbors_future, mask), args)

        loss["loss"] = (
            args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
            + args.alpha_planning_loss * loss["ego_planning_loss"]
            + loss["turn_indicator_loss"]
            + args.coeff_road_border_loss * loss["road_border_loss"]
            + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
            + getattr(args, "coeff_comfort_loss", 0.0) * loss["comfort_loss"]
            + getattr(args, "coeff_progress_loss", 0.0) * loss["progress_loss"]
        )
        if getattr(args, "use_dfp_decoder", False):
            loss["loss"] = loss["loss"] + (
                args.dfp_lambda_future * loss["dfp_future_loss"]
                + args.dfp_lambda_hist * loss["dfp_history_loss"]
                + args.dfp_lambda_current * loss["dfp_current_loss"]
            )
            if "ego_original_planning_loss" in loss:
                loss["loss"] = loss["loss"] + (
                    args.dfp_lambda_original_ego * loss["ego_original_planning_loss"]
                )

        should_step = ((step_idx + 1) % accum_steps == 0) or ((step_idx + 1) == len(data_loader))
        sync_context = (
            model.no_sync()
            if getattr(args, "ddp", False) and hasattr(model, "no_sync") and not should_step
            else nullcontext()
        )
        with sync_context:
            (loss["loss"] / accum_steps).backward()

        if should_step:
            grad_logs = {}
            joint_out_weight = None
            if getattr(args, "dfp_decoder_mode", "") == "joint_temporal_agent":
                joint_out_weight = _get_named_parameter(
                    model, args.ddp, "decoder.joint_temporal_final_layer.proj.4.weight"
                )
                if joint_out_weight is not None:
                    if joint_out_weight.grad is None:
                        grad_logs["grad_norm_joint_temporal_out_weight"] = torch.tensor(
                            0.0, device=joint_out_weight.device
                        )
                        grad_logs["grad_missing_joint_temporal_out_weight"] = torch.tensor(
                            1.0, device=joint_out_weight.device
                        )
                    else:
                        grad_logs["grad_norm_joint_temporal_out_weight"] = (
                            joint_out_weight.grad.detach().norm()
                        )
                        grad_logs["grad_missing_joint_temporal_out_weight"] = torch.tensor(
                            0.0, device=joint_out_weight.device
                        )
                    grad_logs["param_norm_joint_temporal_out_weight_before_step"] = (
                        joint_out_weight.detach().norm()
                    )

            clip_stats = _sanitize_and_clip_grad_norm(model.parameters(), 5.0)
            log_device = loss["loss"].device if torch.is_tensor(loss["loss"]) else args.device
            for key, value in clip_stats.items():
                grad_logs[key] = torch.tensor(float(value), device=log_device)

            optimizer.step()

            if joint_out_weight is not None:
                grad_logs["param_norm_joint_temporal_out_weight_after_step"] = (
                    joint_out_weight.detach().norm()
                )
            loss.update(grad_logs)

            ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            args._wandb_global_step += 1

        if should_step and log_step and args._wandb_global_step % step_log == 0:
            wandb.log(
                {
                    f"train_step/{k}": (v.item() if torch.is_tensor(v) else v)
                    for k, v in loss.items()
                    if k != "loss" or torch.is_tensor(v)
                }
                | {"train_step/global_step": args._wandb_global_step},
                commit=True,
            )

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
