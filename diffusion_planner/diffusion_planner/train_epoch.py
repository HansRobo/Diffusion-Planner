import torch
import wandb
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.masks import neighbor_future_padding_mask
from diffusion_planner.utils.train_utils import (
    compute_grad_stats,
    finalize_epoch_loss_sums,
    update_epoch_loss_sums,
)


def compose_supervised_total_loss(loss, args):
    return (
        loss["ego_planning_loss"]
        + args.planning_hybrid_loss
        * loss.get("ego_planning_hybrid_loss", torch.zeros_like(loss["ego_planning_loss"]))
        + loss["turn_indicator_loss"]
        + args.coeff_road_border_loss * loss["road_border_loss"]
        + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
    )


def heading_to_cos_sin(x, *, preserve_zero_padding=False):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))

    Idempotent: a [..., 4] input that is already (x, y, cos, sin) is returned
    unchanged in value. A clone is returned so callers can safely mask it in place
    without mutating a batch-owned tensor. ``preserve_zero_padding`` must be used
    for raw padded poses: converting an all-zero (x, y, heading) row otherwise turns
    it into (0, 0, 1, 0), which is a valid-looking pose to downstream masks.
    """
    if x.shape[-1] not in (3, 4):
        raise ValueError(f"Expected pose features with last dimension 3 or 4, got {x.shape}")
    zero_padding = None
    if preserve_zero_padding:
        zero_padding = torch.all(x == 0, dim=-1, keepdim=True)
    if x.shape[-1] == 4:
        converted = x.clone()
    else:
        converted = torch.cat(
            [
                x[..., :2],
                x[..., 2:3].cos(),
                x[..., 2:3].sin(),
            ],
            dim=-1,
        )
    if zero_padding is not None:
        converted.masked_fill_(zero_padding, 0.0)
    return converted


def prepare_neighbor_supervision(
    neighbors_future: torch.Tensor,
    action_neighbor_num: int,
    include_collision_futures: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Prepare only the neighbor futures consumed by the configured training losses."""
    if action_neighbor_num == 0 and not include_collision_futures:
        batch_size, _, future_len, _ = neighbors_future.shape
        empty_future = neighbors_future.new_zeros((batch_size, 0, future_len, 4))
        empty_mask = torch.zeros(
            (batch_size, 0, future_len), dtype=torch.bool, device=neighbors_future.device
        )
        return empty_future, empty_mask, None

    if include_collision_futures:
        all_neighbor_mask = neighbor_future_padding_mask(neighbors_future)
        all_neighbors_future = heading_to_cos_sin(neighbors_future)
        all_neighbors_future[all_neighbor_mask] = 0.0
        return (
            all_neighbors_future[:, :action_neighbor_num],
            all_neighbor_mask[:, :action_neighbor_num],
            (all_neighbors_future, all_neighbor_mask),
        )

    action_future = neighbors_future[:, :action_neighbor_num]
    action_mask = neighbor_future_padding_mask(action_future)
    action_future = heading_to_cos_sin(action_future)
    action_future[action_mask] = 0.0
    return action_future, action_mask, None


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation = None):
    epoch_loss_sums = {}
    epoch_loss_counts = {}

    model.train()
    step_log = getattr(args, "wandb_step_log_interval", 0)
    log_step = args.use_wandb and step_log > 0 and ddp.get_rank() == 0
    if not hasattr(args, "_wandb_global_step"):
        args._wandb_global_step = 0
    current_epoch = getattr(args, "_current_epoch", 0)
    total_epochs = getattr(args, "_train_epochs", 0)

    num_batches = len(data_loader)
    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    if args.ddp:
        torch.cuda.synchronize()

    # Grad stats are diagnostics only: computing them every step concatenates ALL
    # gradients and calls .item() five times (five device syncs per step). Sample them
    # on the wandb logging cadence instead (or every 100 steps when step logging is off).
    stats_interval = step_log if step_log > 0 else 100
    needs_neighbor_futures = args.coeff_neighbor_collision_loss > 0
    net = ddp.get_model(model, args.ddp)
    turn_params = list(net.decoder.turn_indicator_predictor.parameters())
    turn_param_ids = {id(param) for param in turn_params}
    policy_params = [param for param in net.parameters() if id(param) not in turn_param_ids]

    for batch_idx, inputs in enumerate(data_loader, start=1):
        neighbor_future_cpu = inputs.pop("neighbor_agents_future", None)
        if needs_neighbor_futures and neighbor_future_cpu is None:
            raise KeyError("neighbor_agents_future is required for collision supervision")
        inputs = {key: value.to(args.device, non_blocking=True) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(
            inputs["ego_agent_past"], preserve_zero_padding=True
        )
        inputs["goal_pose"] = heading_to_cos_sin(
            inputs["goal_pose"], preserve_zero_padding=True
        )

        ego_future = inputs["ego_agent_future"]
        if neighbor_future_cpu is None:
            batch_size, future_len = ego_future.shape[:2]
            neighbors_future = ego_future.new_zeros(
                (batch_size, 0, future_len, ego_future.shape[-1])
            )
        else:
            neighbors_future = neighbor_future_cpu.to(args.device, non_blocking=True)
        # Normalize to ego-centric
        if aug is not None:
            inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

        # heading to cos sin
        ego_future = heading_to_cos_sin(ego_future)

        action_neighbor_num = 0
        neighbors_future, mask, collision_futures = prepare_neighbor_supervision(
            neighbors_future,
            action_neighbor_num,
            include_collision_futures=args.coeff_neighbor_collision_loss > 0,
        )
        # Future GT is not an encoder input. In ego-only training this also releases the
        # full 320-agent tensor before normalization and the model forward.
        inputs = args.observation_normalizer(inputs)

        # call the model
        optimizer.zero_grad()

        loss = compute_training_loss(
            model,
            inputs,
            (ego_future, neighbors_future, mask),
            args,
            collision_futures=collision_futures,
        )

        loss["loss"] = compose_supervised_total_loss(loss, args)

        # loss backward
        loss["loss"].backward()

        # Gradient statistics (computed before clipping so that exploding
        # gradients are not masked by clip_grad_norm_). Sampled on the logging
        # cadence — see stats_interval above; the counter is identical on all
        # ranks so the branch is DDP-consistent.
        if (args._wandb_global_step + 1) % stats_interval == 0:
            loss.update(compute_grad_stats(model.parameters()))

        nn.utils.clip_grad_norm_(policy_params, 5)
        nn.utils.clip_grad_norm_(turn_params, 5)
        optimizer.step()

        if ema is not None:
            ema.update(net)
        args._wandb_global_step += 1

        if log_step and args._wandb_global_step % step_log == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    f"train_step/{key}": (value.item() if torch.is_tensor(value) else value)
                    for key, value in loss.items()
                    if key != "loss" or torch.is_tensor(value)
                }
                | {
                    "epoch": current_epoch,
                    "global_step": args._wandb_global_step,
                    "lr/lr": current_lr,
                    "train_step/global_step": args._wandb_global_step,
                    "train_step/epoch": current_epoch,
                    "train_step/total_epochs": total_epochs,
                    "train_step/batch": batch_idx,
                    "train_step/num_batches": num_batches,
                    "train_step/epoch_progress": batch_idx / max(num_batches, 1),
                    "train_step/lr": current_lr,
                },
                commit=True,
            )

        # NOTE: no per-step torch.cuda.synchronize() here — it stalled the whole
        # pipeline every iteration for no correctness benefit (DDP synchronizes via
        # its allreduce; kernels are stream-ordered).
        update_epoch_loss_sums(epoch_loss_sums, epoch_loss_counts, loss)

    epoch_mean_loss = finalize_epoch_loss_sums(epoch_loss_sums, epoch_loss_counts)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
