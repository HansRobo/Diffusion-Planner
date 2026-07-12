"""HDP reward-weighted RL-Hybrid epoch loop.

The only supported RL path is the HDP reward-weighted hybrid loss:

  1. sample a group of trajectories per scene;
  2. score them with the HDP risk/follow/lane reward adapted to Tier IV NPZ signals;
  3. group-normalize the reward and weight the HDP hybrid diffusion loss with
     exp(beta * normalized_reward).
"""

import torch
import wandb
from torch import nn
from tqdm import tqdm

from diffusion_planner.hdp_rl_utils import (
    compute_hdp_reward,
    compute_reward_weighted_loss,
    compute_reward_weights,
    distributed_valid_sample_count,
    expand_batch,
    sample_group,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp
from diffusion_planner.utils.masks import neighbor_future_padding_mask
from diffusion_planner.utils.train_utils import finalize_epoch_loss_sums, update_epoch_loss_sums


def _set_hdp_rl_train_mode(model, args):
    if getattr(args, "rl_train_scope", "decoder") != "decoder":
        return
    net = ddp.get_model(model, args.ddp)
    encoder = getattr(net, "encoder", None)
    if encoder is not None:
        encoder.eval()


def _neighbor_future_world(neighbor_future_raw: torch.Tensor):
    mask = neighbor_future_padding_mask(neighbor_future_raw)
    neighbors_future = heading_to_cos_sin(neighbor_future_raw)
    neighbors_future.masked_fill_(mask.unsqueeze(-1), 0.0)
    return neighbors_future


def _policy_observation_inputs(norm_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop supervision-only futures before candidate batch expansion."""
    return {
        key: value
        for key, value in norm_inputs.items()
        if key not in {"ego_agent_future", "neighbor_agents_future"}
    }


def _hdp_rl_step(raw_inputs, model, optimizer, trainable_params, args, ema, aug, profile=False):
    n = args.num_generations
    timing_events = None
    if profile and torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        timing_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        timing_events[0].record()

    raw_inputs = dict(raw_inputs)
    if aug is not None:
        raw_inputs["ego_agent_past"] = heading_to_cos_sin(raw_inputs["ego_agent_past"])
        raw_inputs["goal_pose"] = heading_to_cos_sin(raw_inputs["goal_pose"])
        raw_inputs, ego_future_aug, neighbors_future_aug = aug(
            raw_inputs, raw_inputs["ego_agent_future"], raw_inputs["neighbor_agents_future"]
        )
        raw_inputs["ego_agent_future"] = ego_future_aug
        raw_inputs["neighbor_agents_future"] = neighbors_future_aug
    else:
        raw_inputs["ego_agent_past"] = heading_to_cos_sin(raw_inputs["ego_agent_past"])
        raw_inputs["goal_pose"] = heading_to_cos_sin(raw_inputs["goal_pose"])

    reward_neighbors_raw = _neighbor_future_world(raw_inputs["neighbor_agents_future"])
    norm_inputs = args.observation_normalizer(raw_inputs)
    if getattr(args, "rl_train_scope", "decoder") == "decoder":
        decoder_inputs = {"ego_current_state": norm_inputs["ego_current_state"]}
    else:
        decoder_inputs = _policy_observation_inputs(norm_inputs)
    norm_exp = expand_batch(decoder_inputs, n)
    if getattr(args, "rl_train_scope", "decoder") == "decoder":
        # Keep one route tensor per scene. Decoder compresses it inside the DDP forward,
        # then repeats only the small global condition across the candidate group.
        norm_exp["route_lanes"] = norm_inputs["route_lanes"]
        norm_exp["_global_route_repeat_interleave"] = n
    batch_size = norm_exp["ego_current_state"].shape[0]
    num_scenes = batch_size // n

    # Eq. (AWR) draws actions from the previous policy. The EMA shadow is the stable
    # previous-policy snapshot; the live model receives the reward-weighted update.
    rollout_model = ema.ema if ema is not None else model
    ego_world, rollout_encoding = sample_group(
        rollout_model,
        norm_exp,
        args.rl_noise_scale,
        args.device,
        scene_norm_inputs=norm_inputs,
        group_size=n,
        use_bf16=getattr(args, "amp_dtype", "off") == "bf16",
        sample_steps=getattr(args, "rl_rollout_steps", 6),
        return_encoding=True,
    )
    if timing_events is not None:
        timing_events[1].record()
    _set_hdp_rl_train_mode(model, args)

    reward, reward_metrics = compute_hdp_reward(
        ego_world, raw_inputs, reward_neighbors_raw, num_scenes, n, args
    )
    if timing_events is not None:
        timing_events[2].record()

    reward_weights, valid_sample = compute_reward_weights(
        reward,
        num_scenes,
        n,
        args.rl_reward_normalize,
        getattr(args, "rl_reward_beta", 1.0),
        args.advantage_eps,
    )
    global_valid_count, ddp_world_size = distributed_valid_sample_count(
        valid_sample,
        bool(getattr(args, "ddp", False)),
    )
    has_valid_group = bool(global_valid_count > 0)
    bc_weight = float(getattr(args, "rl_bc_weight", 0.0))
    has_optimizer_update = has_valid_group or bc_weight > 0.0

    optimizer.zero_grad(set_to_none=True)
    if has_optimizer_update:
        decoder_only = getattr(args, "rl_train_scope", "decoder") == "decoder"
        expert_norm_inputs = (
            {
                "ego_current_state": norm_inputs["ego_current_state"],
                "route_lanes": norm_inputs["route_lanes"],
            }
            if decoder_only
            else _policy_observation_inputs(norm_inputs)
        )
        loss_dict = compute_reward_weighted_loss(
            model,
            norm_exp,
            ego_world,
            reward,
            num_scenes,
            n,
            args,
            cached_encoding=(rollout_encoding if decoder_only else None),
            reward_weights=reward_weights,
            valid_sample=valid_sample,
            global_valid_count=global_valid_count,
            ddp_world_size=ddp_world_size,
            expert_norm_inputs=expert_norm_inputs,
            expert_ego_gt=heading_to_cos_sin(raw_inputs["ego_agent_future"]),
            expert_cached_encoding=rollout_encoding[::n] if decoder_only else None,
        )
        loss_dict["loss"].backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, 5)
        optimizer.step()
        if ema is not None:
            ema.update(model)
    else:
        # The paper discards identical-reward groups. Skipping the optimizer as well prevents
        # AdamW weight decay from changing the policy when an entire distributed batch is invalid.
        zero = reward.new_zeros(())
        grad_norm = zero
        loss_dict = {
            "loss": zero,
            "rl_loss": zero,
            "reward_weighted_loss": zero,
            "bc_loss": zero,
            "ego_hdp_diffusion_loss": zero,
            "ego_hdp_waypoint_loss": zero,
            "reward_weight_mean": reward_weights.mean(),
            "reward_weight_max": reward_weights.max(),
            "reward_weight_min": reward_weights.min(),
            "valid_group_fraction": zero,
        }
    if timing_events is not None:
        timing_events[3].record()
        timing_events[3].synchronize()

    grouped_reward = reward.reshape(num_scenes, n)
    endpoints = ego_world[:, -1, :2].reshape(num_scenes, n, 2)
    endpoint_distances = torch.linalg.vector_norm(
        endpoints[:, :, None] - endpoints[:, None, :], dim=-1
    )
    endpoint_diversity = endpoint_distances.sum() / max(num_scenes * n * (n - 1), 1)
    result = {
        "loss": loss_dict["loss"].detach(),
        "rl_loss": loss_dict["rl_loss"].detach(),
        "bc_loss": loss_dict["bc_loss"].detach(),
        "reward_weighted_loss": loss_dict["reward_weighted_loss"],
        "reward_mean": reward.mean().detach(),
        "reward_std": grouped_reward.std(dim=1).mean().detach(),
        "reward_group_range": (grouped_reward.max(dim=1).values - grouped_reward.min(dim=1).values)
        .mean()
        .detach(),
        "reward_max": grouped_reward.max(dim=1).values.mean().detach(),
        "rollout_endpoint_diversity_m": endpoint_diversity.detach(),
        "grad_norm": grad_norm.detach(),
        "ego_hdp_diffusion_loss": loss_dict["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_dict["ego_hdp_waypoint_loss"],
        "reward_weight_mean": loss_dict["reward_weight_mean"].detach(),
        "reward_weight_max": loss_dict["reward_weight_max"].detach(),
        "reward_weight_min": loss_dict["reward_weight_min"].detach(),
        "valid_group_fraction": loss_dict["valid_group_fraction"].detach(),
        "optimizer_step_fraction": reward.new_tensor(float(has_optimizer_update)),
    }
    if timing_events is not None:
        rollout_s = timing_events[0].elapsed_time(timing_events[1]) / 1000.0
        reward_s = timing_events[1].elapsed_time(timing_events[2]) / 1000.0
        update_s = timing_events[2].elapsed_time(timing_events[3]) / 1000.0
        total_s = rollout_s + reward_s + update_s
        result.update(
            {
                "time_rollout_s": rollout_s,
                "time_reward_s": reward_s,
                "time_update_s": update_s,
                "throughput_scenes_per_s": num_scenes / max(total_s, 1e-9),
                "throughput_candidates_per_s": batch_size / max(total_s, 1e-9),
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
        )
    result.update({key: value.detach() for key, value in reward_metrics.items()})
    return result


def train_hdp_rl_epoch(data_loader, model, optimizer, trainable_params, args, ema, aug):
    epoch_loss_sums = {}
    epoch_loss_counts = {}

    model.train()
    _set_hdp_rl_train_mode(model, args)

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="HDP-RL", unit="batch")

    step_log = getattr(args, "wandb_step_log_interval", 0)
    log_step = args.use_wandb and step_log > 0 and ddp.get_rank() == 0
    if not hasattr(args, "_wandb_global_step"):
        args._wandb_global_step = 0

    for batch_idx, raw_inputs in enumerate(data_loader, start=1):
        raw_inputs = {
            key: value.to(args.device, non_blocking=True) for key, value in raw_inputs.items()
        }
        profile_step = (
            args.use_wandb and step_log > 0 and (args._wandb_global_step + 1) % step_log == 0
        )
        step_loss = _hdp_rl_step(
            raw_inputs,
            model,
            optimizer,
            trainable_params,
            args,
            ema,
            aug,
            profile=profile_step,
        )
        args._wandb_global_step += 1
        if log_step and args._wandb_global_step % step_log == 0:
            wandb.log(
                {
                    **{
                        f"train_step/{key}": (value.item() if torch.is_tensor(value) else value)
                        for key, value in step_loss.items()
                    },
                    "train_step/global_step": args._wandb_global_step,
                    "train_step/batch": batch_idx,
                    "train_step/num_batches": len(data_loader),
                    "train_step/epoch_progress": batch_idx / max(len(data_loader), 1),
                    "train_step/lr": optimizer.param_groups[0]["lr"],
                }
            )
        update_epoch_loss_sums(epoch_loss_sums, epoch_loss_counts, step_loss)

    epoch_mean_loss = finalize_epoch_loss_sums(epoch_loss_sums, epoch_loss_counts)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['reward_mean']=:.4f}")
        print(f"{epoch_mean_loss['reward_max']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
