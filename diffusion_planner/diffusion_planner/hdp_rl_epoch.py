"""HDP reward-weighted RL-Hybrid epoch loop.

The only supported RL path is the HDP reward-weighted hybrid loss:

  1. sample a group of trajectories per scene;
  2. score them with the HDP risk/follow/lane reward adapted to Tier IV NPZ signals;
  3. group-normalize the reward and weight the HDP hybrid diffusion loss with
     exp(beta * normalized_reward).
"""

import copy
import time

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


@torch.no_grad()
def commit_ema_policy_update(model, ema, optimizer, use_ddp: bool):
    """Commit one conservative policy iteration and reset the live proposal."""
    live = ddp.get_model(model, use_ddp)
    old_policy = ddp.get_model(ema.ema, use_ddp)
    first_param = next(live.parameters())
    delta_sq = first_param.new_zeros((), dtype=torch.float32)
    reference_sq = first_param.new_zeros((), dtype=torch.float32)
    for live_param, old_param in zip(live.parameters(), old_policy.parameters(), strict=True):
        if not live_param.requires_grad:
            continue
        delta_sq.add_((live_param.detach().float() - old_param.detach().float()).square().sum())
        reference_sq.add_(old_param.detach().float().square().sum())
    proposal_relative_l2 = (delta_sq / reference_sq.clamp_min(1e-12)).sqrt()

    ema.update(model)
    accepted_policy = ddp.get_model(ema.ema, use_ddp)
    live.load_state_dict(accepted_policy.state_dict())
    optimizer.state.clear()
    return proposal_relative_l2


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
        getattr(args, "rl_reward_beta", 0.5),
        args.advantage_eps,
    )
    global_valid_count, ddp_world_size = distributed_valid_sample_count(
        valid_sample,
        bool(getattr(args, "ddp", False)),
    )
    has_valid_group = bool(global_valid_count > 0)
    bc_weight = float(getattr(args, "rl_bc_weight", 0.0))
    has_optimizer_update = has_valid_group or bc_weight > 0.0

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
        loss_sums = {}
        grad_norm = reward.new_zeros(())
        grad_norm_max = reward.new_zeros(())
        grad_clip_count = reward.new_zeros(())
        updates_per_rollout = int(getattr(args, "rl_updates_per_rollout", 1))
        expert_ego_gt = heading_to_cos_sin(raw_inputs["ego_agent_future"])
        for _ in range(updates_per_rollout):
            optimizer.zero_grad(set_to_none=True)
            update_loss = compute_reward_weighted_loss(
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
                expert_ego_gt=expert_ego_gt,
                expert_cached_encoding=rollout_encoding[::n] if decoder_only else None,
            )
            update_loss["loss"].backward()
            update_grad_norm = nn.utils.clip_grad_norm_(trainable_params, 5).detach()
            grad_norm = grad_norm + update_grad_norm
            grad_norm_max = torch.maximum(grad_norm_max, update_grad_norm)
            grad_clip_count = grad_clip_count + (update_grad_norm > 5).to(reward.dtype)
            optimizer.step()
            for key, value in update_loss.items():
                loss_sums[key] = loss_sums.get(key, reward.new_zeros(())) + value.detach()
        loss_dict = {key: value / updates_per_rollout for key, value in loss_sums.items()}
        grad_norm = grad_norm / updates_per_rollout
        grad_clip_fraction = grad_clip_count / updates_per_rollout
        optimizer_steps = updates_per_rollout
    else:
        # The paper discards identical-reward groups. Skipping the optimizer as well prevents
        # AdamW weight decay from changing the policy when an entire distributed batch is invalid.
        zero = reward.new_zeros(())
        grad_norm = zero
        grad_norm_max = zero
        grad_clip_fraction = zero
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
        optimizer_steps = 0
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
        "grad_norm_max_per_rollout": grad_norm_max.detach(),
        "grad_clip_fraction": grad_clip_fraction.detach(),
        "ego_hdp_diffusion_loss": loss_dict["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_dict["ego_hdp_waypoint_loss"],
        "reward_weight_mean": loss_dict["reward_weight_mean"].detach(),
        "reward_weight_max": loss_dict["reward_weight_max"].detach(),
        "reward_weight_min": loss_dict["reward_weight_min"].detach(),
        "valid_group_fraction": loss_dict["valid_group_fraction"].detach(),
        "optimizer_step_fraction": reward.new_tensor(float(has_optimizer_update)),
        "optimizer_steps_per_rollout": reward.new_tensor(float(optimizer_steps)),
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
                "throughput_candidate_updates_per_s": batch_size
                * optimizer_steps
                / max(total_s, 1e-9),
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
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

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    local_scene_count = 0

    for batch_idx, raw_inputs in enumerate(data_loader, start=1):
        local_scene_count += int(raw_inputs["ego_current_state"].shape[0])
        raw_inputs = {
            key: value.to(args.device, non_blocking=True) for key, value in raw_inputs.items()
        }
        requested_updates = int(getattr(args, "rl_updates_per_rollout", 1))
        next_log_step = args._wandb_global_step + requested_updates
        profile_step = (
            args.use_wandb
            and step_log > 0
            and next_log_step // step_log > args._wandb_global_step // step_log
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
        optimizer_steps = int(step_loss["optimizer_steps_per_rollout"].item())
        previous_global_step = args._wandb_global_step
        args._wandb_global_step += optimizer_steps
        crossed_log_interval = (
            step_log > 0 and args._wandb_global_step // step_log > previous_global_step // step_log
        )
        if log_step and crossed_log_interval:
            wandb.log(
                {
                    **{
                        f"train_step/{key}": (value.item() if torch.is_tensor(value) else value)
                        for key, value in step_loss.items()
                    },
                    "train_step/global_step": args._wandb_global_step,
                    "optimizer_step": args._wandb_global_step,
                    "train_step/batch": batch_idx,
                    "train_step/num_batches": len(data_loader),
                    "train_step/epoch_progress": batch_idx / max(len(data_loader), 1),
                    "train_step/lr": optimizer.param_groups[0]["lr"],
                }
            )
        update_epoch_loss_sums(epoch_loss_sums, epoch_loss_counts, step_loss)

    epoch_mean_loss = finalize_epoch_loss_sums(epoch_loss_sums, epoch_loss_counts)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_stats = torch.tensor(
        [time.perf_counter() - wall_start, float(local_scene_count)],
        dtype=torch.float64,
        device=device,
    )
    if args.ddp:
        wall_elapsed = wall_stats[:1].clone()
        global_scenes = wall_stats[1:].clone()
        torch.distributed.all_reduce(wall_elapsed, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(global_scenes, op=torch.distributed.ReduceOp.SUM)
        wall_stats = torch.cat((wall_elapsed, global_scenes))
    epoch_mean_loss["wall_time_s"] = wall_stats[0].float()
    epoch_mean_loss["wall_throughput_global_scenes_per_s"] = (
        wall_stats[1] / wall_stats[0].clamp_min(1e-9)
    ).float()
    if device.type == "cuda":
        memory_peaks = torch.tensor(
            [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
            dtype=torch.float64,
            device=device,
        ) / (1024**3)
        if args.ddp:
            torch.distributed.all_reduce(memory_peaks, op=torch.distributed.ReduceOp.MAX)
        epoch_mean_loss["max_memory_allocated_gb"] = memory_peaks[0].float()
        epoch_mean_loss["max_memory_reserved_gb"] = memory_peaks[1].float()

    if ema is not None:
        proposal_relative_l2 = commit_ema_policy_update(model, ema, optimizer, args.ddp)
        epoch_mean_loss["policy_proposal_relative_l2"] = proposal_relative_l2
        epoch_mean_loss["policy_accepted_relative_l2"] = proposal_relative_l2 * float(
            getattr(args, "rl_ema_update_rate", 0.05)
        )

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['reward_mean']=:.4f}")
        print(f"{epoch_mean_loss['reward_max']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]


@torch.no_grad()
def validate_hdp_reward_policy(data_loader, model, args):
    """Evaluate the rollout policy reward on a fixed held-out scene distribution."""
    n = int(args.rl_eval_num_generations)
    device = torch.device(args.device)
    eval_reward_args = copy.copy(args)
    for name, default in (
        ("risk", 1.0),
        ("follow", 3.0),
        ("lane", 2.5),
        ("progress", 3.0),
    ):
        setattr(
            eval_reward_args,
            f"rl_reward_w_{name}",
            float(getattr(args, f"rl_eval_reward_w_{name}", default)),
        )
    # reward sum, candidate count, per-group max sum, scene count, invalid reward count,
    # invalid diagnostic count. Invalid values are reported after the collective so every
    # rank exits together instead of leaving peers blocked in all_reduce.
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    metric_totals = None
    metric_keys = None
    iterator = (
        tqdm(data_loader, desc="RL-reward-valid", unit="batch")
        if ddp.get_rank() == 0
        else data_loader
    )
    rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        seed = int(getattr(args, "seed", 3407)) + 100_003 + ddp.get_rank()
        torch.manual_seed(seed)
        for raw_inputs in iterator:
            raw_inputs = {
                key: value.to(device, non_blocking=True) for key, value in raw_inputs.items()
            }
            raw_inputs["ego_agent_past"] = heading_to_cos_sin(raw_inputs["ego_agent_past"])
            raw_inputs["goal_pose"] = heading_to_cos_sin(raw_inputs["goal_pose"])
            reward_neighbors = _neighbor_future_world(raw_inputs["neighbor_agents_future"])
            norm_inputs = args.observation_normalizer(raw_inputs)
            decoder_inputs = {"ego_current_state": norm_inputs["ego_current_state"]}
            norm_exp = expand_batch(decoder_inputs, n)
            norm_exp["route_lanes"] = norm_inputs["route_lanes"]
            norm_exp["_global_route_repeat_interleave"] = n
            ego_world = sample_group(
                model,
                norm_exp,
                args.rl_eval_noise_scale,
                device,
                scene_norm_inputs=norm_inputs,
                group_size=n,
                use_bf16=getattr(args, "amp_dtype", "off") == "bf16",
                sample_steps=args.diffusion_sample_steps,
            )
            num_scenes = raw_inputs["ego_current_state"].shape[0]
            reward, reward_metrics = compute_hdp_reward(
                ego_world, raw_inputs, reward_neighbors, num_scenes, n, eval_reward_args
            )
            finite_reward = torch.isfinite(reward)
            safe_reward = torch.where(finite_reward, reward, torch.zeros_like(reward))
            grouped = safe_reward.view(num_scenes, n)
            candidate_count = reward.numel()
            totals[0] += safe_reward.double().sum()
            totals[1] += candidate_count
            totals[2] += grouped.max(dim=1).values.double().sum()
            totals[3] += num_scenes
            totals[4] += (~finite_reward).sum()
            current_keys = tuple(sorted(reward_metrics))
            if metric_keys is None:
                metric_keys = current_keys
                metric_totals = torch.zeros(len(metric_keys), dtype=torch.float64, device=device)
            elif current_keys != metric_keys:
                raise RuntimeError("RL reward metric keys changed between validation batches")
            for offset, key in enumerate(metric_keys):
                metric = reward_metrics[key]
                totals[5] += (~torch.isfinite(metric)).sum()
                metric_totals[offset] += (
                    torch.nan_to_num(metric.double(), nan=0.0, posinf=0.0, neginf=0.0)
                    * candidate_count
                )

    if metric_totals is None or metric_keys is None:
        raise ValueError("RL reward validation loader produced no batches")
    if args.ddp:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(metric_totals, op=torch.distributed.ReduceOp.SUM)
    if totals[4].item() > 0 or totals[5].item() > 0:
        raise FloatingPointError(
            "RL reward validation produced non-finite values: "
            f"rewards={int(totals[4].item())}, diagnostics={int(totals[5].item())}"
        )
    candidate_count = totals[1].clamp_min(1.0)
    scene_count = totals[3].clamp_min(1.0)
    return {
        "mean": (totals[0] / candidate_count).float(),
        "group_max": (totals[2] / scene_count).float(),
        **{
            key.removeprefix("reward_").removesuffix("_score"): (
                metric_totals[offset] / candidate_count
            ).float()
            for offset, key in enumerate(metric_keys)
        },
    }
