"""HDP reward-weighted RL-Hybrid building blocks for the diffusion planner.

This module is the RL counterpart of ``train_epoch.py``/``decoder.compute_training_loss``.
The single supported pipeline is:

  1. ``expand_batch``        - replicate every scene in the batch ``N`` times so we can
                               draw a *group* of ``N`` trajectories per scene in a single
                               multi-batch inference pass.
  2. ``sample_group``        - run the model in inference mode with random initial noise to
                               produce ``N`` diverse ego trajectories per scene.
  3. ``compute_epdms_style_reward`` - score each trajectory with the Tier IV NPZ adaptation
                               of the official HDP multi-reward setting.
  4. ``compute_official_reward_weighted_loss`` - apply exp(beta * group-normalized reward)
                               to the HDP hybrid diffusion loss.
"""

import random

import torch

from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.loss import (
    hybrid_loss_components,
    inverse_normalize_ego_velocity,
    normalize_ego_velocity,
    sample_diffusion_time,
    vp_supervision_elementwise_loss,
    weighted_waypoint_dpm_loss,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import generate_prefix_mask
from planner_metrics.aggregate import compute_subscores_scene_batch
from planner_metrics.config import RewardConfig

_RL_REWARD_CONFIG = RewardConfig()


def expand_batch(inputs: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    """Replicate every scene ``n`` times along the batch dimension.

    Scene ``b`` ends up occupying rows ``[b * n, (b + 1) * n)`` so a group of ``n`` samples
    for the same scene is contiguous (handy for the group-relative advantage reshape).
    """
    return {key: value.repeat_interleave(n, dim=0) for key, value in inputs.items()}


@torch.no_grad()
def sample_group(
    model,
    norm_inputs: dict[str, torch.Tensor],
    noise_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate one ego trajectory per (already replicated) row via inference sampling.

    Args:
        model: the Diffusion_Planner (or DDP-wrapped) model.
        norm_inputs: observation-normalized inputs, batch dimension already expanded.
        noise_scale: rollout sampling temperature. ``0`` stays deterministic.
        device: target device.

    Returns:
        ego_world: [B*N, T, 4] ego trajectories in the ego-centric world frame
            (x, y, cos_yaw, sin_yaw).
    """
    was_training = model.training
    model.eval()

    B = norm_inputs["ego_current_state"].shape[0]
    inference_inputs = dict(norm_inputs)
    # Official HDP-RL samples rollouts with a fixed sampling temperature.
    if noise_scale == 0.0:
        sampled = torch.zeros(B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, device=device)
    else:
        sampled = (
            torch.randn(B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, device=device)
            * float(noise_scale)
        )
    inference_inputs["sampled_trajectories"] = sampled
    inference_inputs["delay"] = torch.zeros(B, dtype=torch.float32, device=device)

    _, outputs = model(inference_inputs)
    ego_world = outputs["prediction"][:, 0].detach()  # [B*N, T, 4]

    if was_training:
        model.train()
    return ego_world


def _group_reference_path_length(ego_future_gt: torch.Tensor, n: int) -> torch.Tensor:
    gt_xy = ego_future_gt[..., :2]
    valid = gt_xy.abs().sum(dim=-1) > 1e-6
    step_valid = valid[:, 1:] & valid[:, :-1]
    step = torch.linalg.norm(gt_xy[:, 1:] - gt_xy[:, :-1], dim=-1)
    path_len = (step * step_valid.float()).sum(dim=-1)
    return path_len[:, None].expand(-1, n)


def _group_ade_to_gt(
    ego_world_group: torch.Tensor,
    ego_future_gt: torch.Tensor,
) -> torch.Tensor:
    gt_xy = ego_future_gt[:, None, :, :2]
    pred_xy = ego_world_group[..., :2]
    T = min(pred_xy.shape[2], gt_xy.shape[2])
    pred_xy = pred_xy[:, :, :T]
    gt_xy = gt_xy[:, :, :T]
    valid = gt_xy.abs().sum(dim=-1) > 1e-6
    dist = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    valid_count = valid.sum(dim=-1).clamp_min(1)
    masked = (dist * valid.float()).sum(dim=-1) / valid_count
    unmasked = dist.mean(dim=-1)
    return torch.where(valid.any(dim=-1), masked, unmasked)


@torch.no_grad()
def compute_epdms_style_reward(
    ego_world: torch.Tensor,
    scene_inputs: dict[str, torch.Tensor],
    neighbors_future: torch.Tensor,
    num_scenes: int,
    n: int,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Tier IV NPZ reward with official HDP multi-reward semantics.

    The official HDP RL objective consumes a scalar reward but does not require
    the reward backend to be NAVSIM-specific. This reward mirrors the paper's
    multi-reward grouping on the tensors available in Diffusion-Planner NPZs:
    risk, route/GT following, and lane keeping. It reuses the same EPDMS-style
    subscore implementation as validation, then applies only monotonic
    normalizations before the official group-normalized exp weighting.
    """
    ego_group = ego_world.reshape(num_scenes, n, ego_world.shape[1], ego_world.shape[2])

    scene_data = {}
    for key in (
        "ego_shape",
        "neighbor_agents_past",
        "route_lanes",
        "lanes",
        "line_strings",
        "goal_pose",
        "ego_agent_future",
    ):
        if key in scene_inputs:
            scene_data[key] = scene_inputs[key]
    if "ego_agent_future" not in scene_data:
        raise ValueError("epdms_style RL reward requires ego_agent_future in the training batch.")

    scene_data["neighbor_agents_future"] = neighbors_future.reshape(
        num_scenes, n, neighbors_future.shape[1], neighbors_future.shape[2], neighbors_future.shape[3]
    )[:, 0]

    subscores = compute_subscores_scene_batch(ego_group, scene_data, _RL_REWARD_CONFIG)

    safety_raw = subscores["safety"].float()
    collision_score = torch.where(
        safety_raw <= 0.5 * _RL_REWARD_CONFIG.collision_penalty,
        torch.zeros_like(safety_raw),
        torch.exp(safety_raw.clamp(max=0.0)),
    ).clamp(0.0, 1.0)
    ttc_score = subscores["ttc"].float().clamp(0.0, 1.0)
    dac_score = subscores["rb_crossing_gate"].float().clamp(0.0, 1.0)
    red_light_score = torch.where(
        subscores["red_light"].float() <= 0.5 * _RL_REWARD_CONFIG.red_light_penalty,
        torch.zeros_like(safety_raw),
        torch.ones_like(safety_raw),
    )
    kinematic_gate = subscores["kinematic_gate"].float().clamp(0.0, 1.0)
    risk_score = collision_score * ttc_score * dac_score * red_light_score * kinematic_gate

    gt_path_len = _group_reference_path_length(scene_data["ego_agent_future"], n)
    progress_score = (
        subscores["progress"].float().clamp_min(0.0) / gt_path_len.clamp_min(1.0)
    ).clamp(0.0, 1.0)
    gt_follow_score = torch.exp(-_group_ade_to_gt(ego_group, scene_data["ego_agent_future"]) / 2.0)
    comfort_score = torch.exp((subscores["comfort"].float() / 20.0).clamp(min=-20.0, max=0.0))
    feasibility_score = torch.exp(
        (subscores["feasibility"].float() / 10.0).clamp(min=-20.0, max=0.0)
    ) * kinematic_gate
    follow_score = (
        0.45 * progress_score + 0.35 * gt_follow_score + 0.20 * comfort_score * feasibility_score
    ).clamp(0.0, 1.0)

    centerline_score = torch.exp(subscores["centerline"].float().clamp(min=-20.0, max=0.0))
    lane_score = centerline_score.clamp(0.0, 1.0) * subscores["lane_crossing_gate"].float().clamp(
        0.0, 1.0
    )

    reward = (
        args.rl_reward_w_risk * risk_score
        + args.rl_reward_w_follow * follow_score
        + args.rl_reward_w_lane * lane_score
    )

    flat_metrics = {
        "reward_risk_score": risk_score.mean(),
        "reward_follow_score": follow_score.mean(),
        "reward_lane_score": lane_score.mean(),
        "reward_collision_score": collision_score.mean(),
        "reward_ttc_score": ttc_score.mean(),
        "reward_dac_score": dac_score.mean(),
        "reward_progress_score": progress_score.mean(),
        "reward_gt_follow_score": gt_follow_score.mean(),
        "reward_comfort_score": comfort_score.mean(),
        "reward_feasibility_score": feasibility_score.mean(),
        "reward_centerline_score": centerline_score.mean(),
        "neighbor_collision_penalty": (1.0 - collision_score).mean(),
        "road_border_penalty": (1.0 - dac_score).mean(),
    }
    return reward.reshape(-1), flat_metrics


def compute_official_reward_weights(
    reward: torch.Tensor,
    num_scenes: int,
    n: int,
    normalize: str,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        raise ValueError("HDP-RL requires num_generations >= 2 for group reward normalization.")
    grouped = reward.view(num_scenes, n)
    group_std = grouped.std(dim=1, keepdim=True)
    finite_group = torch.isfinite(grouped).all(dim=1, keepdim=True)
    if normalize == "group":
        mean = grouped.mean(dim=1, keepdim=True)
        valid_group = finite_group
        reward_norm = torch.where(
            valid_group,
            (grouped - mean) / (group_std + eps),
            torch.zeros_like(grouped),
        ).reshape(-1)
        valid_sample = valid_group.expand(-1, n).reshape(-1)
    elif normalize == "batch":
        finite = torch.isfinite(reward)
        if finite.any():
            finite_reward = reward[finite]
            std = finite_reward.std()
            if torch.isfinite(std) and std > eps:
                reward_norm = (reward - finite_reward.mean()) / (std + eps)
            else:
                reward_norm = torch.zeros_like(reward)
        else:
            reward_norm = torch.zeros_like(reward)
        valid_sample = finite
    elif normalize == "none":
        reward_norm = reward
        valid_sample = torch.isfinite(reward)
    else:
        raise ValueError(f"Unsupported official_reward_normalize={normalize!r}")
    reward_norm = torch.nan_to_num(reward_norm, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.where(valid_sample, torch.exp(beta * reward_norm), torch.zeros_like(reward_norm))
    return weights.detach(), valid_sample


def _compute_policy_ego_loss_per_sample(
    model,
    norm_inputs: dict[str, torch.Tensor],
    ego_pseudo_gt: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    args,
) -> dict[str, torch.Tensor]:
    vp_model_types = {"x_start", "noise", "score", "v"}
    if args.diffusion_model_type not in vp_model_types:
        raise NotImplementedError(
            f"RL loss only supports VP diffusion model types {sorted(vp_model_types)}, got "
            f"'{args.diffusion_model_type}'."
        )

    norm = args.state_normalizer
    model_type = args.diffusion_model_type
    supervision_type = getattr(args, "diffusion_supervision_type", model_type)
    if supervision_type not in vp_model_types:
        raise ValueError(f"Unsupported diffusion_supervision_type={supervision_type!r}")
    use_velocity = args.use_velocity_representation
    if use_velocity and (model_type != "x_start" or supervision_type != "x_start"):
        raise NotImplementedError(
            "HDP velocity representation is enabled only for x_start prediction with x_start supervision."
        )
    ego_target = ego_pseudo_gt.detach()

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    device = ego_pseudo_gt.device

    ego_current = norm_inputs["ego_current_state"][:, :4]
    neighbors_current = norm_inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    # norm_inputs are observation-normalized; convert back to m/s (see decoder.py, same fix).
    _lv_mean, _lv_std = args.observation_normalizer.stats("ego_current_state")
    longitudinal_velocity = norm_inputs["ego_current_state"][:, 4:5] * float(_lv_std[4]) + float(
        _lv_mean[4]
    )

    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )  # [B, Pn, T+1]

    gt_future = torch.cat([ego_target[:, None, :, :], neighbors_future], dim=1)  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    eps = 1e-3
    t = sample_diffusion_time(
        B,
        device,
        eps,
        getattr(args, "diffusion_time_sample_method", "uniform"),
    )
    t = t.view(B, 1, 1, 1).expand(B, P, T + 1, 1)
    z = torch.randn_like(gt_future)

    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=device)
    prefix_mask = generate_prefix_mask(delay, P, T + 1)  # [B, P, T+1, 1]
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    waypoint_gt = torch.cat(
        [current_states[:, :, None, :], norm(gt_future)], dim=2
    )  # [B, P, T+1, 4]
    all_gt = waypoint_gt.clone()
    if use_velocity:
        ego_velocity_gt = waypoints_to_velocity(ego_target)
        all_gt[:, 0, 1:, :] = normalize_ego_velocity(ego_velocity_gt, norm)
    all_gt[:, 1:] = all_gt[:, 1:].masked_fill(neighbor_mask.unsqueeze(-1), 0.0)

    model_ref = getattr(model, "module", model)
    sde = getattr(model_ref, "sde", None)
    if sde is None:
        sde = VPSDE_linear()
    t_future = t[..., 1:, :]
    # Same schedule values as marginal_prob(ones_like(...)) without the full-size ones tensor.
    alpha = sde.marginal_alpha(t_future)
    std = sde.marginal_prob_std(t_future)
    x0_target = all_gt[:, :, 1:, :]
    xT_future = alpha * x0_target + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT_future], dim=2)
    xT = torch.where(prefix_mask, all_gt, xT)
    xT_future = xT[:, :, 1:, :]

    merged_inputs = {
        **norm_inputs,
        "gt_trajectories": all_gt,
        "turn_indicator_trajectories": waypoint_gt,
        "sampled_trajectories": xT,
        "diffusion_time": t,
        "prefix_mask": prefix_mask,
    }
    # Same amp scoping as decoder.compute_training_loss: forward only, losses stay fp32.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=getattr(args, "amp_dtype", "off") == "bf16",
    ):
        _, decoder_output = model(merged_inputs)
    model_output = decoder_output["model_output"][:, :, 1:, :].float()  # [B, P, T, 4]

    pred_x_start = sde.transform(f"{model_type}->x_start", model_output, t_future, xT_future)
    supervised_prediction = sde.transform(
        f"{model_type}->{supervision_type}", model_output, t_future, xT_future
    )

    if use_velocity:
        # Guarded at function entry: velocity mode implies model_type == supervision_type
        # == "x_start" (mirrors decoder.compute_training_loss).
        ego_diffusion_loss = torch.sum(
            (supervised_prediction[:, 0] - x0_target[:, 0]) ** 2, dim=-1
        )
        ego_pred_velocity_raw = inverse_normalize_ego_velocity(pred_x_start[:, 0], norm)
        _, ego_waypoint_loss = hybrid_loss_components(
            pred_x_start[:, 0],
            x0_target[:, 0],
            ego_pred_velocity_raw,
            ego_target,
            W=args.hybrid_loss_window,
        )
        ego_reconstruction = ego_diffusion_loss + args.planning_hybrid_loss * ego_waypoint_loss
        ego_valid = ~prefix_mask[:, 0, 1 : 1 + args.ego_prediction_horizon, 0]
        ego_loss_per_sample = (
            ego_reconstruction[:, : args.ego_prediction_horizon].masked_fill(~ego_valid, 0.0).sum(dim=-1)
            / ego_valid.sum(dim=-1).clamp_min(1)
        )
    elif supervision_type == "x_start":
        dpm_loss = weighted_waypoint_dpm_loss(
            pred_x_start,
            x0_target,
            longitudinal_velocity,
            args.coeff_position_lat_loss,
            args.coeff_position_lon_loss,
            args.coeff_heading_l2_loss,
            args.coeff_velocity,
            args.coeff_timestep,
        )
        ego_loss_per_sample = dpm_loss[:, 0, : args.ego_prediction_horizon].mean(dim=-1)
        ego_diffusion_loss = torch.zeros_like(ego_loss_per_sample)
        ego_waypoint_loss = torch.zeros_like(ego_loss_per_sample)
    else:
        dpm_loss = vp_supervision_elementwise_loss(
            supervised_prediction, z, std, supervision_type, sde, t_future, xT_future
        )
        ego_loss_per_sample = dpm_loss[:, 0, : args.ego_prediction_horizon].mean(dim=-1)
        ego_diffusion_loss = torch.zeros_like(ego_loss_per_sample)
        ego_waypoint_loss = torch.zeros_like(ego_loss_per_sample)

    return {
        "ego_loss_per_sample": ego_loss_per_sample,
        "ego_hdp_diffusion_loss": ego_diffusion_loss.mean().detach(),
        "ego_hdp_waypoint_loss": ego_waypoint_loss.mean().detach(),
    }


def compute_official_reward_weighted_loss(
    model,
    norm_inputs: dict[str, torch.Tensor],
    ego_pseudo_gt: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    reward: torch.Tensor,
    num_scenes: int,
    n: int,
    args,
) -> dict[str, torch.Tensor]:
    loss_terms = _compute_policy_ego_loss_per_sample(
        model,
        norm_inputs,
        ego_pseudo_gt,
        neighbors_future,
        neighbor_future_mask,
        args,
    )
    ego_loss_per_sample = loss_terms["ego_loss_per_sample"]

    weights, valid_sample = compute_official_reward_weights(
        reward,
        num_scenes,
        n,
        args.official_reward_normalize,
        getattr(args, "official_reward_beta", 1.0),
        args.advantage_eps,
    )
    valid_weight = valid_sample.to(ego_loss_per_sample.dtype)
    denom = valid_weight.sum().clamp_min(1.0)
    loss = (weights * valid_weight * ego_loss_per_sample).sum() / denom

    return {
        "loss": loss,
        "official_reward_weighted_loss": loss.detach(),
        "ego_diffusion_loss": ego_loss_per_sample.mean().detach(),
        "ego_hdp_diffusion_loss": loss_terms["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_terms["ego_hdp_waypoint_loss"],
        "official_reward_weight_mean": weights.mean().detach(),
        "official_reward_weight_max": weights.max().detach(),
        "official_reward_weight_min": weights.min().detach(),
        "official_valid_group_fraction": valid_sample.view(num_scenes, n)[:, 0].float().mean().detach(),
    }
