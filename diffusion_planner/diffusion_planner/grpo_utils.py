"""GRPO (Group Relative Policy Optimization) building blocks for the diffusion planner.

This module is the GRPO counterpart of ``train_epoch.py``/``decoder.compute_training_loss``.
The pipeline is:

  1. ``expand_batch``        - replicate every scene in the batch ``N`` times so we can
                               draw a *group* of ``N`` trajectories per scene in a single
                               multi-batch inference pass.
  2. ``sample_group``        - run the model in inference mode with random initial noise to
                               produce ``N`` diverse ego trajectories per scene.
  3. ``compute_collision_reward`` - score each trajectory with a collision-based reward
                               (neighbor collision + optional road-border), reusing the same
                               penalty functions used by the supervised trainer.
  4. ``compute_group_advantages`` - normalise rewards within each group of ``N``.
  5. ``compute_grpo_loss``   - advantage-weighted diffusion (denoising) loss. Treating the
                               diffusion reconstruction loss of a generated trajectory as a
                               proxy for its negative log-likelihood, minimising
                               ``mean(advantage_i * loss_i)`` pushes probability mass toward
                               higher-reward (lower-collision) trajectories.

The policy loss mirrors the supervised HDP loss-space path: the ego target may use
velocity representation, diffusion prediction/supervision can be x_start/noise/score/v,
and the official reward-weighted objective applies exp(beta * normalized_reward).
"""

import random

import torch

from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
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
from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory

# Denser collision sampling for the RL reward than the training-loss default: with only
# 5 eval steps a collision entered and exited between them contributes zero penalty
# (cummax carries only *detected* hits forward). Every 5th step plus the final step closes
# that blind spot; the extra SAT cost is negligible next to the rollout's denoiser passes.
_RL_COLLISION_EVAL_STEPS = sorted(set(range(0, OUTPUT_T, 5)) | {OUTPUT_T - 1})


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
        noise_scale: the *maximum* initial-noise std. Each row draws its own scale uniformly
            from ``U[0, noise_scale]`` (fresh every call), so a group spans a range of
            diversities rather than all sharing one fixed scale. ``noise_scale=0`` stays
            deterministic (temperature 0).
        device: target device.

    Returns:
        ego_world: [B*N, T, 4] ego trajectories in the ego-centric world frame
            (x, y, cos_yaw, sin_yaw).
    """
    was_training = model.training
    model.eval()

    B = norm_inputs["ego_current_state"].shape[0]
    inference_inputs = dict(norm_inputs)
    # Per-row noise std ~ U[0, noise_scale]: each generated trajectory uses its own initial-noise
    # magnitude, drawn fresh each call.
    per_row_scale = torch.rand(B, 1, 1, 1, device=device) * noise_scale
    inference_inputs["sampled_trajectories"] = (
        torch.randn(B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, device=device) * per_row_scale
    )
    inference_inputs["delay"] = torch.zeros(B, dtype=torch.float32, device=device)

    _, outputs = model(inference_inputs)
    ego_world = outputs["prediction"][:, 0].detach()  # [B*N, T, 4]

    if was_training:
        model.train()
    return ego_world


@torch.no_grad()
def compute_collision_reward(
    ego_world: torch.Tensor,
    norm_inputs: dict[str, torch.Tensor],
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collision-based reward for a batch of generated ego trajectories.

    Higher reward == fewer/less-severe collisions. The reward is the negative of the
    accumulated hinge penalties used by the supervised trainer, so the units are directly
    comparable to the ``neighbor_collision_loss`` / ``road_border_loss`` metrics.

    Args:
        ego_world: [B*N, T, 4] generated ego trajectories (world frame).
        norm_inputs: observation-normalized inputs (batch already expanded).
        neighbors_future: [B*N, Pn, T, 4] neighbor futures (world frame, cos/sin).
        neighbors_future_valid: [B*N, Pn, T] validity mask.
        args: namespace with reward weights / margins.

    Returns:
        reward: [B*N] scalar reward per trajectory.
        nc_penalty: [B*N, T] neighbor-collision penalty per timestep (for logging).
        rb_penalty: [B*N, T] road-border penalty per timestep (for logging).
    """
    denorm_inputs = args.observation_normalizer.inverse(norm_inputs)

    ego_edge_points = compute_ego_edge_points(
        ego_world, norm_inputs["ego_shape"], n_interp=args.road_border_n_interp
    )

    nc_penalty = compute_neighbor_collision_penalty(
        ego_edge_points,
        neighbors_future,
        neighbors_future_valid,
        denorm_inputs["neighbor_agents_past"],
        margin_vehicle=args.neighbor_collision_margin_vehicle,
        margin_pedestrian=args.neighbor_collision_margin_pedestrian,
        margin_bicycle=args.neighbor_collision_margin_bicycle,
        eval_steps=_RL_COLLISION_EVAL_STEPS,
    )  # [B*N, T]

    if args.w_road_border > 0.0:
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B*N, T]
    else:
        rb_penalty = torch.zeros_like(nc_penalty)

    # Once a collision/border violation occurs it cannot be "undone": carry each penalty forward
    # as a running max over time (penalty_t = max(penalty_{0..t})). This makes the per-timestep
    # penalty monotonically non-decreasing, so a trajectory that briefly hits then speeds through
    # is never cheaper than one that stops at the hit -- it removes the "plow through quickly"
    # exploit. (Equivalently, the per-step reward -penalty is a running min.)
    nc_penalty = torch.cummax(nc_penalty, dim=-1).values
    rb_penalty = torch.cummax(rb_penalty, dim=-1).values

    reward = -(
        args.w_collision * nc_penalty.sum(dim=-1) + args.w_road_border * rb_penalty.sum(dim=-1)
    )
    return reward, nc_penalty, rb_penalty


@torch.no_grad()
def compute_gt_l2_distance(
    ego_world: torch.Tensor,
    ego_future_gt: torch.Tensor,
) -> torch.Tensor:
    """ADE (mean per-waypoint L2) from each generated trajectory to the scene's own GT future.

    Penalising this distance keeps the generated ego trajectory close to the real recorded
    maneuver, so the policy cannot reward-hack the collision term by, e.g., standing still.

    Args:
        ego_world: [B, T, 4] generated ego trajectories (x, y, cos, sin), ego frame, metres.
        ego_future_gt: [B, Tg, >=2] GT ego future (x, y, ...), same ego-centric frame.

    Returns:
        ade: [B] mean L2 distance over valid (non-zero-padded) GT waypoints.
    """
    gen_xy = ego_world[..., :2]
    gt_xy = ego_future_gt[..., :2]
    T = min(gen_xy.shape[1], gt_xy.shape[1])
    gen_xy, gt_xy = gen_xy[:, :T], gt_xy[:, :T]
    dist = torch.linalg.norm(gen_xy - gt_xy, dim=-1)  # [B, T]
    valid = (gt_xy.abs().sum(dim=-1) > 1e-6).float()  # mask zero-padded GT waypoints
    valid_count = valid.sum(dim=-1)  # [B]
    masked_ade = (dist * valid).sum(dim=-1) / valid_count.clamp_min(1.0)  # [B]
    # An all-"padding" ego GT means a stationary ego (the ego-frame future stays at the
    # origin), not missing data: fall back to the unmasked ADE so stop scenes keep their
    # realism anchor instead of handing out ADE=0 for any generated motion.
    return torch.where(valid_count > 0, masked_ade, dist.mean(dim=-1))


@torch.no_grad()
def compute_kinematic_consistency_penalty(
    ego_world: torch.Tensor,
    ego_agent_past_4d: torch.Tensor,
    ego_current_state: torch.Tensor,
) -> torch.Tensor:
    """Round-trip drift of a generated trajectory through the (accel, curvature) action space.

    A kinematically feasible trajectory survives a trajectory -> action -> trajectory round-trip
    almost unchanged, because the unicycle action space can represent it exactly. An infeasible /
    jerky trajectory (e.g. teleporting, instantaneous heading flips) cannot be reproduced from any
    smooth control sequence, so the reconstruction drifts away. Penalising this drift pushes the
    policy toward dynamically realisable trajectories without needing a reference GT.

    Args:
        ego_world: [B, T, 4] generated ego trajectory (x, y, cos, sin), ego-centric frame, metres.
        ego_agent_past_4d: [B, Th, 4] ego past trajectory (x, y, cos, sin), same frame. The last
            history step is assumed to be the (zeroed) current pose.
        ego_current_state: [B, >=5] ego current state; index 4 is the longitudinal velocity used
            as the integration's initial speed.

    Returns:
        drift: [B] mean per-waypoint L2 distance between the trajectory and its round-trip.
    """
    smoothed = smoothing_future_trajectory(ego_agent_past_4d, ego_current_state, ego_world)
    drift = torch.linalg.norm(ego_world[..., :2] - smoothed[..., :2], dim=-1)  # [B, T]
    return drift.mean(dim=-1)  # [B]


def compute_group_advantages(reward: torch.Tensor, num_scenes: int, n: int, eps: float):
    """Group-relative advantages: normalise rewards within each group of ``n`` samples.

    Args:
        reward: [B*N] rewards laid out as ``num_scenes`` contiguous groups of ``n``.
        num_scenes: number of distinct scenes (B).
        n: group size (N).
        eps: numerical floor for the per-group std.

    Returns:
        advantages: [B*N] normalised advantages (mean 0, std ~1 within each group).
    """
    grouped = reward.view(num_scenes, n)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    valid_group = torch.isfinite(grouped).all(dim=1, keepdim=True) & (std > eps)
    advantages = torch.nan_to_num((grouped - mean) / (std + eps), nan=0.0, posinf=0.0, neginf=0.0)
    advantages = torch.where(valid_group, advantages, torch.zeros_like(grouped))
    return advantages.reshape(-1)


def compute_official_reward_weights(
    reward: torch.Tensor,
    num_scenes: int,
    n: int,
    normalize: str,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = reward.view(num_scenes, n)
    group_std = grouped.std(dim=1, keepdim=True)
    finite_group = torch.isfinite(grouped).all(dim=1, keepdim=True)
    if normalize == "group":
        mean = grouped.mean(dim=1, keepdim=True)
        reward_norm = torch.where(
            group_std > eps,
            (grouped - mean) / (group_std + eps),
            torch.zeros_like(grouped),
        ).reshape(-1)
        valid_sample = finite_group.expand(-1, n).reshape(-1)
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
    alpha, std = sde.marginal_prob(torch.ones_like(all_gt[..., 1:, :]), t_future)
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
    _, decoder_output = model(merged_inputs)
    model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]

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


def compute_grpo_loss(
    model,
    norm_inputs: dict[str, torch.Tensor],
    ego_pseudo_gt: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    advantages: torch.Tensor,
    args,
) -> dict[str, torch.Tensor]:
    """Advantage-weighted diffusion loss (the GRPO policy-gradient surrogate).

    This mirrors the ``x_start`` branch of :func:`decoder.compute_training_loss`, but:
      * the ego target is the *generated* trajectory (``ego_pseudo_gt``) rather than GT,
      * the per-sample ego loss is weighted by its group-relative advantage,
      * neighbor / turn-indicator / penalty terms are dropped (GRPO only shapes the ego policy).

    Args:
        model: the Diffusion_Planner (training mode forward).
        norm_inputs: observation-normalized inputs (batch already expanded to B*N).
        ego_pseudo_gt: [B*N, T, 4] generated ego trajectory (world frame), detached.
        neighbors_future: [B*N, Pn, T, 4] neighbor futures (world frame).
        neighbor_future_mask: [B*N, Pn, T] True where a neighbor timestep is invalid.
        advantages: [B*N] group-relative advantages.
        args: namespace with normalizers, loss coefficients, horizon.

    Returns:
        dict with ``loss`` (scalar, to backprop) plus detached scalar diagnostics.
    """
    loss_terms = _compute_policy_ego_loss_per_sample(
        model,
        norm_inputs,
        ego_pseudo_gt,
        neighbors_future,
        neighbor_future_mask,
        args,
    )
    ego_loss_per_sample = loss_terms["ego_loss_per_sample"]

    # GRPO policy-gradient surrogate: minimise advantage-weighted reconstruction loss.
    grpo_loss = (advantages * ego_loss_per_sample).mean()

    return {
        "loss": grpo_loss,
        "ego_diffusion_loss": ego_loss_per_sample.mean().detach(),
        "ego_hdp_diffusion_loss": loss_terms["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_terms["ego_hdp_waypoint_loss"],
        "mean_advantage": advantages.mean().detach(),
        "abs_advantage": advantages.abs().mean().detach(),
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
