import math
import random
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.comfort_constants import (
    MAX_ABS_LAT_ACCEL,
    MAX_ABS_LON_JERK,
    MAX_ABS_MAG_JERK,
    MAX_ABS_YAW_ACCEL,
    MAX_ABS_YAW_RATE,
    MAX_LON_ACCEL,
    MIN_LON_ACCEL,
)
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    hybrid_loss,
    loss_func,
    make_turn_indicator_gt,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.unicycle_accel_curvature import ratan2
from diffusion_planner.model.module.dfp import (
    DFPDiT,
    DFPFinalLayer,
    TimestepEmbedder,
    inverse_normalize_ego_trajectory,
    inverse_normalize_trajectory_by_agent,
    normalize_ego_trajectory,
    normalize_neighbor_trajectory,
    normalize_trajectory_by_agent,
    vp_alpha_sigma,
)
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def generate_prefix_mask(delay: torch.Tensor, num_agents: int, max_len: int) -> torch.Tensor:
    """Generates a prefix mask based on a delay tensor.

    Args:
        delay: A 1D tensor of shape (B,) with delay values.
        num_agents: The number of agents (P).
        max_len: The maximum length of the sequence (T+1 or T_plus_1).

    Returns:
        A 4D boolean tensor of shape (B, num_agents, max_len, 1) where mask[i, :, j, 0] is True if j <= delay[i].
    """
    # Create steps tensor (1, 1, max_len, 1)
    steps = torch.arange(max_len, device=delay.device).view(1, 1, -1, 1)
    # Reshape delay to (B, 1, 1, 1) for broadcasting
    reshaped_delay = delay.reshape(delay.shape[0], 1, 1, 1)
    # Perform the comparison, result is (B, 1, max_len, 1)
    mask = steps <= reshaped_delay
    ego_mask = mask.expand(-1, 1, -1, -1)
    neighbor_mask = torch.zeros(
        (delay.shape[0], num_agents - 1, max_len, 1), dtype=torch.bool, device=delay.device
    )
    return torch.cat([ego_mask, neighbor_mask], dim=1)


def replace_current_state(x: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
    """Return a trajectory tensor with the first timestep replaced."""
    return torch.cat([current_states[:, :, None, :], x[:, :, 1:, :]], dim=2)


def add_current_xy(future: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
    """Add current xy position to future xy channels without mutating the input."""
    xy = future[..., :2] + current_states[:, :, None, :2]
    return torch.cat([xy, future[..., 2:]], dim=-1)


def _angle_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    diff = lhs - rhs
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _time_gradient(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[1] <= 1:
        return torch.zeros_like(values)
    grad = torch.empty_like(values)
    grad[:, 0] = (values[:, 1] - values[:, 0]) / dt
    grad[:, -1] = (values[:, -1] - values[:, -2]) / dt
    if values.shape[1] > 2:
        grad[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * dt)
    return grad


def _heading_gradient(heading: torch.Tensor, dt: float) -> torch.Tensor:
    if heading.shape[1] <= 1:
        return torch.zeros_like(heading)
    grad = torch.empty_like(heading)
    grad[:, 0] = _angle_diff(heading[:, 1], heading[:, 0]) / dt
    grad[:, -1] = _angle_diff(heading[:, -1], heading[:, -2]) / dt
    if heading.shape[1] > 2:
        grad[:, 1:-1] = _angle_diff(heading[:, 2:], heading[:, :-2]) / (2.0 * dt)
    return grad


def compute_comfort_surrogate_loss(ego_traj: torch.Tensor, dt: float = 0.1) -> dict[str, torch.Tensor]:
    """Differentiable surrogate for the PDMS/NAVSIM binary comfort metric.

    PDMS comfort is 1 only when all six kinematic signals stay within fixed
    NAVSIM bounds at every timestep. This loss uses the same physical signals
    and bounds, but replaces the hard all-timestep boolean with normalized
    squared hinge violations so it can train the planner.
    """
    xy = ego_traj[..., :2]
    heading = ratan2(ego_traj[..., 3], ego_traj[..., 2])

    vx = _time_gradient(xy[..., 0], dt)
    vy = _time_gradient(xy[..., 1], dt)
    ax = _time_gradient(vx, dt)
    ay = _time_gradient(vy, dt)

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    acc_lon = ax * cos_h + ay * sin_h
    acc_lat = -ax * sin_h + ay * cos_h
    acc_mag = torch.sqrt(torch.clamp(ax * ax + ay * ay, min=1.0e-8))

    jerk_mag = _time_gradient(acc_mag, dt)
    jerk_lon = _time_gradient(acc_lon, dt)
    yaw_rate = _heading_gradient(heading, dt)
    yaw_accel = _time_gradient(yaw_rate, dt)

    def upper(values: torch.Tensor, bound: float) -> torch.Tensor:
        return (torch.relu(values - bound) / max(abs(bound), 1.0e-6)).pow(2).mean()

    def lower(values: torch.Tensor, bound: float) -> torch.Tensor:
        return (torch.relu(bound - values) / max(abs(bound), 1.0e-6)).pow(2).mean()

    def abs_upper(values: torch.Tensor, bound: float) -> torch.Tensor:
        return (torch.relu(values.abs() - bound) / max(abs(bound), 1.0e-6)).pow(2).mean()

    lon_accel_loss = upper(acc_lon, MAX_LON_ACCEL) + lower(acc_lon, MIN_LON_ACCEL)
    lat_accel_loss = abs_upper(acc_lat, MAX_ABS_LAT_ACCEL)
    mag_jerk_loss = abs_upper(jerk_mag, MAX_ABS_MAG_JERK)
    lon_jerk_loss = abs_upper(jerk_lon, MAX_ABS_LON_JERK)
    yaw_accel_loss = abs_upper(yaw_accel, MAX_ABS_YAW_ACCEL)
    yaw_rate_loss = abs_upper(yaw_rate, MAX_ABS_YAW_RATE)
    total = (
        lon_accel_loss
        + lat_accel_loss
        + mag_jerk_loss
        + lon_jerk_loss
        + yaw_accel_loss
        + yaw_rate_loss
    )
    return {
        "comfort_loss": total,
        "comfort_lon_accel_loss": lon_accel_loss.detach(),
        "comfort_lat_accel_loss": lat_accel_loss.detach(),
        "comfort_mag_jerk_loss": mag_jerk_loss.detach(),
        "comfort_lon_jerk_loss": lon_jerk_loss.detach(),
        "comfort_yaw_accel_loss": yaw_accel_loss.detach(),
        "comfort_yaw_rate_loss": yaw_rate_loss.detach(),
    }


def compute_progress_surrogate_loss(
    ego_pred: torch.Tensor,
    ego_gt: torch.Tensor,
    threshold: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Differentiable under-progress surrogate for the PDMS ego-progress term.

    The exact metric projects the predicted start/end points onto the GT
    LineString and gates out references with <=5 m progress. That projection is
    not convenient for backprop, so this surrogate uses two smooth proxies on
    the same gated samples: path-length ratio and final displacement along the
    GT start-to-end direction. The ordinary supervised trajectory loss keeps
    the path shape anchored, while this term specifically discourages short
    plans that under-drive the expert route.
    """
    pred_xy = ego_pred[..., :2]
    gt_xy = ego_gt[..., :2]
    eps = 1.0e-6

    pred_step = torch.linalg.vector_norm(pred_xy[:, 1:] - pred_xy[:, :-1], dim=-1)
    gt_step = torch.linalg.vector_norm(gt_xy[:, 1:] - gt_xy[:, :-1], dim=-1)
    pred_len = pred_step.sum(dim=-1)
    gt_len = gt_step.sum(dim=-1)

    gt_disp = gt_xy[:, -1] - gt_xy[:, 0]
    pred_disp = pred_xy[:, -1] - pred_xy[:, 0]
    gt_chord = torch.linalg.vector_norm(gt_disp, dim=-1).clamp_min(eps)
    gt_dir = gt_disp / gt_chord.unsqueeze(-1)
    pred_forward = (pred_disp * gt_dir).sum(dim=-1)

    valid = gt_len > threshold
    if not bool(valid.any().item()):
        zero = torch.tensor(0.0, device=ego_pred.device, dtype=ego_pred.dtype)
        return {
            "progress_loss": zero,
            "progress_under_path_loss": zero,
            "progress_under_forward_loss": zero,
            "progress_over_path_loss": zero,
            "progress_valid_fraction": zero,
        }

    gt_len_safe = gt_len.clamp_min(eps)
    under_path = torch.relu((gt_len - pred_len) / gt_len_safe)
    under_forward = torch.relu((gt_chord - pred_forward) / gt_chord)
    over_path = torch.relu((pred_len - 1.10 * gt_len) / gt_len_safe)

    under_path_loss = under_path[valid].pow(2).mean()
    under_forward_loss = under_forward[valid].pow(2).mean()
    over_path_loss = over_path[valid].pow(2).mean()
    total = under_path_loss + under_forward_loss + 0.1 * over_path_loss
    return {
        "progress_loss": total,
        "progress_under_path_loss": under_path_loss.detach(),
        "progress_under_forward_loss": under_forward_loss.detach(),
        "progress_over_path_loss": over_path_loss.detach(),
        "progress_valid_fraction": valid.float().mean().detach(),
    }


def build_dfp_training_inputs(
    inputs: dict[str, torch.Tensor],
    ego_future: torch.Tensor,
    norm: StateNormalizer,
    args: Namespace,
    eps: float = 1e-3,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build paper-style DFP chunk noising inputs without replacing original diffusion inputs."""
    B, T, _ = ego_future.shape
    chunk_len = args.dfp_chunk_len
    history_len = args.dfp_history_len
    assert history_len == chunk_len, "DFP expects history_len == chunk_len"
    assert T % chunk_len == 0, "future_len must be divisible by dfp_chunk_len"
    future_chunks = T // chunk_len

    raw_inputs = args.observation_normalizer.inverse(inputs) if hasattr(args, "observation_normalizer") else inputs
    history = _last_history_chunk(raw_inputs["ego_agent_past"], history_len)

    current = raw_inputs["ego_current_state"][:, :4]
    current = current[:, None, :].expand(B, chunk_len, 4)

    history = normalize_ego_trajectory(norm, history)
    current = normalize_ego_trajectory(norm, current)
    future = normalize_ego_trajectory(norm, ego_future)

    clean_chunks = torch.cat(
        [
            history[:, None],
            current[:, None],
            future.reshape(B, future_chunks, chunk_len, 4),
        ],
        dim=1,
    )

    beta = torch.distributions.Beta(args.dfp_history_beta_a, args.dfp_history_beta_b)
    history_t = beta.sample((B, 1)).to(clean_chunks.device, dtype=clean_chunks.dtype)
    history_t = torch.clamp(history_t, eps, 1.0 - eps)
    current_t = torch.zeros(B, 1, device=clean_chunks.device, dtype=clean_chunks.dtype)
    future_t = torch.rand(B, future_chunks, device=clean_chunks.device, dtype=clean_chunks.dtype)
    future_t = future_t * (1.0 - eps) + eps
    t = torch.cat([history_t, current_t, future_t], dim=1)

    alpha, sigma = vp_alpha_sigma(t)
    sampled_chunks = alpha[:, :, None, None] * clean_chunks + sigma[:, :, None, None] * torch.randn_like(clean_chunks)
    sampled_chunks[:, 1] = clean_chunks[:, 1]

    return {"dfp_sampled_chunks": sampled_chunks, "dfp_diffusion_time": t}, clean_chunks


def _last_history_chunk(past: torch.Tensor, history_len: int) -> torch.Tensor:
    if past.shape[-2] >= history_len + 1:
        return past[..., -history_len - 1 : -1, :4]
    pad_shape = list(past.shape[:-2]) + [history_len + 1 - past.shape[-2], 4]
    pad = past[..., :1, :4].expand(*pad_shape)
    return torch.cat([pad, past[..., :4]], dim=-2)[..., -history_len - 1 : -1, :]


def build_joint_dfp_training_inputs(
    inputs: dict[str, torch.Tensor],
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    norm: StateNormalizer,
    args: Namespace,
    eps: float = 1e-3,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    B, T, _ = ego_future.shape
    Pn = neighbors_future.shape[1]
    chunk_len = args.dfp_chunk_len
    history_len = args.dfp_history_len
    assert history_len == chunk_len, "Joint DFP expects history_len == chunk_len"
    assert T % chunk_len == 0, "future_len must be divisible by dfp_chunk_len"
    future_chunks = T // chunk_len

    raw_inputs = args.observation_normalizer.inverse(inputs) if hasattr(args, "observation_normalizer") else inputs
    ego_history = _last_history_chunk(raw_inputs["ego_agent_past"], history_len)[:, None]
    neighbor_history = _last_history_chunk(
        raw_inputs["neighbor_agents_past"][:, :Pn, :, :4], history_len
    )
    history = torch.cat([ego_history, neighbor_history], dim=1)

    ego_current = raw_inputs["ego_current_state"][:, :4]
    neighbor_current = raw_inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    current_states = torch.cat([ego_current[:, None], neighbor_current], dim=1)
    current = current_states[:, :, None, :].expand(B, 1 + Pn, chunk_len, 4)

    future = torch.cat([ego_future[:, None], neighbors_future], dim=1)
    future_chunks_tensor = future.reshape(B, 1 + Pn, future_chunks, chunk_len, 4)

    history = normalize_trajectory_by_agent(norm, history)
    current = normalize_trajectory_by_agent(norm, current)
    future_chunks_tensor = normalize_trajectory_by_agent(norm, future_chunks_tensor)

    clean_chunks = torch.cat([history[:, :, None], current[:, :, None], future_chunks_tensor], dim=2)

    ego_agent_valid = torch.ones(B, 1, dtype=torch.bool, device=clean_chunks.device)
    neighbor_current_valid = torch.sum(torch.ne(neighbor_current[..., :4], 0), dim=-1) != 0
    agent_valid = torch.cat([ego_agent_valid, neighbor_current_valid], dim=1)

    ego_history_valid = torch.ones(B, 1, chunk_len, dtype=torch.bool, device=clean_chunks.device)
    neighbor_history_valid = torch.sum(torch.ne(neighbor_history[..., :4], 0), dim=-1) != 0
    history_valid = torch.cat([ego_history_valid, neighbor_history_valid], dim=1)
    current_valid = agent_valid[:, :, None].expand(B, 1 + Pn, chunk_len)
    ego_future_valid = torch.ones(B, 1, T, dtype=torch.bool, device=clean_chunks.device)
    future_valid = torch.cat([ego_future_valid, ~neighbor_future_mask], dim=1).reshape(
        B, 1 + Pn, future_chunks, chunk_len
    )
    valid_mask = torch.cat(
        [history_valid[:, :, None], current_valid[:, :, None], future_valid], dim=2
    )
    clean_chunks = clean_chunks.masked_fill(~valid_mask[..., None], 0.0)

    beta = torch.distributions.Beta(args.dfp_history_beta_a, args.dfp_history_beta_b)
    history_t = beta.sample((B, 1 + Pn, 1)).to(clean_chunks.device, dtype=clean_chunks.dtype)
    history_t = history_t.clamp(eps, 1.0 - eps)
    current_t = torch.zeros(B, 1 + Pn, 1, device=clean_chunks.device, dtype=clean_chunks.dtype)
    future_t = torch.rand(
        B, 1 + Pn, future_chunks, device=clean_chunks.device, dtype=clean_chunks.dtype
    )
    future_t = future_t * (1.0 - eps) + eps
    t = torch.cat([history_t, current_t, future_t], dim=2)

    alpha, sigma = vp_alpha_sigma(t)
    sampled_chunks = alpha[..., None, None] * clean_chunks + sigma[..., None, None] * torch.randn_like(clean_chunks)
    sampled_chunks[:, :, 1] = clean_chunks[:, :, 1]
    sampled_chunks = sampled_chunks.masked_fill(~valid_mask[..., None], 0.0)

    return (
        {
            "joint_sampled_chunks": sampled_chunks,
            "joint_diffusion_time": t,
            "joint_valid_mask": valid_mask,
            "joint_agent_valid_mask": agent_valid,
        },
        clean_chunks,
        valid_mask,
    )


def _modulate_joint(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class JointTemporalAgentBlock(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_agent = nn.LayerNorm(dim)
        self.agent_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))

    @staticmethod
    def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clone()
        all_masked = torch.all(mask, dim=1)
        if torch.any(all_masked):
            mask[all_masked, 0] = False
        return mask

    def forward(self, x, scene, y, agent_valid_mask, token_valid_mask, scene_mask):
        B, P, C, D = x.shape
        shift_time, scale_time, shift_mlp, scale_mlp = self.adaLN_modulation(y).chunk(4, dim=-1)

        temporal_mask = self._safe_mask(~token_valid_mask.reshape(B * P, C))
        xt = _modulate_joint(self.norm_temporal(x), shift_time, scale_time).reshape(B * P, C, D)
        x = x + self.temporal_attn(
            xt, xt, xt, key_padding_mask=temporal_mask, need_weights=False
        )[0].reshape(B, P, C, D)
        x = x.masked_fill(~token_valid_mask[..., None], 0.0)

        xa = self.norm_agent(x).permute(0, 2, 1, 3).reshape(B * C, P, D)
        # Agent attention is performed independently for each chunk. Mask invalid
        # agents at that specific chunk, not only agents missing at the current
        # frame; many neighbor future chunks are padded and otherwise dilute the
        # valid ego/neighbor interaction softmax.
        agent_mask = self._safe_mask((~token_valid_mask).permute(0, 2, 1).reshape(B * C, P))
        xa = self.agent_attn(xa, xa, xa, key_padding_mask=agent_mask, need_weights=False)[0]
        x = x + xa.reshape(B, C, P, D).permute(0, 2, 1, 3)
        x = x.masked_fill(~token_valid_mask[..., None], 0.0)

        xf = self.norm_cross(x).reshape(B, P * C, D)
        out = self.cross_attn(
            xf, scene, scene, key_padding_mask=scene_mask, need_weights=False
        )[0].reshape(B, P, C, D)
        x = (x + out).masked_fill(~token_valid_mask[..., None], 0.0)

        xm = _modulate_joint(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + self.mlp(xm)
        return x.masked_fill(~token_valid_mask[..., None], 0.0)


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    norm = args.state_normalizer
    model_type = args.diffusion_model_type
    use_velocity = args.use_velocity_representation
    hybrid_omega = args.hybrid_loss_omega
    hybrid_window = args.hybrid_loss_window

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, V]

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    longitudinal_velocity = inputs["ego_current_state"][:, 4:5]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat(
        [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    eps = 1e-3
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    t = t.view(B, 1, 1, 1)
    t = t.expand(B, P, T + 1, 1)
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=gt_future.device)  # [B,]
    prefix_mask = generate_prefix_mask(delay, 1 + Pn, T + 1)  # (B, P, T+1, 1)
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=gt_future.device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    if use_velocity:
        full_traj = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [B, P, T+1, 4]
        gt_velocity = waypoints_to_velocity(full_traj)  # [B, P, T, 4]
        all_gt = torch.cat([current_states[:, :, None, :], gt_velocity], dim=2)
    else:
        all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    dfp_clean_chunks = None
    dfp_valid_mask = None
    orig_model_output = None

    if model_type == "x_start":
        if getattr(args, "dfp_decoder_mode", "") == "joint_temporal_agent":
            joint_inputs, dfp_clean_chunks, dfp_valid_mask = build_joint_dfp_training_inputs(
                inputs, ego_future, neighbors_future, neighbor_future_mask, norm, args
            )
            merged_inputs = {
                **inputs,
                **joint_inputs,
                "gt_trajectories": all_gt,
            }
            _, decoder_output = model(merged_inputs)
            model_output = decoder_output["model_output"][:, :, 1:, :]
            gt_target = all_gt[:, :, 1:, :]

            if use_velocity:
                dpm_loss = hybrid_loss(
                    model_output,
                    gt_target,
                    omega=hybrid_omega,
                    W=hybrid_window,
                )
            else:
                loss_dict = loss_func(model_output, gt_target)
                heading_l2_loss = loss_dict["heading_l2_loss"]
                position_lat_loss = loss_dict["position_lat_loss"]
                position_lon_loss = loss_dict["position_lon_loss"]

                velocity_weight = longitudinal_velocity * args.coeff_velocity
                velocity_weight = torch.abs(velocity_weight)
                velocity_weight = torch.clamp_min(velocity_weight, 1.0)
                velocity_weight = velocity_weight.unsqueeze(-1)
                position_lon_loss = position_lon_loss / velocity_weight

                timestep_weight = args.coeff_timestep
                assert T % len(timestep_weight) == 0, (
                    f"Timestep {T} is not divisible by the number of timestep weights {len(timestep_weight)}"
                )
                unit = T // len(timestep_weight)
                for i in range(len(timestep_weight)):
                    position_lat_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                    position_lon_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                    heading_l2_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]

                dpm_loss = (
                    args.coeff_position_lat_loss * position_lat_loss
                    + args.coeff_position_lon_loss * position_lon_loss
                    + args.coeff_heading_l2_loss * heading_l2_loss
                )
        else:
            mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t[..., 1:, :])
            # mean([B, P, T, D]), std([B, 1, T, 1]), z([B, P, T, D])
            xT = mean + std * z

            xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
            xT = torch.where(prefix_mask, all_gt, xT)  # [B, P, 1 + T, 4]

            merged_inputs = {
                **inputs,
                "gt_trajectories": all_gt,
                "sampled_trajectories": xT,
                "diffusion_time": t,
                "prefix_mask": prefix_mask,
            }
            if getattr(args, "use_dfp_decoder", False):
                dfp_inputs, dfp_clean_chunks = build_dfp_training_inputs(inputs, ego_future, norm, args)
                merged_inputs.update(dfp_inputs)
            _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
            model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]
            orig_model_output = decoder_output.get("model_output_orig")
            if orig_model_output is not None:
                orig_model_output = orig_model_output[:, :, 1:, :]

            gt_target = all_gt[:, :, 1:, :]  # [B, P, T, 4]

            if use_velocity:
                # Hybrid loss: velocity L2 + omega * waypoint L2 (with detach window)
                dpm_loss = hybrid_loss(
                    model_output,
                    gt_target,
                    omega=hybrid_omega,
                    W=hybrid_window,
                )  # [B, P, T]
            else:
                loss_dict = loss_func(model_output, gt_target)
                heading_l2_loss = loss_dict["heading_l2_loss"]  # [B, P, T]
                position_lat_loss = loss_dict["position_lat_loss"]  # [B, P, T]
                position_lon_loss = loss_dict["position_lon_loss"]  # [B, P, T]

                # velocity weight
                velocity_weight = longitudinal_velocity * args.coeff_velocity
                velocity_weight = torch.abs(velocity_weight)
                velocity_weight = torch.clamp_min(velocity_weight, 1.0)
                velocity_weight = velocity_weight.unsqueeze(-1)  # [B, 1, 1]
                position_lon_loss = position_lon_loss / velocity_weight

                # timestep weight
                timestep_weight = args.coeff_timestep
                assert T % len(timestep_weight) == 0, (
                    f"Timestep {T} is not divisible by the number of timestep weights {len(timestep_weight)}"
                )
                unit = T // len(timestep_weight)
                for i in range(len(timestep_weight)):
                    position_lat_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                    position_lon_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                    heading_l2_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]

                dpm_loss = (
                    args.coeff_position_lat_loss * position_lat_loss
                    + args.coeff_position_lon_loss * position_lon_loss
                    + args.coeff_heading_l2_loss * heading_l2_loss
                )  # [B, P, T]

    elif model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        t = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t) * z + t * all_gt[:, :, 1:, :]  # [B, P, T, 4]
        t = t.reshape(-1)  # [B,]

        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        if getattr(args, "use_dfp_decoder", False):
            dfp_inputs, dfp_clean_chunks = build_dfp_training_inputs(inputs, ego_future, norm, args)
            merged_inputs.update(dfp_inputs)
        _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
        model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]

        target_v = all_gt[:, :, 1:, :] - z
        dpm_loss = torch.sum((model_output - target_v) ** 2, dim=-1)
    else:
        raise NotImplementedError(f"Unknown diffusion model type: {model_type}")

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, : args.ego_prediction_horizon].mean()

    if orig_model_output is not None:
        if use_velocity:
            orig_dpm_loss = hybrid_loss(
                orig_model_output,
                gt_target,
                omega=hybrid_omega,
                W=hybrid_window,
            )
        else:
            orig_loss_dict = loss_func(orig_model_output, gt_target)
            orig_heading_l2_loss = orig_loss_dict["heading_l2_loss"]
            orig_position_lat_loss = orig_loss_dict["position_lat_loss"]
            orig_position_lon_loss = orig_loss_dict["position_lon_loss"]

            velocity_weight = longitudinal_velocity * args.coeff_velocity
            velocity_weight = torch.abs(velocity_weight)
            velocity_weight = torch.clamp_min(velocity_weight, 1.0)
            velocity_weight = velocity_weight.unsqueeze(-1)
            orig_position_lon_loss = orig_position_lon_loss / velocity_weight

            timestep_weight = args.coeff_timestep
            unit = T // len(timestep_weight)
            for i in range(len(timestep_weight)):
                orig_position_lat_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                orig_position_lon_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                orig_heading_l2_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]

            orig_dpm_loss = (
                args.coeff_position_lat_loss * orig_position_lat_loss
                + args.coeff_position_lon_loss * orig_position_lon_loss
                + args.coeff_heading_l2_loss * orig_heading_l2_loss
            )
        loss["ego_original_planning_loss"] = orig_dpm_loss[
            :, 0, : args.ego_prediction_horizon
        ].mean()

    if dfp_clean_chunks is not None and "joint_dfp_x0" in decoder_output:
        dfp_pred = decoder_output["joint_dfp_x0"]
        dfp_loss = torch.sum((dfp_pred - dfp_clean_chunks) ** 2, dim=-1)
        valid = dfp_valid_mask

        def _masked_mean(values, mask):
            selected = values[mask]
            if selected.numel() == 0:
                return torch.tensor(0.0, device=values.device, dtype=values.dtype)
            return selected.mean()

        dfp_ego_history_loss = _masked_mean(dfp_loss[:, 0, 0], valid[:, 0, 0])
        dfp_neighbor_history_loss = _masked_mean(dfp_loss[:, 1:, 0], valid[:, 1:, 0])
        dfp_ego_current_loss = _masked_mean(dfp_loss[:, 0, 1], valid[:, 0, 1])
        dfp_neighbor_current_loss = _masked_mean(dfp_loss[:, 1:, 1], valid[:, 1:, 1])
        dfp_ego_future_loss = _masked_mean(dfp_loss[:, 0, 2:], valid[:, 0, 2:])
        dfp_neighbor_future_loss = _masked_mean(dfp_loss[:, 1:, 2:], valid[:, 1:, 2:])

        loss["dfp_history_loss"] = dfp_ego_history_loss + args.alpha_neighbor_loss * dfp_neighbor_history_loss
        loss["dfp_current_loss"] = dfp_ego_current_loss + args.alpha_neighbor_loss * dfp_neighbor_current_loss
        loss["dfp_future_loss"] = dfp_ego_future_loss + args.alpha_neighbor_loss * dfp_neighbor_future_loss
        loss["dfp_ego_future_loss"] = dfp_ego_future_loss.detach()
        loss["dfp_neighbor_future_loss"] = dfp_neighbor_future_loss.detach()
    elif dfp_clean_chunks is not None:
        dfp_pred = decoder_output["dfp_x0"]
        dfp_loss = torch.sum((dfp_pred - dfp_clean_chunks) ** 2, dim=-1)
        loss["dfp_history_loss"] = dfp_loss[:, 0].mean()
        loss["dfp_current_loss"] = dfp_loss[:, 1].mean()
        loss["dfp_future_loss"] = dfp_loss[:, 2:].mean()

    if "dfp_gate" in decoder_output:
        dfp_gate = decoder_output["dfp_gate"].detach()
        loss["dfp_gate_mean"] = dfp_gate.mean()
        loss["dfp_gate_std"] = dfp_gate.std(unbiased=False)
        loss["dfp_gate_min"] = dfp_gate.amin()
        loss["dfp_gate_max"] = dfp_gate.amax()
    if "dfp_original_gate" in decoder_output:
        original_gate = decoder_output["dfp_original_gate"].detach()
        loss["dfp_original_gate_mean"] = original_gate.mean()
        loss["dfp_original_gate_std"] = original_gate.std(unbiased=False)
        loss["dfp_original_gate_min"] = original_gate.amin()
        loss["dfp_original_gate_max"] = original_gate.amax()

    # Compute denormalized ego trajectory for physical penalty losses.
    need_ego_world = model_type == "x_start" and (
        args.coeff_road_border_loss > 0
        or args.coeff_neighbor_collision_loss > 0
        or getattr(args, "coeff_comfort_loss", 0.0) > 0
        or getattr(args, "coeff_progress_loss", 0.0) > 0
    )
    need_ego_edge = model_type == "x_start" and (
        args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    )
    if need_ego_world:
        ego_pred = model_output[:, 0]  # [B, T, 4]
        if use_velocity:
            ego_current_raw = current_states[:, 0]  # [B, 4]
            ego_pred_world = velocity_to_waypoints(ego_pred)
            ego_pred_world[..., :2] = ego_pred_world[..., :2] + ego_current_raw[:, None, :2]
        else:
            ego_pred_world = ego_pred * norm.std[0].to(model_output.device) + norm.mean[0].to(
                model_output.device
            )  # [B, T, 4]
    if need_ego_edge:
        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        denorm_inputs = args.observation_normalizer.inverse(inputs)

    if getattr(args, "coeff_comfort_loss", 0.0) > 0 and model_type == "x_start":
        loss.update(compute_comfort_surrogate_loss(ego_pred_world))
    else:
        loss["comfort_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    if getattr(args, "coeff_progress_loss", 0.0) > 0 and model_type == "x_start":
        loss.update(compute_progress_surrogate_loss(ego_pred_world, ego_future))
    else:
        loss["progress_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Road border collision loss (ego only, x_start mode)
    if args.coeff_road_border_loss > 0 and model_type == "x_start":
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Neighbor collision loss (ego only, x_start mode)
    if args.coeff_neighbor_collision_loss > 0 and model_type == "x_start":
        nc_loss = compute_neighbor_collision_penalty(
            ego_edge_points,
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin=args.neighbor_collision_margin,
        )  # [B, T]
        loss["neighbor_collision_loss"] = nc_loss.mean()
    else:
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, TURN_INDICATOR_OUTPUT_KEEP]
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
    turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
    loss["turn_indicator_loss"] = turn_indicator_loss

    with torch.no_grad():
        turn_indicator_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        loss["turn_indicator_accuracy"] = turn_indicator_accuracy

    return loss


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._use_dfp_decoder = getattr(config, "use_dfp_decoder", False)
        self._dfp_decoder_mode = getattr(config, "dfp_decoder_mode", "additive")
        self._dfp_use_inference = getattr(config, "dfp_use_inference", False)
        self._dfp_history_len = getattr(config, "dfp_history_len", 20)
        self._dfp_chunk_len = getattr(config, "dfp_chunk_len", 20)
        self._dfp_future_chunks = self._future_len // self._dfp_chunk_len
        self._dfp_num_chunks = 2 + self._dfp_future_chunks
        self._dfp_guidance_w = getattr(config, "dfp_guidance_w", 0.2)
        self._dfp_guidance_beta = getattr(config, "dfp_guidance_beta", 2.0)
        self._dfp_sampler_steps = getattr(config, "dfp_sampler_steps", 10)
        self._dfp_fusion_mode = getattr(config, "dfp_fusion_mode", "none")
        self._dfp_fusion_residual_scale = getattr(config, "dfp_fusion_residual_scale", 1.0)
        self._dfp_gate_alpha_init = float(getattr(config, "dfp_gate_alpha_init", 0.1))
        self._dfp_interaction = self._dfp_decoder_mode == "shared_stack_interaction_ego"
        self._dfp_joint_temporal = self._dfp_decoder_mode == "joint_temporal_agent"
        self._dfp_interaction_detach = bool(getattr(config, "dfp_interaction_detach", True))
        self._dfp_shared_stack = self._dfp_decoder_mode in ("shared_stack_unified_ego", "shared_stack_gated_ego", "shared_stack_interaction_ego")
        if self._dfp_fusion_mode == "residual" and self._dfp_decoder_mode == "additive":
            # Backward compatibility for earlier fusion scripts.
            self._dfp_decoder_mode = "fusion"
        if self._use_dfp_decoder:
            assert config.diffusion_model_type == "x_start", "DFP branch is implemented for x_start training"
            assert self._dfp_history_len == self._dfp_chunk_len
            assert self._future_len % self._dfp_chunk_len == 0
        if self._dfp_decoder_mode not in (
            "additive",
            "fusion",
            "unified_ego",
            "shared_stack_unified_ego",
            "shared_stack_gated_ego",
            "shared_stack_interaction_ego",
            "joint_temporal_agent",
        ):
            raise ValueError(f"Unknown dfp_decoder_mode={self._dfp_decoder_mode}")
        if self._dfp_fusion_mode not in ("none", "residual"):
            raise ValueError(f"Unknown dfp_fusion_mode={self._dfp_fusion_mode}")
        if self._dfp_decoder_mode in ("fusion", "unified_ego", "shared_stack_unified_ego", "shared_stack_gated_ego", "shared_stack_interaction_ego", "joint_temporal_agent"):
            assert self._use_dfp_decoder, "DFP decoder mode requires use_dfp_decoder=True"
        if self._dfp_decoder_mode == "fusion" and self._dfp_fusion_mode != "residual":
            raise ValueError("dfp_decoder_mode=fusion requires dfp_fusion_mode=residual")

        self.dit = (
            None
            if self._dfp_joint_temporal
            else DiT(
                depth=config.decoder_depth,
                output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
                hidden_dim=config.hidden_dim,
                heads=config.num_heads,
                dropout=dpr,
            )
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )
        self.dfp_dit = (
            DFPDiT(
                num_chunks=self._dfp_num_chunks,
                chunk_len=self._dfp_chunk_len,
                depth=config.decoder_depth,
                hidden_dim=config.hidden_dim,
                heads=config.num_heads,
                dropout=dpr,
            )
            if self._use_dfp_decoder and not self._dfp_shared_stack and not self._dfp_joint_temporal
            else None
        )
        self.dfp_shared_preproj = None
        self.dfp_shared_t_embedder = None
        self.dfp_shared_chunk_pos_embed = None
        self.dfp_shared_final_layer = None
        if self._dfp_shared_stack:
            self.dfp_shared_preproj = nn.Sequential(
                nn.Linear(self._dfp_chunk_len * 4, 512),
                nn.GELU(approximate="tanh"),
                nn.Linear(512, config.hidden_dim),
            )
            self.dfp_shared_t_embedder = TimestepEmbedder(config.hidden_dim)
            self.dfp_shared_chunk_pos_embed = nn.Parameter(
                torch.zeros(1, self._dfp_num_chunks, config.hidden_dim)
            )
            self.dfp_shared_final_layer = DFPFinalLayer(
                config.hidden_dim, self._dfp_chunk_len * 4
            )
        self.dfp_fusion_head = (
            nn.Sequential(
                nn.Linear(8 + config.hidden_dim, config.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, 4),
            )
            if self._dfp_decoder_mode == "fusion"
            else None
        )
        self.dfp_gate_head = (
            nn.Sequential(
                nn.Linear(8 + config.hidden_dim, config.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, 1),
            )
            if self._dfp_decoder_mode == "shared_stack_gated_ego"
            else None
        )
        self.dfp_gate_alpha_logit = None
        if self._dfp_decoder_mode == "shared_stack_gated_ego":
            alpha_init = min(max(self._dfp_gate_alpha_init, 1.0e-4), 1.0 - 1.0e-4)
            self.dfp_gate_alpha_logit = nn.Parameter(
                torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32)
            )
        self.dfp_interaction_preproj = None
        self.dfp_interaction_chunk_pos_embed = None
        if self._dfp_interaction:
            self.dfp_interaction_preproj = nn.Sequential(
                nn.Linear(self._dfp_chunk_len * 4, config.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(config.hidden_dim),
            )
            self.dfp_interaction_chunk_pos_embed = nn.Parameter(
                torch.zeros(1, 1, self._dfp_future_chunks, config.hidden_dim)
            )
        self.joint_temporal_preproj = None
        self.joint_temporal_t_embedder = None
        self.joint_temporal_agent_type_embed = None
        self.joint_temporal_chunk_pos_embed = None
        self.joint_temporal_blocks = None
        self.joint_temporal_final_layer = None
        if self._dfp_joint_temporal:
            self.joint_temporal_preproj = nn.Sequential(
                nn.Linear(self._dfp_chunk_len * 4, config.hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.joint_temporal_t_embedder = TimestepEmbedder(config.hidden_dim)
            self.joint_temporal_agent_type_embed = nn.Embedding(2, config.hidden_dim)
            self.joint_temporal_chunk_pos_embed = nn.Parameter(
                torch.zeros(1, 1, self._dfp_num_chunks, config.hidden_dim)
            )
            self.joint_temporal_blocks = nn.ModuleList(
                [
                    JointTemporalAgentBlock(
                        dim=config.hidden_dim,
                        heads=config.num_heads,
                        dropout=dpr,
                    )
                    for _ in range(config.decoder_depth)
                ]
            )
            self.joint_temporal_final_layer = DFPFinalLayer(
                config.hidden_dim, self._dfp_chunk_len * 4
            )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = (
            config.guidance_fn if config.__dict__.get("guidance_fn") is not None else None
        )
        self._guidance_scale = config.guidance_scale
        self._model_type = config.diffusion_model_type
        self._use_velocity = config.use_velocity_representation

        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)
        if self.dfp_shared_chunk_pos_embed is not None:
            nn.init.normal_(self.dfp_shared_chunk_pos_embed, std=0.02)
        if self.dfp_interaction_chunk_pos_embed is not None:
            nn.init.normal_(self.dfp_interaction_chunk_pos_embed, std=0.02)
        if self.joint_temporal_chunk_pos_embed is not None:
            nn.init.normal_(self.joint_temporal_chunk_pos_embed, std=0.02)

        # Zero-out output layers:
        if self.dit is not None:
            nn.init.constant_(self.dit.final_layer.proj[-1].weight, 0)
            nn.init.constant_(self.dit.final_layer.proj[-1].bias, 0)
        if self.dfp_dit is not None:
            nn.init.constant_(self.dfp_dit.final_layer.proj[-1].weight, 0)
            nn.init.constant_(self.dfp_dit.final_layer.proj[-1].bias, 0)
        if self.dfp_shared_final_layer is not None:
            nn.init.constant_(self.dfp_shared_final_layer.proj[-1].weight, 0)
            nn.init.constant_(self.dfp_shared_final_layer.proj[-1].bias, 0)
        if self.joint_temporal_final_layer is not None:
            # Joint temporal training starts from scratch and replaces the original
            # DiT output head. A tiny non-zero output init lets gradients reach the
            # temporal/interaction stack from step 1 while keeping initial outputs
            # close to zero for diffusion stability.
            nn.init.xavier_uniform_(self.joint_temporal_final_layer.proj[-1].weight, gain=0.01)
            nn.init.constant_(self.joint_temporal_final_layer.proj[-1].bias, 0)
        if self.dfp_fusion_head is not None:
            nn.init.constant_(self.dfp_fusion_head[-1].weight, 0)
            nn.init.constant_(self.dfp_fusion_head[-1].bias, 0)
        if self.dfp_gate_head is not None:
            nn.init.constant_(self.dfp_gate_head[-1].weight, 0)
            nn.init.constant_(self.dfp_gate_head[-1].bias, 0)

    def _prepare_current_states(self, inputs):
        """Extract and prepare current states for ego and neighbors.

        Args:
            inputs: Dict containing ego_current_state and neighbor_agents_past

        Returns:
            Tuple of (current_states, neighbor_current_mask, ego_current, neighbors_current)
                - current_states: [B, P, 4] concatenated ego and neighbor current states
                - neighbor_current_mask: [B, Pn] mask for invalid neighbors
                - ego_current: [B, 1, 4] ego current state
                - neighbors_current: [B, Pn, 4] neighbor current states
        """
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self._predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B, P, 4]

        return current_states, neighbor_current_mask, ego_current, neighbors_current

    def _compute_turn_indicator(self, ego_trajectory, encoding_pooled):
        """Compute turn indicator logit from ego trajectory and encoding.

        Args:
            ego_trajectory: [B, 2 * (T // 10)] flattened ego trajectory positions
            encoding_pooled: [B, D] pooled encoding

        Returns:
            turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
        """
        turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
        return self.turn_indicator_predictor(turn_indicator_input)

    def _dfp_future_from_chunks(self, dfp_x0):
        return dfp_x0[:, 2:].reshape(dfp_x0.shape[0], self._future_len, 4)

    def _has_dfp_path(self):
        return self.dfp_dit is not None or self.dfp_shared_final_layer is not None

    def _decode_dfp_chunks(self, chunks, t, encoding, interaction_tokens=None, interaction_mask=None):
        if self._dfp_shared_stack:
            return self._decode_dfp_chunks_shared_stack(
                chunks, t, encoding, interaction_tokens, interaction_mask
            )
        if self.dfp_dit is None:
            raise RuntimeError("DFP decoder is not initialized")
        return self.dfp_dit(chunks, t, encoding)

    def _build_dfp_interaction_tokens(self, neighbor_future, neighbor_current_mask):
        if not self._dfp_interaction or self.dfp_interaction_preproj is None:
            return None, None
        if self._dfp_interaction_detach:
            neighbor_future = neighbor_future.detach()
        B, Pn, T, D = neighbor_future.shape
        assert T == self._future_len, f"{T=} expected {self._future_len}"
        chunks = neighbor_future.reshape(
            B, Pn, self._dfp_future_chunks, self._dfp_chunk_len, D
        )
        x = chunks.reshape(B, Pn, self._dfp_future_chunks, self._dfp_chunk_len * D)
        tokens = self.dfp_interaction_preproj(x)
        tokens = tokens + self.dfp_interaction_chunk_pos_embed[:, :, : self._dfp_future_chunks]
        mask = neighbor_current_mask[:, :, None].expand(B, Pn, self._dfp_future_chunks)
        tokens = tokens.masked_fill(mask[..., None], 0.0)
        return tokens.reshape(B, Pn * self._dfp_future_chunks, -1), mask.reshape(
            B, Pn * self._dfp_future_chunks
        )

    def _decode_dfp_chunks_shared_stack(
        self, chunks, t, encoding, interaction_tokens=None, interaction_mask=None
    ):
        B, N, L, D = chunks.shape
        assert N == self._dfp_num_chunks, f"{N=} expected {self._dfp_num_chunks}"
        assert L == self._dfp_chunk_len, f"{L=} expected {self._dfp_chunk_len}"
        assert D == 4

        x = chunks.reshape(B, N, L * D)
        x = self.dfp_shared_preproj(x) + self.dfp_shared_chunk_pos_embed[:, :N]
        y = self.dfp_shared_t_embedder(t.reshape(B * N)).reshape(B, N, -1)

        attn_mask = torch.zeros((B, N), dtype=torch.bool, device=chunks.device)
        cross_c = encoding
        cross_attn_mask = torch.all(encoding == 0, dim=-1)
        if interaction_tokens is not None:
            cross_c = torch.cat([cross_c, interaction_tokens], dim=1)
            if interaction_mask is None:
                interaction_mask = torch.zeros(
                    interaction_tokens.shape[:2], dtype=torch.bool, device=chunks.device
                )
            cross_attn_mask = torch.cat([cross_attn_mask, interaction_mask], dim=1)
        all_masked = torch.all(cross_attn_mask, dim=1)
        if torch.any(all_masked):
            cross_attn_mask = cross_attn_mask.clone()
            cross_attn_mask[all_masked, 0] = False

        for block in self.dit.blocks:
            x = block(x, cross_c, y, attn_mask, cross_attn_mask)

        x = self.dfp_shared_final_layer(x, y)
        return x.reshape(B, N, L, D)

    def _fuse_dfp_ego(self, ego_original, ego_dfp, encoding_pooled):
        if self.dfp_fusion_head is None:
            return ego_original
        B, T, _ = ego_original.shape
        scene = encoding_pooled[:, None].expand(B, T, encoding_pooled.shape[-1])
        fusion_input = torch.cat([ego_original, ego_dfp, scene], dim=-1)
        delta = torch.tanh(self.dfp_fusion_head(fusion_input))
        return ego_original + self._dfp_fusion_residual_scale * delta

    def _gate_dfp_ego(self, ego_original, ego_dfp, encoding_pooled):
        if self.dfp_gate_head is None or self.dfp_gate_alpha_logit is None:
            dfp_weight = torch.ones_like(ego_dfp[..., :1])
            original_weight = torch.zeros_like(dfp_weight)
            return ego_dfp, dfp_weight, original_weight
        B, T, _ = ego_original.shape
        scene = encoding_pooled[:, None].expand(B, T, encoding_pooled.shape[-1])
        gate_input = torch.cat([ego_original, ego_dfp, scene], dim=-1)
        local_original_gate = torch.sigmoid(self.dfp_gate_head(gate_input))
        alpha = torch.sigmoid(self.dfp_gate_alpha_logit).to(
            device=ego_dfp.device, dtype=ego_dfp.dtype
        )
        original_weight = alpha * local_original_gate
        dfp_weight = 1.0 - original_weight
        ego = dfp_weight * ego_dfp + original_weight * ego_original
        return ego, dfp_weight, original_weight

    def _decode_joint_temporal_chunks(self, chunks, t, encoding, agent_valid_mask, valid_mask):
        B, P, C, L, D = chunks.shape
        assert C == self._dfp_num_chunks, f"{C=} expected {self._dfp_num_chunks}"
        assert L == self._dfp_chunk_len, f"{L=} expected {self._dfp_chunk_len}"
        x = chunks.reshape(B, P, C, L * D)
        x = self.joint_temporal_preproj(x)
        x = x + self.joint_temporal_chunk_pos_embed[:, :, :C]
        type_ids = torch.ones(P, dtype=torch.long, device=chunks.device)
        type_ids[0] = 0
        x = x + self.joint_temporal_agent_type_embed(type_ids)[None, :, None, :]
        y = self.joint_temporal_t_embedder(t.reshape(B * P * C)).reshape(B, P, C, -1)
        token_valid_mask = valid_mask.any(dim=-1) & agent_valid_mask[:, :, None]
        x = x.masked_fill(~token_valid_mask[..., None], 0.0)
        scene_mask = torch.all(encoding == 0, dim=-1)
        all_scene_masked = torch.all(scene_mask, dim=1)
        if torch.any(all_scene_masked):
            scene_mask = scene_mask.clone()
            scene_mask[all_scene_masked, 0] = False
        for block in self.joint_temporal_blocks:
            x = block(x, encoding, y, agent_valid_mask, token_valid_mask, scene_mask)
        x = self.joint_temporal_final_layer(x, y)
        return x.reshape(B, P, C, L, D).masked_fill(~valid_mask[..., None], 0.0)

    def _forward_joint_temporal_training(
        self, encoding, inputs, neighbor_current_mask, encoding_pooled
    ):
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        chunks = inputs["joint_sampled_chunks"]
        t = inputs["joint_diffusion_time"]
        valid_mask = inputs["joint_valid_mask"]
        agent_valid_mask = inputs["joint_agent_valid_mask"]
        x0 = self._decode_joint_temporal_chunks(chunks, t, encoding, agent_valid_mask, valid_mask)
        future = x0[:, :, 2:].reshape(B, P, self._future_len, 4)
        raw_inputs = self._observation_normalizer.inverse(inputs)
        ego_current = raw_inputs["ego_current_state"][:, :4]
        neighbors_current = raw_inputs["neighbor_agents_past"][:, : self._predicted_neighbor_num, -1, :4]
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
        current_states = normalize_trajectory_by_agent(self._state_normalizer, current_states)
        model_output = torch.cat([current_states[:, :, None], future], dim=2)
        if "gt_trajectories" in inputs:
            ego_trajectory = inputs["gt_trajectories"][:, 0, 1::10, :2].reshape(
                B, 2 * (self._future_len // 10)
            )
        else:
            ego_trajectory = future[:, 0, ::10, :2].reshape(B, 2 * (self._future_len // 10))
        return {
            "model_output": model_output,
            "joint_dfp_x0": x0,
            "turn_indicator_logit": self._compute_turn_indicator(ego_trajectory, encoding_pooled),
        }

    def _joint_condition_chunks(self, inputs, B, P, device, dtype):
        raw_inputs = self._observation_normalizer.inverse(inputs)
        ego_history = _last_history_chunk(raw_inputs["ego_agent_past"], self._dfp_history_len)[:, None]
        neighbor_history = _last_history_chunk(
            raw_inputs["neighbor_agents_past"][:, : self._predicted_neighbor_num, :, :4],
            self._dfp_history_len,
        )
        history = torch.cat([ego_history, neighbor_history], dim=1)
        ego_current = raw_inputs["ego_current_state"][:, :4]
        neighbor_current = raw_inputs["neighbor_agents_past"][:, : self._predicted_neighbor_num, -1, :4]
        current_states = torch.cat([ego_current[:, None], neighbor_current], dim=1)
        current = current_states[:, :, None, :].expand(B, P, self._dfp_chunk_len, 4)
        history = normalize_trajectory_by_agent(self._state_normalizer, history).to(device=device, dtype=dtype)
        current = normalize_trajectory_by_agent(self._state_normalizer, current).to(device=device, dtype=dtype)
        agent_valid = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=device),
                (torch.sum(torch.ne(neighbor_current[..., :4], 0), dim=-1) != 0).to(device),
            ],
            dim=1,
        )
        history_valid = torch.cat(
            [
                torch.ones(B, 1, self._dfp_chunk_len, dtype=torch.bool, device=device),
                (torch.sum(torch.ne(neighbor_history[..., :4], 0), dim=-1) != 0).to(device),
            ],
            dim=1,
        )
        current_valid = agent_valid[:, :, None].expand(B, P, self._dfp_chunk_len)
        future_valid = agent_valid[:, :, None, None].expand(
            B, P, self._dfp_future_chunks, self._dfp_chunk_len
        )
        valid_mask = torch.cat(
            [history_valid[:, :, None], current_valid[:, :, None], future_valid], dim=2
        )
        return history[:, :, None], current[:, :, None], agent_valid, valid_mask

    def _forward_joint_temporal_inference(
        self, encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
    ):
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        device = encoding.device
        dtype = encoding.dtype
        history_chunk, current_chunk, agent_valid, valid_mask = self._joint_condition_chunks(
            inputs, B, P, device, dtype
        )
        future_xt = torch.randn(
            B, P, self._dfp_future_chunks, self._dfp_chunk_len, 4, device=device, dtype=dtype
        )
        eps = 1.0e-3
        timesteps = torch.linspace(1.0, eps, self._dfp_sampler_steps + 1, device=device, dtype=dtype)
        x0_future = future_xt
        for step in range(self._dfp_sampler_steps):
            t_s = timesteps[step]
            t_next = timesteps[step + 1]
            future_t = t_s.expand(B, P, self._dfp_future_chunks)
            hist_noise = torch.randn_like(history_chunk)
            unguided_chunks = torch.cat([hist_noise, current_chunk, future_xt], dim=2)
            unguided_t = torch.cat(
                [
                    torch.ones(B, P, 1, device=device, dtype=dtype),
                    torch.zeros(B, P, 1, device=device, dtype=dtype),
                    future_t,
                ],
                dim=2,
            )
            x0_unguided = self._decode_joint_temporal_chunks(
                unguided_chunks, unguided_t, encoding, agent_valid, valid_mask
            )
            t_hist = torch.clamp(t_s.pow(self._dfp_guidance_beta), min=eps)
            hist_alpha, hist_sigma = vp_alpha_sigma(t_hist)
            guided_history = hist_alpha * history_chunk + hist_sigma * torch.randn_like(history_chunk)
            guided_chunks = torch.cat([guided_history, current_chunk, future_xt], dim=2)
            guided_t = torch.cat(
                [
                    t_hist.expand(B, P, 1),
                    torch.zeros(B, P, 1, device=device, dtype=dtype),
                    future_t,
                ],
                dim=2,
            )
            x0_guided = self._decode_joint_temporal_chunks(
                guided_chunks, guided_t, encoding, agent_valid, valid_mask
            )
            x0 = x0_unguided + self._dfp_guidance_w * (x0_guided - x0_unguided)
            x0_future = x0[:, :, 2:]
            alpha_s, sigma_s = vp_alpha_sigma(t_s)
            alpha_next, sigma_next = vp_alpha_sigma(t_next)
            eps_pred = (future_xt - alpha_s * x0_future) / torch.clamp(sigma_s, min=1.0e-6)
            future_xt = alpha_next * x0_future + sigma_next * eps_pred
            future_xt = future_xt.masked_fill(~valid_mask[:, :, 2:, :, None], 0.0)
        future_norm = x0_future.reshape(B, P, self._future_len, 4)
        prediction = inverse_normalize_trajectory_by_agent(self._state_normalizer, future_norm)
        future_valid = valid_mask[:, :, 2:].reshape(B, P, self._future_len)
        prediction = prediction.masked_fill(~future_valid[..., None], 0.0)
        ego_trajectory = future_norm[:, 0, ::10, :2].reshape(B, 2 * (self._future_len // 10))
        return {
            "prediction": prediction,
            "turn_indicator_logit": self._compute_turn_indicator(ego_trajectory, encoding_pooled),
        }

    def _forward_training(self, encoding, inputs, neighbor_current_mask, encoding_pooled):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing model_output and turn_indicator_logit
        """
        if self._dfp_joint_temporal:
            return self._forward_joint_temporal_training(
                encoding, inputs, neighbor_current_mask, encoding_pooled
            )

        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num


        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len), 4
        )
        diffusion_time = inputs["diffusion_time"]

        gt_trajectories = inputs["gt_trajectories"].reshape(B, P, (1 + self._future_len), 4)
        ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)

        model_output = self.dit(
            sampled_trajectories,
            diffusion_time,
            encoding,
            neighbor_current_mask,
        ).reshape(B, P, -1, 4)

        outputs = {
            "model_output": model_output,
            "turn_indicator_logit": turn_indicator_logit,
        }
        if self._has_dfp_path() and "dfp_sampled_chunks" in inputs:
            interaction_tokens, interaction_mask = self._build_dfp_interaction_tokens(
                model_output[:, 1:, 1:], neighbor_current_mask
            )
            dfp_x0 = self._decode_dfp_chunks(
                inputs["dfp_sampled_chunks"],
                inputs["dfp_diffusion_time"],
                encoding,
                interaction_tokens,
                interaction_mask,
            )
            outputs["dfp_x0"] = dfp_x0
            ego_dfp = self._dfp_future_from_chunks(dfp_x0)
            if self._dfp_decoder_mode == "shared_stack_gated_ego":
                # Gated DFP ego decoder: learn a per-timestep interpolation between
                # the original DiT ego future and the DFP ego future while keeping
                # the original DiT neighbor head unchanged.
                gated_ego, dfp_gate, original_gate = self._gate_dfp_ego(
                    model_output[:, 0, 1:],
                    ego_dfp,
                    encoding_pooled,
                )
                gated_output = model_output.clone()
                gated_output[:, 0, 1:] = gated_ego
                outputs["model_output_orig"] = model_output
                outputs["model_output"] = gated_output
                outputs["dfp_gate"] = dfp_gate
                outputs["dfp_original_gate"] = original_gate
                ego_trajectory = gated_ego[:, ::10, :2].reshape(
                    B, 2 * (self._future_len // 10)
                )
                outputs["turn_indicator_logit"] = self._compute_turn_indicator(
                    ego_trajectory, encoding_pooled
                )
            elif self._dfp_decoder_mode in ("unified_ego", "shared_stack_unified_ego", "shared_stack_interaction_ego"):
                # Unified DFP ego decoder: the main ego output used by planner loss,
                # road-border/collision losses, validation, and inference is DFP x0.
                # The original DiT remains only as the neighbor-future head and optional
                # compatibility regularizer; it no longer supplies the primary ego future.
                unified_output = model_output.clone()
                unified_output[:, 0, 1:] = ego_dfp
                outputs["model_output_orig"] = model_output
                outputs["model_output"] = unified_output
                ego_trajectory = ego_dfp[:, ::10, :2].reshape(
                    B, 2 * (self._future_len // 10)
                )
                outputs["turn_indicator_logit"] = self._compute_turn_indicator(
                    ego_trajectory, encoding_pooled
                )
            elif self.dfp_fusion_head is not None:
                fused_output = model_output.clone()
                fused_output[:, 0, 1:] = self._fuse_dfp_ego(
                    model_output[:, 0, 1:],
                    ego_dfp,
                    encoding_pooled,
                )
                outputs["model_output_orig"] = model_output
                outputs["model_output"] = fused_output
                ego_trajectory = fused_output[:, 0, 1::10, :2].reshape(
                    B, 2 * (self._future_len // 10)
                )
                outputs["turn_indicator_logit"] = self._compute_turn_indicator(
                    ego_trajectory, encoding_pooled
                )
        return outputs

    def _inference_flow_matching(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using Flow Matching approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        x = sampled_trajectories
        NUM_STEP = 10
        func = partial(
            self.dit,
            cross_c=encoding,
            neighbor_current_mask=neighbor_current_mask,
        )
        x = euler_integration(func, x, NUM_STEP)
        # x = heun_integration(func, x, NUM_STEP)
        # x = rk4_integration(func, x, NUM_STEP)
        x = x.reshape(B, P, (1 + self._future_len), 4)
        ego_trajectory = x[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        if self._use_velocity:
            future = velocity_to_waypoints(x[:, :, 1:, :])
            future = add_current_xy(future, current_states)
            x = future  # [B, P, T, 4]
        else:
            x = self._state_normalizer.inverse(x)[:, :, 1:]
        return {"prediction": x, "turn_indicator_logit": turn_indicator_logit}

    def _inference_x_start(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        action_prefix = sampled_trajectories.reshape(B, P, 1 + self._future_len, 4)
        action_prefix = replace_current_state(action_prefix, current_states)
        xT = action_prefix.reshape(B, P, (1 + self._future_len) * 4)

        B, P, T_plus_1, D = action_prefix.shape

        delay = inputs["delay"].to(device=action_prefix.device)
        mask = generate_prefix_mask(delay, P, T_plus_1)  # (B, P, T_plus_1, 1)

        def prefix_constraint(xt, t, step):
            xt = xt.reshape(B, P, 1 + self._future_len, 4)
            xt = replace_current_state(xt, current_states)
            return xt

        model_wrapper_params = {
            "classifier_fn": self._guidance_fn,
            "classifier_kwargs": {
                "model": self.dit,
                "model_condition": {
                    "cross_c": encoding,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                "inputs": inputs,
                "observation_normalizer": self._observation_normalizer,
                "state_normalizer": self._state_normalizer,
            },
            "guidance_scale": self._guidance_scale,
            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
        }

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type=self._model_type,
            model_kwargs={
                "cross_c": encoding,
                "neighbor_current_mask": neighbor_current_mask,
            },
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, correcting_xt_fn=prefix_constraint)

        x0 = dpm_solver.sample(xT, steps=10, prefix_mask=mask, skip_type="logSNR")

        x0 = x0.reshape(B, P, (1 + self._future_len), 4)
        ego_trajectory = x0[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        if self._use_velocity:
            future = velocity_to_waypoints(x0[:, :, 1:, :])
            future = add_current_xy(future, current_states)
            x0 = future  # [B, P, T, 4]
        else:
            x0 = self._state_normalizer.inverse(x0)[:, :, 1:]

        return {"prediction": x0, "turn_indicator_logit": turn_indicator_logit}


    def _dfp_clean_chunks(self, inputs, B):
        raw_inputs = self._observation_normalizer.inverse(inputs)
        history = _last_history_chunk(raw_inputs["ego_agent_past"], self._dfp_history_len)

        current = raw_inputs["ego_current_state"][:, :4]
        current = current[:, None, :].expand(B, self._dfp_chunk_len, 4)
        history = normalize_ego_trajectory(self._state_normalizer, history)
        current = normalize_ego_trajectory(self._state_normalizer, current)
        return history[:, None], current[:, None]

    def _dfp_sample_ego_future(
        self, encoding, inputs, B, device, dtype, interaction_tokens=None, interaction_mask=None
    ):
        history_chunk, current_chunk = self._dfp_clean_chunks(inputs, B)
        future_xt = torch.randn(
            B,
            self._dfp_future_chunks,
            self._dfp_chunk_len,
            4,
            device=device,
            dtype=dtype,
        )
        eps = 1.0e-3
        timesteps = torch.linspace(1.0, eps, self._dfp_sampler_steps + 1, device=device, dtype=dtype)
        x0_future = future_xt
        for step in range(self._dfp_sampler_steps):
            t_s = timesteps[step]
            t_next = timesteps[step + 1]
            future_t = t_s.expand(B, self._dfp_future_chunks)

            hist_noise = torch.randn_like(history_chunk)
            unguided_chunks = torch.cat([hist_noise, current_chunk, future_xt], dim=1)
            unguided_t = torch.cat(
                [
                    torch.ones(B, 1, device=device, dtype=dtype),
                    torch.zeros(B, 1, device=device, dtype=dtype),
                    future_t,
                ],
                dim=1,
            )
            x0_unguided = self._decode_dfp_chunks(
                unguided_chunks, unguided_t, encoding, interaction_tokens, interaction_mask
            )

            t_hist = torch.clamp(t_s.pow(self._dfp_guidance_beta), min=eps)
            hist_alpha, hist_sigma = vp_alpha_sigma(t_hist)
            guided_history = hist_alpha * history_chunk + hist_sigma * torch.randn_like(history_chunk)
            guided_chunks = torch.cat([guided_history, current_chunk, future_xt], dim=1)
            guided_t = torch.cat(
                [
                    t_hist.expand(B, 1),
                    torch.zeros(B, 1, device=device, dtype=dtype),
                    future_t,
                ],
                dim=1,
            )
            x0_guided = self._decode_dfp_chunks(
                guided_chunks, guided_t, encoding, interaction_tokens, interaction_mask
            )

            x0 = x0_unguided + self._dfp_guidance_w * (x0_guided - x0_unguided)
            x0_future = x0[:, 2:]
            alpha_s, sigma_s = vp_alpha_sigma(t_s)
            alpha_next, sigma_next = vp_alpha_sigma(t_next)
            eps_pred = (future_xt - alpha_s * x0_future) / torch.clamp(sigma_s, min=1.0e-6)
            future_xt = alpha_next * x0_future + sigma_next * eps_pred

        future = x0_future.reshape(B, self._future_len, 4)
        return inverse_normalize_ego_trajectory(self._state_normalizer, future)

    def _maybe_apply_dfp_inference(self, output, encoding, inputs, encoding_pooled):
        if not self._has_dfp_path() or not self._dfp_use_inference:
            return output
        B = encoding.shape[0]
        prediction = output["prediction"].clone()
        interaction_tokens, interaction_mask = None, None
        if self._dfp_interaction:
            neighbor_future = normalize_neighbor_trajectory(self._state_normalizer, prediction[:, 1:])
            neighbor_current_mask = inputs.get("neighbor_current_mask")
            if neighbor_current_mask is None:
                neighbors_current = inputs["neighbor_agents_past"][
                    :, : self._predicted_neighbor_num, -1, :4
                ]
                neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
            interaction_tokens, interaction_mask = self._build_dfp_interaction_tokens(
                neighbor_future, neighbor_current_mask
            )
        future = self._dfp_sample_ego_future(
            encoding,
            inputs,
            B,
            prediction.device,
            prediction.dtype,
            interaction_tokens,
            interaction_mask,
        )
        if self._dfp_decoder_mode == "shared_stack_gated_ego" and self.dfp_gate_head is not None:
            original_future = normalize_ego_trajectory(self._state_normalizer, prediction[:, 0])
            dfp_future = normalize_ego_trajectory(self._state_normalizer, future)
            future_norm, dfp_gate, original_gate = self._gate_dfp_ego(
                original_future, dfp_future, encoding_pooled
            )
            future = inverse_normalize_ego_trajectory(self._state_normalizer, future_norm)
            output["dfp_gate"] = dfp_gate
            output["dfp_original_gate"] = original_gate
        elif self._dfp_decoder_mode == "fusion" and self.dfp_fusion_head is not None:
            original_future = normalize_ego_trajectory(self._state_normalizer, prediction[:, 0])
            dfp_future = normalize_ego_trajectory(self._state_normalizer, future)
            future_norm = self._fuse_dfp_ego(original_future, dfp_future, encoding_pooled)
            future = inverse_normalize_ego_trajectory(self._state_normalizer, future_norm)
        else:
            future_norm = normalize_ego_trajectory(self._state_normalizer, future)
        prediction[:, 0] = future
        ego_trajectory = future_norm[:, ::10, :2].reshape(B, 2 * (self._future_len // 10))
        output = {**output, "prediction": prediction}
        output["turn_indicator_logit"] = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        return output

    def _forward_inference(
        self, encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
    ):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        if self._dfp_joint_temporal:
            return self._forward_joint_temporal_inference(
                encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
            )

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len) * 4
        )

        if self._model_type == "flow_matching":
            output = self._inference_flow_matching(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                encoding_pooled,
                sampled_trajectories,
            )
        elif self._model_type == "x_start":
            output = self._inference_x_start(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                encoding_pooled,
                sampled_trajectories,
            )
        else:
            raise NotImplementedError(f"Unknown model type {self._model_type}")
        return self._maybe_apply_dfp_inference(output, encoding, inputs, encoding_pooled)

    def forward(self, encoding, inputs):
        """
        Diffusion decoder process.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,
                    "neighbor_agent_past": past and current neighbor states,

                    "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + self._future_len, 4]
                    "delay": number of initial steps to keep fixed (>=0),
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "model_output": Predicted future states, [B, P, 1 + self._future_len, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, self._future_len, 4]
                    "turn_indicator_logit": Turn indicator prediction, [B, TURN_INDICATOR_OUTPUT_DIM]
                    ...
                }

        """
        # Common preprocessing
        current_states, neighbor_current_mask, ego_current, neighbors_current = (
            self._prepare_current_states(inputs)
        )

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Pool encoding to get a fixed-size representation
        encoding_mask = torch.any(torch.ne(encoding, 0), dim=-1, keepdim=True)
        encoding_pooled = (encoding * encoding_mask).sum(dim=1) / encoding_mask.sum(dim=1).clamp_min(1)

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, neighbor_current_mask, encoding_pooled)
        else:
            return self._forward_inference(
                encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
            )
