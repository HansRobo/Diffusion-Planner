import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from planner_metrics.pdms_proxy import pdms_proxy
except Exception as exc:  # optional metric dependency; raised only when enabled
    pdms_proxy = None
    _PDMS_IMPORT_ERROR = exc
else:
    _PDMS_IMPORT_ERROR = None

from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    loss_func,
    make_turn_indicator_gt,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp


TRAJECTORY_CONSISTENCY_DT = 0.1
TRAJECTORY_CONSISTENCY_VEL_TOL = 1.0
TRAJECTORY_CONSISTENCY_ACC_TOL = 2.0
TRAJECTORY_CONSISTENCY_JERK_TOL = 5.0
TRAJECTORY_CONSISTENCY_HEADING_TOL = 0.2
TRAJECTORY_CONSISTENCY_HEADING_RATE_TOL = 0.5


def _angle_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    diff = lhs - rhs
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def trajectory_consistency_metrics(
    prediction: torch.Tensor,
    ego_past: torch.Tensor,
    ego_current: torch.Tensor,
    dt: float = TRAJECTORY_CONSISTENCY_DT,
) -> dict[str, torch.Tensor]:
    """Measure single-output temporal consistency of the predicted ego trajectory.

    This is an online-validation proxy for Autoware's trajectory consistency/stability
    metrics. Autoware compares different historical trajectories at a common future time;
    the training validator only has one model output per sample, so this metric instead
    checks whether the predicted trajectory connects smoothly to observed ego history and
    remains dynamically continuous inside the predicted horizon.

    Returns per-sample lower-is-better tensors.
    """
    ego_pred = prediction[:, 0]
    pred_xy = ego_pred[..., :2]
    current_xy = ego_current[:, :2]
    B = pred_xy.shape[0]
    device = pred_xy.device
    dtype = pred_xy.dtype

    points = torch.cat([current_xy[:, None], pred_xy], dim=1)
    velocity = (points[:, 1:] - points[:, :-1]) / dt
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt if velocity.shape[1] > 1 else None
    jerk = (
        (acceleration[:, 1:] - acceleration[:, :-1]) / dt
        if acceleration is not None and acceleration.shape[1] > 1
        else None
    )

    if ego_past.shape[1] >= 2:
        past_velocity = (current_xy - ego_past[:, -2, :2]) / dt
        velocity_boundary_error = torch.linalg.vector_norm(velocity[:, 0] - past_velocity, dim=-1)
    else:
        past_velocity = torch.zeros(B, 2, device=device, dtype=dtype)
        velocity_boundary_error = torch.zeros(B, device=device, dtype=dtype)

    if ego_past.shape[1] >= 3 and acceleration is not None and acceleration.shape[1] > 0:
        previous_past_velocity = (ego_past[:, -2, :2] - ego_past[:, -3, :2]) / dt
        past_acceleration = (past_velocity - previous_past_velocity) / dt
        acceleration_boundary_error = torch.linalg.vector_norm(
            acceleration[:, 0] - past_acceleration, dim=-1
        )
    else:
        acceleration_boundary_error = torch.zeros(B, device=device, dtype=dtype)

    if jerk is not None and jerk.numel() > 0:
        jerk_rms = torch.sqrt(torch.clamp(torch.sum(jerk**2, dim=-1).mean(dim=1), min=0.0))
    else:
        jerk_rms = torch.zeros(B, device=device, dtype=dtype)

    if ego_pred.shape[-1] >= 4 and ego_current.shape[-1] >= 4:
        pred_heading = torch.atan2(ego_pred[..., 3], ego_pred[..., 2])
        current_heading = torch.atan2(ego_current[:, 3], ego_current[:, 2])
        heading = torch.cat([current_heading[:, None], pred_heading], dim=1)
        heading_step = _angle_diff(heading[:, 1:], heading[:, :-1])
        heading_boundary_error = torch.abs(heading_step[:, 0])
        heading_rate = heading_step / dt
        if heading_rate.shape[1] > 1:
            heading_rate_change = torch.abs(heading_rate[:, 1:] - heading_rate[:, :-1]).mean(dim=1)
        else:
            heading_rate_change = torch.zeros(B, device=device, dtype=dtype)
    else:
        heading_boundary_error = torch.zeros(B, device=device, dtype=dtype)
        heading_rate_change = torch.zeros(B, device=device, dtype=dtype)

    consistency_error = torch.sqrt(
        (
            (velocity_boundary_error / TRAJECTORY_CONSISTENCY_VEL_TOL) ** 2
            + (acceleration_boundary_error / TRAJECTORY_CONSISTENCY_ACC_TOL) ** 2
            + (jerk_rms / TRAJECTORY_CONSISTENCY_JERK_TOL) ** 2
            + (heading_boundary_error / TRAJECTORY_CONSISTENCY_HEADING_TOL) ** 2
            + (heading_rate_change / TRAJECTORY_CONSISTENCY_HEADING_RATE_TOL) ** 2
        )
        / 5.0
    )

    return {
        "ego_trajectory_consistency_error": consistency_error,
        "ego_trajectory_consistency_velocity_boundary": velocity_boundary_error,
        "ego_trajectory_consistency_acceleration_boundary": acceleration_boundary_error,
        "ego_trajectory_consistency_jerk": jerk_rms,
        "ego_trajectory_consistency_heading_boundary": heading_boundary_error,
        "ego_trajectory_consistency_heading_rate_change": heading_rate_change,
    }


def _line_strings_to_pdms_border_lines(line_strings: torch.Tensor) -> list[list[np.ndarray]]:
    """Extract road-border polylines for PDMS DAC from DP line_strings.

    This intentionally mirrors loss.compute_road_border_penalty: a line string is a
    road border when channel 3 is active on any point. Points with zero xy are padding.
    Returned polylines stay in the same ego-centric frame as prediction/GT.
    """
    arr = line_strings.detach().cpu().numpy()
    out: list[list[np.ndarray]] = []
    for sample in arr:
        sample_lines: list[np.ndarray] = []
        rb_mask = (sample[..., 3] > 0.5).any(axis=-1)
        for row in sample[rb_mask]:
            pts = row[:, :2]
            valid = np.abs(pts).sum(axis=-1) > 1e-6
            pts = pts[valid]
            if pts.shape[0] >= 2:
                sample_lines.append(np.asarray(pts, dtype=np.float64))
        out.append(sample_lines)
    return out


def _neighbors_to_pdms_agent_boxes(
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    neighbor_agents_past: torch.Tensor,
    dt: float,
) -> list[list[np.ndarray]]:
    """Convert DP neighbor GT futures into PDMS-compatible per-timestep boxes.

    OnePlanner's full PDMS uses .gt.tar boxes. DP NPZs already carry future neighbor
    centers/headings plus current width/length in neighbor_agents_past, so we construct
    the same [x, y, z, w, l, h, yaw, vx, vy] layout. This enables NC/TTC as an
    open-loop proxy without requiring extra sidecar map/GT files.
    """
    nf = neighbors_future.detach().cpu().numpy()
    valid = neighbors_future_valid.detach().cpu().numpy().astype(bool)
    past = neighbor_agents_past.detach().cpu().numpy()
    B, Pn, T, _ = nf.shape
    result: list[list[np.ndarray]] = []
    for b in range(B):
        width = past[b, :Pn, -1, 6] if past.shape[-1] > 6 else np.full(Pn, 2.0)
        length = past[b, :Pn, -1, 7] if past.shape[-1] > 7 else np.full(Pn, 4.5)
        width = np.where(np.abs(width) > 1e-3, np.abs(width), 2.0)
        length = np.where(np.abs(length) > 1e-3, np.abs(length), 4.5)
        pos = nf[b, :, :, :2]
        vx = np.zeros((Pn, T), dtype=np.float64)
        vy = np.zeros((Pn, T), dtype=np.float64)
        if T >= 2:
            vx[:, 1:] = (pos[:, 1:, 0] - pos[:, :-1, 0]) / dt
            vy[:, 1:] = (pos[:, 1:, 1] - pos[:, :-1, 1]) / dt
            vx[:, 0] = vx[:, 1]
            vy[:, 0] = vy[:, 1]
        sample_boxes: list[np.ndarray] = []
        for t in range(T):
            idx = np.where(valid[b, :, t])[0]
            boxes = np.zeros((idx.shape[0], 9), dtype=np.float64)
            if idx.shape[0] > 0:
                boxes[:, 0] = nf[b, idx, t, 0]
                boxes[:, 1] = nf[b, idx, t, 1]
                boxes[:, 3] = width[idx]
                boxes[:, 4] = length[idx]
                boxes[:, 5] = 1.5
                boxes[:, 6] = np.arctan2(nf[b, idx, t, 3], nf[b, idx, t, 2])
                boxes[:, 7] = vx[idx, t]
                boxes[:, 8] = vy[idx, t]
            sample_boxes.append(boxes)
        result.append(sample_boxes)
    return result


def _pdms_eval_metrics(
    prediction: torch.Tensor,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    denorm_inputs: dict[str, torch.Tensor],
    args,
) -> dict[str, torch.Tensor]:
    if pdms_proxy is None:
        raise RuntimeError(
            'PDMS eval is enabled, but planner_metrics.pdms_proxy could not be imported: '
            f'{_PDMS_IMPORT_ERROR!r}'
        )
    ego_pred = prediction[:, 0]
    ego_dims = denorm_inputs['ego_shape'].detach().cpu().numpy()
    border_lines = None
    if getattr(args, 'pdms_eval_use_road_border', True) and 'line_strings' in denorm_inputs:
        border_lines = _line_strings_to_pdms_border_lines(denorm_inputs['line_strings'])
    agent_boxes = None
    if getattr(args, 'pdms_eval_use_agent_boxes', True):
        agent_boxes = _neighbors_to_pdms_agent_boxes(
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs['neighbor_agents_past'],
            TRAJECTORY_CONSISTENCY_DT,
        )
    out = pdms_proxy(
        ego_pred,
        ego_future,
        agent_boxes_per_t=agent_boxes,
        ego_dims=ego_dims,
        border_lines=border_lines,
        dt=TRAJECTORY_CONSISTENCY_DT,
    )
    return {k: v if torch.is_tensor(v) else torch.as_tensor(v, device=ego_pred.device) for k, v in out.items()}


@torch.no_grad()
def validate_model(model, val_loader, args, return_pred=False) -> tuple[float, float]:
    """return: ave_loss_ego, ave_loss_neighbor"""
    device = args.device
    model.eval()
    total_loss_ego = 0.0
    total_loss_neighbor = 0.0
    total_samples_ego = 0
    total_samples_neighbor = 0

    predictions = []
    turn_indicators = []
    loss_ego_list = []

    total_result_dict = defaultdict(list)
    turn_indicator_correct = 0.0
    turn_indicator_total = 0
    turn_indicator_change_correct = 0.0
    turn_indicator_change_total = 0

    delay = 0

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
    cuda_device = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        cuda_device = torch.cuda.current_device()
        cuda_rng_state = torch.cuda.get_rng_state(cuda_device)
    torch.manual_seed(getattr(args, "seed", 0) + 20260627)
    if cuda_rng_state is not None:
        torch.cuda.manual_seed_all(getattr(args, "seed", 0) + 20260627)

    for inputs in tqdm(val_loader, desc="validate", disable=ddp.get_rank() != 0):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        B = inputs["ego_current_state"].shape[0]

        turn_indicator_seq = inputs["turn_indicators"]

        inputs["sampled_trajectories"] = torch.zeros(
            B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=device
        )
        inputs["delay"] = torch.full((B,), delay, dtype=torch.float32, device=device)

        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
        ego_past_for_consistency = inputs["ego_agent_past"].clone()

        ego_future = inputs["ego_agent_future"]
        ego_future = heading_to_cos_sin(ego_future)  # (B, T, 4)
        neighbors_future = inputs["neighbor_agents_future"]
        neighbor_future_mask = (
            torch.sum(torch.ne(neighbors_future[..., :4], 0), dim=-1) == 0
        )  # (B, Pn, T)
        neighbors_future = heading_to_cos_sin(neighbors_future)  # (B, Pn, T, 4)
        neighbors_future[neighbor_future_mask] = 0.0

        B, Pn, T, _ = neighbors_future.shape
        ego_current, neighbors_current = (
            inputs["ego_current_state"][:, :4],
            inputs["neighbor_agents_past"][:, :Pn, -1, :4],
        )
        ego_current_for_consistency = ego_current.clone()
        inputs = args.observation_normalizer(inputs)

        _, outputs = model(inputs)

        neighbor_current_mask = (
            torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        )  # (B, Pn)
        neighbor_mask = torch.concat(
            (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
        )  # (B, Pn, T + 1)

        gt_future = torch.cat(
            [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
        )  # (B, Pn + 1, T, 4)
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
        # (B, Pn + 1, 4)

        all_gt = torch.cat(
            [current_states[:, :, None, :], gt_future], dim=2
        )  # (B, Pn + 1, T + 1, 4)
        all_gt[:, 1:][neighbor_mask] = 0.0

        prediction = outputs["prediction"]
        if "dfp_gate" in outputs:
            dfp_gate = outputs["dfp_gate"].detach().squeeze(-1)
            total_result_dict["gate_dfp_weight"].append(dfp_gate.cpu())
            total_result_dict["gate_dfp_weight_std_per_sample"].append(
                dfp_gate.std(dim=1, unbiased=False).cpu()
            )
            total_result_dict["gate_dfp_weight_min_per_sample"].append(dfp_gate.amin(dim=1).cpu())
            total_result_dict["gate_dfp_weight_max_per_sample"].append(dfp_gate.amax(dim=1).cpu())
        if "dfp_original_gate" in outputs:
            original_gate = outputs["dfp_original_gate"].detach().squeeze(-1)
            total_result_dict["gate_original_weight"].append(original_gate.cpu())
            total_result_dict["gate_original_weight_std_per_sample"].append(
                original_gate.std(dim=1, unbiased=False).cpu()
            )
            total_result_dict["gate_original_weight_min_per_sample"].append(original_gate.amin(dim=1).cpu())
            total_result_dict["gate_original_weight_max_per_sample"].append(original_gate.amax(dim=1).cpu())
        consistency_dict = trajectory_consistency_metrics(
            prediction, ego_past_for_consistency, ego_current_for_consistency
        )
        for key, val in consistency_dict.items():
            total_result_dict[key].append(val.cpu())

        turn_indicator_logit = outputs["turn_indicator_logit"]
        turn_indicator = turn_indicator_logit.argmax(dim=-1)
        turn_indicator_gt = make_turn_indicator_gt(turn_indicator_seq)
        correct = (turn_indicator == turn_indicator_gt).long()
        turn_indicator_correct += correct.sum().item()
        turn_indicator_total += correct.numel()
        change_mask = turn_indicator_seq[:, -1] != turn_indicator_seq[:, -2]
        change_count = change_mask.sum().item()
        if change_count > 0:
            turn_indicator_change_correct += correct[change_mask].sum().item()
            turn_indicator_change_total += change_count
        if return_pred:
            predictions.append(prediction.cpu())
            turn_indicators.append(turn_indicator.cpu())

        neighbors_future_valid = ~neighbor_future_mask
        all_gt = all_gt[:, :, 1:, :]  # (B, Pn + 1, T, 4)
        loss_tensor = (prediction - all_gt) ** 2
        loss_ego = loss_tensor[:, 0, :]
        loss_ego_list.append(loss_ego.cpu())
        loss_nei = loss_tensor[:, 1:, :]
        loss_nei = loss_nei[neighbors_future_valid]
        total_loss_ego += loss_ego.mean().item() * B
        total_samples_ego += B
        if loss_nei.shape[0] > 0:
            nei_B = loss_nei.shape[0]
            total_loss_neighbor += loss_nei.mean().item() * nei_B
            total_samples_neighbor += nei_B

        loss_dict = loss_func(prediction, all_gt)
        for key, val in loss_dict.items():
            # val : (B, Pn + 1, T)
            total_result_dict[f"ego_{key}"].append(val[:, 0, :].cpu())  # (B, T)

        # Compute ego edge points for penalty metrics
        ego_edge_points = compute_ego_edge_points(
            prediction[:, 0], inputs["ego_shape"], n_interp=args.road_border_n_interp
        )

        denorm_inputs = args.observation_normalizer.inverse(inputs)
        neighbor_penalty = compute_neighbor_collision_penalty(
            ego_edge_points,
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin=args.neighbor_collision_margin,
        )
        total_result_dict["ego_neighbor_margin_loss"].append(neighbor_penalty.cpu())

        # Road border collision metric
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )
        total_result_dict["ego_road_border_loss"].append(rb_penalty.cpu())


        if getattr(args, "enable_pdms_eval", False):
            pdms_dict = _pdms_eval_metrics(
                prediction,
                ego_future,
                neighbors_future,
                neighbors_future_valid,
                denorm_inputs,
                args,
            )
            for key, val in pdms_dict.items():
                total_result_dict[f"pdms_{key}"].append(val.detach().cpu())

    avg_loss_ego = total_loss_ego / total_samples_ego
    avg_loss_neighbor = total_loss_neighbor / max(total_samples_neighbor, 1)
    loss_ego = torch.cat(loss_ego_list, dim=0)

    for key, val in total_result_dict.items():
        total_result_dict[key] = torch.cat(val, dim=0)  # (total_samples, T)

    if return_pred:
        predictions = torch.cat(predictions, dim=0)
        turn_indicators = torch.cat(turn_indicators, dim=0)

    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None and cuda_device is not None:
        torch.cuda.set_rng_state(cuda_rng_state, cuda_device)

    turn_indicator_accuracy = (
        turn_indicator_correct / turn_indicator_total if turn_indicator_total > 0 else 0.0
    )
    turn_indicator_change_accuracy = (
        turn_indicator_change_correct / turn_indicator_change_total
        if turn_indicator_change_total > 0
        else 0.0
    )
    return {
        "avg_loss_ego": avg_loss_ego,
        "avg_loss_neighbor": avg_loss_neighbor,
        "loss_ego": loss_ego,
        "predictions": predictions,
        "turn_indicators": turn_indicators,
        "turn_indicator_accuracy": turn_indicator_accuracy,
        "turn_indicator_change_accuracy": turn_indicator_change_accuracy,
        "turn_indicator_change_total": turn_indicator_change_total,
        **total_result_dict,
    }
