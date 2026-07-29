from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

try:
    from planner_metrics.pdms_proxy import add_synthetic_epdms, pdms_proxy
except Exception as exc:  # optional metric dependency; raised only when enabled
    add_synthetic_epdms = None
    pdms_proxy = None
    _EPDMS_IMPORT_ERROR = exc
else:
    _EPDMS_IMPORT_ERROR = None

from diffusion_planner.dimensions import OUTPUT_T, POSE_DIM, TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    loss_func,
    make_turn_indicator_gt,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp
from diffusion_planner.utils.masks import neighbor_future_padding_mask

EPDMS_DT = 0.1
MULTISAMPLE_ADE_THRESHOLD_M = 4.0
MULTISAMPLE_FDE_THRESHOLD_M = 8.0
TURN_INDICATOR_CLASS_NAMES = ("disable", "enable_left", "enable_right")
# Reliability-diagram resolution for the intent head. The deployment state machine gates
# on absolute probabilities, so how well those probabilities mean what they say is a
# first-class metric here, not a post-hoc curiosity.
TURN_INDICATOR_CALIBRATION_BINS = 10


def turn_indicator_metrics_from_confusion(confusion: list[int]) -> dict[str, float]:
    """Compute robust state/activation metrics from a flattened 3x3 matrix."""
    if len(confusion) != TURN_INDICATOR_OUTPUT_DIM**2:
        raise ValueError(
            f"Expected {TURN_INDICATOR_OUTPUT_DIM**2} confusion entries, got {len(confusion)}"
        )
    matrix = [
        confusion[row * TURN_INDICATOR_OUTPUT_DIM : (row + 1) * TURN_INDICATOR_OUTPUT_DIM]
        for row in range(TURN_INDICATOR_OUTPUT_DIM)
    ]
    total = sum(confusion)
    correct = sum(matrix[index][index] for index in range(TURN_INDICATOR_OUTPUT_DIM))
    recalls = []
    f1_scores = []
    for index in range(TURN_INDICATOR_OUTPUT_DIM):
        tp = matrix[index][index]
        actual = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(TURN_INDICATOR_OUTPUT_DIM))
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        recalls.append(recall)
        f1_scores.append(
            2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )

    active_true = sum(sum(matrix[row]) for row in (1, 2))
    active_predicted = sum(matrix[row][column] for row in range(3) for column in (1, 2))
    active_detected = sum(matrix[row][column] for row in (1, 2) for column in (1, 2))
    active_precision = active_detected / active_predicted if active_predicted else 0.0
    active_recall = active_detected / active_true if active_true else 0.0
    return {
        "turn_indicator_accuracy": correct / total if total else 0.0,
        "turn_indicator_balanced_accuracy": sum(recalls) / len(recalls),
        "turn_indicator_macro_f1": sum(f1_scores) / len(f1_scores),
        "turn_indicator_active_precision": active_precision,
        "turn_indicator_active_recall": active_recall,
        "turn_indicator_active_f1": (
            2.0 * active_precision * active_recall / (active_precision + active_recall)
            if active_precision + active_recall > 0
            else 0.0
        ),
        # Requires both activation and the correct side; unlike active recall,
        # a LEFT/RIGHT confusion is counted as incorrect.
        "turn_indicator_direction_accuracy": (
            (matrix[1][1] + matrix[2][2]) / active_true if active_true else 0.0
        ),
    }


def turn_indicator_calibration_counts(
    logit: torch.Tensor,  # [B, TURN_INDICATOR_OUTPUT_DIM]
    target: torch.Tensor,  # [B] dense OFF/LEFT/RIGHT labels
) -> list[float]:
    """Reliability increments for one batch, laid out as a flat all-reducible list.

    Layout: ``[samples, nll_sum, per-bin count..., per-bin confidence sum...,
    per-bin correct count...]``, binned on the top-class probability the state machine
    actually thresholds.
    """
    probabilities = torch.softmax(logit.detach().float(), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    nll_sum = float(
        -probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        .log()
        .gather(1, target.view(-1, 1))
        .sum()
    )
    bins = TURN_INDICATOR_CALIBRATION_BINS
    # A confidence of exactly 1.0 belongs to the last bin, not to a nonexistent bin+1.
    index = (confidence * bins).long().clamp(max=bins - 1)
    counts = torch.bincount(index, minlength=bins).double()
    confidence_sums = torch.bincount(index, weights=confidence.double(), minlength=bins)
    correct = (prediction == target).double()
    correct_sums = torch.bincount(index, weights=correct, minlength=bins)
    return [
        float(target.numel()),
        nll_sum,
        *(
            value
            for tensor in (counts, confidence_sums, correct_sums)
            for value in tensor.cpu().tolist()
        ),
    ]


def turn_indicator_calibration_metrics(counts: list[float]) -> dict[str, float]:
    """Mean NLL and expected calibration error from accumulated reliability counts."""
    bins = TURN_INDICATOR_CALIBRATION_BINS
    if len(counts) != 2 + 3 * bins:
        raise ValueError(f"Expected {2 + 3 * bins} calibration entries, got {len(counts)}")
    samples, nll_sum = counts[0], counts[1]
    bin_counts = counts[2 : 2 + bins]
    bin_confidence = counts[2 + bins : 2 + 2 * bins]
    bin_correct = counts[2 + 2 * bins :]
    if samples <= 0:
        return {"turn_indicator_nll": 0.0, "turn_indicator_ece": 0.0}
    ece = sum(
        abs(bin_correct[index] - bin_confidence[index]) / samples
        for index in range(bins)
        if bin_counts[index] > 0
    )
    return {"turn_indicator_nll": nll_sum / samples, "turn_indicator_ece": ece}


def _valid_xy(points: np.ndarray) -> np.ndarray:
    return np.abs(points[..., :2]).sum(axis=-1) > 1e-6


def _line_strings_to_epdms_border_lines(line_strings: torch.Tensor) -> list[list[np.ndarray]]:
    """Extract road-border polylines from DP line_strings."""
    arr = line_strings.detach().cpu().numpy()
    out: list[list[np.ndarray]] = []
    for sample in arr:
        sample_lines: list[np.ndarray] = []
        road_border_mask = (sample[..., 3] > 0.5).any(axis=-1)
        for row in sample[road_border_mask]:
            pts = row[:, :2]
            # Channel 3 is the explicit per-point road-border validity flag.
            # Keep flagged points even when their valid coordinate is exactly
            # the ego origin; using only ``(x, y) != 0`` drops that legitimate
            # endpoint and biases border-based validation metrics.
            point_valid = (row[:, 3] > 0.5) | _valid_xy(pts)
            pts = pts[point_valid]
            if pts.shape[0] >= 2:
                sample_lines.append(np.asarray(pts, dtype=np.float64))
        out.append(sample_lines)
    return out


def _lane_tensor_to_polygons(lanes: torch.Tensor) -> list[list[np.ndarray]]:
    """Convert DP lane tensors to ego-frame lane polygons."""
    arr = lanes.detach().cpu().numpy()
    out: list[list[np.ndarray]] = []
    for sample in arr:
        sample_polys: list[np.ndarray] = []
        for lane in sample:
            # Lane tensors use the first eight geometry channels as their
            # validity mask.  A valid lane point can sit at (0, 0) as long as
            # its direction or boundary offsets are populated.
            valid = np.abs(lane[..., :8]).sum(axis=-1) > 1e-6
            if valid.sum() < 2:
                continue
            center = lane[valid, :2]
            left = center + lane[valid, 4:6]
            right = center + lane[valid, 6:8]
            ring = np.concatenate([left, right[::-1]], axis=0)
            if ring.shape[0] >= 3:
                sample_polys.append(np.asarray(ring, dtype=np.float64))
        out.append(sample_polys)
    return out


def _lane_tensor_to_centerlines(lanes: torch.Tensor) -> list[list[np.ndarray]]:
    arr = lanes.detach().cpu().numpy()
    out: list[list[np.ndarray]] = []
    for sample in arr:
        sample_lines: list[np.ndarray] = []
        for lane in sample:
            pts = lane[:, :2]
            valid = np.abs(lane[..., :8]).sum(axis=-1) > 1e-6
            pts = pts[valid]
            if pts.shape[0] >= 2:
                sample_lines.append(np.asarray(pts, dtype=np.float64))
        out.append(sample_lines)
    return out


def _polygons_to_rings(polygons: torch.Tensor) -> list[list[np.ndarray]]:
    arr = polygons.detach().cpu().numpy()
    out: list[list[np.ndarray]] = []
    for sample in arr:
        sample_rings: list[np.ndarray] = []
        for poly in sample:
            pts = poly[:, :2]
            pts = pts[_valid_xy(pts)]
            if pts.shape[0] >= 3:
                sample_rings.append(np.asarray(pts, dtype=np.float64))
        out.append(sample_rings)
    return out


def _neighbors_to_epdms_agent_boxes(
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    neighbor_agents_past: torch.Tensor,
    dt: float,
) -> list[list[np.ndarray]]:
    """Convert DP neighbor GT futures to [N, 9] per-timestep boxes."""
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


@torch.no_grad()
def _multisample_metrics(
    model,
    norm_inputs: dict[str, torch.Tensor],
    cached_encoding: torch.Tensor,
    ego_future: torch.Tensor,
    args,
    batch_step: int,
) -> dict[str, torch.Tensor]:
    """Seeded multi-sample minADE/minFDE open-loop metrics."""
    num_samples = int(getattr(args, "multisample_eval_num_samples", 0))
    if num_samples <= 0:
        return {}

    B = ego_future.shape[0]
    device = ego_future.device
    noise_scale = float(getattr(args, "multisample_eval_noise_scale", 0.1))
    base_seed = int(getattr(args, "multisample_eval_seed", 3407))
    sample_steps = int(getattr(args, "multisample_eval_sample_steps", 6))

    net = getattr(model, "module", model)
    decoder = getattr(net, "decoder", None)
    action_agent_num = 1 + decoder._predicted_neighbor_num
    old_sample_steps = getattr(decoder, "_sample_steps", None)
    if old_sample_steps is not None:
        decoder._sample_steps = sample_steps

    sampled_noise = []
    try:
        for sample_idx in range(num_samples):
            generator = torch.Generator(device=device)
            generator.manual_seed(
                base_seed + ddp.get_rank() * 1_000_003 + batch_step * num_samples + sample_idx
            )
            sampled_noise.append(
                torch.randn(
                    B,
                    action_agent_num,
                    OUTPUT_T,
                    POSE_DIM,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                * noise_scale
            )
        sampled = torch.stack(sampled_noise, dim=1).reshape(
            B * num_samples, action_agent_num, OUTPUT_T, POSE_DIM
        )
        repeated_encoding = (
            cached_encoding[:, None]
            .expand(B, num_samples, *cached_encoding.shape[1:])
            .reshape(B * num_samples, *cached_encoding.shape[1:])
        )
        ego_current_state = (
            norm_inputs["ego_current_state"][:, None]
            .expand(B, num_samples, -1)
            .reshape(B * num_samples, -1)
        )
        use_bf16 = getattr(args, "amp_dtype", "off") == "bf16" and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            global_route_condition = decoder.global_route_encoder(norm_inputs["route_lanes"])
        repeated_global_route_condition = (
            global_route_condition[:, None].expand(B, num_samples, -1).reshape(B * num_samples, -1)
        )
        inference_inputs = {
            "sampled_trajectories": sampled,
            "ego_current_state": ego_current_state,
            "_cached_encoding": repeated_encoding,
            "_cached_global_route_condition": repeated_global_route_condition,
            # Multi-sample metrics only consume the denoised trajectory. Avoid
            # evaluating the auxiliary classifier once per candidate.
            "_skip_turn_indicator": True,
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _, outputs = model(inference_inputs)
        sampled_xy = outputs["prediction"][:, 0, :, :2].reshape(B, num_samples, OUTPUT_T, 2)
    finally:
        if old_sample_steps is not None:
            decoder._sample_steps = old_sample_steps

    gt_xy = ego_future[..., :2]
    T = min(sampled_xy.shape[2], gt_xy.shape[1])
    sampled_xy = sampled_xy[:, :, :T]
    gt_xy = gt_xy[:, :T]

    distance = torch.linalg.norm(sampled_xy - gt_xy[:, None], dim=-1)
    # Ego futures have a fixed full-horizon contract and no padding mask. A stopped vehicle can
    # legitimately have x=y=yaw=0, so inferring validity from non-zero xy would drop precisely the
    # stationary scenes whose behavior the planner must preserve.
    ade = distance.mean(dim=-1)
    fde = distance[..., -1]

    min_ade = ade.min(dim=1).values
    min_fde = fde.min(dim=1).values
    endpoints = sampled_xy[..., -1, :]
    endpoint_center = endpoints.mean(dim=1, keepdim=True)
    endpoint_divergence = torch.linalg.norm(endpoints - endpoint_center, dim=-1).mean(dim=1)
    return {
        "multisample_minADE": min_ade,
        "multisample_minFDE": min_fde,
        "multisample_ADE_score": 100.0
        * (1.0 - min_ade / MULTISAMPLE_ADE_THRESHOLD_M).clamp(0.0, 1.0),
        "multisample_FDE_score": 100.0
        * (1.0 - min_fde / MULTISAMPLE_FDE_THRESHOLD_M).clamp(0.0, 1.0),
        "multisample_endpoint_divergence": endpoint_divergence,
    }


def _epdms_eval_metrics(
    prediction: torch.Tensor,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    denorm_inputs: dict[str, torch.Tensor],
    args,
) -> dict[str, torch.Tensor]:
    if pdms_proxy is None or add_synthetic_epdms is None:
        raise RuntimeError(
            "EPDMS eval is enabled, but planner_metrics.pdms_proxy could not be imported: "
            f"{_EPDMS_IMPORT_ERROR!r}"
        )

    ego_pred = prediction[:, 0]
    ego_dims = denorm_inputs["ego_shape"].detach().cpu().numpy()

    border_lines = None
    if getattr(args, "epdms_eval_use_road_border", True) and "line_strings" in denorm_inputs:
        border_lines = _line_strings_to_epdms_border_lines(denorm_inputs["line_strings"])

    agent_boxes = None
    if getattr(args, "epdms_eval_use_agent_boxes", True):
        agent_boxes = _neighbors_to_epdms_agent_boxes(
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            EPDMS_DT,
        )

    lane_rings = (
        _lane_tensor_to_polygons(denorm_inputs["lanes"]) if "lanes" in denorm_inputs else None
    )
    route_polys = (
        _lane_tensor_to_polygons(denorm_inputs["route_lanes"])
        if "route_lanes" in denorm_inputs
        else None
    )
    route_centerlines = (
        _lane_tensor_to_centerlines(denorm_inputs["route_lanes"])
        if "route_lanes" in denorm_inputs
        else None
    )
    intersection_rings = (
        _polygons_to_rings(denorm_inputs["polygons"]) if "polygons" in denorm_inputs else None
    )

    kwargs = dict(
        agent_boxes_per_t=agent_boxes,
        ego_dims=ego_dims,
        border_lines=border_lines,
        route_polys=route_polys,
        lane_rings=lane_rings,
        intersection_rings=intersection_rings,
        route_centerlines=route_centerlines,
        dt=EPDMS_DT,
    )
    agent = pdms_proxy(ego_pred, ego_future, add_aggregation=False, **kwargs)
    human = pdms_proxy(
        ego_future,
        ego_future,
        add_aggregation=False,
        self_reference_progress=True,
        **kwargs,
    )
    add_synthetic_epdms(agent, human)
    return {
        key: value if torch.is_tensor(value) else torch.as_tensor(value, device=ego_pred.device)
        for key, value in agent.items()
    }


@torch.no_grad()
def validate_model(model, val_loader, args, return_pred=False) -> tuple[float, float]:
    """return: ave_loss_ego, ave_loss_neighbor"""
    device = args.device
    # The production policy stage has an intentionally untrained/frozen intent
    # head. Do not spend validation time running it or publish random head metrics
    # as if they measured the trajectory policy. Joint and head-only stages still
    # evaluate the head normally.
    evaluate_turn_indicator = getattr(args, "supervised_training_stage", "policy") != "policy"
    model.eval()
    total_loss_ego = 0.0
    total_loss_neighbor = 0.0
    total_samples_ego = 0
    total_samples_neighbor = 0

    predictions = []
    turn_indicators = []
    turn_indicator_logits = []
    loss_ego_list = []

    total_result_dict = defaultdict(list)
    turn_indicator_confusion = [0] * (TURN_INDICATOR_OUTPUT_DIM**2)
    turn_indicator_calibration = [0.0] * (2 + 3 * TURN_INDICATOR_CALIBRATION_BINS)

    total_batches = len(val_loader)
    pbar = tqdm(total=total_batches, desc="validate", disable=ddp.get_rank() != 0)
    net = getattr(model, "module", model)
    action_neighbor_num = net.decoder._predicted_neighbor_num
    action_agent_num = 1 + action_neighbor_num
    for step, inputs in enumerate(val_loader):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        B = inputs["ego_current_state"].shape[0]

        turn_indicator_seq = inputs["turn_indicators"] if evaluate_turn_indicator else None

        inputs["sampled_trajectories"] = torch.zeros(
            B, action_agent_num, OUTPUT_T, POSE_DIM, dtype=torch.float32, device=device
        )
        # Origin-frame ego/goal poses are valid when x=y=heading=0.
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        ego_future = heading_to_cos_sin(ego_future)  # (B, T, 4)
        all_neighbors_future = inputs["neighbor_agents_future"]
        all_neighbor_future_mask = neighbor_future_padding_mask(all_neighbors_future)
        all_neighbors_future = heading_to_cos_sin(all_neighbors_future)
        all_neighbors_future[all_neighbor_future_mask] = 0.0
        neighbors_future = all_neighbors_future[:, :action_neighbor_num]
        neighbor_future_mask = all_neighbor_future_mask[:, :action_neighbor_num]

        B, Pn, T, _ = neighbors_future.shape
        ego_current, neighbors_current = (
            inputs["ego_current_state"][:, :4],
            inputs["neighbor_agents_past"][:, :Pn, -1, :4],
        )
        inputs = args.observation_normalizer(inputs)
        if not evaluate_turn_indicator:
            inputs["_skip_turn_indicator"] = True

        use_bf16 = getattr(args, "amp_dtype", "off") == "bf16" and str(device).startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            encoder_outputs, outputs = model(inputs)

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
        turn_indicator_logit = outputs.get("turn_indicator_logit")
        turn_logit_finite = turn_indicator_logit is not None and bool(
            torch.isfinite(turn_indicator_logit).all()
        )
        if not torch.isfinite(prediction).all() or (
            evaluate_turn_indicator and not turn_logit_finite
        ):
            raise FloatingPointError(
                f"Non-finite validation model output at batch {step}: "
                f"prediction_finite={bool(torch.isfinite(prediction).all())}, "
                f"turn_logit_finite={turn_logit_finite}"
            )
        multisample_metrics = _multisample_metrics(
            model, inputs, encoder_outputs, ego_future, args, step
        )
        for key, value in multisample_metrics.items():
            total_result_dict[key].append(value.cpu())
        if evaluate_turn_indicator:
            turn_indicator = turn_indicator_logit.argmax(dim=-1)
            turn_indicator_gt = make_turn_indicator_gt(turn_indicator_seq)
            batch_confusion = (
                torch.bincount(
                    turn_indicator_gt * TURN_INDICATOR_OUTPUT_DIM + turn_indicator,
                    minlength=TURN_INDICATOR_OUTPUT_DIM**2,
                )
                .cpu()
                .tolist()
            )
            turn_indicator_confusion = [
                old + int(new)
                for old, new in zip(turn_indicator_confusion, batch_confusion, strict=False)
            ]
            batch_calibration = turn_indicator_calibration_counts(
                turn_indicator_logit, turn_indicator_gt
            )
            turn_indicator_calibration = [
                old + new
                for old, new in zip(turn_indicator_calibration, batch_calibration, strict=True)
            ]
        if return_pred:
            predictions.append(prediction.cpu())
            if evaluate_turn_indicator:
                turn_indicators.append(turn_indicator.cpu())
                turn_indicator_logits.append(turn_indicator_logit.float().cpu())

        neighbors_future_valid = ~neighbor_future_mask
        all_neighbors_future_valid = ~all_neighbor_future_mask
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
            all_neighbors_future,
            all_neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin_vehicle=args.neighbor_collision_margin_vehicle,
            margin_pedestrian=args.neighbor_collision_margin_pedestrian,
            margin_bicycle=args.neighbor_collision_margin_bicycle,
        )
        total_result_dict["ego_neighbor_margin_loss"].append(neighbor_penalty.cpu())

        # Road border collision metric
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )
        total_result_dict["ego_road_border_loss"].append(rb_penalty.cpu())

        if getattr(args, "enable_epdms_eval", False):
            epdms_dict = _epdms_eval_metrics(
                prediction,
                ego_future,
                all_neighbors_future,
                all_neighbors_future_valid,
                denorm_inputs,
                args,
            )
            for key, val in epdms_dict.items():
                total_result_dict[f"epdms_{key}"].append(val.detach().cpu())

        if ddp.get_rank() == 0:
            pbar.update(1)
    pbar.close()

    avg_loss_ego = total_loss_ego / total_samples_ego
    avg_loss_neighbor = total_loss_neighbor / max(total_samples_neighbor, 1)
    loss_ego = torch.cat(loss_ego_list, dim=0)

    for key, val in total_result_dict.items():
        total_result_dict[key] = torch.cat(val, dim=0)  # (total_samples, T)

    if return_pred:
        predictions = torch.cat(predictions, dim=0)
        if evaluate_turn_indicator:
            turn_indicators = torch.cat(turn_indicators, dim=0)
            turn_indicator_logits = torch.cat(turn_indicator_logits, dim=0)
        else:
            turn_indicators = torch.empty(0, dtype=torch.long)
            turn_indicator_logits = torch.empty(0, TURN_INDICATOR_OUTPUT_DIM)
    turn_metrics = turn_indicator_metrics_from_confusion(turn_indicator_confusion)
    turn_metrics.update(turn_indicator_calibration_metrics(turn_indicator_calibration))
    return {
        "avg_loss_ego": avg_loss_ego,
        "avg_loss_neighbor": avg_loss_neighbor,
        "loss_ego": loss_ego,
        "predictions": predictions,
        "turn_indicators": turn_indicators,
        "turn_indicator_logits": turn_indicator_logits,
        **turn_metrics,
        # Raw per-rank accumulators, kept so callers that run validation on ALL ranks
        # (non-padding distributed eval shards) can all-reduce them into global metrics
        # via aggregate_valid_metrics(). validate_model itself stays collective-free so
        # it remains safe to call on rank 0 only (as train.py / HDP-RL do at some sites).
        "_loss_ego_sum": total_loss_ego,
        "_samples_ego": total_samples_ego,
        "_loss_neighbor_sum": total_loss_neighbor,
        "_samples_neighbor": total_samples_neighbor,
        "_turn_confusion": turn_indicator_confusion,
        "_turn_calibration": turn_indicator_calibration,
        "_turn_indicator_evaluated": evaluate_turn_indicator,
        **total_result_dict,
    }


def aggregate_valid_metrics(valid_dict, device):
    """All-reduce the scalar validation metrics across DDP ranks.

    COLLECTIVE: must be called by every rank that participated in the matching
    ``validate_model`` call. Returns a dict of globally-aggregated scalars; the
    per-sample tensors in ``valid_dict`` are left untouched (callers still use them
    to save per-data-point files). In single-process runs this is a no-op pass-through.

    Callers use ``DistributedEvalSampler``, so shards are disjoint and no metric receives
    duplicate-padding bias when dataset size is not divisible by world size.
    """
    loss_ego_sum = ddp.all_reduce_sum(valid_dict["_loss_ego_sum"], device)
    samples_ego = ddp.all_reduce_sum(valid_dict["_samples_ego"], device)
    loss_nei_sum = ddp.all_reduce_sum(valid_dict["_loss_neighbor_sum"], device)
    samples_nei = ddp.all_reduce_sum(valid_dict["_samples_neighbor"], device)
    turn_confusion = [
        int(value) for value in ddp.all_reduce_sum_values(valid_dict["_turn_confusion"], device)
    ]
    turn_calibration = ddp.all_reduce_sum_values(
        valid_dict.get("_turn_calibration", [0.0] * (2 + 3 * TURN_INDICATOR_CALIBRATION_BINS)),
        device,
    )
    turn_indicator_evaluated = bool(valid_dict.get("_turn_indicator_evaluated", True))
    turn_matrix = [
        turn_confusion[row * TURN_INDICATOR_OUTPUT_DIM : (row + 1) * TURN_INDICATOR_OUTPUT_DIM]
        for row in range(TURN_INDICATOR_OUTPUT_DIM)
    ]
    turn_class_correct = [turn_matrix[index][index] for index in range(TURN_INDICATOR_OUTPUT_DIM)]
    turn_class_total = [sum(turn_matrix[index]) for index in range(TURN_INDICATOR_OUTPUT_DIM)]
    turn_class_accuracy = {
        name: turn_class_correct[index] / max(turn_class_total[index], 1)
        for index, name in enumerate(TURN_INDICATOR_CLASS_NAMES)
    }
    turn_class_count = {
        name: int(turn_class_total[index]) for index, name in enumerate(TURN_INDICATOR_CLASS_NAMES)
    }

    ego_means = {}
    for key, val in valid_dict.items():
        if key.startswith("ego_"):
            local_sum = ddp.all_reduce_sum(val.sum().item(), device)
            local_cnt = ddp.all_reduce_sum(val.numel(), device)
            ego_means[key] = local_sum / max(local_cnt, 1)

    epdms_means = {}
    for key, val in valid_dict.items():
        if not key.startswith("epdms_"):
            continue
        metric = key.removeprefix("epdms_")
        tensor = val.float()
        if metric.endswith("_available"):
            local_sum = ddp.all_reduce_sum(tensor.sum().item(), device)
            local_cnt = ddp.all_reduce_sum(tensor.numel(), device)
            epdms_means[metric] = local_sum / max(local_cnt, 1)
            continue

        available = valid_dict.get(f"{key}_available")
        if available is None:
            finite = torch.isfinite(tensor)
            local_sum = ddp.all_reduce_sum(
                tensor[finite].sum().item() if finite.any() else 0.0, device
            )
            local_cnt = ddp.all_reduce_sum(finite.sum().item(), device)
            epdms_means[metric] = local_sum / local_cnt if local_cnt > 0 else float("nan")
            continue

        mask = available.float() > 0.5
        local_sum = ddp.all_reduce_sum(tensor[mask].sum().item() if mask.any() else 0.0, device)
        local_cnt = ddp.all_reduce_sum(mask.sum().item(), device)
        local_total = ddp.all_reduce_sum(mask.numel(), device)
        epdms_means[f"{metric}_coverage"] = local_cnt / max(local_total, 1)
        epdms_means[metric] = local_sum / local_cnt if local_cnt > 0 else float("nan")

    multisample_means = {}
    for key, val in valid_dict.items():
        if not key.startswith("multisample_"):
            continue
        local_sum = ddp.all_reduce_sum(val.float().sum().item(), device)
        local_cnt = ddp.all_reduce_sum(val.numel(), device)
        multisample_means[key.removeprefix("multisample_")] = (
            local_sum / local_cnt if local_cnt > 0 else float("nan")
        )

    return {
        "avg_loss_ego": loss_ego_sum / max(samples_ego, 1),
        "avg_loss_neighbor": loss_nei_sum / max(samples_nei, 1),
        **turn_indicator_metrics_from_confusion(turn_confusion),
        **turn_indicator_calibration_metrics(turn_calibration),
        "turn_indicator_class_accuracy": turn_class_accuracy,
        "turn_indicator_class_count": turn_class_count,
        "turn_indicator_evaluated": turn_indicator_evaluated,
        "ego_means": ego_means,
        "epdms_means": epdms_means,
        "multisample_means": multisample_means,
    }
