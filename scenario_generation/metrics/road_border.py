"""Per-step road-border (curb) clearance for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.geometry import point_inside_any_lane_polygon
from planner_metrics.subscores import compute_road_border_penalty


def _as_float_tensor(value, device: str) -> torch.Tensor:
    """numpy array OR torch tensor -> float32 tensor on ``device``, no copy
    when it already matches (lets the rollout pass GPU-resident batch slices
    instead of forcing a host->device upload every step)."""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(value), dtype=torch.float32, device=device)


def _ego_inside_lane(np_dict: dict, device: str) -> bool | None:
    """Is the ego origin inside any lane polygon? ``None`` when ``lanes`` is absent
    (sign cannot be determined, e.g. older callers/tests that only pass line_strings)."""
    if "lanes" not in np_dict:
        return None
    lanes_t = _as_float_tensor(np_dict["lanes"], device)
    if lanes_t.dim() == 4:
        lanes_t = lanes_t[0]
    origin = torch.zeros(2, dtype=torch.float32, device=device)
    return point_inside_any_lane_polygon(origin, lanes_t)


def _ego_intersects_border_segments(
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
    ego_shape: torch.Tensor,
) -> bool:
    """Whether any border segment intersects the current ego OBB.

    ``score_road_border_step`` evaluates the current pose, whose reference
    point is the origin and whose heading is +x.  In that frame the ego OBB is
    an axis-aligned rectangle.  This exact segment/rectangle test avoids
    missing an intersection between the 80 sampled perimeter points.
    """
    if seg_p1.shape[0] == 0:
        return False

    length = ego_shape[1]
    width = ego_shape[2]
    wheelbase = ego_shape[0]
    rear_overhang = (length - wheelbase) / 2.0
    xmin = -rear_overhang
    xmax = length - rear_overhang
    ymin = -width / 2.0
    ymax = width / 2.0

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def point_in_rect(p: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        return (
            (p[:, 0] >= xmin - eps)
            & (p[:, 0] <= xmax + eps)
            & (p[:, 1] >= ymin - eps)
            & (p[:, 1] <= ymax + eps)
        )

    # An endpoint inside is an intersection.
    if point_in_rect(seg_p1).any() or point_in_rect(seg_p2).any():
        return True

    rect = torch.stack(
        [
            torch.stack([xmin, ymin]),
            torch.stack([xmax, ymin]),
            torch.stack([xmax, ymax]),
            torch.stack([xmin, ymax]),
        ]
    )
    rect_next = torch.roll(rect, shifts=-1, dims=0)

    # Inclusive segment intersection, vectorized over border segments and
    # rectangle edges.  The epsilon also treats touching as intersection.
    a = seg_p1[:, None, :]
    b = seg_p2[:, None, :]
    c = rect[None, :, :]
    d = rect_next[None, :, :]
    ab = b - a
    cd = d - c
    ac = c - a
    ad = d - a
    ca = a - c
    cb = b - c
    o1 = cross(ab, ac)
    o2 = cross(ab, ad)
    o3 = cross(cd, ca)
    o4 = cross(cd, cb)
    eps = 1e-6
    bbox_overlap = (
        (torch.minimum(a[..., 0], b[..., 0]) <= torch.maximum(c[..., 0], d[..., 0]) + eps)
        & (torch.maximum(a[..., 0], b[..., 0]) + eps >= torch.minimum(c[..., 0], d[..., 0]))
        & (torch.minimum(a[..., 1], b[..., 1]) <= torch.maximum(c[..., 1], d[..., 1]) + eps)
        & (torch.maximum(a[..., 1], b[..., 1]) + eps >= torch.minimum(c[..., 1], d[..., 1]))
    )
    proper = (o1 * o2 <= eps) & (o3 * o4 <= eps) & bbox_overlap
    return bool(proper.any().item())


def score_road_border_step(np_dict: dict, *, device: str) -> dict:
    """Ego-to-road-border distance at the current step.

    Uses ``line_strings`` channel 3 as road border. Collision / miss masks are
    derived later in ``_finalize`` from the distance series. Current pose is the
    ego-frame origin; history is not needed.

    Signed by lane containment when ``lanes`` is available: positive while the ego
    origin is inside a lane polygon (normal clearance), negative once it has crossed
    outside (magnitude = how far past the border) -- this is the opposite convention
    from ``_point_to_segments_signed_min_dist`` (lane departure: +outside/-inside),
    chosen to match how ``rb_dist_m`` is consumed downstream: smaller/more negative
    already reads as "worse" (collision thresholds, ``clearance_min_m``/``p5``), so
    inside=positive keeps that ordering intact once crossings go negative.
    """
    rb_dist_m = float("inf")
    if "line_strings" in np_dict and "ego_shape" in np_dict:
        ego_shape_t = _as_float_tensor(np_dict["ego_shape"], device).reshape(-1)[:3]
        traj = torch.zeros(1, 1, 4, dtype=torch.float32, device=device)
        traj[..., 2] = 1.0  # origin, heading +x
        ls = _as_float_tensor(np_dict["line_strings"], device)
        if ls.dim() == 4:
            ls = ls[0]
        data = {"line_strings": ls}
        _gate, _near, _wide, _steps, _cont, per_ts_min = compute_road_border_penalty(
            traj, ego_shape_t, data, config=RewardConfig()
        )
        rb_dist_m = float(per_ts_min[0, 0].item())
        if np.isfinite(rb_dist_m):
            inside = _ego_inside_lane(np_dict, device)
            # A border crossing/contact takes precedence over the signed
            # clearance: the requested value at intersection is exactly 0.
            if "line_strings" in np_dict:
                ls_for_intersection = _as_float_tensor(np_dict["line_strings"], device)
                if ls_for_intersection.dim() == 4:
                    ls_for_intersection = ls_for_intersection[0]
                border_xy = ls_for_intersection[..., :2]
                valid = (ls_for_intersection[..., 3] > 0.5) & (border_xy.norm(dim=-1) > 1e-3)
                valid_pair = valid[:, :-1] & valid[:, 1:]
                idx = torch.where(valid_pair.reshape(-1))[0]
                seg_p1 = border_xy[:, :-1].reshape(-1, 2)[idx]
                seg_p2 = border_xy[:, 1:].reshape(-1, 2)[idx]
                if _ego_intersects_border_segments(seg_p1, seg_p2, ego_shape_t):
                    rb_dist_m = 0.0
            if rb_dist_m != 0.0 and inside is False:
                rb_dist_m = -rb_dist_m

    return {"rb_dist_m": rb_dist_m}
