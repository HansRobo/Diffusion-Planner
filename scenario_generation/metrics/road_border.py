"""Per-step road-border (curb) clearance for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_road_border_penalty
from scenario_generation.metrics.context import StepMetricContext


def score_road_border_step(
    ctx: StepMetricContext,
    *,
    near_miss_thresh_m: float = 0.3,
) -> dict:
    """Ego-to-road-border distance and collision/miss flags at the current step.

    Uses ``line_strings`` channel 3 as road border. Collision is hard contact
    (``rb_dist_m <= 0``); miss uses ``near_miss_thresh_m`` only.
    """
    np_dict = ctx.np_dict
    rb_dist_m = float("inf")
    if "line_strings" in np_dict and "ego_shape" in np_dict:
        ego_shape_t = torch.tensor(
            np.asarray(np_dict["ego_shape"]).reshape(-1)[:3],
            dtype=torch.float32,
            device=ctx.device,
        )
        traj = ctx.ego_traj_ego_frame(n_steps=1)
        ls = np.asarray(np_dict["line_strings"])
        if ls.ndim == 4:
            ls = ls[0]
        data = {"line_strings": torch.tensor(ls, dtype=torch.float32, device=ctx.device)}
        _gate, _near, _wide, _steps, _cont, per_ts_min = compute_road_border_penalty(
            traj, ego_shape_t, data, config=RewardConfig()
        )
        rb_dist_m = float(per_ts_min[0, 0].item())

    thresh = float(near_miss_thresh_m)
    return {
        "rb_dist_m": rb_dist_m,
        "rb_collision": bool(np.isfinite(rb_dist_m) and rb_dist_m <= 0.0),
        "rb_miss": bool(np.isfinite(rb_dist_m) and rb_dist_m <= thresh),
    }
