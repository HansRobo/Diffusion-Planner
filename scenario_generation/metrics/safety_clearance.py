"""Ego-to-road-border distance for closed-loop steps."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_road_border_penalty


def road_border_distance(np_dict: dict, device: str) -> float:
    """Ego-to-road-border distance (m) at the current step, or ``inf`` if ``np_dict``
    carries no lane geometry (``line_strings``/``ego_shape``) to check against.
    """
    if "line_strings" not in np_dict or "ego_shape" not in np_dict:
        return float("inf")
    ego_shape_t = torch.tensor(
        np.asarray(np_dict["ego_shape"]).reshape(-1)[:3],
        dtype=torch.float32,
        device=device,
    )
    traj = torch.zeros(1, 1, 4, dtype=torch.float32, device=device)
    traj[0, 0, 2] = 1.0  # heading +x in ego frame
    data = {
        "line_strings": torch.tensor(np_dict["line_strings"], dtype=torch.float32, device=device)
    }
    _gate, _near, _wide, _steps, _cont, per_ts_min = compute_road_border_penalty(
        traj, ego_shape_t, data, config=RewardConfig()
    )
    return float(per_ts_min[0, 0].item())
