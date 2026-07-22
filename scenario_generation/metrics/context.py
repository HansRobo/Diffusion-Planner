"""Shared per-step context for closed-loop metric scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from planner_metrics.config import RewardConfig


@dataclass
class StepMetricContext:
    """Everything a per-step (or short-window) metric scorer may need.

    Built once per sim tick by the rollout. Instantaneous metrics use the current
    observation; scorers that need speed / short history read ``ego_speed_mps`` /
    ``ego_hist`` / ``ego_traj_ego_frame`` instead of inventing their own windows.

    Metric-specific thresholds (near-miss, strong-brake, …) are **not** stored here —
    pass them as scorer kwargs so unused metrics stay decoupled.
    """

    np_dict: dict
    device: str
    k: int
    ego_speed_mps: float
    live_pose: np.ndarray  # (3,) world x, y, yaw
    ego_hist: np.ndarray  # (PAST, 3+) recent ego states in world/sim frame
    ego_accel_mps2: float | None = None  # set once this step's accel is known

    def ego_traj_ego_frame(self, n_steps: int = 1) -> torch.Tensor:
        """Ego-centric trajectory tensor ``(1, n_steps, 4)`` for planner_metrics APIs.

        Current pose is always the origin with heading +x. When ``n_steps >= 2``,
        earlier steps are placed along -x using ``ego_speed_mps * dt`` so finite-
        difference speed gates (e.g. red-light) see the live ego speed.
        """
        n = max(1, int(n_steps))
        traj = torch.zeros(1, n, 4, dtype=torch.float32, device=self.device)
        traj[..., 2] = 1.0  # cos yaw = +x
        if n >= 2:
            dt = float(RewardConfig().dt)
            # steps [0 .. n-2] behind current; last step at origin
            for i in range(n - 1):
                age = n - 1 - i
                traj[0, i, 0] = -float(self.ego_speed_mps) * dt * age
        return traj
