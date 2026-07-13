"""Is the human trajectory typical of the planner's sample cloud?

Shrinkage-Mahalanobis (Ledoit-Wolf) on 8-point trajectory features, reported
as an empirical percentile against the samples' own distances. Shared-fit
percentile (not leave-one-out) — good enough for a prototype ranking signal.
"""
import numpy as np
from sklearn.covariance import LedoitWolf

from human_match_prototype.metrics import derive_speed

N_POINTS = 8


def traj_features(traj: np.ndarray) -> np.ndarray:
    """traj: (80, 3) [x, y, yaw] -> (32,) = 8 x [x, y, speed, accel]."""
    idx = np.linspace(9, 79, N_POINTS).astype(int)  # ~every 1 s
    xy = traj[:, :2]
    v = derive_speed(xy)                      # (79,)
    a = np.diff(v) / 0.1                      # (78,)
    v_pad = np.concatenate([v, v[-1:]])       # (80,)
    a_pad = np.concatenate([a, a[-1:], a[-1:]])  # (80,)
    return np.concatenate([xy[idx, 0], xy[idx, 1], v_pad[idx], a_pad[idx]])


def typicality(human: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    feats = np.stack([traj_features(s) for s in samples])
    hf = traj_features(human)
    lw = LedoitWolf().fit(feats)
    d_samples = lw.mahalanobis(feats)
    d_human = float(lw.mahalanobis(hf[None])[0])
    return {
        "maha_dist": d_human,
        "maha_pct": float(100.0 * (d_samples <= d_human).mean()),
    }
