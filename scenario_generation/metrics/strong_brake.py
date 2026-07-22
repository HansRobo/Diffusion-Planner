"""Per-step strong-brake metric from realized tangential acceleration."""

from __future__ import annotations

import numpy as np


def score_strong_brake_step(
    *,
    ego_accel_mps2: float,
    thresh_mps2: float = -4.0,
) -> dict:
    """Flag whether this step's accel is at or below the strong-brake threshold.

    ``thresh_mps2`` is negative (e.g. -4.0). Accel itself is produced by the
    rollout dynamics; this scorer only applies the metric threshold.
    """
    accel = float(ego_accel_mps2)
    return {
        "accel_mps2": accel,
        "strong_brake": bool(accel <= float(thresh_mps2)),
    }


def strong_brake_mask(
    accels: np.ndarray,
    *,
    thresh_mps2: float = -4.0,
) -> np.ndarray:
    """Vectorized ``score_strong_brake_step`` over a 1-D accel series."""
    return np.asarray(accels, dtype=np.float32) <= float(thresh_mps2)
