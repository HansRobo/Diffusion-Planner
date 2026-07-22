"""Strong-brake metric from realized tangential acceleration."""

from __future__ import annotations

import numpy as np


def strong_brake_mask(
    accels: np.ndarray,
    *,
    thresh_mps2: float = -4.0,
) -> np.ndarray:
    """Vectorized strong-brake flag over a 1-D accel series (``accel <= thresh``)."""
    return np.asarray(accels, dtype=np.float32) <= float(thresh_mps2)
