# human_match_prototype/metrics.py
"""Coverage metrics: is the human future covered by the planner sample cloud?"""
import numpy as np

DT = 0.1
HORIZONS = {"2s": 20, "4s": 40, "8s": 80}
CLOSE_ADE_THRESHOLDS = {"2s": 0.5, "4s": 1.0, "8s": 2.0}


def derive_speed(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    """xy: (..., T, 2) -> speed (..., T-1) from point-to-point distance."""
    return np.linalg.norm(np.diff(xy, axis=-2), axis=-1) / dt


def _frenet_errors(human_xy: np.ndarray, sample_xy: np.ndarray) -> tuple[float, float]:
    """Mean |longitudinal| and |lateral| offset of sample vs human, along the
    human trajectory's local heading. Both (T, 2)."""
    tang = np.gradient(human_xy, axis=0)
    norm = np.linalg.norm(tang, axis=-1, keepdims=True)
    default = np.zeros_like(tang)
    default[:, 0] = 1.0
    tang = np.where(norm > 1e-6, tang / np.maximum(norm, 1e-6), default)
    nvec = np.stack([-tang[:, 1], tang[:, 0]], -1)
    diff = sample_xy - human_xy
    lon = np.abs((diff * tang).sum(-1)).mean()
    lat = np.abs((diff * nvec).sum(-1)).mean()
    return float(lon), float(lat)


def coverage_metrics(human: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    """human: (80, 3) [x, y, yaw]; samples: (N, 80, 3). Flat metric dict."""
    out: dict[str, float] = {}
    h_xy, s_xy = human[:, :2], samples[:, :, :2]
    for name, T in HORIZONS.items():
        d = np.linalg.norm(s_xy[:, :T] - h_xy[None, :T], axis=-1)  # (N, T)
        ade, fde = d.mean(axis=1), d[:, T - 1]
        thr = CLOSE_ADE_THRESHOLDS[name]
        out[f"min_ade_{name}"] = float(ade.min())
        out[f"p10_ade_{name}"] = float(np.percentile(ade, 10))
        out[f"median_ade_{name}"] = float(np.median(ade))
        out[f"max_ade_{name}"] = float(ade.max())
        out[f"min_fde_{name}"] = float(fde.min())
        out[f"median_fde_{name}"] = float(np.median(fde))
        out[f"frac_close_{name}"] = float((ade < thr).mean())
        out[f"mismatch_{name}"] = float((ade >= thr).all())
        ends = s_xy[:, T - 1]
        out[f"spread_{name}"] = float(
            np.linalg.norm(ends - ends.mean(axis=0), axis=-1).mean()
        )
        best = int(ade.argmin())
        lon, lat = _frenet_errors(h_xy[:T], s_xy[best, :T])
        out[f"best_lon_err_{name}"] = lon
        out[f"best_lat_err_{name}"] = lat
        dyaw = samples[best, :T, 2] - human[:T, 2]
        out[f"best_heading_err_{name}"] = float(
            np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))).mean()
        )
        out[f"best_speed_err_{name}"] = float(
            np.abs(derive_speed(s_xy[best, :T]) - derive_speed(h_xy[:T])).mean()
        )
    return out
