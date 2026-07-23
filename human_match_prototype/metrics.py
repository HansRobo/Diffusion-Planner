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
        out[f"spread_{name}"] = float(np.linalg.norm(ends - ends.mean(axis=0), axis=-1).mean())
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


def multi_human_metrics(
    human_futures: list[np.ndarray],
    dp_samples: np.ndarray,
    test_human: np.ndarray,
) -> dict[str, float]:
    """Compare DP sample cloud against multiple matched human trajectories.

    human_futures: list of (80, 3) [x, y, yaw] in test-ego coords.
    dp_samples: (N, 80, 3) planner samples.
    test_human: (80, 3) the test frame's own human trajectory.
    """
    out: dict[str, float] = {"n_humans": len(human_futures)}
    if not human_futures or len(dp_samples) == 0:
        for name in HORIZONS:
            for key in (
                "human_spread",
                "dp_human_coverage",
                "human_dp_coverage",
                "human_consensus",
                "test_human_typicality",
            ):
                out[f"{key}_{name}"] = float("nan")
        return out

    h_stack = np.stack(human_futures)  # (M, 80, 3)
    s_xy = dp_samples[:, :, :2]  # (N, 80, 2)

    # Normalize all trajectories to start at origin so we compare shapes, not positions.
    # Training humans come from different ego positions on the same lanelet.
    h_origins = h_stack[:, 0:1, :2]  # (M, 1, 2)
    h_norm = h_stack[:, :, :2] - h_origins  # (M, 80, 2)
    s_origins = s_xy[:, 0:1, :]  # (N, 1, 2)
    s_norm = s_xy - s_origins  # (N, 80, 2)
    test_origin = test_human[0:1, :2]  # (1, 2)
    test_norm = test_human[:, :2] - test_origin  # (80, 2)

    for name, T in HORIZONS.items():
        thr = CLOSE_ADE_THRESHOLDS[name]
        h_xy = h_norm[:, :T]  # (M, T, 2) — origin-normalized
        s_xy_t = s_norm[:, :T]  # (N, T, 2) — origin-normalized

        # human_spread: mean distance of endpoints from centroid (on normalized trajs)
        h_ends = h_xy[:, T - 1]  # (M, 2)
        centroid = h_ends.mean(axis=0)
        out[f"human_spread_{name}"] = float(np.linalg.norm(h_ends - centroid, axis=-1).mean())

        # dp_human_coverage: fraction of humans covered by at least one DP sample
        covered_humans = 0
        for h in h_xy:
            ade_per_sample = np.linalg.norm(s_xy_t - h[None, :, :], axis=-1).mean(axis=1)
            if ade_per_sample.min() < thr:
                covered_humans += 1
        out[f"dp_human_coverage_{name}"] = covered_humans / len(human_futures)

        # human_dp_coverage: fraction of DP samples close to at least one human
        covered_samples = 0
        for s in s_xy_t:
            ade_per_human = np.linalg.norm(h_xy - s[None, :, :], axis=-1).mean(axis=1)
            if ade_per_human.min() < thr:
                covered_samples += 1
        out[f"human_dp_coverage_{name}"] = covered_samples / len(dp_samples)

        # human_consensus: low spread indicates consensus
        out[f"human_consensus_{name}"] = float(out[f"human_spread_{name}"] < 3.0)

        # test_human_typicality: Mahalanobis CDF of test human endpoint
        test_end = test_norm[T - 1]  # (2,) — origin-normalized
        if len(human_futures) >= 3:
            cov = np.cov(h_ends.T)
            if np.linalg.det(cov) > 1e-12:
                inv_cov = np.linalg.inv(cov)
                diff_test = test_end - centroid
                test_d = float(np.sqrt(diff_test @ inv_cov @ diff_test))
                human_diffs = h_ends - centroid
                human_dists = np.array([float(np.sqrt(d @ inv_cov @ d)) for d in human_diffs])
                out[f"test_human_typicality_{name}"] = float((human_dists < test_d).mean())
            else:
                test_d_eucl = np.linalg.norm(test_end - centroid)
                human_eucl = np.linalg.norm(h_ends - centroid, axis=-1)
                out[f"test_human_typicality_{name}"] = float((human_eucl < test_d_eucl).mean())
        else:
            out[f"test_human_typicality_{name}"] = float("nan")

    return out
