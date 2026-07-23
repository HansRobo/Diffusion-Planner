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


def extract_features(traj: np.ndarray, T: int, dt: float = DT) -> np.ndarray:
    """Extract 5D feature vector from a single trajectory at horizon T.

    Returns: (5,) array [x, y, cos_heading, sin_heading, speed].
    """
    endpoint = traj[T - 1]  # [x, y, yaw]
    speed = np.linalg.norm(traj[T - 1, :2] - traj[T - 2, :2]) / dt if T >= 2 else 0.0
    return np.array(
        [
            endpoint[0],
            endpoint[1],
            np.cos(endpoint[2]),
            np.sin(endpoint[2]),
            speed,
        ]
    )


def _energy_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Unbiased energy distance between two sample sets. P: (n, d), Q: (m, d)."""
    from scipy.spatial.distance import cdist

    n, m = len(P), len(Q)
    if n < 2 or m < 2:
        return float("nan")
    pq = cdist(P, Q).mean()
    pp = cdist(P, P).sum() / (n * (n - 1))
    qq = cdist(Q, Q).sum() / (m * (m - 1))
    return float(2 * pq - pp - qq)


def _mmd_rbf(P: np.ndarray, Q: np.ndarray, bandwidth: float | None = None) -> float:
    """MMD with RBF kernel. Median heuristic for bandwidth if not given."""
    from scipy.spatial.distance import cdist

    n, m = len(P), len(Q)
    if n < 2 or m < 2:
        return float("nan")
    if bandwidth is None:
        pooled = np.vstack([P, Q])
        dists = cdist(pooled, pooled)
        nonzero = dists[dists > 0]
        bandwidth = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
    if bandwidth < 1e-12:
        bandwidth = 1.0
    gamma = 1.0 / (2 * bandwidth**2)

    kpp = np.exp(-gamma * cdist(P, P) ** 2)
    kqq = np.exp(-gamma * cdist(Q, Q) ** 2)
    kpq = np.exp(-gamma * cdist(P, Q) ** 2)

    np.fill_diagonal(kpp, 0)
    np.fill_diagonal(kqq, 0)
    mmd = kpp.sum() / (n * (n - 1)) + kqq.sum() / (m * (m - 1)) - 2 * kpq.mean()
    return float(max(mmd, 0.0))


def _mmd_rbf_subset(P: np.ndarray, Q: np.ndarray, indices: list[int], bandwidth: float) -> float:
    """MMD on a subset of feature dimensions."""
    return _mmd_rbf(P[:, indices], Q[:, indices], bandwidth)


def _frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Frechet distance between two sample sets assuming Gaussian."""
    from sklearn.covariance import LedoitWolf

    if len(P) < 3 or len(Q) < 3:
        return float("nan")
    mu_p, mu_q = P.mean(axis=0), Q.mean(axis=0)
    try:
        cov_p = LedoitWolf().fit(P).covariance_
        cov_q = LedoitWolf().fit(Q).covariance_
    except Exception:
        return float("nan")
    from scipy.linalg import sqrtm

    diff = mu_p - mu_q
    covmean = sqrtm(cov_p @ cov_q)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_p + cov_q - 2 * covmean))


def _human_outlier_score(test_feat: np.ndarray, human_feats: np.ndarray, k: int = 5) -> float:
    """k-NN distance of test feature to human cloud."""
    if len(human_feats) < k:
        return float("nan")
    dists = np.linalg.norm(human_feats - test_feat, axis=-1)
    return float(np.sort(dists)[:k].mean())


def _planner_energy_score(test_feat: np.ndarray, planner_feats: np.ndarray) -> float:
    """Energy score: E||P - y|| - 0.5 * E||P - P'||."""
    n = len(planner_feats)
    if n < 2:
        return float("nan")
    py = np.linalg.norm(planner_feats - test_feat, axis=-1).mean()
    from scipy.spatial.distance import cdist

    pp = cdist(planner_feats, planner_feats).sum() / (n * (n - 1))
    return float(py - 0.5 * pp)


def multi_human_metrics(
    human_futures: list[np.ndarray],
    dp_samples: np.ndarray,
    test_human: np.ndarray,
) -> dict[str, float]:
    """Compare DP sample distribution against human trajectory distribution.

    human_futures: list of (80, 3) [x, y, yaw] in test-ego coords.
    dp_samples: (N, 80, 3) planner samples.
    test_human: (80, 3) the test frame's own human trajectory.
    """
    import ot

    nan_keys = [
        "energy_dist",
        "mmd",
        "mmd_position",
        "mmd_heading",
        "mmd_speed",
        "sinkhorn",
        "sliced_w",
        "fid",
        "human_outlier",
        "planner_energy_score",
    ]
    out: dict[str, float] = {"n_humans": len(human_futures)}
    if not human_futures or len(dp_samples) == 0:
        for name in HORIZONS:
            for key in nan_keys:
                out[f"{key}_{name}"] = float("nan")
        return out

    for name, T in HORIZONS.items():
        h_feats = np.array([extract_features(h, T) for h in human_futures])  # (M, 5)
        s_feats = np.array([extract_features(s, T) for s in dp_samples])  # (N, 5)
        t_feat = extract_features(test_human, T)  # (5,)

        # 1. Energy Distance
        out[f"energy_dist_{name}"] = _energy_distance(s_feats, h_feats)

        # 2. MMD + diagnostic split
        pooled = np.vstack([s_feats, h_feats])
        from scipy.spatial.distance import cdist

        all_dists = cdist(pooled, pooled)
        nonzero = all_dists[all_dists > 0]
        bw = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
        if bw < 1e-12:
            bw = 1.0
        out[f"mmd_{name}"] = _mmd_rbf(s_feats, h_feats, bw)
        out[f"mmd_position_{name}"] = _mmd_rbf_subset(s_feats, h_feats, [0, 1], bw)
        out[f"mmd_heading_{name}"] = _mmd_rbf_subset(s_feats, h_feats, [2, 3], bw)
        out[f"mmd_speed_{name}"] = _mmd_rbf_subset(s_feats, h_feats, [4], bw)

        # 3. Sinkhorn Divergence
        try:
            s64 = s_feats.astype(np.float64)
            h64 = h_feats.astype(np.float64)
            sink = ot.bregman.empirical_sinkhorn_divergence(
                s64,
                h64,
                reg=1.0,
                metric="euclidean",
            )
            out[f"sinkhorn_{name}"] = float(max(sink, 0.0))
        except Exception:
            out[f"sinkhorn_{name}"] = float("nan")

        # 4. Sliced Wasserstein
        try:
            out[f"sliced_w_{name}"] = float(
                ot.sliced.sliced_wasserstein_distance(
                    s_feats.astype(np.float64),
                    h_feats.astype(np.float64),
                    n_projections=256,
                )
            )
        except Exception:
            out[f"sliced_w_{name}"] = float("nan")

        # 5. FID
        out[f"fid_{name}"] = _frechet_distance(s_feats, h_feats)

        # 6. Diagnosis
        out[f"human_outlier_{name}"] = _human_outlier_score(t_feat, h_feats)
        out[f"planner_energy_score_{name}"] = _planner_energy_score(t_feat, s_feats)

    return out
