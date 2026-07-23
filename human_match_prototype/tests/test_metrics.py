# human_match_prototype/tests/test_metrics.py
import numpy as np

from human_match_prototype.metrics import coverage_metrics, derive_speed


def straight(v, yaw=0.0, T=80, dt=0.1):
    t = np.arange(1, T + 1) * dt
    xy = np.stack([v * t * np.cos(yaw), v * t * np.sin(yaw)], -1)
    return np.concatenate([xy, np.full((T, 1), yaw)], -1)


def test_identical_sample_gives_zero_min_ade():
    human = straight(5.0)
    samples = np.stack([human, straight(5.0, yaw=0.05)])
    m = coverage_metrics(human, samples)
    assert m["min_ade_8s"] < 1e-9
    assert m["frac_close_8s"] >= 0.5
    assert m["mismatch_8s"] == 0


def test_all_far_flags_mismatch_and_lat_lon_split():
    human = straight(5.0)
    lat_shift = human.copy()
    lat_shift[:, 1] += 5.0  # pure lateral offset
    samples = np.stack([lat_shift, lat_shift])
    m = coverage_metrics(human, samples)
    assert m["mismatch_2s"] == 1 and m["mismatch_8s"] == 1
    assert abs(m["min_ade_8s"] - 5.0) < 1e-6
    assert m["best_lat_err_8s"] > 4.9 and m["best_lon_err_8s"] < 0.1


def test_longitudinal_error_dominates_for_speed_mismatch():
    human = straight(5.0)
    fast = straight(7.0)  # same path, faster
    m = coverage_metrics(human, np.stack([fast, fast]))
    assert m["best_lon_err_8s"] > 5 * m["best_lat_err_8s"]
    assert m["best_speed_err_8s"] > 1.5


def test_derive_speed():
    v = derive_speed(straight(5.0)[:, :2])
    assert np.allclose(v, 5.0, atol=1e-6)


from human_match_prototype.metrics import extract_features, multi_human_metrics


def test_extract_features():
    """Extract 5D features from a trajectory."""
    traj = straight(5.0, yaw=0.3)  # (80, 3)
    feat = extract_features(traj, T=40, dt=0.1)
    assert feat.shape == (5,)
    # x, y = endpoint at t=40
    assert abs(feat[0] - traj[39, 0]) < 1e-6
    assert abs(feat[1] - traj[39, 1]) < 1e-6
    # cos_h, sin_h from yaw
    assert abs(feat[2] - np.cos(traj[39, 2])) < 1e-6
    assert abs(feat[3] - np.sin(traj[39, 2])) < 1e-6
    # speed > 0
    assert feat[4] > 0


def test_distributional_identical():
    """Identical distributions -> all metrics near zero."""
    base = straight(5.0)
    humans = [base.copy() for _ in range(30)]
    samples = np.stack([base.copy() for _ in range(30)])
    m = multi_human_metrics(humans, samples, base)
    assert m["n_humans"] == 30
    assert m["energy_dist_4s"] < 0.1
    assert m["mmd_4s"] < 0.1
    assert m["sinkhorn_4s"] < 0.1
    assert m["sliced_w_4s"] < 0.1
    assert m["fid_4s"] < 0.1


def test_distributional_different():
    """Different distributions -> all metrics significantly positive."""
    base = straight(5.0)
    turned = straight(5.0, yaw=1.0)
    humans = [turned.copy() for _ in range(30)]
    samples = np.stack([base.copy() for _ in range(30)])
    m = multi_human_metrics(humans, samples, base)
    assert m["energy_dist_4s"] > 1.0
    assert m["mmd_4s"] > 0.1
    assert m["sinkhorn_4s"] > 1.0
    assert m["sliced_w_4s"] > 1.0
    assert m["fid_4s"] > 1.0


def test_distributional_diagnosis():
    """Diagnosis quantities are populated."""
    base = straight(5.0)
    humans = [straight(5.0 + i * 0.1) for i in range(30)]
    samples = np.stack([base.copy() for _ in range(20)])
    m = multi_human_metrics(humans, samples, base)
    assert "human_outlier_4s" in m
    assert "planner_energy_score_4s" in m
    assert not np.isnan(m["human_outlier_4s"])
    assert not np.isnan(m["planner_energy_score_4s"])


def test_distributional_empty():
    """No matched humans -> NaN metrics."""
    base = straight(5.0)
    samples = np.stack([base.copy()])
    m = multi_human_metrics([], samples, base)
    assert m["n_humans"] == 0
    assert np.isnan(m["energy_dist_4s"])
    assert np.isnan(m["mmd_4s"])


def test_mmd_diagnostic_split():
    """MMD split into position/heading/speed components."""
    base = straight(5.0)
    humans = [straight(5.0, yaw=0.5 + i * 0.01) for i in range(30)]
    samples = np.stack([base.copy() for _ in range(30)])
    m = multi_human_metrics(humans, samples, base)
    # Heading component should dominate since only yaw differs
    assert m["mmd_heading_4s"] > m["mmd_speed_4s"]
