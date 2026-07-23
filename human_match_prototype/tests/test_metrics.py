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


from human_match_prototype.metrics import multi_human_metrics


def test_multi_human_perfect_coverage():
    """All humans identical to planner samples -> full coverage."""
    base = straight(5.0)
    humans = [base.copy() for _ in range(10)]
    samples = np.stack([base.copy() for _ in range(8)])
    m = multi_human_metrics(humans, samples, base)
    assert m["n_humans"] == 10
    assert m["dp_human_coverage_4s"] == 1.0
    assert m["human_dp_coverage_4s"] == 1.0
    assert m["human_spread_4s"] < 0.01


def test_multi_human_no_coverage():
    """Humans with different trajectory shape -> zero coverage."""
    base = straight(5.0)
    # Different direction (90 deg) — shape differs after origin normalization
    turning = straight(5.0, yaw=1.5)
    humans = [turning.copy() for _ in range(5)]
    samples = np.stack([base.copy() for _ in range(8)])
    m = multi_human_metrics(humans, samples, base)
    assert m["dp_human_coverage_4s"] == 0.0
    assert m["human_dp_coverage_4s"] == 0.0


def test_multi_human_partial_coverage():
    """Some humans same shape as planner, some different shape."""
    base = straight(5.0)
    close = straight(5.1)  # slightly different speed, same direction
    turning = straight(5.0, yaw=1.5)  # different direction
    humans = [close, turning]
    samples = np.stack([base.copy() for _ in range(4)])
    m = multi_human_metrics(humans, samples, base)
    assert m["n_humans"] == 2
    assert m["dp_human_coverage_4s"] == 0.5
    assert m["human_dp_coverage_4s"] == 1.0


def test_multi_human_typicality_typical():
    """Test human is at the centroid of training humans."""
    base = straight(5.0)
    humans = [base.copy() for _ in range(20)]
    for i, h in enumerate(humans):
        h[:, 0] += (i - 10) * 0.1
    samples = np.stack([base.copy()])
    m = multi_human_metrics(humans, samples, base)
    assert m["test_human_typicality_4s"] < 0.5


def test_multi_human_empty():
    """No matched humans -> NaN metrics."""
    base = straight(5.0)
    samples = np.stack([base.copy()])
    m = multi_human_metrics([], samples, base)
    assert m["n_humans"] == 0
    assert np.isnan(m["dp_human_coverage_4s"])
