import numpy as np
from human_match_prototype.typicality import traj_features, typicality


def noisy_straight(rng, v=5.0, T=80, dt=0.1):
    t = np.arange(1, T + 1) * dt
    v_i = v + rng.normal(0, 0.3)
    xy = np.stack([v_i * t, rng.normal(0, 0.05) * t], -1)
    return np.concatenate([xy, np.zeros((T, 1))], -1)


def test_feature_shape():
    rng = np.random.default_rng(0)
    assert traj_features(noisy_straight(rng)).shape == (32,)


def test_member_is_typical_outlier_is_not():
    rng = np.random.default_rng(0)
    samples = np.stack([noisy_straight(rng) for _ in range(64)])
    member = samples[0]
    outlier = noisy_straight(rng)
    outlier[:, 1] += np.linspace(0, 8, 80)  # hard lateral departure
    t_member = typicality(member, samples)
    t_outlier = typicality(outlier, samples)
    assert t_outlier["maha_dist"] > t_member["maha_dist"]
    assert t_outlier["maha_pct"] == 100.0
    assert t_member["maha_pct"] < 60.0
