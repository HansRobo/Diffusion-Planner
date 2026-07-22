import numpy as np
import pandas as pd

from human_match_prototype.cluster_report import aggregate


def make_df():
    rows = []
    for i in range(4):  # cluster A: half mismatch
        rows.append(
            dict(
                category="cluster_idA",
                mismatch_4s=float(i % 2),
                min_ade_4s=1.0 + i,
                frac_close_4s=0.5,
                best_lon_err_4s=2.0,
                best_lat_err_4s=0.2,
                spread_4s=0.3,
                latent_knn_mean=0.7,
            )
        )
    for i in range(2):  # cluster B: no mismatch
        rows.append(
            dict(
                category="cluster_idB",
                mismatch_4s=0.0,
                min_ade_4s=0.1,
                frac_close_4s=1.0,
                best_lon_err_4s=0.1,
                best_lat_err_4s=0.1,
                spread_4s=0.2,
                latent_knn_mean=0.5,
            )
        )
    return pd.DataFrame(rows)


def test_aggregate_per_cluster_stats_and_order():
    agg = aggregate(make_df())
    assert list(agg.cluster) == ["cluster_idA", "cluster_idB"]  # sorted by mismatch desc
    a = agg.iloc[0]
    assert a.n == 4 and abs(a.mismatch_rate_4s - 0.5) < 1e-9
    assert abs(a.median_min_ade_4s - 2.5) < 1e-9
    assert abs(a.median_lon_err_4s - 2.0) < 1e-9
    b = agg.iloc[1]
    assert b.n == 2 and b.mismatch_rate_4s == 0.0


def test_aggregate_with_multi_human_columns():
    """aggregate includes multi-human stats when present."""
    from human_match_prototype.cluster_report import aggregate

    df = pd.DataFrame(
        {
            "category": ["c0"] * 10,
            "mismatch_4s": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            "min_ade_4s": np.random.rand(10),
            "frac_close_4s": np.random.rand(10),
            "best_lon_err_4s": np.random.rand(10),
            "best_lat_err_4s": np.random.rand(10),
            "spread_4s": np.random.rand(10),
            "latent_knn_mean": np.random.rand(10),
            "n_humans": [20, 3, 15, 0, 8, 25, 4, 10, 2, 30],
            "dp_human_coverage_4s": [0.8, 0.1, 0.9, float("nan"), 0.5, 0.7, 0.3, 0.6, 0.0, 0.95],
        }
    )
    agg = aggregate(df)
    assert "median_n_humans" in agg.columns
    assert "median_dp_human_coverage_4s" in agg.columns
    assert "frac_insufficient_data" in agg.columns
    assert agg.iloc[0]["frac_insufficient_data"] == 0.4  # 4 frames with n_humans < 5
