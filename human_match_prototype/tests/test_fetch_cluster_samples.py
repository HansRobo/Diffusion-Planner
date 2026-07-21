# human_match_prototype/tests/test_fetch_cluster_samples.py
import pytest

from human_match_prototype.fetch_cluster_samples import check_capacity, select_paths


def fake_clusters():
    return {
        "cluster_id1": [f"/srv/a/{i}.npz" for i in range(100)],
        "cluster_id0": [f"/srv/b/{i}.npz" for i in range(3)],
        "cluster_id2": [f"/srv/c/{i}.npz" for i in range(50)],
    }


def test_select_paths_counts_and_determinism():
    sel1 = select_paths(fake_clusters(), per_cluster=10, seed=0)
    sel2 = select_paths(fake_clusters(), per_cluster=10, seed=0)
    assert sel1 == sel2
    assert len(sel1["cluster_id1"]) == 10
    assert len(sel1["cluster_id0"]) == 3  # smaller cluster: take all
    assert set(sel1["cluster_id0"]) == set(fake_clusters()["cluster_id0"])
    assert select_paths(fake_clusters(), 10, seed=1) != sel1


def test_select_paths_limit_clusters_numeric_order():
    sel = select_paths(fake_clusters(), 5, seed=0, limit_clusters=2)
    assert sorted(sel) == ["cluster_id0", "cluster_id1"]


def test_check_capacity():
    check_capacity(est_bytes=1_000, free_bytes=5_000)  # exactly 5x: OK
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        check_capacity(est_bytes=1_000, free_bytes=4_999)
