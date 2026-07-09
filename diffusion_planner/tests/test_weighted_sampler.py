import json
import tempfile
from pathlib import Path

from diffusion_planner.utils.weighted_sampler import ClusterWeightedDistributedSampler


def _make_cluster_json(tmp_dir: str, data_list: list[str]) -> tuple[str, dict]:
    """Create a cluster JSON where first 2 paths are 'rare' and rest are 'common'."""
    clusters = {
        "cluster_id0": data_list[:2],
        "cluster_id1": data_list[2:],
    }
    path = str(Path(tmp_dir) / "clusters.json")
    with open(path, "w") as f:
        json.dump(clusters, f)
    return path, clusters


class TestWeightComputation:
    def test_rare_cluster_gets_higher_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )

            rare_weight = sampler.weights[0].item()
            common_weight = sampler.weights[2].item()
            assert rare_weight > common_weight, (
                f"Rare cluster weight {rare_weight} should be > common {common_weight}"
            )

    def test_weights_length_matches_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert len(sampler.weights) == len(data_list)


class TestIteration:
    def test_yields_correct_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            indices = list(sampler)
            assert len(indices) == len(data_list)

    def test_indices_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            indices = list(sampler)
            assert all(0 <= idx < len(data_list) for idx in indices)

    def test_rare_cluster_sampled_more(self):
        """Over many samples, rare cluster indices should appear more than their natural rate."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            indices = list(sampler)
            rare_count = sum(1 for idx in indices if idx < 2)
            # Natural rate is 2/100 = 2%. With inverse-frequency weighting,
            # rare cluster should be sampled at ~50% (2 clusters, equal weight).
            assert rare_count > 10, (
                f"Rare cluster appeared {rare_count} times in 100 samples, expected >> 2"
            )


class TestDDP:
    def test_two_ranks_cover_full_dataset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            s0 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=0, seed=42
            )
            s1 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=1, seed=42
            )
            i0, i1 = list(s0), list(s1)
            assert len(i0) == 10
            assert len(i1) == 10

    def test_set_epoch_changes_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(50)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            sampler.set_epoch(0)
            indices_e0 = list(sampler)
            sampler.set_epoch(1)
            indices_e1 = list(sampler)
            assert indices_e0 != indices_e1, "Different epochs should produce different orderings"


class TestEdgeCases:
    def test_missing_paths_get_default_weight(self):
        """Paths not in cluster JSON get weight 1.0 (neutral)."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            clusters = {"cluster_id0": data_list[:2]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert len(sampler.weights) == 5
            indices = list(sampler)
            assert len(indices) == 5
