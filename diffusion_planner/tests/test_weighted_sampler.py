import json
import tempfile
import warnings
from pathlib import Path

import pytest
import torch
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

    def test_two_ranks_get_different_shards(self):
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
            assert i0 != i1, "Different ranks should receive different index shards"

    def test_padding_with_non_divisible_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(21)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            s0 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=0, seed=42
            )
            s1 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=1, seed=42
            )
            assert len(list(s0)) == 11
            assert len(list(s1)) == 11

    def test_tiny_dataset_ddp_equal_shards(self):
        """Every rank gets equal shard length even when total_size < num_replicas."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(2)]
            clusters = {"cluster_id0": data_list[:1], "cluster_id1": data_list[1:]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            num_replicas = 8
            for rank in range(num_replicas):
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=num_replicas, rank=rank, seed=42
                )
                indices = list(sampler)
                assert len(indices) == sampler.num_samples, (
                    f"Rank {rank}: got {len(indices)} indices, expected {sampler.num_samples}"
                )

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
    def test_multinomial_size_guard(self):
        """Datasets exceeding torch.multinomial's 2^24 category limit are rejected early."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            class BigList(list):
                def __len__(self):
                    return 2**24 + 1

            with pytest.raises(ValueError, match="2\\^24"):
                ClusterWeightedDistributedSampler(
                    BigList(data_list), cluster_path, num_replicas=1, rank=0
                )

    def test_missing_paths_get_neutral_weight(self):
        """Paths not in any cluster get the mean of matched weights (neutral rate).

        After normalization: rare > unmatched > common.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            clusters = {
                "cluster_id0": data_list[:2],
                "cluster_id1": data_list[2:6],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )
                matching_warnings = [x for x in w if "paths matched cluster JSON" in str(x.message)]
                assert len(matching_warnings) == 1

            assert len(sampler.weights) == 10
            assert sampler.matched_count == 6

            rare_weight = sampler.weights[0].item()
            common_weight = sampler.weights[3].item()
            unmatched_weight = sampler.weights[7].item()

            assert rare_weight > unmatched_weight > common_weight, (
                f"Expected rare ({rare_weight:.3f}) > unmatched ({unmatched_weight:.3f}) "
                f"> common ({common_weight:.3f})"
            )
            for i in range(6, 10):
                assert sampler.weights[i].item() == pytest.approx(unmatched_weight)

    def test_all_empty_clusters_raises_valueerror(self):
        """All-empty cluster JSON should raise ValueError, not ZeroDivisionError."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            clusters = {"cluster_id0": [], "cluster_id1": []}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with pytest.raises(ValueError, match="Cluster JSON contains no paths"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )

    def test_no_matching_paths_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            clusters = {"cluster_id0": ["/other/path_0.npz", "/other/path_1.npz"]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with pytest.raises(ValueError, match="No paths in data_list matched"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )

    def test_low_match_rate_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            clusters = {"cluster_id0": data_list[:2]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )
                matching_warnings = [x for x in w if "paths matched cluster JSON" in str(x.message)]
                assert len(matching_warnings) == 1
                assert sampler.matched_count == 2

    def test_cluster_counts_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.cluster_counts == {"cluster_id0": 2, "cluster_id1": 18}
            assert sampler.matched_count == 20

    def test_cluster_counts_reflect_live_data_list(self):
        """cluster_counts should reflect the live data_list, not raw JSON totals."""
        with tempfile.TemporaryDirectory() as tmp:
            all_paths = [f"/data/sample_{i}.npz" for i in range(20)]
            clusters = {
                "cluster_id0": all_paths[:10],
                "cluster_id1": all_paths[10:],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            # data_list only includes 2 from cluster_id0 and 8 from cluster_id1
            data_list = all_paths[:2] + all_paths[10:18]

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.cluster_counts == {"cluster_id0": 2, "cluster_id1": 8}
            assert sampler.matched_count == 10

    def test_prefix_mismatched_paths_still_match(self):
        """Cluster JSON with different path prefix still matches via canonicalization."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"data/train/20230101/1200/frame_{i}.npz" for i in range(10)]
            clusters = {
                "cluster_id0": [
                    f"/mnt/nfs/data/train/20230101/1200/frame_{i}.npz" for i in range(2)
                ],
                "cluster_id1": [
                    f"/mnt/nfs/data/train/20230101/1200/frame_{i}.npz" for i in range(2, 10)
                ],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.matched_count == 10, (
                f"Expected all 10 paths to match, got {sampler.matched_count}"
            )


class TestAlpha:
    def test_default_alpha_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            default = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            explicit = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=1.0
            )
            assert default.alpha == 1.0
            assert torch.equal(default.weights, explicit.weights)

    def test_alpha_zero_gives_uniform_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.0
            )
            for w in sampler.weights.tolist():
                assert w == pytest.approx(1.0)

    def test_alpha_half_is_sqrt_of_full_ratio(self):
        """The rare/common weight ratio must go from R to R**alpha."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            full = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=1.0
            )
            half = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.5
            )
            # index 0 is in the rare cluster (2 samples), index 2 in the common one (98)
            ratio_full = full.weights[0].item() / full.weights[2].item()
            ratio_half = half.weights[0].item() / half.weights[2].item()

            assert ratio_full == pytest.approx(49.0, rel=1e-4)
            assert ratio_half == pytest.approx(ratio_full**0.5, rel=1e-4)

    def test_alpha_reduces_oversampling_multiplier(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            full = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=1.0
            )
            soft = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.25
            )
            assert soft.cluster_multipliers["cluster_id0"] < full.cluster_multipliers["cluster_id0"]
            assert soft.cluster_multipliers["cluster_id0"] > 1.0

    def test_negative_alpha_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="alpha"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=-0.5
                )

    def test_cluster_multipliers_match_normalized_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert set(sampler.cluster_multipliers) == {"cluster_id0", "cluster_id1"}
            # index 0 -> cluster_id0, index 2 -> cluster_id1
            assert sampler.cluster_multipliers["cluster_id0"] == pytest.approx(
                sampler.weights[0].item()
            )
            assert sampler.cluster_multipliers["cluster_id1"] == pytest.approx(
                sampler.weights[2].item()
            )

    def test_multipliers_average_to_one(self):
        """Multipliers weighted by cluster size average to 1.0 — no net inflation."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.5
            )
            weighted = sum(
                sampler.cluster_multipliers[cid] * count
                for cid, count in sampler.cluster_counts.items()
            )
            assert weighted / len(data_list) == pytest.approx(1.0, rel=1e-6)
