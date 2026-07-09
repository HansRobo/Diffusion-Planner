import json
import math
import warnings

import torch
from torch.utils.data import Sampler


class ClusterWeightedDistributedSampler(Sampler):
    """Inverse-frequency weighted sampler compatible with DDP.

    Reads a cluster assignment JSON (Fumiya's cluster.py output format) and assigns
    each sample a weight of 1/cluster_frequency. Rare clusters are sampled more often;
    no data is discarded.
    """

    def __init__(
        self,
        data_list: list[str],
        cluster_json_path: str,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        self.data_list = data_list
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.total_size = len(data_list)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

        self.weights, self.cluster_counts, self.matched_count = self._compute_weights(
            cluster_json_path
        )

    def _compute_weights(
        self, cluster_json_path: str
    ) -> tuple[torch.Tensor, dict[str, int], int]:
        with open(cluster_json_path, "r") as f:
            clusters = json.load(f)

        path_to_cluster: dict[str, str] = {}
        for cluster_id, paths in clusters.items():
            for p in paths:
                path_to_cluster[p] = cluster_id

        cluster_counts: dict[str, int] = {}
        for cluster_id, paths in clusters.items():
            cluster_counts[cluster_id] = len(paths)

        total_in_clusters = sum(cluster_counts.values())
        cluster_freq = {cid: count / total_in_clusters for cid, count in cluster_counts.items()}

        matched = 0
        weights = torch.ones(len(self.data_list), dtype=torch.float64)
        for i, path in enumerate(self.data_list):
            cluster_id = path_to_cluster.get(path)
            if cluster_id is not None:
                matched += 1
                weights[i] = 1.0 / (cluster_freq[cluster_id] + 1e-8)

        if matched == 0:
            raise ValueError(
                f"No paths in data_list matched cluster JSON "
                f"({len(path_to_cluster)} cluster entries). Check path formats."
            )
        if matched < len(self.data_list) // 2:
            warnings.warn(
                f"Only {matched}/{len(self.data_list)} paths matched cluster JSON. "
                f"Check path formats."
            )

        weights = weights / weights.mean()
        return weights, cluster_counts, matched

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        ).tolist()

        # Pad indices to ensure every rank gets exactly num_samples indices.
        # Pad by repeating from the front to reach num_samples * num_replicas length.
        padded_length = self.num_samples * self.num_replicas
        if len(indices) < padded_length:
            indices.extend(indices[: padded_length - len(indices)])

        # Shard across ranks
        indices = indices[self.rank : padded_length : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
