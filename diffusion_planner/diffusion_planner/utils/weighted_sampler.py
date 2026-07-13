import math
import warnings

import torch
from torch.utils.data import Sampler

from diffusion_planner.utils.path_key import data_path_to_rel
from diffusion_planner.utils.train_utils import openjson


class ClusterWeightedDistributedSampler(Sampler):
    """Inverse-frequency weighted sampler compatible with DDP.

    Reads a cluster assignment JSON (Fumiya's cluster.py output format) and assigns
    each sample a weight of 1/cluster_frequency. Rare clusters are sampled more often;
    no data is discarded.

    The cluster JSON must contain the exact same path strings as the training
    data_list JSON. Run cluster.py with the same --data_list file used for
    --train_set_list to ensure paths match.
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
        if len(data_list) > 2**24:
            raise ValueError(
                f"data_list has {len(data_list)} entries, exceeding "
                f"torch.multinomial's 2^24 ({2**24:,}) category limit on CPU."
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.total_size = len(data_list)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

        self.weights, self.cluster_counts, self.matched_count = self._compute_weights(
            cluster_json_path
        )

    def _compute_weights(self, cluster_json_path: str) -> tuple[torch.Tensor, dict[str, int], int]:
        clusters = openjson(cluster_json_path)

        path_to_cluster: dict[str, str] = {}
        for cluster_id, paths in clusters.items():
            for p in paths:
                path_to_cluster[str(data_path_to_rel(p))] = cluster_id

        if not path_to_cluster:
            raise ValueError(
                f"Cluster JSON contains no paths "
                f"({len(clusters)} clusters, all empty). Check cluster JSON."
            )

        # First pass: identify cluster membership and count live matches
        sample_cluster = [None] * len(self.data_list)
        live_counts: dict[str, int] = {}
        matched = 0

        for i, path in enumerate(self.data_list):
            cid = path_to_cluster.get(str(data_path_to_rel(path)))
            if cid is not None:
                matched += 1
                sample_cluster[i] = cid
                live_counts[cid] = live_counts.get(cid, 0) + 1

        if matched == 0:
            raise ValueError(
                f"No paths in data_list matched cluster JSON "
                f"({len(path_to_cluster)} cluster entries). Check path formats."
            )

        # Compute frequencies from live counts
        cluster_freq = {cid: count / matched for cid, count in live_counts.items()}

        # Second pass: assign weights using live frequencies
        weights = torch.ones(len(self.data_list), dtype=torch.float64)
        for i, cid in enumerate(sample_cluster):
            if cid is not None:
                weights[i] = 1.0 / (cluster_freq[cid] + 1e-8)

        if matched < len(self.data_list):
            matched_mask = torch.tensor(
                [c is not None for c in sample_cluster], dtype=torch.bool
            )
            mean_matched = weights[matched_mask].mean().item()
            weights[~matched_mask] = mean_matched
            warnings.warn(
                f"{matched}/{len(self.data_list)} paths matched cluster JSON. "
                f"Unmatched paths assigned neutral weight."
            )

        weights = weights / weights.mean()
        return weights, live_counts, matched

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        padded_length = self.num_samples * self.num_replicas
        indices = torch.multinomial(
            self.weights, padded_length, replacement=True, generator=g
        ).tolist()

        indices = indices[self.rank : padded_length : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
