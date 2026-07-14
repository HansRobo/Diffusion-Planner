# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate an HTML diagnostic report for cluster-weighted sampling.

Usage:
    python visualize_cluster_report.py \\
        --cluster_json /path/to/cluster_result.json \\
        --output_dir   /path/to/report_output/ \\
        [--max_videos 3] [--workers 1] [--seed 42]

Reads the cluster assignment JSON from cluster.py, computes sampling
statistics, renders BEV video examples per cluster via render-video-txt
(clip-review-tool), and assembles an HTML report.
"""

from __future__ import annotations

import json


def load_cluster_json(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_cluster_stats(clusters: dict[str, list[str]]) -> list[dict]:
    sorted_ids = sorted(clusters.keys(), key=lambda x: int(x.replace("cluster_id", "")))
    total = sum(len(v) for v in clusters.values())

    raw_weights = {}
    for cid in sorted_ids:
        freq = len(clusters[cid]) / total
        raw_weights[cid] = 1.0 / (freq + 1e-8)

    mean_weight = sum(raw_weights[cid] * len(clusters[cid]) for cid in sorted_ids) / total
    weights = {cid: w / mean_weight for cid, w in raw_weights.items()}

    total_weight = sum(weights[cid] * len(clusters[cid]) for cid in sorted_ids)

    stats = []
    for cid in sorted_ids:
        count = len(clusters[cid])
        w = weights[cid]
        sampling_rate = (w * count) / total_weight
        draws = sampling_rate * total
        stats.append({
            "cluster_id": cid,
            "count": count,
            "pct": 100.0 * count / total,
            "weight": w,
            "sampling_rate": sampling_rate,
            "draws_per_epoch": draws,
            "repeats_per_sample": draws / count if count > 0 else 0.0,
        })
    return stats
