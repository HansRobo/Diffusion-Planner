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

import base64
import io
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def render_bar_chart(stats: list[dict]) -> str:
    ids = [s["cluster_id"].replace("cluster_id", "") for s in stats]
    counts = [s["count"] for s in stats]

    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.6), 4))
    bars = ax.bar(ids, counts, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Sample Count")
    ax.set_title("Cluster Size Distribution")
    ax.tick_params(axis="x", rotation=45 if len(ids) > 15 else 0)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def subsample_cluster_paths(
    clusters: dict[str, list[str]], max_videos: int, seed: int
) -> dict[str, list[str]]:
    result = {}
    for cid, paths in clusters.items():
        rng = random.Random(seed + int(cid.replace("cluster_id", "")))
        if len(paths) <= max_videos:
            result[cid] = list(paths)
        else:
            result[cid] = rng.sample(paths, max_videos)
    return result


def render_cluster_videos(
    subsampled: dict[str, list[str]], output_dir: str, workers: int
) -> tuple[dict[str, list[str]], list[dict]]:
    if not shutil.which("render-video-txt"):
        raise RuntimeError(
            "render-video-txt not found on PATH. "
            "Install clip-review-tool: pip install -e /path/to/clip-review-tool"
        )

    videos_dir = Path(output_dir) / "videos"
    rendered: dict[str, list[str]] = {}
    errors: list[dict] = []

    for cluster_id, paths in subsampled.items():
        if not paths:
            rendered[cluster_id] = []
            continue

        cluster_dir = videos_dir / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            for p in paths:
                f.write(p + "\n")
            txt_path = f.name

        try:
            subprocess.run(
                ["render-video-txt", txt_path, str(cluster_dir),
                 "--workers", str(workers)],
                check=False,
            )
        finally:
            Path(txt_path).unlink(missing_ok=True)

        mp4s = sorted(str(p) for p in cluster_dir.glob("*.mp4"))
        rendered[cluster_id] = mp4s

        log_path = cluster_dir / "render_log.jsonl"
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("status") == "error":
                    errors.append({
                        "cluster_id": cluster_id,
                        "file": entry.get("file", ""),
                        "reason": entry.get("reason", "unknown"),
                    })

    return rendered, errors
