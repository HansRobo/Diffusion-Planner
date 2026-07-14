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

"""Tests for sampling/visualize_cluster_report.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SAMPLING_DIR = Path(__file__).resolve().parent.parent / "sampling"
sys.path.insert(0, str(SAMPLING_DIR))

from visualize_cluster_report import compute_cluster_stats, load_cluster_json


class TestLoadClusterJson:
    def test_loads_valid_json(self, tmp_path):
        data = {
            "cluster_id0": ["/a/0.npz", "/a/1.npz"],
            "cluster_id1": ["/a/2.npz"],
        }
        p = tmp_path / "clusters.json"
        p.write_text(json.dumps(data))
        result = load_cluster_json(str(p))
        assert result == data

    def test_raises_on_missing_file(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_cluster_json("/nonexistent/path.json")


class TestComputeClusterStats:
    def test_two_clusters(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(90)],
            "cluster_id1": [f"/b/{i}.npz" for i in range(10)],
        }
        stats = compute_cluster_stats(clusters)

        assert len(stats) == 2
        s0 = next(s for s in stats if s["cluster_id"] == "cluster_id0")
        s1 = next(s for s in stats if s["cluster_id"] == "cluster_id1")

        assert s0["count"] == 90
        assert s1["count"] == 10
        assert abs(s0["pct"] - 90.0) < 0.01
        assert abs(s1["pct"] - 10.0) < 0.01

        # Rare cluster should have higher weight
        assert s1["weight"] > s0["weight"]

        # Weights normalized to mean 1.0
        mean_w = (s0["weight"] * 90 + s1["weight"] * 10) / 100
        assert abs(mean_w - 1.0) < 0.01

        # Expected draws: rare cluster gets more than natural 10%
        assert s1["draws_per_epoch"] > 10
        # Expected repeats: rare cluster repeats more per sample
        assert s1["repeats_per_sample"] > s0["repeats_per_sample"]

    def test_sorted_by_cluster_id(self):
        clusters = {
            "cluster_id2": ["/a.npz"],
            "cluster_id0": ["/b.npz"],
            "cluster_id1": ["/c.npz"],
        }
        stats = compute_cluster_stats(clusters)
        ids = [s["cluster_id"] for s in stats]
        assert ids == ["cluster_id0", "cluster_id1", "cluster_id2"]
