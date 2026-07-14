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

from unittest.mock import MagicMock, patch

from visualize_cluster_report import (
    compute_cluster_stats,
    generate_html_report,
    load_cluster_json,
    render_bar_chart,
    render_cluster_videos,
    subsample_cluster_paths,
)



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


class TestRenderBarChart:
    def test_returns_base64_data_uri(self):
        stats = [
            {"cluster_id": "cluster_id0", "count": 90, "pct": 90.0,
             "weight": 0.5, "sampling_rate": 0.5, "draws_per_epoch": 50,
             "repeats_per_sample": 0.56},
            {"cluster_id": "cluster_id1", "count": 10, "pct": 10.0,
             "weight": 4.5, "sampling_rate": 0.5, "draws_per_epoch": 50,
             "repeats_per_sample": 5.0},
        ]
        result = render_bar_chart(stats)
        assert result.startswith("data:image/png;base64,")
        # Should be a reasonable-length base64 string
        assert len(result) > 100


class TestSubsampleClusterPaths:
    def test_caps_at_max_videos(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(100)],
        }
        result = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        assert len(result["cluster_id0"]) == 3

    def test_keeps_all_if_under_max(self):
        clusters = {
            "cluster_id0": ["/a/0.npz", "/a/1.npz"],
        }
        result = subsample_cluster_paths(clusters, max_videos=5, seed=42)
        assert len(result["cluster_id0"]) == 2

    def test_deterministic_with_seed(self):
        clusters = {"cluster_id0": [f"/a/{i}.npz" for i in range(50)]}
        r1 = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        r2 = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self):
        clusters = {"cluster_id0": [f"/a/{i}.npz" for i in range(50)]}
        r1 = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        r2 = subsample_cluster_paths(clusters, max_videos=3, seed=99)
        assert r1 != r2


class TestRenderClusterVideos:
    def test_calls_render_video_txt_per_cluster(self, tmp_path):
        subsampled = {
            "cluster_id0": ["/a/0.npz", "/a/1.npz"],
            "cluster_id1": ["/b/0.npz"],
        }
        with patch("visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"), \
             patch("visualize_cluster_report.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rendered, errors = render_cluster_videos(subsampled, str(tmp_path), workers=1)
            assert mock_run.call_count == 2

    def test_creates_cluster_subdirectories(self, tmp_path):
        subsampled = {
            "cluster_id0": ["/a/0.npz"],
        }
        with patch("visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"), \
             patch("visualize_cluster_report.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rendered, errors = render_cluster_videos(subsampled, str(tmp_path), workers=1)
            assert (tmp_path / "videos" / "cluster_id0").is_dir()

    def test_checks_render_video_txt_available(self):
        with patch("visualize_cluster_report.shutil.which", return_value=None):
            import pytest
            with pytest.raises(RuntimeError, match="render-video-txt not found"):
                render_cluster_videos({"cluster_id0": ["/a.npz"]}, "/tmp/out", workers=1)

    def test_collects_errors_from_render_log(self, tmp_path):
        subsampled = {"cluster_id0": ["/a/0.npz"]}
        log_content = '{"file": "a__0", "status": "error", "reason": "corrupt file"}\n'

        def fake_run(cmd, **kwargs):
            log_path = tmp_path / "videos" / "cluster_id0" / "render_log.jsonl"
            log_path.write_text(log_content)
            return MagicMock(returncode=0)

        with patch("visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"), \
             patch("visualize_cluster_report.subprocess.run", side_effect=fake_run):
            rendered, errors = render_cluster_videos(subsampled, str(tmp_path), workers=1)
            assert len(errors) == 1
            assert errors[0]["cluster_id"] == "cluster_id0"
            assert errors[0]["reason"] == "corrupt file"


class TestGenerateHtmlReport:
    def test_creates_report_html(self, tmp_path):
        stats = [
            {"cluster_id": "cluster_id0", "count": 90, "pct": 90.0,
             "weight": 0.56, "sampling_rate": 0.5, "draws_per_epoch": 50,
             "repeats_per_sample": 0.56},
            {"cluster_id": "cluster_id1", "count": 10, "pct": 10.0,
             "weight": 4.5, "sampling_rate": 0.5, "draws_per_epoch": 50,
             "repeats_per_sample": 5.0},
        ]
        chart_uri = "data:image/png;base64,AAAA"
        rendered = {
            "cluster_id0": [str(tmp_path / "videos/cluster_id0/a.mp4")],
            "cluster_id1": [],
        }
        errors = [{"cluster_id": "cluster_id1", "file": "b__0", "reason": "corrupt"}]
        path = generate_html_report(
            stats, chart_uri, rendered, errors, "/fake/cluster.json", str(tmp_path)
        )
        assert Path(path).exists()
        html = Path(path).read_text()
        assert "cluster_id0" in html
        assert "cluster_id1" in html
        assert "90.0" in html
        assert 'preload="none"' in html
        assert "Sampling Behavior" in html
        assert "corrupt" in html

    def test_video_tags_use_relative_paths(self, tmp_path):
        stats = [
            {"cluster_id": "cluster_id0", "count": 1, "pct": 100.0,
             "weight": 1.0, "sampling_rate": 1.0, "draws_per_epoch": 1,
             "repeats_per_sample": 1.0},
        ]
        vid_path = tmp_path / "videos" / "cluster_id0" / "sample.mp4"
        vid_path.parent.mkdir(parents=True)
        vid_path.touch()
        rendered = {"cluster_id0": [str(vid_path)]}
        path = generate_html_report(
            stats, "data:image/png;base64,AAAA", rendered, [],
            "/fake/cluster.json", str(tmp_path)
        )
        html = Path(path).read_text()
        assert "videos/cluster_id0/sample.mp4" in html
        # Should NOT contain absolute path
        assert str(tmp_path) not in html.split('src="')[1].split('"')[0]
