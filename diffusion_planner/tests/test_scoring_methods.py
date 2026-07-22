from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusion_planner.utils.scoring_methods import MahalanobisScorer


def _random_l2(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-8)).astype(np.float32)


class TestMahalanobisFit:
    def test_fit_stores_params(self):
        emb = _random_l2(100, 32)
        scorer = MahalanobisScorer.fit(emb)
        assert scorer.mu.shape == (32,)
        assert scorer.precision.shape == (32, 32)
        assert scorer.whitening.shape == (32, 32)

    def test_condition_number_logged(self, capsys):
        emb = _random_l2(100, 32)
        MahalanobisScorer.fit(emb)
        captured = capsys.readouterr()
        assert "condition number" in captured.out.lower()


class TestMahalanobisScore:
    def test_returns_expected_key(self):
        emb = _random_l2(100, 16)
        scorer = MahalanobisScorer.fit(emb)
        result = scorer.score(torch.randn(2, 16))
        assert "mahalanobis" in result
        assert result["mahalanobis"].shape == (2,)

    def test_score_is_squared(self):
        emb = _random_l2(100, 8)
        scorer = MahalanobisScorer.fit(emb)
        result = scorer.score(torch.randn(1, 8))
        assert result["mahalanobis"].item() >= 0

    def test_inlier_lower_than_outlier(self):
        rng = np.random.RandomState(0)
        cluster = rng.randn(200, 8).astype(np.float32) * 0.1
        cluster[:, 0] += 5.0
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        cluster = (cluster / norms).astype(np.float32)
        scorer = MahalanobisScorer.fit(cluster)

        inlier = torch.from_numpy(cluster[:1])
        outlier = torch.from_numpy(-cluster[:1])
        assert (
            scorer.score(inlier)["mahalanobis"].item() < scorer.score(outlier)["mahalanobis"].item()
        )

    def test_1d_input_auto_batched(self):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        result = scorer.score(torch.randn(8))
        assert result["mahalanobis"].shape == (1,)


class TestMahalanobisNearest:
    def test_returns_k_neighbors(self):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        results = scorer.nearest(torch.from_numpy(emb[:1]), k=3)
        assert len(results) == 1
        assert len(results[0]) == 3
        assert "distance" in results[0][0]
        assert "index" in results[0][0]

    def test_self_is_nearest(self):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        results = scorer.nearest(torch.from_numpy(emb[0:1]), k=1)
        assert results[0][0]["index"] == 0
        assert results[0][0]["distance"] < 1e-4


class TestMahalanobisSaveLoad:
    def test_roundtrip(self, tmp_path):
        emb = _random_l2(50, 8)
        scorer = MahalanobisScorer.fit(emb)
        scorer.save(tmp_path / "maha.npz")

        loaded = MahalanobisScorer.load(tmp_path / "maha.npz")
        z = torch.randn(2, 8)
        torch.testing.assert_close(scorer.score(z)["mahalanobis"], loaded.score(z)["mahalanobis"])
