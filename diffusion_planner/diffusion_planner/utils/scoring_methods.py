from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class MahalanobisScorer:
    def __init__(
        self,
        mu: np.ndarray,
        precision: np.ndarray,
        whitening: np.ndarray,
        whitened_bank: np.ndarray | None = None,
    ):
        self.mu = mu
        self.precision = precision
        self.whitening = whitening
        self.whitened_bank = whitened_bank

    @classmethod
    def fit(cls, embeddings_l2: np.ndarray, eps: float = 1e-5) -> MahalanobisScorer:
        mu = embeddings_l2.mean(axis=0)
        centered = embeddings_l2 - mu
        cov = (centered.T @ centered) / (len(embeddings_l2) - 1)
        cov_reg = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)

        cond = np.linalg.cond(cov_reg)
        print(f"  Mahalanobis covariance condition number: {cond:.2e} (eps={eps})")

        precision = np.linalg.inv(cov_reg).astype(np.float32)

        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        whitening = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)

        whitened_bank = (centered @ whitening.T).astype(np.float32)

        return cls(
            mu=mu.astype(np.float32),
            precision=precision,
            whitening=whitening,
            whitened_bank=whitened_bank,
        )

    def score(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        mu_t = torch.from_numpy(self.mu).to(z.device)
        prec_t = torch.from_numpy(self.precision).to(z.device)
        centered = z - mu_t
        mahal = (centered @ prec_t * centered).sum(dim=-1)
        return {"mahalanobis": mahal}

    def nearest(self, z: torch.Tensor, k: int = 5) -> list[list[dict]]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        mu_t = torch.from_numpy(self.mu).to(z.device)
        w_t = torch.from_numpy(self.whitening).to(z.device)
        bank_t = torch.from_numpy(self.whitened_bank).to(z.device)

        centered = z - mu_t
        whitened_q = centered @ w_t.T

        dists = torch.cdist(whitened_q, bank_t)
        k = min(k, bank_t.shape[0])
        topk_dists, topk_idx = torch.topk(dists, k, dim=-1, largest=False)

        results = []
        for b in range(z.shape[0]):
            neighbors = []
            for i in range(k):
                neighbors.append(
                    {
                        "distance": topk_dists[b, i].item(),
                        "index": topk_idx[b, i].item(),
                    }
                )
            results.append(neighbors)
        return results

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mu=self.mu,
            precision=self.precision,
            whitening=self.whitening,
            whitened_bank=self.whitened_bank,
        )

    @classmethod
    def load(cls, path: Path) -> MahalanobisScorer:
        data = np.load(path)
        return cls(
            mu=data["mu"],
            precision=data["precision"],
            whitening=data["whitening"],
            whitened_bank=data.get("whitened_bank"),
        )
