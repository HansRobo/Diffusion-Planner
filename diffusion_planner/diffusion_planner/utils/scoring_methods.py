from __future__ import annotations

from pathlib import Path

import faiss
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


class SphericalKMeansScorer:
    def __init__(
        self,
        centroids: np.ndarray,
        assignments: np.ndarray,
        k: int,
    ):
        self.centroids = centroids
        self.assignments = assignments
        self.k = k

    @classmethod
    def fit(cls, embeddings_l2: np.ndarray, k: int = 32) -> SphericalKMeansScorer:
        d = embeddings_l2.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, spherical=True)
        kmeans.train(embeddings_l2.astype(np.float32))
        centroids = kmeans.centroids.copy()

        _, assignments = kmeans.index.search(embeddings_l2.astype(np.float32), 1)
        assignments = assignments.ravel()

        return cls(centroids=centroids, assignments=assignments, k=k)

    def score(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        c_t = torch.from_numpy(self.centroids).to(z.device)
        cos_sim = z @ c_t.T
        dists = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0))

        sorted_dists, sorted_idx = torch.sort(dists, dim=-1)
        return {
            "kmeans_dist": sorted_dists[:, 0],
            "cluster_id": sorted_idx[:, 0],
            "margin": sorted_dists[:, 1] - sorted_dists[:, 0],
        }

    def nearest(self, z: torch.Tensor, embeddings_l2: np.ndarray, k: int = 5) -> list[list[dict]]:
        z = F.normalize(z.float(), dim=-1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        c_t = torch.from_numpy(self.centroids).to(z.device)
        cos_sim = z @ c_t.T
        dists_to_c = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0))
        _, top2_clusters = torch.topk(dists_to_c, 2, dim=-1, largest=False)

        bank_t = torch.from_numpy(embeddings_l2).to(z.device)
        assignments_t = torch.from_numpy(self.assignments).to(z.device)

        results = []
        for b in range(z.shape[0]):
            c0 = top2_clusters[b, 0].item()
            c1 = top2_clusters[b, 1].item()
            mask = (assignments_t == c0) | (assignments_t == c1)
            candidate_idx = torch.where(mask)[0]

            if len(candidate_idx) == 0:
                results.append([])
                continue

            candidate_emb = bank_t[candidate_idx]
            cos_sim_cand = z[b : b + 1] @ candidate_emb.T
            cand_dists = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim_cand, min=0.0)).squeeze(0)

            actual_k = min(k, len(candidate_idx))
            topk_dists, topk_local = torch.topk(cand_dists, actual_k, largest=False)

            neighbors = []
            for i in range(actual_k):
                neighbors.append(
                    {
                        "distance": topk_dists[i].item(),
                        "index": candidate_idx[topk_local[i]].item(),
                    }
                )
            results.append(neighbors)
        return results

    def save(self, path: Path) -> None:
        np.savez(
            path,
            centroids=self.centroids,
            assignments=self.assignments,
            k=self.k,
        )

    @classmethod
    def load(cls, path: Path) -> SphericalKMeansScorer:
        data = np.load(path)
        return cls(
            centroids=data["centroids"],
            assignments=data["assignments"],
            k=int(data["k"]),
        )
