#!/usr/bin/env python3
"""Build a latent OOD embedding bank from Diffusion Planner training data.

Usage:
    python scripts/build_latent_ood_bank.py \
        --model_path /path/to/best_model.pth \
        --args_path /path/to/args.json \
        --train_list /path/to/train.json \
        --output_dir /path/to/latent_ood_bank \
        [--val_list /path/to/val.json] \
        [--batch_size 64] \
        [--num_workers 4] \
        [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.latent_ood import LatentOODScorer
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=Path, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--args_path", type=Path, required=True, help="Path to training args.json")
    p.add_argument("--train_list", type=Path, required=True, help="JSON list of training .npz paths")
    p.add_argument("--output_dir", type=Path, required=True, help="Output bank directory")
    p.add_argument("--val_list", type=Path, default=None, help="Optional JSON list of validation .npz for calibration")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_model(args_path: Path, model_path: Path, device: str):
    config_obj = Config(str(args_path))
    model = Diffusion_Planner(config_obj)
    model.eval()
    ckpt = torch.load(str(model_path), map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    obs_norm = config_obj.observation_normalizer
    return model, obs_norm, config_obj


def extract_embeddings(model, obs_norm, data_list: Path, batch_size: int, num_workers: int, device: str):
    dataset = DiffusionPlannerData(str(data_list))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(device != "cpu"))

    with open(data_list) as f:
        npz_paths = json.load(f)

    embeddings = []
    idx = 0
    records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = obs_norm(batch)
            encoding = model.encoder(batch)
            z = encoding.mean(dim=1)
            z = F.normalize(z, dim=-1)
            embeddings.append(z.cpu().numpy())

            for i in range(z.shape[0]):
                path = npz_paths[idx] if idx < len(npz_paths) else f"unknown_{idx}"
                records.append({"row": idx, "npz_path": path})
                idx += 1

    return np.concatenate(embeddings, axis=0).astype(np.float32), records


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    model, obs_norm, config_obj = load_model(args.args_path, args.model_path, args.device)
    hidden_dim = config_obj.hidden_dim
    print(f"  hidden_dim={hidden_dim}")

    print(f"Extracting training embeddings from {args.train_list}")
    t0 = time.time()
    embeddings, records = extract_embeddings(
        model, obs_norm, args.train_list, args.batch_size, args.num_workers, args.device
    )
    print(f"  Extracted {embeddings.shape[0]} embeddings in {time.time() - t0:.1f}s")

    metadata = {
        "model_path": str(args.model_path),
        "args_path": str(args.args_path),
        "embedding_source": "encoder_outputs.mean(dim=1)",
        "embedding_dim": int(embeddings.shape[1]),
        "pooling": "mean_all_tokens_after_encoder_mask_zeroing",
        "num_embeddings": int(embeddings.shape[0]),
        "l2_normalized": True,
        "git_sha": git_sha(),
        "train_list": str(args.train_list),
    }

    scorer = LatentOODScorer.build(embeddings, records, metadata, device=args.device)

    if args.val_list:
        print(f"Calibrating on validation set {args.val_list}")
        val_emb, _ = extract_embeddings(
            model, obs_norm, args.val_list, args.batch_size, args.num_workers, args.device
        )
        val_z = torch.from_numpy(val_emb).to(args.device)
        val_scores = []
        for i in tqdm(range(0, val_z.shape[0], args.batch_size), desc="Scoring validation"):
            chunk = val_z[i : i + args.batch_size]
            result = scorer.score(chunk, k=10)
            val_scores.append(result["knn_mean"].cpu().numpy())
        val_scores = np.concatenate(val_scores)
        scorer.calibrate(val_scores)
        print(f"  warning={scorer._calibration['thresholds']['warning']:.4f}")
        print(f"  high={scorer._calibration['thresholds']['high']:.4f}")

    print(f"Saving bank to {args.output_dir}")
    scorer.save(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
