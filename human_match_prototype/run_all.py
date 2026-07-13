"""Score frames: coverage + typicality + latent OOD -> scores.csv.

Usage:
  .venv/bin/python -m human_match_prototype.run_all \
    --npz_list data/or_scene_npz/path_list_test.json \
    --output data/human_match/scores_or.csv \
    [--limit 20] [--num_samples 64] [--seed 0] [--self_check]
"""

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import torch
from diffusion_planner.utils.latent_ood import LatentOODScorer
from tqdm import tqdm

from human_match_prototype.metrics import coverage_metrics
from human_match_prototype.sampler import TrajectorySampler
from human_match_prototype.typicality import typicality

MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")
BANK_DIR = "data/latent_ood_bank_5k"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_list", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--self_check",
        action="store_true",
        help="Replace the human with planner sample 0, scored against samples 1..N "
        "(expect: covered / typical)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    paths = json.load(open(args.npz_list))
    if args.limit:
        paths = paths[: args.limit]
    sampler = TrajectorySampler(
        MODEL_DIR / "args.json", MODEL_DIR / "diffusion_planner.onnx", args.device
    )
    latent = LatentOODScorer.load(BANK_DIR, device=args.device)

    rows, skipped = [], 0
    for path in tqdm(paths):
        try:
            r = sampler.sample(path, args.num_samples, args.seed)
            human, samples = r.human_future, r.ego_samples
            if args.self_check:
                human, samples = samples[0], samples[1:]
            row = {
                "npz_path": path,
                "category": Path(path).parent.name,
                "self_check": int(args.self_check),
            }
            row.update(coverage_metrics(human, samples))
            row.update(typicality(human, samples))
            z = torch.from_numpy(r.embedding).unsqueeze(0).to(args.device)
            row["latent_knn_mean"] = float(latent.score(z, k=10)["knn_mean"][0])
            rows.append(row)
        except Exception:
            skipped += 1
            traceback.print_exc()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
