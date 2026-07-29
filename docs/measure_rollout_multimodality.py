"""Measure our policy's multimodality on the real mine, with the paper's own metric.

The question this answers: does our RL rollout distribution have enough sample
diversity for AWR to have anything to reweight, or is it mode-collapsed and in need
of the released `augment_trajectory_batch` to manufacture spread?

The paper defines two quantities we can reproduce exactly:

  * Section 2.2 "Trajectory Divergence": the average PAIRWISE Euclidean distance
    among samples drawn for the same scene.
  * Appendix "Divergence Score" (Eq. after L1200): the average distance of trajectory
    ENDPOINTS to their geometric centre, over 64 generations.

We have 32 generations per scene, not 64, so the endpoint version is reported as-is
and the pairwise version alongside it. Both are computed on `ego_world`, which is what
the scorer and the AWR loss actually consume.

It also measures the firing rate of the paper's own remedy for a collapsed group --
"we discard samples in which all actions receive identical rewards" (ap:implementation)
-- because that rate IS the paper's definition of how often diversity has failed.

Read-only. CPU only. Runs on the live cache of the arm currently training.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-dir", required=True)
    parser.add_argument("--shards", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    paths = sorted(glob.glob(f"{args.cycle_dir}/rank00_shard*.pt"))
    # Spread the sample over the whole mine rather than taking the first N, so the
    # numbers are not biased toward whatever the policy looked like at mine start.
    if len(paths) > args.shards:
        idx = np.linspace(0, len(paths) - 1, args.shards).round().astype(int)
        paths = [paths[i] for i in idx]

    k = args.group_size
    pair_div, end_div = [], []
    sd_step = {1: [], 5: [], 10: [], 20: [], 40: [], 80: []}
    reward_sd, reward_ptp = [], []
    groups = 0
    dead_groups = 0
    distinct_x1, distinct_y1, distinct_x80 = [], [], []
    step1_abs_x = []

    for path in paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ego = blob["ego_world"].double()  # (S*k, 80, 4)
        reward = blob["reward"].double()  # (S*k,)
        n = ego.shape[0] // k
        ego = ego[: n * k].view(n, k, ego.shape[1], ego.shape[2])
        reward = reward[: n * k].view(n, k)

        xy = ego[..., :2]  # (n, k, 80, 2)

        # Paper, Appendix: endpoint to geometric centre.
        end = xy[:, :, -1, :]  # (n, k, 2)
        centre = end.mean(dim=1, keepdim=True)
        end_div.append((end - centre).norm(dim=-1).mean(dim=1).numpy())

        # Paper, Section 2.2: average pairwise distance among samples.
        d = (end.unsqueeze(2) - end.unsqueeze(1)).norm(dim=-1)  # (n, k, k)
        iu = torch.triu_indices(k, k, offset=1)
        pair_div.append(d[:, iu[0], iu[1]].mean(dim=1).numpy())

        for step in sd_step:
            sd_step[step].append(xy[:, :, step - 1, 0].std(dim=1).numpy())

        reward_sd.append(reward.std(dim=1).numpy())
        reward_ptp.append((reward.max(dim=1).values - reward.min(dim=1).values).numpy())
        groups += n
        # The paper's own discard criterion, at the trainer's own epsilon.
        dead_groups += int((reward.std(dim=1) <= 1e-6).sum())

        for g in range(n):
            distinct_x1.append(len(torch.unique(xy[g, :, 0, 0])))
            distinct_y1.append(len(torch.unique(xy[g, :, 0, 1])))
            distinct_x80.append(len(torch.unique(xy[g, :, -1, 0])))
        step1_abs_x.append(xy[:, :, 0, 0].abs().mean(dim=1).numpy())

    def stat(chunks, scale=1.0):
        a = np.concatenate(chunks) * scale
        return {
            "mean": float(a.mean()),
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    report = {
        "cycle_dir": args.cycle_dir,
        "shards_read": len(paths),
        "shards_available_rank00": len(sorted(glob.glob(f"{args.cycle_dir}/rank00_shard*.pt"))),
        "groups": groups,
        "generations_per_group": k,
        "paper_metrics": {
            "trajectory_divergence_pairwise_m": stat(pair_div),
            "divergence_score_endpoint_to_centre_m": stat(end_div),
        },
        "candidate_spread_sd_x_mm": {f"step_{s}": stat(v, 1000.0) for s, v in sd_step.items()},
        "reward_within_group": {
            "sd": stat(reward_sd),
            "peak_to_peak": stat(reward_ptp),
            "identical_reward_groups": dead_groups,
            "identical_reward_group_fraction": dead_groups / max(groups, 1),
        },
        "distinct_values_per_group": {
            "step1_x": stat([np.asarray(distinct_x1, dtype=np.float64)]),
            "step1_y": stat([np.asarray(distinct_y1, dtype=np.float64)]),
            "step80_x": stat([np.asarray(distinct_x80, dtype=np.float64)]),
        },
        "step1_abs_x_mm": stat(step1_abs_x, 1000.0),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
