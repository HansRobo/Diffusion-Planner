#!/usr/bin/env python3
"""Audit the exact reward/weight signal in a salvaged T4 AWR replay cache.

Only the small ``rewards.npy`` and ``weights.npy`` memmaps are read.  The
multi-terabyte trajectory and encoder arrays remain untouched.  Outputs are a
machine-readable summary and a 16:9 figure suitable for the conference report.
This establishes what signal entered training; it is not post-training model
performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="T4 full replay: exact AWR learning signal")
    return parser.parse_args()


def _distribution(values: np.ndarray) -> dict:
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(values.max()),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _render(summary: dict, arrays: dict[str, np.ndarray], output_dir: Path, title: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
            "figure.facecolor": "#f5f7fb",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="#f5f7fb")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.10, wspace=0.22, hspace=0.32)
    fig.suptitle(title, x=0.06, y=0.965, ha="left", fontsize=22, fontweight="bold", color="#102a43")
    fig.text(
        0.06,
        0.915,
        f"{summary['scene_groups']:,} real scene-groups · exact frozen-refresh reward/weight memmaps · K={summary['group_size']} stochastic candidates",
        fontsize=11,
        color="#52677d",
    )

    ax = axes[0, 0]
    values = arrays["best_minus_candidate0"]
    ax.hist(np.clip(values, 0, 0.25), bins=70, color="#14b8a6", alpha=0.9)
    ax.set_title("1 · Candidate headroom: best − first stochastic sample", loc="left")
    ax.set_xlabel("reward gap (values >0.25 clipped into last bin)")
    ax.set_ylabel("scene-groups")
    ax.grid(axis="y", alpha=0.16)
    ax.text(
        0.98,
        0.94,
        f">0.01: {_pct(summary['fractions']['best_minus_candidate0_gt_0.01'])}\n"
        f">0.05: {_pct(summary['fractions']['best_minus_candidate0_gt_0.05'])}\n"
        f"p50: {summary['distributions']['best_minus_candidate0']['p50']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=.45", "fc": "#ecfeff", "ec": "#99f6e4"},
    )

    ax = axes[0, 1]
    gain = arrays["weighted_reward_gain"]
    lo, hi = np.quantile(gain, [0.001, 0.999])
    ax.hist(np.clip(gain, lo, hi), bins=70, color="#2563eb", alpha=0.88)
    ax.axvline(0, color="#dc2626", lw=1.5, ls="--")
    ax.set_title("2 · AWR direction: weighted reward − uniform reward", loc="left")
    ax.set_xlabel("within-group reward lift (0.1% tails clipped)")
    ax.set_ylabel("scene-groups")
    ax.grid(axis="y", alpha=0.16)
    ax.text(
        0.98,
        0.94,
        f"positive: {_pct(summary['fractions']['weighted_gain_positive'])}\n"
        f">0.01: {_pct(summary['fractions']['weighted_gain_gt_0.01'])}\n"
        f"mean: {summary['distributions']['weighted_reward_gain']['mean']:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=.45", "fc": "#eff6ff", "ec": "#bfdbfe"},
    )

    ax = axes[1, 0]
    ess = arrays["ess"]
    ax.hist(ess, bins=np.linspace(1, summary["group_size"], 55), color="#7c3aed", alpha=0.88)
    ax.axvline(summary["distributions"]["ess"]["p50"], color="#4c1d95", lw=1.5)
    ax.set_title("3 · Weight concentration: effective candidates (ESS)", loc="left")
    ax.set_xlabel(f"ESS out of K={summary['group_size']} (higher = less concentrated)")
    ax.set_ylabel("scene-groups")
    ax.grid(axis="y", alpha=0.16)
    ax.text(
        0.02,
        0.94,
        f"p05 / p50 / p95\n{summary['distributions']['ess']['p05']:.2f} / "
        f"{summary['distributions']['ess']['p50']:.2f} / {summary['distributions']['ess']['p95']:.2f}\n"
        f"ESS <5: {_pct(summary['fractions']['ess_lt_5'])}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=.45", "fc": "#f5f3ff", "ec": "#ddd6fe"},
    )

    ax = axes[1, 1]
    labels = [
        "Any reward=0\ncandidate",
        "All K rewards=0",
        "Reward-tied\ngroup",
        "First stochastic sample\nis best (ties count)",
    ]
    vals = [
        summary["fractions"]["any_zero_reward"],
        summary["fractions"]["all_zero_reward"],
        summary["fractions"]["reward_tied"],
        summary["fractions"]["candidate0_is_best"],
    ]
    colors = ["#f59e0b", "#dc2626", "#64748b", "#10b981"]
    bars = ax.bar(labels, np.asarray(vals) * 100.0, color=colors, width=0.66)
    ax.set_title("4 · Where ranking signal exists—or does not", loc="left")
    ax.set_ylabel("scene-groups [%]")
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, _pct(value), ha="center", fontweight="bold")

    fig.text(
        0.06,
        0.035,
        "Contract: mining used 10 independent N(0,I) samples at noise scale 0.5 plus HDP-style trajectory augmentation std 0.5; no zero-noise anchor. "
        "This figure audits training targets, not checkpoint improvement.",
        fontsize=9.5,
        color="#52677d",
    )
    for suffix, dpi in ((".png", 150), (".svg", 1)):
        fig.savefig(output_dir / f"full_replay_signal{suffix}", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = _args()
    rank_dirs = sorted(args.replay_root.glob("rank_*"))
    if not rank_dirs:
        raise FileNotFoundError(f"no rank_* directories under {args.replay_root}")

    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "best_minus_candidate0",
            "best_minus_uniform",
            "group_mean_reward",
            "best_group_reward",
            "weighted_reward_gain",
            "reward_std",
            "ess",
            "top_weight_share",
            "mean_weight",
            "any_zero",
            "all_zero",
            "tied",
            "zero_weight_group",
            "candidate0_is_best",
        )
    }
    rank_counts = []
    top_index_counts: np.ndarray | None = None
    group_size = None
    candidate_sum = candidate_sq_sum = 0.0
    candidate_count = 0
    candidate_min = float("inf")
    candidate_max = float("-inf")
    source_manifests = []

    for rank_dir in rank_dirs:
        manifest_path = rank_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        n = int(manifest["scene_count"])
        k = int(manifest["group_size"])
        if group_size is None:
            group_size = k
            top_index_counts = np.zeros(k, dtype=np.int64)
        if k != group_size:
            raise RuntimeError(f"mixed group size: {k} versus {group_size}")
        rewards = np.asarray(np.load(rank_dir / "rewards.npy", mmap_mode="r")[:n], dtype=np.float32)
        weights = np.asarray(np.load(rank_dir / "weights.npy", mmap_mode="r")[:n], dtype=np.float32)
        if rewards.shape != (n, k) or weights.shape != (n, k):
            raise RuntimeError(f"shape mismatch in {rank_dir}: {rewards.shape}, {weights.shape}")
        if not (np.isfinite(rewards).all() and np.isfinite(weights).all()):
            raise RuntimeError(f"non-finite reward/weight in {rank_dir}")

        uniform = rewards.mean(axis=1)
        best = rewards.max(axis=1)
        weight_sum = weights.sum(axis=1)
        weighted = (rewards * weights).sum(axis=1) / np.maximum(weight_sum, 1e-12)
        reward_std = rewards.std(axis=1, ddof=1)
        ess = weight_sum * weight_sum / np.maximum((weights * weights).sum(axis=1), 1e-12)
        collected["best_minus_candidate0"].append(best - rewards[:, 0])
        collected["best_minus_uniform"].append(best - uniform)
        collected["group_mean_reward"].append(uniform)
        collected["best_group_reward"].append(best)
        collected["weighted_reward_gain"].append(weighted - uniform)
        collected["reward_std"].append(reward_std)
        collected["ess"].append(ess)
        collected["top_weight_share"].append(weights.max(axis=1) / np.maximum(weight_sum, 1e-12))
        collected["mean_weight"].append(weights.mean(axis=1))
        zero = rewards <= 1e-8
        collected["any_zero"].append(zero.any(axis=1))
        collected["all_zero"].append(zero.all(axis=1))
        collected["tied"].append(reward_std <= 1e-6)
        collected["zero_weight_group"].append(weight_sum <= 1e-12)
        collected["candidate0_is_best"].append(rewards[:, 0] >= best - 1e-7)
        top_index_counts += np.bincount(np.argmax(rewards, axis=1), minlength=k)
        candidate_sum += float(rewards.astype(np.float64).sum())
        candidate_sq_sum += float(np.square(rewards.astype(np.float64)).sum())
        candidate_count += int(rewards.size)
        candidate_min = min(candidate_min, float(rewards.min()))
        candidate_max = max(candidate_max, float(rewards.max()))
        rank_counts.append({"rank": int(manifest["rank"]), "scene_count": n})
        source_manifests.append(str(manifest_path.resolve()))
        print(f"{rank_dir.name}: {n:,} groups", flush=True)

    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    scene_groups = len(arrays["ess"])
    strata_masks = {
        "no_zero_candidate": ~arrays["any_zero"].astype(bool),
        "mixed_zero_and_nonzero": arrays["any_zero"].astype(bool)
        & ~arrays["all_zero"].astype(bool),
        "all_zero": arrays["all_zero"].astype(bool),
    }
    strata = {}
    for name, mask in strata_masks.items():
        count = int(mask.sum())
        def masked_mean(key: str) -> float | None:
            return float(arrays[key][mask].mean()) if count else None

        strata[name] = {
            "scene_groups": count,
            "fraction": count / scene_groups,
            "mean_best_minus_candidate0": masked_mean("best_minus_candidate0"),
            "mean_weighted_reward_gain": masked_mean("weighted_reward_gain"),
            "mean_ess": masked_mean("ess"),
            "mean_group_weight": masked_mean("mean_weight"),
        }
    candidate_mean = candidate_sum / candidate_count
    candidate_var = max(candidate_sq_sum / candidate_count - candidate_mean * candidate_mean, 0.0)
    summary = {
        "contract": {
            "evidence": "exact replay reward/weight memmaps used by training",
            "not_evidence_of": "post-training checkpoint improvement",
            "candidate_sampling": "K independent Gaussian initial states; noise scale 0.5; no deterministic anchor",
            "trajectory_augmentation_std_m": 0.5,
            "reward_horizon_steps": 40,
            "weighting": "exp(sample-std group-normalized reward), beta=1, no weight-sum normalization, hdp_mean reduction",
        },
        "replay_root": str(args.replay_root.resolve()),
        "source_manifests": source_manifests,
        "rank_counts": rank_counts,
        "scene_groups": scene_groups,
        "candidate_trajectories": candidate_count,
        "group_size": int(group_size),
        "candidate_reward": {
            "mean": candidate_mean,
            "std": float(np.sqrt(candidate_var)),
            "min": candidate_min,
            "max": candidate_max,
        },
        "distributions": {
            key: _distribution(arrays[key].astype(np.float64))
            for key in (
                "best_minus_candidate0",
                "best_minus_uniform",
                "group_mean_reward",
                "best_group_reward",
                "weighted_reward_gain",
                "reward_std",
                "ess",
                "top_weight_share",
                "mean_weight",
            )
        },
        "fractions": {
            "best_minus_candidate0_gt_0.001": float((arrays["best_minus_candidate0"] > 0.001).mean()),
            "best_minus_candidate0_gt_0.01": float((arrays["best_minus_candidate0"] > 0.01).mean()),
            "best_minus_candidate0_gt_0.05": float((arrays["best_minus_candidate0"] > 0.05).mean()),
            # Replay has no zero-noise deterministic member. These thresholds
            # therefore describe the stochastic group mean and are candidate
            # inputs to a PlannerRFT-style hard-pool ablation, not a claim that
            # they reproduce nuPlan's deterministic Lt90 scene selection.
            "group_mean_reward_lt_0.70": float(
                (arrays["group_mean_reward"] < 0.70).mean()
            ),
            "group_mean_reward_lt_0.80": float(
                (arrays["group_mean_reward"] < 0.80).mean()
            ),
            "group_mean_reward_lt_0.90": float(
                (arrays["group_mean_reward"] < 0.90).mean()
            ),
            "group_mean_reward_lt_0.95": float(
                (arrays["group_mean_reward"] < 0.95).mean()
            ),
            "weighted_gain_positive": float((arrays["weighted_reward_gain"] > 1e-8).mean()),
            "weighted_gain_gt_0.001": float((arrays["weighted_reward_gain"] > 0.001).mean()),
            "weighted_gain_gt_0.01": float((arrays["weighted_reward_gain"] > 0.01).mean()),
            "any_zero_reward": float(arrays["any_zero"].mean()),
            "all_zero_reward": float(arrays["all_zero"].mean()),
            "reward_tied": float(arrays["tied"].mean()),
            "zero_weight_group": float(arrays["zero_weight_group"].mean()),
            "active_preference_group": float(
                1.0 - arrays["zero_weight_group"].mean()
            ),
            "candidate0_is_best": float(arrays["candidate0_is_best"].mean()),
            "ess_lt_3": float((arrays["ess"] < 3.0).mean()),
            "ess_lt_5": float((arrays["ess"] < 5.0).mean()),
            "ess_lt_8": float((arrays["ess"] < 8.0).mean()),
        },
        "reward_zero_strata": strata,
        "correlations": {
            "reward_std_vs_weighted_gain": float(
                np.corrcoef(arrays["reward_std"], arrays["weighted_reward_gain"])[0, 1]
            ),
            "reward_std_vs_ess": float(np.corrcoef(arrays["reward_std"], arrays["ess"])[0, 1]),
            "reward_std_vs_mean_group_weight": float(
                np.corrcoef(arrays["reward_std"], arrays["mean_weight"])[0, 1]
            ),
        },
        "top_reward_candidate_index_counts": top_index_counts.tolist(),
        "top_reward_candidate_index_fractions": (top_index_counts / scene_groups).tolist(),
    }
    if scene_groups != sum(row["scene_count"] for row in rank_counts):
        raise RuntimeError("scene count mismatch after concatenation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full_replay_signal.json").write_text(json.dumps(summary, indent=2) + "\n")
    _render(summary, arrays, args.output_dir, args.title)
    print(json.dumps({"scene_groups": scene_groups, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
