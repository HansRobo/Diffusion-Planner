"""CPU proof-of-concept audit of the 2026-07-23 RL upgrades on real NPZ scenes.

Runs without any model or GPU. Three questions, answered on logged data:

1. Gate calibration: does the first-waypoint gate (with/without the 5 cm tangent
   floor) falsely reject logged expert futures, especially at low speed?
2. Augmentation calibration: are the ramped offsets/stretch velocity-safe in the
   per-step action-normalization units, and do augmented candidates pass the gate?
3. Reward discrimination: on groups of one expert plus augmented variants, do the
   three training objectives (native weighted_sum, native gated_product, ported
   hdp_pdm) produce valid preference signal (group std > eps), sane weight
   concentration (ESS), and expert-consistent rankings?

Usage:
  python util_scripts/audit_rl_thresholds_on_real_data.py \
      --manifest /mnt/storage_rdma/.../path_list_valid_sft_balanced.json \
      --num-scenes 512 --out artifacts/rl_threshold_audit/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusion_planner.hdp_rl_utils import (
    augment_rollout_candidates,
    compute_hdp_reward,
    compute_reward_weights,
    first_waypoint_candidate_gate,
)
from diffusion_planner.pdm_reward_port import compute_pdm_port_reward

SCENE_KEYS = (
    "ego_current_state",
    "ego_shape",
    "ego_agent_future",
    "neighbor_agents_past",
    "neighbor_agents_future",
    "lanes",
    "route_lanes",
    "line_strings",
    "static_objects",
    "turn_indicators",
)


def _load_scene(path: str) -> dict[str, torch.Tensor] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            data = {}
            for key in SCENE_KEYS:
                if key not in archive:
                    return None
                data[key] = torch.from_numpy(np.ascontiguousarray(archive[key])).float()
            return data
    except Exception:
        return None


def _expert_world(future: torch.Tensor) -> torch.Tensor:
    """Logged expert future (T, 3) -> (T, 4) in the current-ego frame."""
    return torch.cat([future[:, :2], torch.cos(future[:, 2:3]), torch.sin(future[:, 2:3])], dim=-1)


def _neighbors_world(neighbor_future: torch.Tensor) -> torch.Tensor:
    xy = neighbor_future[..., :2]
    heading = neighbor_future[..., 2:3]
    out = torch.cat([xy, torch.cos(heading), torch.sin(heading)], dim=-1)
    padding = xy.abs().sum(dim=-1, keepdim=True) <= 1e-6
    return out.masked_fill(padding, 0.0)


def _gate_args(tangent_floor: float) -> SimpleNamespace:
    return SimpleNamespace(
        rl_first_waypoint_gate=True,
        rl_first_waypoint_gate_tangent_min_step_m=tangent_floor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-scenes", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", default="artifacts/rl_threshold_audit/report.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    paths = json.loads(Path(args.manifest).read_text())
    rng.shuffle(paths)

    scenes = []
    for path in paths:
        scene = _load_scene(path)
        if scene is not None and scene["ego_agent_future"].shape[0] >= 80:
            scenes.append(scene)
        if len(scenes) >= args.num_scenes:
            break
    if len(scenes) < args.num_scenes // 2:
        raise RuntimeError(f"loaded only {len(scenes)} usable scenes")

    n = args.group_size
    generator = torch.Generator().manual_seed(args.seed)
    aug_args = SimpleNamespace(
        rl_candidate_aug_prob=1.0,
        rl_candidate_aug_std=0.5,
        rl_candidate_aug_eta_scheme="gaussian",
        rl_candidate_aug_stretch=0.25,
        rl_candidate_aug_ramp_steps=20,
        rl_candidate_aug_keep=1,
        rl_candidate_aug_speed_min_mps=2.0,
    )
    native_args = SimpleNamespace(
        rl_reward_w_risk=1.0,
        rl_reward_w_follow=3.0,
        rl_reward_w_lane=2.5,
        rl_reward_w_progress=3.0,
    )
    gated_args = SimpleNamespace(**vars(native_args), rl_reward_aggregation="gated_product")
    pdm_args = SimpleNamespace(rl_pdm_red_light_gate=True, rl_reward_horizon_steps=40)

    report: dict = {"num_scenes": len(scenes), "group_size": n, "seed": args.seed}

    # ── Part 1: gate on logged experts ────────────────────────────────────
    experts = torch.stack([_expert_world(s["ego_agent_future"][:80]) for s in scenes])
    speeds = torch.stack([s["ego_current_state"][4:6].norm() for s in scenes])
    low_speed = speeds < 1.0
    for name, floor in (("floored_5cm", 0.05), ("unfloored", 1e-9)):
        keep, _ = first_waypoint_candidate_gate(experts, speeds, len(scenes), 1, _gate_args(floor))
        rejected = ~keep
        report[f"gate_{name}_expert_reject_fraction"] = rejected.float().mean().item()
        report[f"gate_{name}_expert_reject_fraction_low_speed"] = (
            rejected[low_speed].float().mean().item() if low_speed.any() else 0.0
        )
    report["low_speed_scene_fraction"] = low_speed.float().mean().item()

    # ── Part 2: augmentation calibration in action-normalization units ────
    groups = experts.repeat_interleave(n, dim=0)
    augmented, aug_metrics = augment_rollout_candidates(
        groups.clone(), speeds, len(scenes), n, aug_args, generator=generator
    )
    delta = augmented[..., :2] - groups[..., :2]
    increments = torch.diff(torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1), dim=1)
    # Per-step action normalization: ego_velocity std (0.5 m dx, 0.25 m dy).
    sigma = increments.abs() / torch.tensor([0.5, 0.25])
    changed = delta.abs().sum(dim=(1, 2)) > 1e-6
    report["aug_scene_fraction"] = aug_metrics["reward_candidate_aug_scene_fraction"].item()
    report["aug_changed_candidate_fraction"] = changed.float().mean().item()
    report["aug_max_step_increment_m_p99"] = (
        increments.norm(dim=-1).amax(dim=1)[changed].quantile(0.99).item() if changed.any() else 0.0
    )
    report["aug_max_step_increment_sigma_p99"] = (
        sigma.amax(dim=(1, 2))[changed].quantile(0.99).item() if changed.any() else 0.0
    )
    gate_keep, _ = first_waypoint_candidate_gate(
        augmented, speeds, len(scenes), n, _gate_args(0.05)
    )
    report["aug_gate_pass_fraction"] = gate_keep.float().mean().item()

    # ── Part 3: reward discrimination on expert+augmented groups ──────────
    rewards: dict[str, list[torch.Tensor]] = {"native": [], "gated": [], "pdm": []}
    failures = {"native": 0, "gated": 0, "pdm": 0}
    for index, scene in enumerate(scenes):
        candidates = augmented[index * n : (index + 1) * n]
        batch = {key: value.unsqueeze(0) for key, value in scene.items()}
        neighbors = _neighbors_world(scene["neighbor_agents_future"][:, :80]).unsqueeze(0)
        for name, fn, fn_args in (
            ("native", compute_hdp_reward, native_args),
            ("gated", compute_hdp_reward, gated_args),
            ("pdm", compute_pdm_port_reward, pdm_args),
        ):
            try:
                value, _ = fn(candidates, batch, neighbors, 1, n, fn_args)
                rewards[name].append(value)
            except Exception:
                failures[name] += 1
                rewards[name].append(torch.full((n,), float("nan")))

    for name, values in rewards.items():
        flat = torch.stack(values)  # [S, n]
        finite = torch.isfinite(flat).all(dim=1)
        report[f"reward_{name}_failures"] = failures[name]
        report[f"reward_{name}_finite_scene_fraction"] = finite.float().mean().item()
        usable = flat[finite]
        if usable.numel() == 0:
            continue
        std = usable.std(dim=1)
        report[f"reward_{name}_valid_group_fraction"] = (std > 1e-6).float().mean().item()
        report[f"reward_{name}_expert_wins_fraction"] = (
            (usable[:, 0] >= usable.max(dim=1).values - 1e-9).float().mean().item()
        )
        weights, valid = compute_reward_weights(
            usable.reshape(-1), usable.shape[0], n, "group", 1.0, 1e-6
        )
        w = weights.view(-1, n)
        ess = (w.sum(dim=1).square() / w.square().sum(dim=1).clamp_min(1e-12)) / n
        active = valid.view(-1, n).any(dim=1)
        report[f"reward_{name}_weight_ess_mean"] = (
            ess[active].mean().item() if active.any() else 0.0
        )
    # Rank agreement between objectives on mutually finite groups.
    native = torch.stack(rewards["native"])
    pdm = torch.stack(rewards["pdm"])
    both = torch.isfinite(native).all(dim=1) & torch.isfinite(pdm).all(dim=1)
    if both.any():
        a = native[both].argsort(dim=1).argsort(dim=1).float()
        b = pdm[both].argsort(dim=1).argsort(dim=1).float()
        num = ((a - a.mean(1, keepdim=True)) * (b - b.mean(1, keepdim=True))).sum(1)
        den = (a.std(1) * b.std(1) * (n - 1)).clamp_min(1e-9)
        report["reward_native_vs_pdm_spearman_mean"] = (num / den).mean().item()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not math.isfinite(report.get("reward_pdm_valid_group_fraction", float("nan"))):
        raise SystemExit("pdm reward produced no usable groups")


if __name__ == "__main__":
    main()
