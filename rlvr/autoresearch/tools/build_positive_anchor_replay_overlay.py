#!/usr/bin/env python3
"""Build a lightweight positive-advantage AWR overlay for a frozen cache.

The source replay remains immutable.  Candidate trajectories, rewards, noise
scales, scene encodings and path manifests are linked read-only; only the small
``weights.npy`` arrays are materialized.  Candidate zero must be the recorded
deterministic behavior trajectory.  Candidates that do not strictly improve
its reward receive zero weight, while the behavior trajectory remains an
explicit retention anchor.  All-zero groups are dropped as in the HDP paper.

A candidate the miner refused to train on (stored weight of zero, e.g. the
Original-DP first-waypoint gate) must never resurrect as a replay target: the
reward total is blind to near-field geometry, so recomputing weights from
rewards alone would hand kinematically infeasible standstill jumps the top
positive-advantage weight.  The recomputed weights are therefore masked by
``source_weights > 0`` before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compute_positive_anchor_weights(
    rewards: np.ndarray,
    *,
    beta: float = 1.0,
    margin: float = 0.0,
    behavior_anchor_weight: float = 1.0,
    unsafe_behavior_anchor_weight: float = 1.0,
    weight_clip: float = 1e9,
    min_group_std: float = 1e-6,
    normalize: bool = False,
    drop_all_zero_groups: bool = True,
) -> np.ndarray:
    """Vectorized equivalent of ``compute_awr_weights`` for cached totals."""

    totals = np.asarray(rewards, dtype=np.float64)
    if totals.ndim != 2 or totals.shape[1] < 1:
        raise ValueError(f"rewards must have shape [B,K], got {totals.shape}")
    if behavior_anchor_weight < 0.0 or unsafe_behavior_anchor_weight < 0.0:
        raise ValueError("behavior anchor weights must be non-negative")

    finite = np.isfinite(totals)
    behavior_finite = finite[:, 0]
    better = finite & (
        totals > totals[:, :1] + max(float(margin), 1e-8)
    )
    better[:, 0] = False
    has_better = better.any(axis=1)
    active = better.copy()
    active[:, 0] = behavior_finite & (
        np.where(
            (np.abs(totals[:, 0]) <= 1e-8) & has_better,
            float(unsafe_behavior_anchor_weight),
            float(behavior_anchor_weight),
        )
        > 0.0
    )

    active_count = active.sum(axis=1)
    active_values = np.where(active, totals, 0.0)
    means = np.divide(
        active_values.sum(axis=1),
        active_count,
        out=np.full(totals.shape[0], np.nan, dtype=np.float64),
        where=active_count > 0,
    )
    centered = np.where(active, totals - means[:, None], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        active_count - 1,
        out=np.zeros(totals.shape[0], dtype=np.float64),
        where=active_count > 1,
    )
    std = np.sqrt(variance)
    z = (totals - means[:, None]) / (
        std[:, None] + max(float(min_group_std), 1e-8)
    )
    log_weights = np.clip(
        float(beta) * z,
        -30.0,
        math.log(max(float(weight_clip), 1.0)),
    )
    weights = np.where(active, np.exp(log_weights), 0.0).astype(np.float32)
    single = active_count == 1
    if single.any():
        weights[single] = active[single].astype(np.float32)

    selected_anchor = np.where(
        (np.abs(totals[:, 0]) <= 1e-8) & has_better,
        float(unsafe_behavior_anchor_weight),
        float(behavior_anchor_weight),
    )
    weights[:, 0] = np.where(active[:, 0], selected_anchor, 0.0)

    if normalize:
        positive = weights > 0.0
        positive_count = positive.sum(axis=1)
        positive_sum = weights.sum(axis=1)
        positive_mean = np.divide(
            positive_sum,
            positive_count,
            out=np.ones_like(positive_sum),
            where=positive_count > 0,
        )
        weights = np.divide(
            weights,
            positive_mean[:, None],
            out=np.zeros_like(weights),
            where=positive,
        )

    if drop_all_zero_groups:
        any_finite = finite.any(axis=1)
        all_zero = any_finite & np.all(
            np.where(finite, np.abs(totals) <= 1e-8, True), axis=1
        )
        weights[all_zero] = 0.0
    return weights


def build_overlay(
    source_replay: Path,
    output_replay: Path,
    *,
    beta: float = 1.0,
    margin: float = 0.0,
    behavior_anchor_weight: float = 1.0,
    unsafe_behavior_anchor_weight: float = 1.0,
    weight_clip: float = 1e9,
    min_group_std: float = 1e-6,
    normalize: bool = False,
    drop_all_zero_groups: bool = True,
) -> dict[str, Any]:
    source_replay = source_replay.expanduser().resolve(strict=True)
    output_replay = output_replay.expanduser().resolve()
    source_run = source_replay.parent
    output_run = output_replay.parent
    if output_replay.exists():
        raise FileExistsError(output_replay)
    source_config_path = source_run / "effective_config.json"
    source_provenance_path = source_run / "provenance.json"
    if not source_config_path.is_file() or not source_provenance_path.is_file():
        raise FileNotFoundError("source replay lacks effective_config/provenance")

    source_config = json.loads(source_config_path.read_text())
    if not bool(source_config.get("awr", {}).get("deterministic_first", False)):
        raise RuntimeError("positive-anchor overlay requires deterministic_first=true")
    rank_dirs = sorted(source_replay.glob("rank_*"))
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {source_replay}")

    temporary = output_replay.with_name(output_replay.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    expert_sidecars: list[bool] = []
    totals = {
        "groups": 0,
        "groups_with_better_candidate": 0,
        "active_targets": 0,
        "dropped_all_zero_groups": 0,
        "source_zero_masked_candidates": 0,
    }
    try:
        for source_rank in rank_dirs:
            manifest = json.loads((source_rank / "manifest.json").read_text())
            count = int(manifest["scene_count"])
            group_size = int(manifest["group_size"])
            rewards = np.load(source_rank / manifest["arrays"]["rewards"], mmap_mode="r")
            if rewards.shape != (count, group_size):
                raise ValueError(f"reward shape mismatch in {source_rank}")
            if not np.isfinite(rewards).all():
                raise ValueError(f"non-finite rewards in {source_rank}")
            source_weights = np.load(
                source_rank / manifest["arrays"]["weights"], mmap_mode="r"
            )
            if source_weights.shape != (count, group_size):
                raise ValueError(f"source weight shape mismatch in {source_rank}")
            weights = compute_positive_anchor_weights(
                rewards,
                beta=beta,
                margin=margin,
                behavior_anchor_weight=behavior_anchor_weight,
                unsafe_behavior_anchor_weight=unsafe_behavior_anchor_weight,
                weight_clip=weight_clip,
                min_group_std=min_group_std,
                normalize=normalize,
                drop_all_zero_groups=drop_all_zero_groups,
            )
            source_vetoed = np.asarray(source_weights) <= 0.0
            masked_candidates = int((source_vetoed & (weights > 0.0)).sum())
            weights = np.where(source_vetoed, 0.0, weights).astype(np.float32)

            output_rank = temporary / source_rank.name
            output_rank.mkdir()
            for name in manifest["arrays"].values():
                if name != manifest["arrays"]["weights"]:
                    os.symlink((source_rank / name).resolve(), output_rank / name)
            paths_name = str(manifest.get("paths", "scene_paths.jsonl"))
            if not (output_rank / paths_name).exists():
                os.symlink((source_rank / paths_name).resolve(), output_rank / paths_name)
            paths_sha256 = _sha256(source_rank / paths_name)
            attachment = dict(manifest.get("decoder_context_attachment", {}))
            attached_paths_sha256 = str(
                attachment.get("expected_paths_sha256", "")
            )
            if attached_paths_sha256 and attached_paths_sha256 != paths_sha256:
                raise RuntimeError(
                    "decoder-context scene order/hash differs from replay paths: "
                    f"{source_rank}: {attached_paths_sha256} != {paths_sha256}"
                )
            # Refresh caches may reuse the immutable decoder context through
            # ``shared_decoder_context_paths``.  Their writer therefore has no
            # local attachment block even though the ordered scene-path file is
            # still the identity contract for that context.  Materialize the
            # verified hash in the derived overlay so compressed-context replay
            # can compare the two orders fail-closed instead of receiving an
            # empty expected hash.
            attachment.update(
                {
                    "candidate_arrays_unchanged": True,
                    "expected_paths_sha256": paths_sha256,
                }
            )
            expected_name = manifest.get("expected_paths")
            if expected_name and (source_rank / expected_name).exists():
                os.symlink((source_rank / expected_name).resolve(), output_rank / expected_name)
            expert_manifest_path = source_rank / "expert_anchor_manifest.json"
            has_expert_sidecar = expert_manifest_path.is_file()
            expert_sidecars.append(has_expert_sidecar)
            if has_expert_sidecar:
                expert_manifest = json.loads(expert_manifest_path.read_text())
                if int(expert_manifest.get("scene_count", -1)) != count:
                    raise RuntimeError(
                        "expert sidecar count differs from replay rank: "
                        f"{source_rank}: "
                        f"{expert_manifest.get('scene_count')} != {count}"
                    )
                expert_arrays = expert_manifest.get("arrays", {})
                expected_expert_arrays = {
                    "expert_trajectories",
                    "expert_rewards",
                    "expert_safe",
                }
                if set(expert_arrays) != expected_expert_arrays:
                    raise RuntimeError(
                        f"incomplete expert sidecar in {source_rank}: "
                        f"{sorted(expert_arrays)}"
                    )
                for filename in expert_arrays.values():
                    source_file = source_rank / str(filename)
                    if not source_file.is_file():
                        raise FileNotFoundError(source_file)
                    os.symlink(source_file.resolve(), output_rank / str(filename))
                os.symlink(
                    expert_manifest_path.resolve(),
                    output_rank / expert_manifest_path.name,
                )
            output_manifest = dict(manifest)
            output_manifest["decoder_context_attachment"] = attachment
            _write_json(output_rank / "manifest.json", output_manifest)
            output_weights = output_rank / manifest["arrays"]["weights"]
            np.save(output_weights, weights)

            better = rewards[:, 1:] > rewards[:, :1] + max(float(margin), 1e-8)
            all_zero = np.all(np.abs(rewards) <= 1e-8, axis=1)
            row = {
                "rank": int(manifest["rank"]),
                "scene_count": count,
                "groups_with_better_candidate": int(better.any(axis=1).sum()),
                "active_targets": int((weights > 0.0).sum()),
                "dropped_all_zero_groups": int(all_zero.sum()) if drop_all_zero_groups else 0,
                "source_zero_masked_candidates": masked_candidates,
                "expert_anchor_sidecar": has_expert_sidecar,
                "weights_sha256": _sha256(output_weights),
            }
            rows.append(row)
            totals["groups"] += count
            totals["groups_with_better_candidate"] += row["groups_with_better_candidate"]
            totals["active_targets"] += row["active_targets"]
            totals["dropped_all_zero_groups"] += row["dropped_all_zero_groups"]
            totals["source_zero_masked_candidates"] += masked_candidates
        if any(expert_sidecars) and not all(expert_sidecars):
            raise RuntimeError(
                "expert sidecar is present on only a subset of replay ranks"
            )
        os.replace(temporary, output_replay)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    effective = json.loads(source_config_path.read_text())
    # Cycle 1 may have been mined by a process started immediately before the
    # explicit sampler-ablation fields were introduced.  Missing keys mean
    # exactly the behavior that process implemented; materialize those
    # defaults in the derived overlay so strict replay validation remains
    # auditable rather than weakening the contract checker.
    effective["awr"].setdefault("plannerrft_reference_mode", "zero_noise")
    effective["awr"].setdefault("plannerrft_reference_noise_scale", 0.5)
    effective["awr"].setdefault(
        "plannerrft_longitudinal_mode", "candidate_stretch"
    )
    effective["awr"].update(
        {
            "beta": float(beta),
            "positive_advantage_only": True,
            "positive_advantage_margin": float(margin),
            "behavior_anchor_weight": float(behavior_anchor_weight),
            "unsafe_behavior_anchor_weight": float(unsafe_behavior_anchor_weight),
            "weight_clip": float(weight_clip),
            "normalize_weights": bool(normalize),
            "drop_all_zero_groups": bool(drop_all_zero_groups),
        }
    )
    _write_json(output_run / "effective_config.json", effective)
    provenance = json.loads(source_provenance_path.read_text())
    provenance.update(
        {
            "replay_weight_overlay": "positive_behavior_anchor_v1",
            "replay_weight_overlay_source": str(source_replay),
        }
    )
    _write_json(output_run / "provenance.json", provenance)
    group_count = totals["groups"]
    payload = {
        "contract": "positive_advantage_with_deterministic_behavior_anchor_v1",
        "source_replay": str(source_replay),
        "output_replay": str(output_replay),
        "parameters": {
            "beta": float(beta),
            "margin": float(margin),
            "behavior_anchor_weight": float(behavior_anchor_weight),
            "unsafe_behavior_anchor_weight": float(unsafe_behavior_anchor_weight),
            "weight_clip": float(weight_clip),
            "min_group_std": float(min_group_std),
            "normalize": bool(normalize),
            "drop_all_zero_groups": bool(drop_all_zero_groups),
            "respect_source_zero_weights": True,
        },
        **totals,
        "groups_with_better_candidate_fraction": (
            totals["groups_with_better_candidate"] / group_count
        ),
        "mean_active_targets": totals["active_targets"] / group_count,
        "bulk_arrays": "absolute read-only symlinks to immutable source",
        "expert_anchor_sidecar": bool(expert_sidecars and all(expert_sidecars)),
        "ranks": rows,
    }
    _write_json(output_run / "overlay_manifest.json", payload)

    for path in output_replay.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    for path in sorted(output_replay.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    output_replay.chmod(output_replay.stat().st_mode & ~0o222)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--behavior-anchor-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-behavior-anchor-weight", type=float, default=1.0)
    args = parser.parse_args()
    payload = build_overlay(
        args.source_replay,
        args.output_replay,
        beta=args.beta,
        margin=args.margin,
        behavior_anchor_weight=args.behavior_anchor_weight,
        unsafe_behavior_anchor_weight=args.unsafe_behavior_anchor_weight,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
