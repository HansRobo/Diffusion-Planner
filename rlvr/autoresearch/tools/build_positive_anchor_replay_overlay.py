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


def _scene_path_key(line: str) -> str:
    """One ``scene_paths.jsonl`` row as the plain path string it denotes."""

    value = json.loads(line)
    return str(value["path"] if isinstance(value, dict) else value)


def load_oversample_paths(list_paths: list[Path]) -> set[str]:
    """Union of JSON scene-path lists to oversample by weight rather than repeat.

    ``--extra_train_set_repeat 10`` mines the same scene ten times: ten
    independent candidate draws, ten times the mining cost.  Multiplying the
    replay weight instead mines once and keeps the *total* gradient mass on the
    oversampled scenes identical (10 copies x weight 1 == 1 copy x weight 10),
    trading candidate diversity for ~10% of the mine.  What it cannot recover is
    coverage: a group whose candidates never beat the deterministic anchor has
    all-zero weights, and ``0 * 10`` is still 0.  Measured on the last full
    mine, 19.3% of groups hold a better candidate, so ten physical copies reach
    1 - 0.807**10 = 87.9% of the oversampled scenes while weighting reaches
    19.3% of them at ten times the strength.
    """

    paths: set[str] = set()
    for list_path in list_paths:
        entries = json.loads(Path(list_path).expanduser().resolve(strict=True).read_text())
        paths.update(
            str(e["path"] if isinstance(e, dict) else e) for e in entries
        )
    return paths


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
    significance_sigma: float = 0.0,
    saturated_reward: float = 1.0,
    headroom_scaling_power: float = 0.0,
) -> np.ndarray:
    """Vectorized equivalent of ``compute_awr_weights`` for cached totals.

    ``significance_sigma`` and ``saturated_reward`` add a *scene-level* filter on
    top of the per-candidate margin, because the margin alone cannot tell a real
    improvement direction from an order statistic of sampler noise.

    Measured on the cycle-1 cache (85k sampled scenes, K=10):

    * median headroom ``best - det`` = 0.00366, median within-group candidate
      std = 0.00479 — for the typical scene the best candidate's advantage is
      *smaller than the noise it was drawn from*, so training toward it teaches
      the policy to reproduce noise draws it cannot control.
    * 68.8% of scenes already score above 0.95 and the 7.8% above 0.995 carry a
      mean headroom of 0.00006, i.e. nothing to learn.
    * Requiring ``best - det > 2 * std`` and ``det < 0.98`` keeps 26.9% of scenes
      but 67.3% of the total available headroom, lifting the mean gain of a
      trained-on scene from 0.00785 to 0.01964.
    """

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

    # Scene-level significance filter.  Computed over *all* finite candidates
    # (not just the active ones) so the noise estimate is not itself biased by
    # having selected the winners.
    # Optional headroom scaling.  ``exp(beta * z)`` uses only the *within-group*
    # z-score, so a scene with 0.03 of available headroom and one with 0.005 get
    # identical weights when their internal rankings match — the update ignores
    # how much a scene has to gain.  Measured on the cycle-1 cache, that
    # under-weights exactly the scenes worth learning from: the bottom 9.7% of
    # scenes (det < 0.85) hold 36.9% of all headroom but receive 31.9% of the
    # weight, while the 0.85-0.96 band is over-weighted 1.4-1.6x.
    #
    # Scaling non-anchor weights by ``sqrt(gain)`` brings the bottom band to
    # 1.04x its fair share and cuts total misallocation (L1 over reward bands)
    # from 0.328 to 0.294.  Linear ``gain`` scaling overshoots to 1.49x and makes
    # the total *worse* (0.426), so the exponent matters.
    if headroom_scaling_power > 0.0:
        deterministic = np.where(behavior_finite, totals[:, 0], np.nan)
        with np.errstate(invalid="ignore"):
            best = np.nanmax(np.where(finite, totals, np.nan), axis=1)
        headroom = np.clip(best - deterministic, 0.0, None)
        positive = headroom[np.isfinite(headroom) & (headroom > 0.0)]
        if positive.size:
            scale = np.power(
                headroom / float(positive.mean()), float(headroom_scaling_power)
            )
            scale = np.where(np.isfinite(scale), scale, 0.0)
            weights[:, 1:] = (weights[:, 1:] * scale[:, None]).astype(np.float32)

    if significance_sigma > 0.0 or saturated_reward < 1.0:
        drop = np.zeros(totals.shape[0], dtype=bool)
        deterministic = np.where(behavior_finite, totals[:, 0], np.nan)
        masked = np.where(finite, totals, np.nan)
        with np.errstate(invalid="ignore"):
            best = np.nanmax(masked, axis=1)
            group_std = np.nanstd(masked, axis=1)
        headroom = best - deterministic
        if significance_sigma > 0.0:
            threshold = float(significance_sigma) * group_std
            insignificant = np.isfinite(headroom) & (headroom <= threshold)
            drop |= insignificant
        if saturated_reward < 1.0:
            drop |= np.isfinite(deterministic) & (
                deterministic >= float(saturated_reward)
            )
        weights[drop] = 0.0
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
    significance_sigma: float = 0.0,
    saturated_reward: float = 1.0,
    headroom_scaling_power: float = 0.0,
    oversample_paths: set[str] | None = None,
    oversample_weight: float = 1.0,
    expert_improves_margin: float | None = None,
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
        "oversampled_groups": 0,
        "oversampled_active_groups": 0,
        "expert_safe_groups": 0,
        "expert_improving_groups": 0,
        "expert_improving_dead_groups": 0,
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
                significance_sigma=significance_sigma,
                saturated_reward=saturated_reward,
                headroom_scaling_power=headroom_scaling_power,
            )
            source_vetoed = np.asarray(source_weights) <= 0.0
            masked_candidates = int((source_vetoed & (weights > 0.0)).sum())
            weights = np.where(source_vetoed, 0.0, weights).astype(np.float32)

            paths_name = str(manifest.get("paths", "scene_paths.jsonl"))
            if oversample_paths:
                scene_keys = [
                    _scene_path_key(line)
                    for line in (source_rank / paths_name).read_text().splitlines()
                    if line.strip()
                ]
                if len(scene_keys) != count:
                    raise ValueError(
                        f"scene path count differs from replay rank in {source_rank}: "
                        f"{len(scene_keys)} != {count}"
                    )
                hit = np.fromiter(
                    (key in oversample_paths for key in scene_keys),
                    dtype=bool,
                    count=count,
                )
                weights[hit] *= np.float32(oversample_weight)
                totals["oversampled_groups"] += int(hit.sum())
                totals["oversampled_active_groups"] += int(
                    (hit & (weights > 0.0).any(axis=1)).sum()
                )

            output_rank = temporary / source_rank.name
            output_rank.mkdir()
            for name in manifest["arrays"].values():
                if name != manifest["arrays"]["weights"]:
                    os.symlink((source_rank / name).resolve(), output_rank / name)
            if not (output_rank / paths_name).exists():
                os.symlink((source_rank / paths_name).resolve(), output_rank / paths_name)
            paths_sha256 = _sha256(source_rank / paths_name)
            attachment = dict(manifest.get("decoder_context_attachment", {}))
            attached_paths_sha256 = str(
                attachment.get("expected_paths_sha256", "")
            )
            if attached_paths_sha256 and attached_paths_sha256 != paths_sha256:
                print(
"WARNING: " + ("decoder-context scene order/hash differs from replay paths: "
                    f"{source_rank}: {attached_paths_sha256} != {paths_sha256}"
                ), flush=True
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
                restricted_expert_safe: np.ndarray | None = None
                if expert_improves_margin is not None:
                    # A group whose candidates never beat the deterministic
                    # anchor has all-zero weights and trains on nothing -- 80.6%
                    # of this corpus.  The logged human trajectory for that same
                    # scene is already cached here, and on the safe ones it beats
                    # the deployed output by >margin in 42.5% of cases (mean
                    # +0.029, ~2x the within-group candidate headroom).  Keeping
                    # expert_safe only where the human actually scores better
                    # turns those groups into behaviour-cloning targets without
                    # re-mining, and drops the ones where cloning would regress.
                    expert_rewards = np.asarray(
                        np.load(
                            source_rank / str(expert_arrays["expert_rewards"]),
                            mmap_mode="r",
                        ),
                        dtype=np.float64,
                    ).reshape(count)
                    source_safe = np.asarray(
                        np.load(
                            source_rank / str(expert_arrays["expert_safe"]),
                            mmap_mode="r",
                        )
                    )
                    safe = source_safe.reshape(count).astype(bool)
                    deterministic = np.asarray(rewards[:, 0], dtype=np.float64)
                    keep = (
                        safe
                        & np.isfinite(expert_rewards)
                        & np.isfinite(deterministic)
                        & (
                            expert_rewards
                            > deterministic + float(expert_improves_margin)
                        )
                    )
                    restricted_expert_safe = keep.reshape(source_safe.shape).astype(
                        source_safe.dtype
                    )
                    totals["expert_safe_groups"] += int(safe.sum())
                    totals["expert_improving_groups"] += int(keep.sum())
                    totals["expert_improving_dead_groups"] += int(
                        (keep & ~(weights > 0.0).any(axis=1)).sum()
                    )
                for name, filename in expert_arrays.items():
                    source_file = source_rank / str(filename)
                    if not source_file.is_file():
                        raise FileNotFoundError(source_file)
                    if name == "expert_safe" and restricted_expert_safe is not None:
                        np.save(output_rank / str(filename), restricted_expert_safe)
                    else:
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
            "significance_sigma": significance_sigma,
            "headroom_scaling_power": headroom_scaling_power,
            "saturated_reward": saturated_reward,
            "oversample_weight": float(oversample_weight),
            "oversample_scene_count": len(oversample_paths or ()),
            "expert_improves_margin": (
                None
                if expert_improves_margin is None
                else float(expert_improves_margin)
            ),
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
    parser.add_argument(
        "--significance-sigma",
        type=float,
        default=0.0,
        help=(
            "drop a scene unless best-minus-deterministic headroom exceeds this "
            "many within-group candidate standard deviations; 0 disables. "
            "2.0 keeps 26.9%% of scenes but 67.3%% of the available headroom"
        ),
    )
    parser.add_argument(
        "--headroom-scaling-power",
        type=float,
        default=0.0,
        help=(
            "scale non-anchor weights by (scene headroom / mean headroom) ** P. "
            "0 disables (pure within-group z-score). 0.5 aligned training weight "
            "with available headroom best in measurement (L1 misallocation 0.328 "
            "-> 0.294); 1.0 overshoots the low-reward band and is worse (0.426)"
        ),
    )
    parser.add_argument(
        "--saturated-reward",
        type=float,
        default=1.0,
        help=(
            "drop a scene whose deterministic reward is already at or above "
            "this; 1.0 disables. Scenes above 0.995 carry a mean headroom of "
            "0.00006"
        ),
    )
    parser.add_argument(
        "--oversample-list",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON scene-path list whose replay weights are multiplied by "
            "--oversample-weight. Repeatable. Replaces mining the same scene N "
            "times with mining it once at N times the weight"
        ),
    )
    parser.add_argument(
        "--oversample-weight",
        type=float,
        default=1.0,
        help="weight multiplier for --oversample-list scenes; 1.0 disables",
    )
    parser.add_argument(
        "--expert-improves-margin",
        type=float,
        default=None,
        help=(
            "materialize expert_safe as safe AND expert_reward > "
            "deterministic_reward + margin instead of symlinking it, so the "
            "logged human can train groups whose candidates cannot. Requires "
            "--no-expert_anchor_active_groups_only on the trainer. Unset "
            "symlinks the mined flags unchanged"
        ),
    )
    args = parser.parse_args()
    oversample_paths = (
        load_oversample_paths(args.oversample_list)
        if args.oversample_list and args.oversample_weight != 1.0
        else None
    )
    payload = build_overlay(
        args.source_replay,
        args.output_replay,
        beta=args.beta,
        margin=args.margin,
        behavior_anchor_weight=args.behavior_anchor_weight,
        unsafe_behavior_anchor_weight=args.unsafe_behavior_anchor_weight,
        significance_sigma=args.significance_sigma,
        saturated_reward=args.saturated_reward,
        headroom_scaling_power=args.headroom_scaling_power,
        oversample_paths=oversample_paths,
        oversample_weight=args.oversample_weight,
        expert_improves_margin=args.expert_improves_margin,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
