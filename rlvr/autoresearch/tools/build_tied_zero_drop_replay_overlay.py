#!/usr/bin/env python3
"""Build a small HDP-faithful weight overlay for an immutable replay cache.

Older Cycle-2 mining retained unit weights when every candidate reward in a
group was zero.  HDP discards groups with no within-group preference.  This
tool keeps the multi-terabyte trajectories/encodings immutable and linked,
while materializing only corrected ``weights.npy`` files and a matching replay
contract.  It fails if a non-zero tied group exists, because that would require
an explicit semantic decision rather than the proven tied-zero equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def build_overlay(source_replay: Path, output_replay: Path) -> dict[str, Any]:
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

    rank_dirs = sorted(path for path in source_replay.glob("rank_*"))
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {source_replay}")

    temporary = output_replay.with_name(output_replay.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    rank_rows: list[dict[str, Any]] = []
    total_groups = 0
    total_dropped = 0
    try:
        for source_rank in rank_dirs:
            manifest = json.loads((source_rank / "manifest.json").read_text())
            count = int(manifest["scene_count"])
            group_size = int(manifest["group_size"])
            rewards = np.load(source_rank / "rewards.npy", mmap_mode="r")
            source_weights = np.load(source_rank / "weights.npy", mmap_mode="r")
            if rewards.shape != (count, group_size):
                raise ValueError(f"reward shape mismatch in {source_rank}")
            if source_weights.shape != rewards.shape:
                raise ValueError(f"weight shape mismatch in {source_rank}")
            if not np.isfinite(rewards).all() or not np.isfinite(source_weights).all():
                raise ValueError(f"non-finite reward/weight in {source_rank}")

            all_zero = (np.asarray(rewards) <= 1e-8).all(axis=1)
            tied = np.ptp(np.asarray(rewards, dtype=np.float32), axis=1) <= 1e-6
            if not np.array_equal(tied, all_zero):
                nonzero_tied = int(np.logical_and(tied, ~all_zero).sum())
                raise RuntimeError(
                    f"{source_rank.name}: found {nonzero_tied} tied non-zero groups; "
                    "refusing to infer their semantics"
                )

            output_rank = temporary / source_rank.name
            output_rank.mkdir()
            for name in manifest["arrays"].values():
                if name == "weights.npy":
                    continue
                os.symlink((source_rank / name).resolve(), output_rank / name)
            paths_name = str(manifest.get("paths", "scene_paths.jsonl"))
            if not (output_rank / paths_name).exists():
                os.symlink((source_rank / paths_name).resolve(), output_rank / paths_name)
            expected_paths = source_rank / "expected_scene_paths.jsonl"
            if expected_paths.exists():
                os.symlink(expected_paths.resolve(), output_rank / expected_paths.name)
            shutil.copy2(source_rank / "manifest.json", output_rank / "manifest.json")

            output_weights_path = output_rank / "weights.npy"
            output_weights = np.lib.format.open_memmap(
                output_weights_path,
                mode="w+",
                dtype=source_weights.dtype,
                shape=source_weights.shape,
            )
            output_weights[:] = source_weights
            output_weights[all_zero] = 0
            output_weights.flush()
            del output_weights
            verified = np.load(output_weights_path, mmap_mode="r")
            if not np.array_equal(verified[~all_zero], source_weights[~all_zero]):
                raise RuntimeError(f"{source_rank.name}: active weights changed")
            if np.any(verified[all_zero] != 0):
                raise RuntimeError(f"{source_rank.name}: tied-zero weights remain nonzero")

            dropped = int(all_zero.sum())
            rank_rows.append(
                {
                    "rank": int(manifest["rank"]),
                    "scene_count": count,
                    "dropped_tied_zero_groups": dropped,
                    "dropped_fraction": dropped / count,
                    "source_weights_sha256": _sha256(source_rank / "weights.npy"),
                    "overlay_weights_sha256": _sha256(output_weights_path),
                }
            )
            total_groups += count
            total_dropped += dropped

        os.replace(temporary, output_replay)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    source_config = json.loads(source_config_path.read_text())
    source_config["awr"]["drop_all_zero_groups"] = True
    _write_json(output_run / "effective_config.json", source_config)
    source_provenance = json.loads(source_provenance_path.read_text())
    source_provenance.update(
        {
            "replay_weight_overlay": "drop_tied_zero_groups_v1",
            "replay_weight_overlay_source": str(source_replay),
        }
    )
    _write_json(output_run / "provenance.json", source_provenance)
    payload = {
        "contract": "HDP_drop_groups_without_within_group_preference",
        "source_replay": str(source_replay),
        "output_replay": str(output_replay),
        "source_effective_config_sha256": _sha256(source_config_path),
        "scene_groups": total_groups,
        "dropped_tied_zero_groups": total_dropped,
        "dropped_fraction": total_dropped / total_groups,
        "active_group_fraction": 1.0 - total_dropped / total_groups,
        "non_dropped_weights_bit_exact": True,
        "bulk_arrays": "absolute read-only symlinks to immutable source",
        "ranks": rank_rows,
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
    args = parser.parse_args()
    payload = build_overlay(args.source_replay, args.output_replay)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
