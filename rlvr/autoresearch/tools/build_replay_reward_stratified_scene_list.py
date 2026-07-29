#!/usr/bin/env python3
"""Build a deterministic train-scene sample stratified by cached Rdet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

BIN_LABELS = (
    "zero",
    "zero_to_0p5",
    "0p5_to_0p8",
    "0p8_to_0p9",
    "0p9_to_0p95",
    "0p95_plus",
)


def _bin_index(reward: float) -> int:
    # Candidate rewards are non-negative. Keep exact terminal zeros separate,
    # then use left-closed score intervals for the remaining strata.
    if abs(float(reward)) <= 1.0e-8:
        return 0
    if reward < 0.5:
        return 1
    if reward < 0.8:
        return 2
    if reward < 0.9:
        return 3
    if reward < 0.95:
        return 4
    return 5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.per_bin < 1:
        raise ValueError("per-bin must be positive")
    replay = args.replay_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_output = output.with_suffix(".manifest.json")
    if output.exists() or manifest_output.exists():
        raise FileExistsError("refusing to overwrite an existing scene sample")

    counts = [0] * len(BIN_LABELS)
    samples: list[list[dict[str, object]]] = [[] for _ in BIN_LABELS]
    generators = [random.Random(args.seed + 1009 * index) for index in range(6)]
    source_manifests = []
    for rank in range(8):
        rank_dir = replay / f"rank_{rank:04d}"
        manifest_path = rank_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["scene_count"]) <= 0:
            raise RuntimeError(f"rank {rank} has an invalid scene count")
        rewards = np.load(
            rank_dir / manifest["arrays"]["rewards"], mmap_mode="r"
        )
        paths_path = rank_dir / manifest["paths"]
        source_manifests.append(
            {
                "rank": rank,
                "scene_count": int(manifest["scene_count"]),
                "manifest_sha256": _sha256_file(manifest_path),
                "paths_size_bytes": int(paths_path.stat().st_size),
                "expected_paths_sha256": manifest.get(
                    "decoder_context_attachment", {}
                ).get("expected_paths_sha256"),
            }
        )
        line_count = 0
        with paths_path.open() as stream:
            for local_index, line in enumerate(stream):
                line_count += 1
                reward = float(rewards[local_index, 0])
                bin_index = _bin_index(reward)
                counts[bin_index] += 1
                row = {
                    "scene_path": json.loads(line),
                    "det_reward_5step_cache": reward,
                    "source_rank": rank,
                    "source_local_index": local_index,
                    "stratum": BIN_LABELS[bin_index],
                }
                bucket = samples[bin_index]
                if len(bucket) < args.per_bin:
                    bucket.append(row)
                    continue
                replacement = generators[bin_index].randrange(counts[bin_index])
                if replacement < args.per_bin:
                    bucket[replacement] = row
        if line_count != int(manifest["scene_count"]):
            raise RuntimeError(
                f"rank {rank} path/reward count differs: "
                f"paths={line_count}, scenes={manifest['scene_count']}"
            )

    if any(len(bucket) != args.per_bin for bucket in samples):
        raise RuntimeError(
            f"one or more reward strata are undersized: {[len(x) for x in samples]}"
        )
    rows = [row for bucket in samples for row in bucket]
    paths = [str(row["scene_path"]) for row in rows]
    if len(set(paths)) != len(paths):
        raise RuntimeError("stratified replay sample contains duplicate scene paths")
    path_hash = hashlib.sha256(
        "".join(path + "\n" for path in paths).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(paths, indent=2) + "\n")
    temporary.replace(output)
    manifest = {
        "version": 1,
        "role": "paired PlannerRFT solver-step candidate audit; not training",
        "source_replay": str(replay),
        "seed": int(args.seed),
        "per_bin": int(args.per_bin),
        "scene_count": len(paths),
        "scene_paths_sha256": path_hash,
        "bin_edges": ["-inf", 0.0, 0.5, 0.8, 0.9, 0.95, "+inf"],
        "bin_labels": list(BIN_LABELS),
        "source_bin_counts": dict(zip(BIN_LABELS, counts, strict=True)),
        "selected_rows": rows,
        "source_manifests": source_manifests,
        "neighbor_future_alignment_offset": 1,
        "x2_legacy_ego_width_m": 2.29156,
    }
    temporary_manifest = manifest_output.with_name(
        f".{manifest_output.name}.tmp"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_output)
    print(output)


if __name__ == "__main__":
    main()
