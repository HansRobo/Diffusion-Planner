#!/usr/bin/env python3
"""Reconstruct the exact missing tails of a salvaged HDP disk replay cache.

The original mining epoch shuffles the filtered corpus with ``seed + epoch``,
pads to ``world_size * scene_batch_size``, and shards by rank using a strided
slice.  A salvaged cache is safe to backfill only if every published
``scene_paths.jsonl`` is an exact prefix of that deterministic rank order.
This tool verifies that contract and writes only the missing tail lists; it
never mutates the multi-terabyte replay arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_list", type=Path, required=True)
    parser.add_argument("--replay_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--mining_epoch", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--scene_batch_size", type=int, default=192)
    return parser.parse_args()


def _sha256_json_list(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl_paths(path: Path) -> list[str]:
    values: list[str] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, str):
                raise TypeError(f"{path}:{line_number}: expected JSON string, got {type(value)!r}")
            values.append(value)
    return values


def main() -> None:
    args = _args()
    paths = json.loads(args.train_list.read_text())
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
        raise ValueError(f"invalid train path list: {args.train_list}")
    original_count = len(paths)
    order = list(paths)
    random.Random(args.seed + args.mining_epoch).shuffle(order)
    pad_multiple = args.world_size * args.scene_batch_size
    remainder = len(order) % pad_multiple
    padding_count = 0 if remainder == 0 else pad_multiple - remainder
    if padding_count:
        order.extend(order[:padding_count])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_records: list[dict[str, object]] = []
    all_missing_real: list[str] = []
    total_cached = 0
    total_cached_padding = 0
    total_missing = 0
    total_missing_padding = 0
    for rank in range(args.world_size):
        expected = order[rank::args.world_size]
        cache_paths_file = args.replay_root / f"rank_{rank:04d}" / "scene_paths.jsonl"
        cached = _read_jsonl_paths(cache_paths_file)
        if len(cached) > len(expected):
            raise RuntimeError(
                f"rank {rank}: cached count {len(cached)} exceeds expected {len(expected)}"
            )
        mismatch = next(
            (index for index, (got, want) in enumerate(zip(cached, expected)) if got != want),
            None,
        )
        if mismatch is not None:
            raise RuntimeError(
                f"rank {rank}: cache is not the expected deterministic prefix at local index {mismatch}: "
                f"cached={cached[mismatch]!r} expected={expected[mismatch]!r}"
            )

        missing = expected[len(cached) :]
        cached_padding_count = sum(
            rank + local_index * args.world_size >= original_count
            for local_index in range(len(cached))
        )
        missing_real: list[str] = []
        missing_padding: list[str] = []
        for local_index, path in enumerate(missing, start=len(cached)):
            global_index = rank + local_index * args.world_size
            if global_index < original_count:
                missing_real.append(path)
            else:
                missing_padding.append(path)

        (args.output_dir / f"rank_{rank:04d}_missing_all.json").write_text(
            json.dumps(missing, indent=2) + "\n"
        )
        (args.output_dir / f"rank_{rank:04d}_missing_real.json").write_text(
            json.dumps(missing_real, indent=2) + "\n"
        )
        rank_records.append(
            {
                "rank": rank,
                "expected_padded_count": len(expected),
                "cached_prefix_count": len(cached),
                "cached_padding_count": cached_padding_count,
                "missing_total_count": len(missing),
                "missing_real_count": len(missing_real),
                "missing_padding_count": len(missing_padding),
                "cached_prefix_sha256": _sha256_json_list(cached),
                "missing_all_file": f"rank_{rank:04d}_missing_all.json",
                "missing_real_file": f"rank_{rank:04d}_missing_real.json",
            }
        )
        all_missing_real.extend(missing_real)
        total_cached += len(cached)
        total_cached_padding += cached_padding_count
        total_missing += len(missing)
        total_missing_padding += len(missing_padding)

    if len(all_missing_real) != len(set(all_missing_real)):
        raise RuntimeError("real missing tails contain duplicate scene paths")
    if total_cached + total_missing != len(order):
        raise RuntimeError("cached + missing does not equal deterministic padded corpus")
    if total_cached_padding + total_missing_padding != padding_count:
        raise RuntimeError(
            "cached plus missing padding does not equal the expected global "
            f"padding: {total_cached_padding} + {total_missing_padding} "
            f"!= {padding_count}"
        )
    (args.output_dir / "missing_real_all_ranks.json").write_text(
        json.dumps(all_missing_real, indent=2) + "\n"
    )
    manifest = {
        "contract": "exact deterministic epoch-1 rank-prefix reconstruction; no replay array mutation",
        "train_list": str(args.train_list.resolve()),
        "train_scene_count": original_count,
        "train_list_sha256": hashlib.sha256(args.train_list.read_bytes()).hexdigest(),
        "seed": args.seed,
        "mining_epoch": args.mining_epoch,
        "shuffle_seed": args.seed + args.mining_epoch,
        "world_size": args.world_size,
        "scene_batch_size": args.scene_batch_size,
        "padded_scene_count": len(order),
        "padding_count": padding_count,
        "cached_padding_count": total_cached_padding,
        "cached_prefix_total": total_cached,
        "missing_total_with_padding": total_missing,
        "missing_real_unique_count": len(all_missing_real),
        "coverage_before_backfill": total_cached / original_count,
        "prefix_contract_verified": True,
        "ranks": rank_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
