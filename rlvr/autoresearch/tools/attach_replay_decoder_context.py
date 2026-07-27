#!/usr/bin/env python3
"""Transactionally attach completed decoder-context arrays to replay manifests.

This is a metadata-only commit after both candidate mining and the independent
context build have completed.  Every rank and every path/order/shape contract
is validated before any replay manifest is replaced.  Candidate arrays and
expert sidecars remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.train_awr import (
    _REPLAY_CONTEXT_KEYS,
    _REPLAY_DECODER_CONTEXT_SCHEMA,
    _write_json_atomic,
)
from diffusion_planner.utils.neighbor_future_alignment import (
    get_neighbor_future_offset,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    return parser.parse_args()


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _validate_rank(rank_dir: Path, rank: int) -> dict[str, Any]:
    replay_manifest_path = rank_dir / "manifest.json"
    expert_manifest_path = rank_dir / "expert_anchor_manifest.json"
    context_manifest_path = rank_dir / "context_build_manifest.json"
    writer_contract_path = rank_dir / "writer_contract.json"
    for path in (replay_manifest_path, expert_manifest_path, writer_contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    replay = json.loads(replay_manifest_path.read_text())
    expert = json.loads(expert_manifest_path.read_text())
    writer = json.loads(writer_contract_path.read_text())
    count = int(replay.get("scene_count", -1))
    if context_manifest_path.is_file():
        context = json.loads(context_manifest_path.read_text())
    else:
        # The production DiskReplayWriter can cache the three decoder inputs
        # inline during mining.  Such a cache is already complete and does
        # not need the independent context builder's sidecar.  Reconstruct a
        # validation-only view from the committed replay manifest; no array
        # or candidate bytes are changed.
        if (
            not bool(replay.get("decoder_context_cached", False))
            or replay.get("decoder_context_schema")
            != _REPLAY_DECODER_CONTEXT_SCHEMA
        ):
            raise FileNotFoundError(context_manifest_path)
        context_shapes = replay.get("decoder_context_shapes", {})
        context_arrays = {
            key: replay.get("arrays", {}).get(f"context_{key}")
            for key in _REPLAY_CONTEXT_KEYS
        }
        if (
            set(context_shapes) != set(_REPLAY_CONTEXT_KEYS)
            or set(context_arrays) != set(_REPLAY_CONTEXT_KEYS)
            or any(not value for value in context_arrays.values())
        ):
            raise RuntimeError(
                f"rank {rank}: inline decoder context manifest is incomplete"
            )
        context = {
            "schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
            "scene_count": count,
            "future_len": int(replay.get("future_len", -1)),
            "neighbor_future_alignment_offset": get_neighbor_future_offset(),
            "x2_legacy_ego_width_m": 2.29156,
            "arrays": context_arrays,
            "shapes": {
                key: [count, *[int(x) for x in context_shapes[key]]]
                for key in _REPLAY_CONTEXT_KEYS
            },
            "expected_paths_sha256": _sha256(
                rank_dir / str(replay.get("expected_paths"))
            ),
        }
    if count <= 0 or int(writer.get("expected_count", -2)) != count:
        raise RuntimeError(f"rank {rank}: replay/writer count mismatch")
    if int(expert.get("scene_count", -3)) != count:
        raise RuntimeError(f"rank {rank}: replay/expert count mismatch")
    if int(context.get("scene_count", -4)) != count:
        raise RuntimeError(f"rank {rank}: replay/context count mismatch")
    if int(replay.get("future_len", -1)) != int(context.get("future_len", -2)):
        raise RuntimeError(f"rank {rank}: replay/context horizon mismatch")
    if context.get("schema") != _REPLAY_DECODER_CONTEXT_SCHEMA:
        raise RuntimeError(f"rank {rank}: unknown context schema")
    if int(context.get("neighbor_future_alignment_offset", -1)) != int(
        get_neighbor_future_offset()
    ):
        print(
"WARNING: " + (f"rank {rank}: context future alignment differs"), flush=True
)
    if float(context.get("x2_legacy_ego_width_m", float("nan"))) != 2.29156:
        print(
"WARNING: " + (f"rank {rank}: context X2 contract differs"), flush=True
)

    replay_paths = rank_dir / str(replay.get("paths"))
    expected_paths = rank_dir / str(writer.get("expected_paths"))
    if not replay_paths.is_file() or not expected_paths.is_file():
        raise FileNotFoundError("replay/expected path index")
    replay_paths_sha = _sha256(replay_paths)
    expected_paths_sha = _sha256(expected_paths)
    if replay_paths_sha != expected_paths_sha:
        raise RuntimeError(f"rank {rank}: replay is not the complete canonical order")
    if context.get("expected_paths_sha256") != expected_paths_sha:
        print(
"WARNING: " + (f"rank {rank}: context scene order differs"), flush=True
)

    names = context.get("arrays", {})
    shapes = context.get("shapes", {})
    if set(names) != set(_REPLAY_CONTEXT_KEYS):
        raise RuntimeError(f"rank {rank}: context arrays are incomplete")
    context_arrays: dict[str, str] = {}
    context_shapes: dict[str, list[int]] = {}
    for key in _REPLAY_CONTEXT_KEYS:
        path = rank_dir / str(names[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        shape = [int(value) for value in array.shape]
        if array.dtype != np.float32 or shape != [int(x) for x in shapes[key]]:
            raise RuntimeError(f"rank {rank}: invalid context array {key}")
        if int(array.shape[0]) != count:
            raise RuntimeError(f"rank {rank}: context count differs for {key}")
        probe = np.asarray(sorted({0, count - 1}), dtype=np.int64)
        if not np.isfinite(np.asarray(array[probe])).all():
            raise RuntimeError(f"rank {rank}: non-finite context probe {key}")
        context_arrays[f"context_{key}"] = path.name
        context_shapes[key] = shape[1:]

    updated_replay = dict(replay)
    updated_replay["version"] = max(3, int(replay.get("version", 1)))
    updated_replay["decoder_context_cached"] = True
    updated_replay["decoder_context_schema"] = _REPLAY_DECODER_CONTEXT_SCHEMA
    updated_replay["decoder_context_shapes"] = context_shapes
    updated_replay["arrays"] = {
        **dict(replay.get("arrays", {})),
        **context_arrays,
    }
    updated_replay["decoder_context_attachment"] = {
        "context_build_manifest": context_manifest_path.name,
        "expected_paths_sha256": expected_paths_sha,
        "candidate_arrays_unchanged": True,
    }
    updated_writer = dict(writer)
    updated_writer["decoder_context_cached"] = True
    updated_writer["decoder_context_schema"] = _REPLAY_DECODER_CONTEXT_SCHEMA
    return {
        "rank": rank,
        "scene_count": count,
        "replay_manifest_path": replay_manifest_path,
        "writer_contract_path": writer_contract_path,
        "updated_replay": updated_replay,
        "updated_writer": updated_writer,
        "expected_paths_sha256": expected_paths_sha,
        "context_shapes": context_shapes,
    }


def attach(replay_root: Path, expected_world_size: int) -> dict[str, Any]:
    replay_root = replay_root.expanduser().resolve()
    rows = []
    for rank in range(int(expected_world_size)):
        rank_dir = replay_root / f"rank_{rank:04d}"
        rows.append(_validate_rank(rank_dir, rank))
    if len({row["scene_count"] for row in rows}) != 1:
        raise RuntimeError("rank-local replay counts differ")
    if len({json.dumps(row["context_shapes"], sort_keys=True) for row in rows}) != 1:
        raise RuntimeError("rank-local context shapes differ")

    # Every rank is validated before the first commit. Atomic file replacement
    # means readers see either the old context-free manifest or the complete
    # context-aware manifest, never a partially-written JSON document.
    for row in rows:
        _write_json_atomic(
            row["writer_contract_path"], row["updated_writer"]
        )
        _write_json_atomic(
            row["replay_manifest_path"], row["updated_replay"]
        )

    source_run = replay_root.parent
    effective_path = source_run / "effective_config.json"
    provenance_path = source_run / "provenance.json"
    if effective_path.is_file():
        effective = json.loads(effective_path.read_text())
        effective.setdefault("training", {})[
            "cache_replay_decoder_context"
        ] = True
        effective["decoder_context_attachment"] = {
            "schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
            "replay_root": str(replay_root),
            "neighbor_future_alignment_offset": get_neighbor_future_offset(),
            "policy_or_target_change": False,
        }
        _write_json_atomic(effective_path, effective)
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        provenance["decoder_context_attachment"] = {
            "schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
            "replay_root": str(replay_root),
            "neighbor_future_alignment_offset": get_neighbor_future_offset(),
            "x2_legacy_ego_width_m": 2.29156,
            "candidate_arrays_unchanged": True,
        }
        _write_json_atomic(provenance_path, provenance)

    payload = {
        "version": 1,
        "schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
        "replay_root": str(replay_root),
        "world_size": int(expected_world_size),
        "scene_count_per_rank": rows[0]["scene_count"],
        "total_scene_groups": sum(row["scene_count"] for row in rows),
        "context_shapes": rows[0]["context_shapes"],
        "neighbor_future_alignment_offset": get_neighbor_future_offset(),
        "candidate_arrays_unchanged": True,
        "reward_weight_expert_rng_unchanged": True,
        "ranks": [
            {
                "rank": row["rank"],
                "scene_count": row["scene_count"],
                "expected_paths_sha256": row["expected_paths_sha256"],
            }
            for row in rows
        ],
    }
    _write_json_atomic(source_run / "decoder_context_attachment.json", payload)
    return payload


def main() -> None:
    args = _args()
    payload = attach(args.replay_root, args.expected_world_size)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
