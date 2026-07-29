#!/usr/bin/env python3
"""Build an exact, resumable decoder-context sidecar for a replay cache.

The candidate miner already fixes one canonical rank-local scene order.  This
tool reads that order from ``expected_scene_paths.jsonl`` and materialises the
three tensors that uncached replay would reopen from every NPZ on every epoch:
ego current state, final neighbor-history frame, and the canonically aligned
neighbor future.  It does not read or modify trajectories, rewards, weights,
expert anchors, model parameters, or RNG state.

Run with ``torchrun --nproc_per_node=8``.  Each process owns one rank directory
and publishes ``context_build_manifest.json`` only after its full array prefix
has been flushed.  Interrupted builds resume from the last flushed position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion_planner.utils.neighbor_future_alignment import (
    get_neighbor_future_offset,
)
from rlvr.train_awr import (
    _REPLAY_CONTEXT_KEYS,
    _REPLAY_DECODER_CONTEXT_SCHEMA,
    _compact_replay_decoder_context,
    _file_sha256,
    _load_scene_batch,
    _write_json_atomic,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--args-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--scene-load-workers", type=int, default=4)
    parser.add_argument("--flush-interval-batches", type=int, default=16)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[str]:
    with path.open() as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if not values or not all(isinstance(value, str) for value in values):
        raise RuntimeError(f"invalid/empty canonical path index: {path}")
    return values


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _building_path(rank_dir: Path, key: str) -> Path:
    return rank_dir / f".context_{key}.npy.building"


def _final_path(rank_dir: Path, key: str) -> Path:
    return rank_dir / f"context_{key}.npy"


def _validate_completed(
    manifest: dict[str, Any],
    *,
    rank_dir: Path,
    expected_count: int,
    expected_paths_sha: str,
    future_len: int,
) -> None:
    if manifest.get("schema") != _REPLAY_DECODER_CONTEXT_SCHEMA:
        raise RuntimeError("completed context sidecar has an unknown schema")
    if int(manifest.get("scene_count", -1)) != expected_count:
        raise RuntimeError("completed context sidecar scene count changed")
    if manifest.get("expected_paths_sha256") != expected_paths_sha:
        raise RuntimeError("completed context sidecar scene order changed")
    if int(manifest.get("future_len", -1)) != future_len:
        raise RuntimeError("completed context sidecar future length changed")
    arrays = manifest.get("arrays", {})
    shapes = manifest.get("shapes", {})
    if set(arrays) != set(_REPLAY_CONTEXT_KEYS):
        raise RuntimeError("completed context sidecar is incomplete")
    for key in _REPLAY_CONTEXT_KEYS:
        path = rank_dir / str(arrays[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if array.dtype != np.float32:
            raise RuntimeError(f"{key} context dtype is {array.dtype}")
        if [int(value) for value in array.shape] != [
            int(value) for value in shapes[key]
        ]:
            raise RuntimeError(f"{key} context shape differs from manifest")
        if int(array.shape[0]) != expected_count:
            raise RuntimeError(f"{key} context count differs")


def main() -> None:
    args = _args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    replay_root = args.replay_root.expanduser().resolve()
    rank_dir = replay_root / f"rank_{rank:04d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    writer_contract_path = rank_dir / "writer_contract.json"
    if not writer_contract_path.is_file():
        raise FileNotFoundError(writer_contract_path)
    writer_contract = json.loads(writer_contract_path.read_text())
    expected_count = int(writer_contract.get("expected_count", -1))
    expected_name = writer_contract.get("expected_paths")
    if expected_count <= 0 or not expected_name:
        raise RuntimeError("replay writer has no canonical expected order")
    expected_paths_path = rank_dir / str(expected_name)
    paths = _read_jsonl(expected_paths_path)
    if len(paths) != expected_count:
        raise RuntimeError(
            f"rank {rank}: expected paths {len(paths)} != {expected_count}"
        )
    expected_paths_sha = _sha256(expected_paths_path)

    args_path = args.args_path.expanduser().resolve()
    args_data = json.loads(args_path.read_text())
    future_len = int(args_data["future_len"])
    predicted_neighbor_num = int(args_data["predicted_neighbor_num"])
    if future_len <= 0 or predicted_neighbor_num <= 0:
        raise RuntimeError("invalid model cardinalities")
    neighbor_future_offset = int(get_neighbor_future_offset())

    completed_path = rank_dir / "context_build_manifest.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text())
        _validate_completed(
            completed,
            rank_dir=rank_dir,
            expected_count=expected_count,
            expected_paths_sha=expected_paths_sha,
            future_len=future_len,
        )
        print(
            f"rank {rank}/{world_size}: decoder context already complete "
            f"({expected_count:,} scenes)",
            flush=True,
        )
        return

    progress_path = rank_dir / "context_build_progress.json"
    progress = (
        json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    )
    position = int(progress.get("position", 0))
    if position < 0 or position > expected_count:
        raise RuntimeError(f"rank {rank}: invalid resume position {position}")
    if position and (
        progress.get("expected_paths_sha256") != expected_paths_sha
        or progress.get("args_sha256") != _file_sha256(args_path)
        or int(progress.get("neighbor_future_alignment_offset", -1))
        != neighbor_future_offset
    ):
        raise RuntimeError("decoder-context resume contract changed")

    batch_size = max(1, int(args.batch_size))
    workers = max(1, int(args.scene_load_workers))
    flush_interval = max(1, int(args.flush_interval_batches))
    arrays: dict[str, np.memmap] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    if position:
        for key in _REPLAY_CONTEXT_KEYS:
            path = _building_path(rank_dir, key)
            if (
                not path.is_file()
                and position == expected_count
                and _final_path(rank_dir, key).is_file()
            ):
                # Recover a crash between per-array atomic renames and the
                # final all-arrays manifest commit.
                path = _final_path(rank_dir, key)
            if not path.is_file():
                raise FileNotFoundError(path)
            array = np.load(path, mmap_mode="r+")
            if array.dtype != np.float32 or int(array.shape[0]) != expected_count:
                raise RuntimeError(f"rank {rank}: invalid resumable {key} array")
            arrays[key] = array
            shapes[key] = tuple(int(value) for value in array.shape)

    start_time = time.time()
    last_flush_time = start_time
    for batch_index, start in enumerate(range(position, expected_count, batch_size)):
        end = min(start + batch_size, expected_count)
        data = _load_scene_batch(
            paths[start:end],
            torch.device("cpu"),
            workers=workers,
            replay_context_only=True,
        )
        compact = _compact_replay_decoder_context(data, predicted_neighbor_num)
        if not arrays:
            for key in _REPLAY_CONTEXT_KEYS:
                value = compact[key]
                shape = (expected_count, *tuple(int(x) for x in value.shape[1:]))
                arrays[key] = np.lib.format.open_memmap(
                    _building_path(rank_dir, key),
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
                shapes[key] = shape
        for key in _REPLAY_CONTEXT_KEYS:
            value = compact[key].detach().cpu().contiguous().numpy()
            if value.dtype != np.float32:
                value = value.astype(np.float32, copy=False)
            if tuple(value.shape[1:]) != tuple(shapes[key][1:]):
                raise RuntimeError(
                    f"rank {rank}: {key} shape changed at {start}: "
                    f"{value.shape[1:]} != {shapes[key][1:]}"
                )
            arrays[key][start:end] = value

        should_flush = (
            (batch_index + 1) % flush_interval == 0 or end == expected_count
        )
        if should_flush:
            for array in arrays.values():
                array.flush()
            now = time.time()
            payload = {
                "version": 1,
                "schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
                "rank": rank,
                "world_size": world_size,
                "position": end,
                "expected_count": expected_count,
                "expected_paths": str(expected_paths_path),
                "expected_paths_sha256": expected_paths_sha,
                "args_path": str(args_path),
                "args_sha256": _file_sha256(args_path),
                "future_len": future_len,
                "predicted_neighbor_num": predicted_neighbor_num,
                "neighbor_future_alignment_offset": neighbor_future_offset,
                "x2_legacy_ego_width_m": 2.29156,
                "shapes": {key: list(shape) for key, shape in shapes.items()},
                "elapsed_seconds": now - start_time,
                "last_flush_seconds": now - last_flush_time,
            }
            _write_json_atomic(progress_path, payload)
            last_flush_time = now
            print(
                f"rank {rank}/{world_size}: context {end:,}/{expected_count:,} "
                f"({100.0 * end / expected_count:.2f}%)",
                flush=True,
            )

    for array in arrays.values():
        array.flush()
    for key in _REPLAY_CONTEXT_KEYS:
        building = _building_path(rank_dir, key)
        final = _final_path(rank_dir, key)
        if building.is_file():
            os.replace(building, final)
        elif not final.is_file():
            raise FileNotFoundError(building)
    completed = {
        "version": 1,
        "schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "scene_count": expected_count,
        "future_len": future_len,
        "predicted_neighbor_num": predicted_neighbor_num,
        "neighbor_future_alignment_offset": neighbor_future_offset,
        "x2_legacy_ego_width_m": 2.29156,
        "expected_paths": str(expected_paths_path),
        "expected_paths_sha256": expected_paths_sha,
        "args_path": str(args_path),
        "args_sha256": _file_sha256(args_path),
        "arrays": {
            key: _final_path(rank_dir, key).name for key in _REPLAY_CONTEXT_KEYS
        },
        "shapes": {key: list(shape) for key, shape in shapes.items()},
        "elapsed_seconds": time.time() - start_time,
        "resume_start_position": position,
        "contract": (
            "byte-equivalent to replay_context_only loader; policy/reward/"
            "candidate/RNG independent"
        ),
    }
    _write_json_atomic(completed_path, completed)
    progress_path.unlink(missing_ok=True)
    _validate_completed(
        completed,
        rank_dir=rank_dir,
        expected_count=expected_count,
        expected_paths_sha=expected_paths_sha,
        future_len=future_len,
    )
    print(
        f"rank {rank}/{world_size}: decoder context committed "
        f"({expected_count:,} scenes, {time.time() - start_time:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
