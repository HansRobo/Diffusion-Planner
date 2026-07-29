#!/usr/bin/env python3
"""Build an exact missing-scene manifest for a salvaged AWR replay cache.

The full train list can be hundreds of megabytes and is commonly stored as a
single JSON line.  This tool streams that array while retaining only the
replay path set, so it does not materialize both multi-million-element lists.
It reads scene-path sidecars only; trajectory and scene-encoding memmaps are
never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--missing-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_json_array(path: Path, chunk_bytes: int) -> Iterator[str]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False
        opened = False
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_bytes)
                position = 0
                eof = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not opened:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON array: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected JSON array in {path}")
                opened = True
                position += 1
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = buffer[position:] + handle.read(chunk_bytes)
                position = 0
                eof = handle.tell() == path.stat().st_size
                continue
            if not isinstance(value, str):
                raise TypeError(f"non-string scene path in {path}: {type(value).__name__}")
            yield value
            position = end


def _load_replay_paths(replay_root: Path) -> tuple[set[str], list[dict], int]:
    paths: set[str] = set()
    rank_rows: list[dict] = []
    record_count = 0
    rank_dirs = sorted(p for p in replay_root.glob("rank_*") if p.is_dir())
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {replay_root}")
    for rank_dir in rank_dirs:
        manifest_path = rank_dir / "manifest.json"
        path_file = rank_dir / "scene_paths.jsonl"
        if not manifest_path.is_file() or not path_file.is_file():
            raise FileNotFoundError(f"incomplete replay sidecar in {rank_dir}")
        manifest = json.loads(manifest_path.read_text())
        before = len(paths)
        rank_records = 0
        with path_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, str):
                    raise TypeError(f"non-string path at {path_file}:{line_number}")
                paths.add(value)
                rank_records += 1
        expected = int(manifest["scene_count"])
        if rank_records != expected:
            raise ValueError(
                f"{rank_dir.name}: manifest={expected}, scene_paths={rank_records}"
            )
        record_count += rank_records
        rank_rows.append(
            {
                "rank": int(manifest["rank"]),
                "scene_records": rank_records,
                "new_unique_paths": len(paths) - before,
                "array_capacity": int(manifest["array_capacity"]),
                "discarded_tail_slots": int(manifest.get("discarded_tail_slots", 0)),
                "salvaged_partial_prefix": int(manifest.get("salvaged_partial_prefix", 0)),
            }
        )
    return paths, rank_rows, record_count


def main() -> None:
    args = _args()
    replay_paths, ranks, replay_record_count = _load_replay_paths(args.replay_root)
    missing: list[str] = []
    source_seen: set[str] = set()
    source_count = 0
    duplicate_source_count = 0
    for path in _iter_json_array(args.train_list, max(64 * 1024, args.chunk_bytes)):
        source_count += 1
        if path in source_seen:
            duplicate_source_count += 1
            continue
        source_seen.add(path)
        if path not in replay_paths:
            missing.append(path)
    unexpected = sorted(replay_paths - source_seen)
    args.missing_output.parent.mkdir(parents=True, exist_ok=True)
    args.missing_output.write_text(json.dumps(missing, indent=2) + "\n")
    payload = {
        "contract": {
            "purpose": "exact backfill list for salvaged replay; scene-path sidecars only",
            "train_list": str(args.train_list.resolve()),
            "train_list_sha256": _sha256(args.train_list),
            "replay_root": str(args.replay_root.resolve()),
            "missing_output": str(args.missing_output.resolve()),
            "missing_output_sha256": _sha256(args.missing_output),
        },
        "source_record_count": source_count,
        "source_unique_count": len(source_seen),
        "duplicate_source_count": duplicate_source_count,
        "replay_record_count": replay_record_count,
        "replay_unique_count": len(replay_paths),
        "duplicate_replay_count": replay_record_count - len(replay_paths),
        "missing_count": len(missing),
        "missing_fraction_of_unique_source": len(missing) / max(1, len(source_seen)),
        "unexpected_replay_count": len(unexpected),
        "unexpected_replay_first_paths": unexpected[:20],
        "ranks": ranks,
    }
    if len(replay_paths) + len(missing) != len(source_seen) + len(unexpected):
        raise AssertionError("set accounting mismatch")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
