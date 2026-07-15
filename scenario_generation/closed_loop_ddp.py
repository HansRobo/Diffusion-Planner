"""DDP sharding helpers for closed-loop evaluation (full-route and grouped)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import aggregate


def shard_items(items: list, rank: int, world_size: int) -> list:
    """Round-robin assignment: rank ``r`` gets indices r, r+world_size, ..."""
    if world_size <= 1:
        return items
    return [items[i] for i in range(rank, len(items), world_size)]


def write_eval_shard(
    out_dir: Path,
    rank: int,
    *,
    mode: str,
    rows: list[dict],
    video_mp4s: list[Path],
    elapsed_sec: float,
    extras: dict[str, Any] | None = None,
    profile: bool = False,
) -> Path:
    """Write a per-rank shard under ``out_dir/ddp_shards/rank_XXX.json``."""
    shard_dir = out_dir / "ddp_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"rank_{rank:03d}.json"
    payload: dict[str, Any] = {
        "mode": mode,
        "rank": rank,
        "rows": rows,
        "video_mp4s": [str(p) for p in video_mp4s],
        "elapsed_sec": elapsed_sec,
        "extras": extras or {},
        "profile": profile,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def merge_full_route_shards(
    out_dir: Path | str,
    world_size: int,
    *,
    npz_root: Path | str,
    near_miss_thresh: float,
) -> dict:
    """Merge per-rank full-route shards and write ``segments.jsonl`` + ``summary.json``."""
    out_dir = Path(out_dir)
    all_rows: list[dict] = []
    video_mp4s: list[Path] = []
    route_keys: list[str] = []
    elapsed_sec = 0.0

    for rank in range(world_size):
        path = out_dir / "ddp_shards" / f"rank_{rank:03d}.json"
        if not path.is_file():
            continue
        shard = json.loads(path.read_text(encoding="utf-8"))
        all_rows.extend(shard.get("rows") or [])
        video_mp4s.extend(Path(p) for p in shard.get("video_mp4s") or [])
        elapsed_sec = max(elapsed_sec, float(shard.get("elapsed_sec", 0.0)))
        route_keys.extend(shard.get("extras", {}).get("route_keys") or [])

    with open(out_dir / "segments.jsonl", "w", encoding="utf-8") as fout:
        for row in all_rows:
            fout.write(json.dumps(row, default=float) + "\n")

    summary = aggregate(all_rows, near_miss_thresh)
    summary["npz_root"] = str(Path(npz_root).resolve())
    summary["n_routes"] = len(set(route_keys))
    summary["elapsed_sec"] = elapsed_sec
    summary["video_mp4s"] = video_mp4s
    summary["segments"] = all_rows
    summary["mode"] = "full"
    summary["ddp_shard"] = True
    summary["ddp_world_size"] = world_size

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in summary.items() if k not in ("video_mp4s", "segments")},
            f,
            indent=4,
        )
    return summary


def ddp_device_for_rank(fallback: str = "cuda") -> str:
    if "LOCAL_RANK" in os.environ:
        return f"cuda:{int(os.environ['LOCAL_RANK'])}"
    return fallback
