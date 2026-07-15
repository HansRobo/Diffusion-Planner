"""DDP sharding helpers for closed-loop evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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


def ddp_device_for_rank(fallback: str = "cuda") -> str:
    if "LOCAL_RANK" in os.environ:
        return f"cuda:{int(os.environ['LOCAL_RANK'])}"
    return fallback
