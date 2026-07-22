"""Match test frames to training data on the same lanelet, fetch and transform trajectories."""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from human_match_prototype.coord_transform import (
    WorldPose,
    pose_from_sidecar,
    transform_trajectory,
)


@dataclass
class MatchResult:
    n_matched: int
    human_futures: list[np.ndarray] = field(default_factory=list)
    match_metadata: list[dict] = field(default_factory=list)


def build_lanelet_lookup(index: list[dict]) -> dict[int, list[dict]]:
    lookup: dict[int, list[dict]] = defaultdict(list)
    for entry in index:
        lookup[entry["lanelet_id"]].append(entry)
    return dict(lookup)


def _heading_diff_deg(a_deg: float, b_deg: float) -> float:
    d = a_deg - b_deg
    d = (d + 180.0) % 360.0 - 180.0
    return abs(d)


def _bag_prefix(npz_path: str) -> str:
    """Extract bag/sequence prefix from NPZ path (everything before the last _NNNNN.npz)."""
    return npz_path.rsplit("_", 1)[0]


def _deduplicate_by_drive(entries: list[dict]) -> list[dict]:
    """Keep one representative frame per driving sequence (bag prefix)."""
    bags: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        bags[_bag_prefix(e["npz_path"])].append(e)
    result = []
    for bag_entries in bags.values():
        bag_entries.sort(key=lambda e: e.get("timestamp", 0))
        result.append(bag_entries[len(bag_entries) // 2])
    return result


def find_matches(
    test_sidecar: dict,
    lanelet_lookup: dict[int, list[dict]],
    test_lanelet_id: int,
    max_heading_diff_deg: float = 45.0,
    max_distance_m: float = 30.0,
    max_per_lanelet: int = 200,
    seed_key: str = "",
) -> list[dict]:
    candidates = lanelet_lookup.get(test_lanelet_id, [])
    if not candidates:
        return []

    tx, ty = test_sidecar["x"], test_sidecar["y"]
    th = test_sidecar["heading_deg"]

    filtered = []
    for c in candidates:
        if _heading_diff_deg(c["heading_deg"], th) > max_heading_diff_deg:
            continue
        dist = math.sqrt((c["x"] - tx) ** 2 + (c["y"] - ty) ** 2)
        if dist > max_distance_m:
            continue
        filtered.append(c)

    filtered = _deduplicate_by_drive(filtered)

    if len(filtered) > max_per_lanelet:
        seed = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(filtered), size=max_per_lanelet, replace=False)
        filtered = [filtered[i] for i in sorted(idx)]

    return filtered
