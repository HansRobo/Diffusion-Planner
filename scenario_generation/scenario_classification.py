"""Load and resolve portable scenario classification JSON (Meta-Repository format)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

AreaEpisode = dict
# keys: area, metric_group, video_start_idx, video_end_idx, labeled_ranges


def load_classification_json(path: Path) -> dict:
    """Load scenario_classification_json document."""
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def time_series_from_doc(doc: dict) -> dict[str, dict]:
    """Per-bag ``time_series`` entries.

    Legacy docs may use top-level ``sequences``; they are normalized to the same shape.
    """
    if "time_series" in doc:
        return dict(doc["time_series"])
    legacy = doc.get("sequences", {})
    return {k: {"name": k, **v} for k, v in legacy.items()}


def load_area_metric_groups(doc: dict) -> dict[str, str]:
    """area_name -> metric_group."""
    if "area_catalog" in doc:
        return {str(k): str(v["metric_group"]) for k, v in doc["area_catalog"].items()}
    if "areas" in doc:
        return {str(k): str(v["metric_group"]) for k, v in doc["areas"].items()}
    out: dict[str, str] = {}
    for entry in time_series_from_doc(doc).values():
        for ep in episodes_from_entry(entry):
            out[str(ep["area"])] = str(ep["metric_group"])
    return out


def _normalize_labeled_ranges(seg: dict) -> list[list[int]]:
    if "labeled_ranges" in seg:
        return [[int(r[0]), int(r[1])] for r in seg["labeled_ranges"]]
    return [[int(seg["start_idx"]), int(seg["end_idx"])]]


def episodes_from_entry(entry: dict) -> list[AreaEpisode]:
    """Return normalized episode list for one bag entry.

    v2.2 episodes bridge unlabeled gaps via ``video_*``; metrics use ``labeled_ranges``.
    v2.1 ``start_idx``/``end_idx`` and legacy ``area_by_idx`` are upgraded on read.
    """
    if "segments" in entry:
        episodes: list[AreaEpisode] = []
        for seg in entry["segments"]:
            if "labeled_ranges" in seg:
                episodes.append(
                    {
                        "area": str(seg["area"]),
                        "metric_group": str(seg["metric_group"]),
                        "video_start_idx": int(seg["video_start_idx"]),
                        "video_end_idx": int(seg["video_end_idx"]),
                        "labeled_ranges": _normalize_labeled_ranges(seg),
                    }
                )
            else:
                start = int(seg["start_idx"])
                end = int(seg["end_idx"])
                episodes.append(
                    {
                        "area": str(seg["area"]),
                        "metric_group": str(seg["metric_group"]),
                        "video_start_idx": start,
                        "video_end_idx": end,
                        "labeled_ranges": [[start, end]],
                    }
                )
        return _merge_v21_episodes(episodes)

    area_by_idx = entry.get("area_by_idx") or []
    groups = entry.get("metric_group_by_idx") or [None] * len(area_by_idx)
    chunks: list[AreaEpisode] = []
    for area, start_idx, end_idx in iter_area_spans(area_by_idx):
        mg = groups[start_idx] if start_idx < len(groups) else None
        chunks.append(
            {
                "area": area,
                "metric_group": str(mg) if mg else "",
                "video_start_idx": start_idx,
                "video_end_idx": end_idx,
                "labeled_ranges": [[start_idx, end_idx]],
            }
        )
    return _merge_v21_episodes(chunks)


def _merge_v21_episodes(episodes: list[AreaEpisode]) -> list[AreaEpisode]:
    """Merge consecutive same-area v2.1-style episodes (bridge unlabeled gaps)."""
    merged: list[AreaEpisode] = []
    for ep in episodes:
        if merged and merged[-1]["area"] == ep["area"]:
            merged[-1]["video_end_idx"] = int(ep["video_end_idx"])
            merged[-1]["labeled_ranges"].extend(ep["labeled_ranges"])
        else:
            merged.append(
                {
                    "area": ep["area"],
                    "metric_group": ep["metric_group"],
                    "video_start_idx": int(ep["video_start_idx"]),
                    "video_end_idx": int(ep["video_end_idx"]),
                    "labeled_ranges": [list(r) for r in ep["labeled_ranges"]],
                }
            )
    return merged


def segments_from_entry(entry: dict) -> list[AreaEpisode]:
    """Alias for :func:`episodes_from_entry` (kept for older call sites)."""
    return episodes_from_entry(entry)


def iter_episodes(entry: dict) -> Iterator[AreaEpisode]:
    """Yield normalized episodes for one bag."""
    yield from episodes_from_entry(entry)


def iter_segments(entry: dict) -> Iterator[tuple[str, int, int]]:
    """Yield ``(area_name, start_idx, end_idx)`` per labeled range (legacy helper)."""
    for ep in episodes_from_entry(entry):
        for start, end in ep["labeled_ranges"]:
            yield str(ep["area"]), int(start), int(end)


def idx_in_labeled_ranges(idx: int, labeled_ranges: list[list[int]]) -> bool:
    return any(int(start) <= idx < int(end) for start, end in labeled_ranges)


def area_at_idx(episodes: list[AreaEpisode], idx: int) -> str | None:
    """Map reproducer frame index to area name on labeled frames only."""
    for ep in episodes:
        if idx_in_labeled_ranges(idx, ep["labeled_ranges"]):
            return str(ep["area"])
    return None


def iter_area_spans(area_by_idx: list[str | None]) -> Iterator[tuple[str, int, int]]:
    """Yield ``(area_name, start_idx, end_idx)`` for each contiguous labeled span (legacy)."""
    i, n = 0, len(area_by_idx)
    while i < n:
        area = area_by_idx[i]
        if area is None:
            i += 1
            continue
        start = i
        i += 1
        while i < n and area_by_idx[i] == area:
            i += 1
        yield str(area), start, i


def validate_classification_npz_root(doc: dict, npz_root: Path) -> list[str]:
    """Return warning strings if JSON date disagrees with ``npz_root`` basename."""
    warnings: list[str] = []
    date = doc.get("date")
    if date and npz_root.name != date:
        warnings.append(f"npz_root basename {npz_root.name!r} != classification date {date!r}")
    return warnings


def classification_json_search_roots() -> list[Path]:
    """Candidate roots for auto-resolving ``scenario_classification_json/<dataset>/<date>.json``."""
    roots: list[Path] = []
    env = os.environ.get("SCENARIO_CLASSIFICATION_JSON_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    repo_root = Path(__file__).resolve().parents[1]
    roots.append(
        repo_root.parent
        / "Diffusion-Planner-Meta-Repository"
        / "dataset"
        / "scenario_classification_json"
    )
    return roots


def resolve_classification_json(
    npz_root: Path | str,
    explicit: str | Path | None = None,
    *,
    dataset_name: str | None = None,
    search_roots: list[Path] | None = None,
) -> Path | None:
    """Resolve classification JSON path for grouped closed-loop eval.

    Resolution order:
    1. ``explicit`` path when the file exists.
    2. ``<search_root>/<dataset_name>/<npz_date>.json`` for each search root.
    """
    npz_root = Path(npz_root).expanduser().resolve()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
        return None

    if not dataset_name:
        return None

    date = npz_root.name
    for root in search_roots or classification_json_search_roots():
        candidate = root.expanduser() / dataset_name / f"{date}.json"
        if candidate.is_file():
            return candidate.resolve()
    return None
