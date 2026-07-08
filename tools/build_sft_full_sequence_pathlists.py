#!/usr/bin/env python3
"""Build full-sequence pathlists constrained by the existing SFT pathlists.

The full-sequence dataset contains many projects/areas. This script prevents data
mixing by using the old SFT list only as a whitelist of route keys:

    (train_or_valid, date, session, route_id)

It then selects all continuous frames from the 20260623_full_sequence list for those
route keys, and finally filters each candidate by the adjacent JSON:

    keep only if is_skipped == false

The output is a new pair of JSON pathlists plus summary/missing-key reports.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


OLD_ROOT = "dfp_matched_20260622_step3"
FULL_ROOT = "20260623_full_sequence"


def load_list(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def route_key(path: str, root_name: str) -> tuple[str, str, str, str]:
    """Return (split, date, session, route_id).

    Handles both new full-sequence paths with explicit train/valid components and
    older SFT paths where split is encoded as xx1_real_train/j6_real_valid.
    """
    parts = path.split("/")
    root_idx = parts.index(root_name)
    rel = parts[root_idx + 1 :]

    split = None
    split_idx = None
    for idx, part in enumerate(rel):
        if part in ("train", "valid"):
            split = part
            split_idx = idx
            break

    if split is None:
        tag = rel[1] if root_name == OLD_ROOT and len(rel) > 1 else ""
        split = "valid" if tag.endswith("_valid") else "train"
        date = rel[-3]
        session = rel[-2]
    else:
        date = rel[split_idx + 1]
        session = rel[split_idx + 2]

    stem = Path(path).stem
    tokens = stem.split("_")
    route_id = tokens[1] if len(tokens) >= 3 else "NA"
    return split, date, session, route_id


LABEL_RE = re.compile(r'"label"\s*:\s*("?[^",}\s]+"?)')


def keep_by_json(npz_path: str) -> tuple[bool, int | str | None]:
    json_path = npz_path[:-4] + ".json" if npz_path.endswith(".npz") else npz_path + ".json"
    try:
        with open(json_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        return False, "missing_json"
    except Exception:
        return False, "bad_json"

    match = LABEL_RE.search(text)
    label: int | str | None = None
    if match:
        raw = match.group(1).strip('"')
        try:
            label = int(raw)
        except ValueError:
            label = raw

    if '"is_skipped": false' in text or '"is_skipped":false' in text:
        return True, label
    if '"is_skipped": true' in text or '"is_skipped":true' in text:
        return False, label
    return False, "missing_is_skipped"


def build_split(
    old_list_path: Path,
    full_list_path: Path,
    output_path: Path,
    missing_path: Path,
    split_name: str,
) -> dict[str, object]:
    old_paths = load_list(old_list_path)
    allowed_keys = {route_key(path, OLD_ROOT) for path in old_paths}

    full_paths = load_list(full_list_path)
    matched_before_filter: list[str] = []
    matched_keys: set[tuple[str, str, str, str]] = set()
    for path in full_paths:
        key = route_key(path, FULL_ROOT)
        if key in allowed_keys:
            matched_before_filter.append(path)
            matched_keys.add(key)

    kept: list[str] = []
    skipped_labels: Counter[int | str | None] = Counter()
    kept_labels: Counter[int | str | None] = Counter()
    for idx, path in enumerate(matched_before_filter, start=1):
        keep, label = keep_by_json(path)
        if keep:
            kept.append(path)
            kept_labels[label] += 1
        else:
            skipped_labels[label] += 1
        if idx % 200000 == 0:
            print(f"{split_name}: json-filtered {idx}/{len(matched_before_filter)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(kept, f, indent=2)

    missing = sorted(allowed_keys - matched_keys)
    with missing_path.open("w") as f:
        for item in missing:
            f.write("\t".join(item) + "\n")

    return {
        "split": split_name,
        "old_npz": len(old_paths),
        "old_route_keys": len(allowed_keys),
        "full_npz_total": len(full_paths),
        "matched_npz_before_json_filter": len(matched_before_filter),
        "matched_route_keys": len(matched_keys),
        "missing_route_keys": len(missing),
        "kept_npz_after_json_filter": len(kept),
        "skipped_by_json": len(matched_before_filter) - len(kept),
        "kept_labels": dict(kept_labels),
        "skipped_labels": dict(skipped_labels),
        "output_path": str(output_path),
        "missing_path": str(missing_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-train-list",
        type=Path,
        default=Path(
            "/mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/"
            "_pathlists/path_list_train_rebuilt_step3.json"
        ),
    )
    parser.add_argument(
        "--old-valid-list",
        type=Path,
        default=Path(
            "/mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/"
            "_pathlists/path_list_valid_rebuilt_step3.json"
        ),
    )
    parser.add_argument(
        "--full-train-list",
        type=Path,
        default=Path(
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "path_list_train_sft.json"
        ),
    )
    parser.add_argument(
        "--full-valid-list",
        type=Path,
        default=Path(
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "path_list_valid_sft.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "_sft_from_20260622_step3"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    train_summary = build_split(
        old_list_path=args.old_train_list,
        full_list_path=args.full_train_list,
        output_path=output_dir / "path_list_train_sft_fullseq_from_20260622_step3.json",
        missing_path=output_dir / "missing_train_route_keys.tsv",
        split_name="train",
    )
    valid_summary = build_split(
        old_list_path=args.old_valid_list,
        full_list_path=args.full_valid_list,
        output_path=output_dir / "path_list_valid_sft_fullseq_from_20260622_step3.json",
        missing_path=output_dir / "missing_valid_route_keys.tsv",
        split_name="valid",
    )

    summary = {
        "old_root": OLD_ROOT,
        "full_root": FULL_ROOT,
        "route_key": ["split", "date", "session", "route_id"],
        "json_filter": "keep only adjacent json with is_skipped == false",
        "train": train_summary,
        "valid": valid_summary,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
