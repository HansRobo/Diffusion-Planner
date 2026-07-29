#!/usr/bin/env python3
"""Merge independently evaluated checkpoint reports into one epoch sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, Any] = {
        "run_dir": None,
        "baseline": None,
        "evidence_scope": "full_filtered_validation",
        "source_reports": [],
        "epochs": {},
    }
    for path in args.reports:
        payload = json.loads(path.read_text())
        merged["source_reports"].append(str(path.resolve()))
        baseline = payload.get("baseline")
        if merged["baseline"] is None:
            merged["baseline"] = baseline
        elif baseline != merged["baseline"]:
            raise RuntimeError(
                f"baseline mismatch: {baseline!r} != {merged['baseline']!r}"
            )
        for epoch, report in payload.get("epochs", {}).items():
            if epoch in merged["epochs"]:
                raise RuntimeError(f"duplicate epoch {epoch} in {path}")
            merged["epochs"][epoch] = report
    if not merged["epochs"]:
        raise RuntimeError("no epoch reports to merge")
    merged["epochs"] = {
        key: merged["epochs"][key]
        for key in sorted(merged["epochs"], key=lambda value: int(value))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
