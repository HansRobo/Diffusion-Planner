#!/usr/bin/env python3
"""Bootstrap a full-validation baseline/checkpoint pair into the sweep report schema.

``eval_awr_full_distributed.py`` writes independent baseline and checkpoint directories.  This
adapter reuses the exact scene-level bootstrap implementation from ``paired_eval_report.py`` so
the 2,048-scene checkpoint sweep and the final 46,068-scene promotion evaluation have the same
metric directions, confidence intervals, and missing-value contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.autoresearch.tools.paired_eval_report import compare, load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--evidence_scope",
        default="full_filtered_validation",
        help="human-readable population/protocol label stored in the report",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="optional metric subset; defaults to the complete paired report",
    )
    args = parser.parse_args()

    if args.metrics:
        from rlvr.autoresearch.tools.paired_eval_report import METRICS

        unknown = sorted(set(args.metrics) - set(METRICS))
        if unknown:
            parser.error(f"unknown metrics: {unknown}")

    baseline = load_rows(args.baseline)
    candidate = load_rows(args.candidate)
    result = compare(
        baseline,
        candidate,
        max(1, int(args.bootstrap)),
        int(args.seed) + int(args.epoch),
        None if not args.metrics else tuple(args.metrics),
    )
    report = {
        "run_dir": None,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "evidence_scope": str(args.evidence_scope),
        "epochs": {str(int(args.epoch)): result},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
