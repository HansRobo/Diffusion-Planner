"""Closed-loop validation of a Diffusion-Planner checkpoint.

Open-loop counterpart: ``valid_predictor.py`` runs ONE forward per sample and scores
the prediction against the recorded GT future. This script instead drives the ego in
CLOSED LOOP through ``scenario_generation``: each tick the model predicts the ego
trajectory, ``PerfectTracker`` advances the ego one step along it, the recorded
neighbors are replayed from the log via the Perception-Reproducer cursor, and the
realized ego footprint is scored against those neighbors with the canonical OBB
(``score_step`` -> collision / near-miss / min clearance).

A *route* = one bag-prefix group of consecutive 10 Hz NPZ frames (``RouteTimeline``);
each route is sliced into ``--seg_len`` segments and rolled out with ``render_segment``
(one GPU forward per tick), which BOTH returns the segment metrics AND writes a per-step
PNG of the live-ego scene. Every run therefore always produces video: one MP4 per segment
(``<route>_<start>_<end>.mp4``). Per-segment metrics are streamed to ``segments.jsonl`` and
aggregated into ``summary.json`` (both next to the checkpoint).

Use ``--mode full`` (default) to evaluate complete routes or ``--mode grouped`` to
evaluate the labeled Odaiba scenario groups from a classification JSON. Only
``--model_path`` and ``--npz_root`` are required for full-route evaluation; outputs
default to ``<model_path dir>/closed_loop/``. Example::

    NPZ=/path/to/closed_loop_npz_dir

    python3 valid_predictor_closed_loop.py \
        --model_path /mnt/nvme/training_result/hdp_rl_run/epoch0001/best_model.pth \
        --npz_root ${NPZ}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scenario_generation.closed_loop_cli import (  # noqa: E402
    add_full_route_args,
    add_grouped_args,
    add_rollout_args,
    print_full_summary,
    run_full_route_eval,
    run_grouped_eval,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("full", "grouped"),
        default="full",
        help="full: entire routes; grouped: labeled scenario areas",
    )
    add_rollout_args(p)
    add_full_route_args(p)
    add_grouped_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "grouped":
        return run_grouped_eval(args)

    summary = run_full_route_eval(args)
    out_dir = args.out_dir
    if out_dir is None:
        from datetime import datetime

        out_dir = args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    print_full_summary(summary, args.near_miss_thresh, Path(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
