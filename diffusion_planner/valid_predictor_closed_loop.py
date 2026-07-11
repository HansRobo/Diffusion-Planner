"""Closed-loop validation of a Diffusion-Planner checkpoint.

Two modes (``--mode``):

* **full** (default) — roll out every route under ``--npz_root`` in ``--seg_len`` chunks;
  same behavior as the original mining validation CLI.
* **grouped** — one full-route closed-loop rollout per sequence, then per-map-area
  metrics and videos split by reproducer frame index (from ``scenario_classification_json``).

Open-loop counterpart: ``valid_predictor.py``.

Examples::

    # Full-route validation
    python diffusion_planner/valid_predictor_closed_loop.py \\
        --model_path ./best_model.pth \\
        --npz_root /path/to/valid/2026-01-15

    # Grouped-by-area validation (after classify_scenario_corpus)
    python diffusion_planner/valid_predictor_closed_loop.py \\
        --mode grouped \\
        --model_path ./best_model.pth \\
        --npz_root /path/to/x2_dev/2231_odaiba.../valid/2026-01-15 \\
        --classification_json_root ../Diffusion-Planner-Meta-Repository/dataset/scenario_classification_json \\
        --out_dir ./odaiba_cl_results \\
        --near_miss_thresh 0.3 \\
        --replan_interval 10 \\
        --draw_every 8
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
        "--model_path",
        type=Path,
        required=True,
        help="checkpoint .pth; args.json must sit next to it (e.g. epoch0001/best_model.pth)",
    )
    p.add_argument(
        "--npz_root",
        type=Path,
        required=True,
        help="dir tree of route NPZ frames (recursively globbed, grouped into routes). "
        "Pose JSON sidecars are read from next to each .npz, falling back to this same tree.",
    )
    # --- tunable knobs (default to the closed-loop mining config) ---
    p.add_argument("--seg_len", type=int, default=6000, help="frames per segment (~60s @10Hz)")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--near_miss_thresh", type=float, default=0.5, help="near-miss clearance (m)")
    p.add_argument(
        "--search_radius", type=float, default=1.5, help="PerceptionReproducer cursor search (m)"
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="steps driven by the recorded GT pose before handing control to the model",
    )
    p.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap the ego to the GT pose ahead after this many no-progress steps (0=off)",
    )
    p.add_argument(
        "--unstick_advance_m", type=float, default=2.5, help="how far ahead to snap when unsticking"
    )
    p.add_argument(
        "--unstick_radius_mult",
        type=float,
        default=3.0,
        help="when stuck, first widen the cursor search_radius to this x nominal so it reaches "
        "frames further ahead (model proceeds on its own); restored to nominal once the ego moves. "
        "<=1 disables this gentle stage (teleport straight away at --unstick_after)",
    )
    p.add_argument(
        "--unstick_teleport_after",
        type=int,
        default=300,
        help="if still stuck this many steps AFTER the radius was widened, fall back to the hard "
        "teleport onto the GT pose ahead (last resort)",
    )
    p.add_argument("--fps", type=int, default=10, help="output video frame rate (10 = realtime)")
    p.add_argument(
        "--replan_interval",
        type=int,
        default=4,
        help="re-run the model every N steps (1 = every step). Between inferences the cached plan "
        "is executed, re-expressed in the current ego frame each step; the ego still steps at 10Hz",
    )
    p.add_argument(
        "--draw_every",
        type=int,
        default=8,
        help="render a PNG only every N steps (1 = every step). PNG rendering (matplotlib) is the "
        "dominant cost; this throttles it without touching the rollout. Frames are encoded at --fps "
        "regardless, so the video also plays N x faster (shorter). For real-time use --fps 10/N",
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
