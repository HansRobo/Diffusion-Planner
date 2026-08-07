"""Map scenario_sim rollout series onto the closed-loop segment-row schema.

``closed_loop_eval.aggregate`` consumes rows made of nested per-category blocks (``object`` /
``road_border`` / ``red_light_violation`` / ``strong_brake`` / ``reproducer``) and fails fast
on a missing block, so a missing metric can never read as a zero. The scenario_sim path
produces the same raw series as the reproducer path -- per-step clearance, collision,
road-border distance and speed -- but through a simulator rather than recorded NPZ frames, so
the mapping onto that schema lives here.

The metric semantics are not re-implemented: ``_clearance_stats``, ``_event_count`` and
``strong_brake_mask`` come from the shared modules, so the two paths' definitions cannot
drift apart.

``mean_gt_deviation_m`` and ``route_completion`` are absent. Both are optional in
``aggregate``, and both need a reference this path does not have: the reproducer measures
deviation from a recorded human drive, and route progress needs the resolved route.
"""

from __future__ import annotations

import numpy as np

from scenario_generation.metrics.strong_brake import strong_brake_mask
from scenario_generation.reproducer_rollout import (
    RB_COLLISION_THRESH_M,
    _clearance_stats,
    _event_count,
)

# scenario_sim has no reproducer cursor: it never expands a window, snaps the ego back or
# repeats a frame. Zeros here are the true measurement, not a placeholder.
_NO_REPRODUCER_CURSOR = {"expand_count": 0, "snap_count": 0, "repeat_steps": 0}


def _red_light_block() -> dict:
    """Red-light violation is not measured on this path.

    ``aggregate`` requires the block, so it is emitted with zero counts plus an explicit
    ``measured`` flag -- otherwise "0 violations" would be indistinguishable from "never
    checked". Detecting them needs stop-line geometry and per-tick traffic-light state.
    """
    return {"steps": 0, "count": 0, "measured": False}


def build_segment_row(
    *,
    n_steps_run: int,
    terminated: str,
    result_kind: str,
    clearances: list[float],
    collisions: list[bool],
    rb_dists: np.ndarray,
    speeds: list[float],
    dt: float,
    near_miss_thresh: float,
    strong_brake_mps2: float,
    progress_m: float,
    extra: dict | None = None,
) -> dict:
    """Build one closed-loop segment row from a scenario_sim rollout's raw series.

    ``rb_dists`` may be empty when the map ships no road-border polylines; the road_border
    block then reports ``inf`` clearances and zero events, which is what ``_clearance_stats``
    does for an all-inf series.
    """
    cl = np.asarray(clearances, dtype=np.float64)
    coll = np.asarray(collisions, dtype=bool)
    rb = np.asarray(rb_dists, dtype=np.float64)

    finite = np.isfinite(cl)
    obj_miss = finite & (cl <= near_miss_thresh)
    rb_finite = np.isfinite(rb)
    rb_coll = rb_finite & (rb < RB_COLLISION_THRESH_M)
    rb_miss = rb_finite & (rb <= near_miss_thresh)

    # Speed is logged per tick, so acceleration is a first difference over the sim step, not
    # wall time. Padded to keep the series as long as clearances/collisions -- otherwise
    # strong_brake.steps would be counted over k-1 steps while object.collision_steps uses k.
    sp = np.asarray(speeds, dtype=np.float64)
    accels = np.zeros(sp.size, dtype=np.float64)
    if sp.size >= 2:
        accels[1:] = np.diff(sp) / dt
    brake_mask = strong_brake_mask(accels, thresh_mps2=float(strong_brake_mps2))

    row = {
        "n_steps_run": int(n_steps_run),
        "terminated": terminated,
        "result_kind": result_kind,
        "progress_m": float(progress_m),
        "object": {
            "miss_thresh_m": float(near_miss_thresh),
            "collision_steps": int(coll.sum()),
            "collision_count": _event_count(coll),
            "miss_steps": int(obj_miss.sum()),
            "miss_count": _event_count(obj_miss),
            **_clearance_stats(cl),
        },
        "road_border": {
            "miss_thresh_m": float(near_miss_thresh),
            "collision_steps": int(rb_coll.sum()),
            "collision_count": _event_count(rb_coll),
            "miss_steps": int(rb_miss.sum()),
            "miss_count": _event_count(rb_miss),
            **_clearance_stats(rb),
        },
        "red_light_violation": _red_light_block(),
        "strong_brake": {
            "thresh_mps2": float(strong_brake_mps2),
            "strongest_mps2": (
                float(accels[brake_mask].min()) if brake_mask.any() else float("inf")
            ),
            "steps": int(brake_mask.sum()),
            "count": _event_count(brake_mask),
        },
        "reproducer": {**_NO_REPRODUCER_CURSOR, "normal_steps": int(n_steps_run)},
    }
    if extra:
        row.update(extra)
    return row


def failed_segment_row(reason: str, near_miss_thresh: float) -> dict:
    """Schema-complete row for a scenario whose worker produced no output.

    Built by the same function as a real row so the two can never drift: ``aggregate`` raises
    on a missing block, so a crashed worker would otherwise take down the whole eval instead
    of being counted as a failure. Empty series give every block its "no samples" values.

    ``near_miss_thresh`` is the configured threshold, not NaN, because
    ``_event_family_block`` copies it straight into the summary as the run's threshold.
    """
    return build_segment_row(
        n_steps_run=0,
        terminated="worker_failed",
        result_kind="",
        clearances=[],
        collisions=[],
        rb_dists=np.zeros(0, dtype=np.float64),
        speeds=[],
        dt=1.0,  # no samples to difference, so the value cannot reach the row
        near_miss_thresh=near_miss_thresh,
        strong_brake_mps2=float("inf"),  # no threshold was applied
        progress_m=0.0,
        extra={"error": reason},
    )
