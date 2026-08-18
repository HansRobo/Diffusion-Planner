"""Map scenario_sim rollout series onto the closed-loop segment-row schema.

``closed_loop_eval.aggregate`` consumes rows made of nested per-category blocks (``object`` /
``road_border`` / ``red_light_violation`` / ``strong_brake`` / ``reproducer``) and fails fast
on a missing block, so a missing metric can never read as a zero. The scenario_sim path
produces the same raw series as the reproducer path -- per-step clearance, collision,
road-border distance and speed -- but through a simulator rather than recorded NPZ frames, so
the mapping onto that schema lives here.

The blocks themselves are not re-implemented: ``clearance_family_block`` and
``strong_brake_block`` are the same builders the reproducer path uses, so neither the metric
semantics nor the key layout can drift apart.

``mean_gt_deviation_m`` and ``route_completion`` are absent. Both are optional in
``aggregate``, and both need a reference this path does not have: the reproducer measures
deviation from a recorded human drive, and route progress needs the resolved route.
"""

from __future__ import annotations

import numpy as np
from diffusion_planner.dimensions import (
    TURN_INDICATOR_OUTPUT_DISABLE,
    TURN_INDICATOR_OUTPUT_ENABLE_LEFT,
    TURN_INDICATOR_OUTPUT_ENABLE_RIGHT,
    TURN_INDICATOR_OUTPUT_KEEP,
)

from scenario_generation.reproducer_rollout import (
    clearance_family_block,
    road_border_collision_mask,
    strong_brake_block,
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


def _turn_indicator_block(states: list[int], asks: list[int | None]) -> dict:
    """Turn-indicator activity over the scored window.

    ``states`` is what the simulator relayed, which is what a scenario's indicator conditions
    judge. ``asks`` is the class the model chose before KEEP was resolved against the held
    state, so ``asked_off_steps`` separates "the signal stayed on" from "the model never asked
    to clear it" -- the two have nothing in common but the same relayed state. An ask is None
    for ticks before the first prediction.

    ``aggregate`` does not consume this block. The NPZ path has no indicator series, so
    requiring it there would fail every reproducer row.
    """
    on = [
        s in (TURN_INDICATOR_OUTPUT_ENABLE_LEFT, TURN_INDICATOR_OUTPUT_ENABLE_RIGHT) for s in states
    ]
    intervals = 0
    longest = 0
    run = 0
    for signalling in on:
        if not signalling:
            run = 0
            continue
        run += 1
        if run == 1:
            intervals += 1
        longest = max(longest, run)
    asked = [a for a in asks if a is not None and a != TURN_INDICATOR_OUTPUT_KEEP]
    return {
        "signal_steps": sum(on),
        "on_intervals": intervals,
        "longest_on_steps": longest,
        "asked_steps": len(asked),
        "asked_off_steps": sum(1 for a in asked if a == TURN_INDICATOR_OUTPUT_DISABLE),
    }


def build_segment_row(
    *,
    n_steps_run: int,
    terminated: str,
    result_kind: str,
    clearances: list[float],
    collisions: list[bool],
    rb_dists: np.ndarray,
    accels: np.ndarray,
    near_miss_thresh: float,
    strong_brake_mps2: float,
    progress_m: float,
    turn_indicators: list[int],
    turn_indicator_asks: list[int | None],
) -> dict:
    """Build one closed-loop segment row from a scenario_sim rollout's raw series.

    Every series covers the same frames: element ``k`` of ``clearances``, ``collisions``,
    ``rb_dists`` and ``accels`` is the same sim tick. The caller differences the speeds,
    because only it knows which sample precedes the window.

    ``n_steps_run`` is the length of those series, which is what ``aggregate`` divides its
    event counts by -- the same definition ``reproducer_rollout._finalize`` uses. Ticks the
    run executed but never scored belong in a flat diagnostic, not here.

    ``rb_dists`` may be empty when the map ships no road-border polylines; the road_border
    block then reports ``inf`` clearances and zero events, which is what a clearance block
    does for an all-inf series.

    The row is schema-complete as returned. Callers add their own flat diagnostic keys by
    merging into the result, never through this builder -- nothing here can overwrite a
    category block.
    """
    cl = np.asarray(clearances, dtype=np.float64)
    rb = np.asarray(rb_dists, dtype=np.float64)
    ac = np.asarray(accels, dtype=np.float64)

    return {
        "n_steps_run": int(n_steps_run),
        "terminated": terminated,
        "result_kind": result_kind,
        "progress_m": float(progress_m),
        "object": clearance_family_block(
            cl, np.asarray(collisions, dtype=bool), miss_thresh=near_miss_thresh
        ),
        "road_border": clearance_family_block(
            rb, road_border_collision_mask(rb), miss_thresh=near_miss_thresh
        ),
        "red_light_violation": _red_light_block(),
        "strong_brake": strong_brake_block(ac, strong_brake_mps2),
        "turn_indicator": _turn_indicator_block(turn_indicators, turn_indicator_asks),
        "reproducer": {**_NO_REPRODUCER_CURSOR, "normal_steps": int(n_steps_run)},
    }


def failed_segment_row(reason: str, near_miss_thresh: float) -> dict:
    """Schema-complete row for a scenario whose worker produced no output.

    Built by the same function as a real row so the two can never drift: ``aggregate`` raises
    on a missing block, so a crashed worker would otherwise take down the whole eval instead
    of being counted as a failure. Empty series give every block its "no samples" values.

    ``near_miss_thresh`` is the configured threshold, not NaN, because
    ``_event_family_block`` copies it straight into the summary as the run's threshold.
    """
    return {
        **build_segment_row(
            n_steps_run=0,
            terminated="worker_failed",
            result_kind="",
            clearances=[],
            collisions=[],
            rb_dists=np.zeros(0, dtype=np.float64),
            accels=np.zeros(0, dtype=np.float64),
            near_miss_thresh=near_miss_thresh,
            strong_brake_mps2=float("inf"),  # no threshold was applied
            progress_m=0.0,
            turn_indicators=[],
            turn_indicator_asks=[],
        ),
        "error": reason,
    }
