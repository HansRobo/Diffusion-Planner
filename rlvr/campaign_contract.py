"""Single source of truth for the PlannerRFT/AWR campaign's numeric contract.

Why this module exists
----------------------
Every campaign used to be a *forked copy* of a 300-400 line shell entrypoint
with its constants edited in place.  Three separate defects on 2026-07-26 came
straight out of that:

* ``beta`` was 1 in the AWR config, 2 in the supervisor's ``REPLAY_BETA``, and
  hardcoded 2 in the entrypoint's overlay call — so cycle 1 trained with the
  weighting the beta probe had already ranked worst.
* The conditional-commit alpha ladder lived in a third script and never
  contained ``0.05``, the only step size with an audited positive result.
* Corpus-size constants were literals in two scripts, so changing the
  oversampling lists failed a cache-shape check hours into a mine.

Shell callers get the same values via ``python -m rlvr.campaign_contract --shell``
and ``eval``, so there is exactly one place to change any of them.

Every value below is tied to recorded evidence; do not adjust one without
re-reading the note next to it.
"""

from __future__ import annotations

import argparse
import math

from preference_optimization.utils import (
    X2_LEGACY_EGO_WIDTH_M,
    XX1_LEGACY_EGO_WIDTH_M,
)

# --- rollout / weighting -----------------------------------------------------

#: AWR reward-weight temperature in ``exp(beta * group-z-score)``.
#: The paired beta probe (same cache, same seed, same updates, 65,536-scene
#: fixed selector) ranked 2 WORST of {0.5, 1, 2} on both reward delta
#: (-1.94e-4 vs -1.14e-4) and kinematic violations (+9.2e-5 vs 0); both earlier
#: campaigns' probes selected 1, and 1 matches the clean public NAVSIM release.
#: See docs/awr_zero_risk_improvement_audit_20260718.md.
REPLAY_BETA = 1

#: Candidates per scene group.  K=10 follows the public NAVSIM release; the
#: stale "G=8, beta=0.5" description must not be used to pick either value.
GROUP_SIZE = 10

#: Positive-advantage margin: a candidate must beat the behaviour anchor by this
#: much to become an active replay target.
POSITIVE_ADVANTAGE_MARGIN = 0.01

# --- conditional commit ------------------------------------------------------

#: Policy-interpolation steps searched when committing a cycle, smallest first.
#: 0.05 is the only step with an audited positive, safety-preserving result: the
#: unscaled policy overshoots.  It was missing from the original
#: (0.10, 0.25, 0.50, 0.75) ladder, so every cycle committed at 0.10-1.0 and
#: collisions rose ~15% relative.
COMMIT_ALPHA_LADDER = (0.02, 0.05, 0.10, 0.25, 0.50, 0.75)

#: Take the SMALLEST strictly-improving step rather than the highest-scoring one.
#: The commit selector is 128 scenes, so inter-alpha differences of ~1e-4 are
#: noise; maximising on that noise is what promoted alpha=0.50-1.0.
COMMIT_STEP_PREFERENCE = "smallest_strictly_improving_alpha"

#: Committing the raw, unscaled replay policy (alpha=1.0) requires an explicit
#: opt-in; it is never the default.
ALLOW_UNSCALED_COMMIT = False

# --- standstill steering gate ------------------------------------------------

#: Reject a low-speed non-anchor candidate whose first step implies a larger
#: front-wheel angle than this.  delta = atan(wheel_base * 2|y| / s**2).
#: Absolute displacement/lateral limits cannot express steering feasibility: a
#: 3.9 cm step with 0.91 cm lateral offset implies ~88 deg yet passes both.
FIRST_WAYPOINT_MAX_IMPLIED_STEER_RAD = 0.64

#: Below this first-step length the geometry is pure sampler noise.  The earlier
#: 0.05 m floor sat ABOVE the measured stop-turn p95 first step (0.039 m), which
#: switched the whole gate off at standstill — the rejection rate was 0 across
#: four cycles.  Verified: 5 mm yields 32.4% low-speed rejection vs 2.7% before.
FIRST_WAYPOINT_TANGENT_MIN_STEP_M = 0.005

#: Speed below which the first-waypoint gate applies at all.
FIRST_WAYPOINT_GATE_SPEED_MPS = 1.0

# --- per-cycle non-regression veto -------------------------------------------

#: The headline reward must be a real improvement, not noise.
VETO_REWARD_MIN_PROBABILITY_IMPROVED = 0.95

#: A safety metric more than this-confidently worse blocks the commit, however
#: much the aggregate reward rose.  Cycles 1-4 committed with
#: P(collision improved) = 0.003 because the audit was report-only.
VETO_SAFETY_MIN_PROBABILITY_IMPROVED = 0.10

#: Jitter is gated on the *fraction* of stop-turn scenes commanding an
#: infeasible angle, not a p95: atan() saturates at pi/2 and sub-floor steps
#: report 0, making the per-scene distribution bimodal.  Baseline on the audited
#: start checkpoint: 0.6593 over 3,452 stop-turn scenes.
VETO_JITTER_TOLERANCE = 0.005

# --- corpus ------------------------------------------------------------------

#: Global rollout batch = ranks x per-rank scene batch.  The corpus is padded up
#: to a whole number of these.
WORLD_SIZE = 8
ROLLOUT_SCENE_BATCH_SIZE = 192
GLOBAL_ROLLOUT_BATCH = WORLD_SIZE * ROLLOUT_SCENE_BATCH_SIZE

#: Times each right-turn oversampling list is repeated into the training set.
EXTRA_TRAIN_REPEAT = 10


def padded_group_count(source_scenes: int) -> int:
    """Scene-group count after DDP tail padding."""

    if source_scenes <= 0:
        raise ValueError(f"source_scenes must be positive, got {source_scenes}")
    batches = math.ceil(source_scenes / GLOBAL_ROLLOUT_BATCH)
    return batches * GLOBAL_ROLLOUT_BATCH


def per_rank_group_count(source_scenes: int) -> int:
    return padded_group_count(source_scenes) // WORLD_SIZE


def shell_assignments() -> str:
    """Emit the contract as shell assignments for ``eval "$(...)"``.

    Keeps the shell entrypoints from re-declaring values that already live here.
    """

    ladder = " ".join(f"{a:g}" for a in COMMIT_ALPHA_LADDER)
    labels = " ".join(f"a{a:g}".replace(".", "p") for a in COMMIT_ALPHA_LADDER)
    lines = [
        f"REPLAY_BETA={REPLAY_BETA:g}",
        f"GROUP_SIZE={GROUP_SIZE}",
        f"POSITIVE_ADVANTAGE_MARGIN={POSITIVE_ADVANTAGE_MARGIN:g}",
        f"COMMIT_ALPHAS=({ladder})",
        f"COMMIT_ALPHA_LABELS=({labels})",
        f"ALLOW_UNSCALED_COMMIT={1 if ALLOW_UNSCALED_COMMIT else 0}",
        f"FIRST_WAYPOINT_MAX_IMPLIED_STEER_RAD={FIRST_WAYPOINT_MAX_IMPLIED_STEER_RAD:g}",
        f"FIRST_WAYPOINT_TANGENT_MIN_STEP_M={FIRST_WAYPOINT_TANGENT_MIN_STEP_M:g}",
        f"WORLD_SIZE={WORLD_SIZE}",
        f"ROLLOUT_SCENE_BATCH_SIZE={ROLLOUT_SCENE_BATCH_SIZE}",
        f"GLOBAL_ROLLOUT_BATCH={GLOBAL_ROLLOUT_BATCH}",
        f"EXTRA_TRAIN_REPEAT={EXTRA_TRAIN_REPEAT}",
        f"X2_LEGACY_EGO_WIDTH_M={X2_LEGACY_EGO_WIDTH_M:g}",
        f"XX1_LEGACY_EGO_WIDTH_M={XX1_LEGACY_EGO_WIDTH_M:g}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shell",
        action="store_true",
        help="emit shell assignments instead of a human-readable dump",
    )
    args = parser.parse_args()
    if args.shell:
        print(shell_assignments())
        return 0
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_"):
            print(f"{name} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
