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

``shell_assignments`` exists so shell callers can ``eval`` these values instead of
re-declaring them.  Be aware that today almost nothing does: the live entrypoints
pull only ``BEHAVIOR_ANCHOR_WEIGHT``, via a direct ``python -c`` import, and no
script evals ``--shell`` at all.  So this module is authoritative for Python and
*documentation* for the shell chain -- which is exactly how ``EXTRA_TRAIN_REPEAT``
came to say 10 while both entrypoints ran 1.  When you change a value here, grep
the scripts for it.

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

#: Weight on the deterministic behaviour trajectory as a *retention* target.
#:
#: 0 = the value the only run that ever produced a cycle-level gain used
#: (plannerrft_prefix_full_cycles02_to10_e100, +0.00108 reward over cycles 1-3).
#:
#: A previous note here claimed 0 was the cause of safety degrading while the
#: aggregate reward rose.  That is refuted: the validated run ran anchor=0 for
#: all 41 epochs and its cycle-5 epochs show the identical pattern (collision
#: 0.001686 -> 0.001924, reward peaking then turning negative).  The
#: reward-up/safety-down trade is inherent to the method; the validated run
#: survives it via cycle-level max-selection, not via an anchor.
#:
#: The measurement that motivated a positive value still stands on its own --
#: with EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=1 only 18.3% of scenes contribute to
#: the loss, and any positive value lifts that to 98.3% without touching the
#: 33,880 improvement targets.  But it has never been shown to beat 0 at the
#: cycle level.  Treat it as a single-variable experiment against this baseline,
#: not as part of it.  0.5 is the cap: 1.0 puts the anchor's total weight at
#: 1.08x the improvement targets' and drowns the signal.
BEHAVIOR_ANCHOR_WEIGHT = 0.0
BEHAVIOR_ANCHOR_WEIGHT_MAX = 0.5

#: Margin by which the logged human must beat the deployed deterministic output
#: before the overlay keeps its expert_safe flag.  None passes the flag through
#: as mined, which is what every cycle up to and including cycle 1 of
#: plannerrft_rewardfix_v3 ran with; 0.01 (= POSITIVE_ADVANTAGE_MARGIN) is the
#: default from cycle 2 on.  The supervisor repeats the number rather than
#: reading it from here -- run_plannerrft_full_to_epoch100.sh does not eval this
#: module -- so the two must be changed together.
#:
#: Measured over all 5,446,656 groups of
#: plannerrft_full_cycle01_mine/20260720-050941 (full census, not a sample):
#:
#:   19.45%  of groups hold a candidate that beats the deterministic output by
#:           POSITIVE_ADVANTAGE_MARGIN, i.e. 80.55% of the corpus trains on
#:           nothing.  This is why replay epochs are nearly inert: the last full
#:           cycle moved train_selection_reward_before_epoch by +0.00024 across
#:           nine of them.
#:   42.52%  of dead *and* expert_safe groups have a logged human that beats the
#:           deterministic output by >0.01, mean gap +0.029 where positive --
#:           about twice the typical within-group candidate headroom
#:           (0.013-0.017).  That target is already cached by the mine.
#:   +32.87pp coverage if those groups anchor on the human: 19.45% -> 52.31%.
#:    0.46%  of groups are dead, safe, and expert-*worse*; gating on
#:           expert > deterministic + margin drops exactly those, so the
#:           coverage is bought without a regression channel.
#:
#: Mass check against the "1.08x drowns the signal" bound above: improvement mass
#: is 0.6555/group (weight_sum 4,019,080.5 over 2,914,556 active targets).  The
#: expert anchor at 0.4 contributes ~0.0743/group today (11.3%) and ~0.2058/group
#: (31.4%) once ungated.  Well inside the bound.
#:
#: That census cache predates db5ad350, so the absolute dead fraction was
#: expected to shift under today's reward.  Re-measured on the first 663,552
#: groups of the live plannerrft_rewardfix_v3 cycle-1 mine, which runs the fixed
#: reward: mean deterministic reward 0.8922 (was 0.9319), mean best-vs-
#: deterministic gain +0.0137 (was +0.0077), trainable 28.92% (was 19.45%).  The
#: fixed reward does discriminate between candidates more, but it does not close
#: the gap: 71.08% of groups still train on nothing, and of the dead-and-safe
#: ones 44.47% have a better logged human -- the 42.52% reproduces.  Coverage
#: with the gate is 28.92% -> 59.26%, so context decoding costs 2.05x, not 2.69x.
#: Those rows are the corpus in mining order, not a random sample.
EXPERT_IMPROVES_MARGIN: float | None = 0.01

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

#: How the three unprotected-right-turn lists get their 10x emphasis.
#:
#: This said REPEAT = 10 while both live entrypoints defaulted to 1, and nothing
#: read this value, so the 10 was wrong and dead at the same time.  What actually
#: runs is repeat=1 with the 10x applied as replay weight.  The invariant is the
#: PRODUCT: REPEAT * WEIGHT = 10 keeps the total gradient mass fixed however the
#: emphasis is split.
#:
#: The split is not cosmetic.  Weight only amplifies a scene that already has a
#: positive-advantage target; repeats give it that many independent mining
#: attempts.  Measured on the live cycle-1 mine over 1,118,400 rows, of which
#: 13,798 are right-turn scenes: they are trainable 28.04% of the time, which is
#: no worse than the rest of the corpus (28.98%), so they carry no special
#: deficit -- and the expert-improves gate lifts them to 53.97%.  At 28% per
#: attempt, repeat=10 would reach ~97% coverage, but it costs 10x the mine time
#: for those scenes (~+1.2 h per cycle) and it is a second variable.  Cycle 2
#: already tests the gate alone, so repeats are the next single-variable
#: candidate, not part of this baseline.
EXTRA_TRAIN_REPEAT = 1
EXTRA_TRAIN_WEIGHT = 10


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
        f"BEHAVIOR_ANCHOR_WEIGHT={BEHAVIOR_ANCHOR_WEIGHT:g}",
        f"COMMIT_ALPHAS=({ladder})",
        f"COMMIT_ALPHA_LABELS=({labels})",
        f"ALLOW_UNSCALED_COMMIT={1 if ALLOW_UNSCALED_COMMIT else 0}",
        f"FIRST_WAYPOINT_MAX_IMPLIED_STEER_RAD={FIRST_WAYPOINT_MAX_IMPLIED_STEER_RAD:g}",
        f"FIRST_WAYPOINT_TANGENT_MIN_STEP_M={FIRST_WAYPOINT_TANGENT_MIN_STEP_M:g}",
        f"WORLD_SIZE={WORLD_SIZE}",
        f"ROLLOUT_SCENE_BATCH_SIZE={ROLLOUT_SCENE_BATCH_SIZE}",
        f"GLOBAL_ROLLOUT_BATCH={GLOBAL_ROLLOUT_BATCH}",
        f"EXTRA_TRAIN_REPEAT={EXTRA_TRAIN_REPEAT}",
        f"EXTRA_TRAIN_WEIGHT={EXTRA_TRAIN_WEIGHT}",
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
