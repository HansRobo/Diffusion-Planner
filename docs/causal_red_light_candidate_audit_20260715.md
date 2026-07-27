# Causal Red-Light Candidate Audit

This audit answers a narrow data question: when the original list filter found
an ego stopped at a red route lane and a future motion onset, is that a
red-to-green release, a red-light violation, or an unrelated stop?

## Evidence and Criteria

The audit never labels a sample a violation from a red one-hot alone.  For each
candidate it:

1. selects the nearest forward red route geometry;
2. associates stop-line segments with that geometry and rejects parallel or
   perpendicular approaches using the route/stop-line direction check;
3. checks whether the logged expert future actually crosses that associated
   stop line; and
4. follows the same geometry through neighboring frames to determine whether
   it becomes green before or after the expert motion onset.

The original NPZs and source manifests are read-only.  The reports are audit
artifacts, not training manifests.

## Observed results

The three full right-turn lists contained 2,051 samples removed by the old
red-plus-motion detector.  All 2,051 matched the same route signal becoming
green before the expert motion onset under the strict stop-line association.
No right-turn candidate was confirmed to cross its associated stop line while
the matched signal remained red.

For a source-order, evenly spaced audit sample of 524 old base candidates from
the first 3.1 million non-overlapping base entries:

| classification | count | interpretation |
| --- | ---: | --- |
| same-geometry red to green before motion | 333 | causal future event; eligible for the strict filter |
| red at motion onset, no associated stop-line crossing | 157 | queue/creep or a downstream red; not a confirmed violation |
| green only after motion, no stop-line crossing | 16 | signal release follows the initial motion; retain conservatively |
| signal evidence unavailable/ambiguous, no crossing | 18 | route window/file evidence is insufficient; retain |

The sample had **zero** confirmed crossings of an associated stop line while
the matched signal remained red.  The 7 apparent red crossings found by the
first diagnostic were all removed after stop-line association; they were
crossings of unrelated nearby lines.

## Interpretation

The remaining red-at-onset records are not evidence of a bad driver label or
an automatic red-light violation.  In the audited data the expert has not yet
crossed the ego-route stop line within the 8-second target.  A red attribute can
also describe a farther downstream lanelet because `route_lanes` is a forward
route window, not a single current-stop-line observation.  A few sequences
show the signal turning green well before the driver actually releases the
brake; those are intentionally retained unless the release is temporally tied
to the motion onset.

The earlier v2 route-transition filter was intentionally conservative about
future green evidence, but the large-sample audit found that its ``any red``
predicate could delete a valid near-green start when a downstream route lane was
also red, and could retain a red/stopped/future-moving sample when no sibling
green frame was available. It is therefore not a production manifest. The v3
filter uses the current-frame nearest forward stop-line association directly:
nearest green, unresolved, and mixed-color cases are retained; only an
unambiguous nearest red signal plus stopped ego and sustained future motion is
removed. The v3 manifest must be regenerated and independently hashed before
training.

## Reproduction

The classifier is in
`diffusion_planner/util_scripts/audit_causal_red_light_candidates.py`.
The right-turn full report and the base audit sample are written below the
v1 audit artifact directory; they can be regenerated without modifying any
source NPZ or list.
