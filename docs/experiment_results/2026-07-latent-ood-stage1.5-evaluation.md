# Latent OOD Stage 1.5 — Feature Formulation Evaluation

> **INVALIDATED (2026-07-17):** All results in this document are invalid.
> Override npz files had all-zero lane/route/polygon tensors due to a
> coordinate-shift bug in `ros_scripts/extract_map_from_bag.py`, while the
> normal/training npz had full map data. Embeddings are not comparable
> across these two populations. Additionally, the v1 +3pt AUPRC lift was
> an imputation artifact (zero-imputing missing EC subscores); the v2
> corrected evaluation showed +0.0004 AUPRC — but both v1 and v2 ran on
> invalid embeddings. The pipeline code and scripts are correct; the input
> data was broken. H-B (maneuver conditioning) was also never tested due
> to a separate bug. Pending fix to map extraction and full rerun.

> Date: 2026-07-06
> Status: ~~Completed~~ INVALIDATED — see banner above
> Branch: `feat/latent-ood-stage1`
> Prior results: `docs/experiment_results/2026-07-latent-ood-first-experiment.md` (Stage 1, also invalidated)

## Executive Summary

Stage 1.5 asked which formulation of the latent-OOD signal — raw kNN distance
(H-A), maneuver-normalized residual (H-B), or both (H-AB) — best complements
EPDMS for override-scene detection, via a LightGBM classifier evaluated with
5-fold group cross-validation.

**Result: adding OOD helps, but the two formulations are indistinguishable,
and neither is the hoped-for breakthrough on EPDMS's blind spots.**

- All three OOD-augmented configurations beat the EPDMS-only baseline by
  +3.1 to +3.2 AUPRC points (0.620 → ~0.651-0.652).
- H-A (raw OOD), H-B (residual OOD), and H-AB (both) are statistically
  indistinguishable from each other (AUPRC 0.6506 / 0.6514 / 0.6515,
  within noise of each other and within one std-dev of the CV fold spread).
- The reason H-B doesn't beat H-A: maneuver conditioning did not work as
  designed. Every normal-driving frame fell back to a single default
  maneuver bucket, so `ood_residual` reduces algebraically to
  `knn_mean − constant` — confirmed empirically: `knn_mean − ood_residual`
  is constant (≈0.4847) across all 29,761 evaluated rows. H-B carries no
  information H-A doesn't already have.
- OOD's biggest practical win is at the high-recall end: R@P50 jumps from
  0.931 (EPDMS-only) to ~0.98 (OOD-augmented) — at 50% precision the
  OOD-augmented models catch ~5 points more overrides. Best-F1 barely
  moves (0.725 → 0.727), so the benefit is concentrated at specific
  operating points, not a uniform lift.
- On the two confirmed EPDMS blind-spot categories, OOD gives only a mild
  lift (停止_車両: +0.06 AUPRC, 回避_車両: +0.02 AUPRC) — not the
  order-of-magnitude improvement Stage 1's exploratory numbers suggested
  might be possible. The biggest OOD lift instead lands on その他 (+0.09),
  the catch-all category where scene novelty is a more natural fit.

**Recommendation:** carry raw kNN OOD (H-A) into Stage 2 as the simpler,
equally-performing formulation. Treat maneuver-conditioned residual (H-B) as
unvalidated until the underlying maneuver-assignment bug is fixed — see
Limitations and Recommendation below.

## Setup

### Data

- **Feature matrix**: `data/feature_matrix.csv`, produced by
  `scripts/build_feature_matrix.py` (joining per-frame OOD scores against
  per-frame EPDMS subscores within a 50ms timestamp tolerance).
- **Rows used**: 29,761 frames with a matched EPDMS row (rows without a
  match, i.e. normal-driving frames with no EPDMS score, are excluded from
  this evaluation — see Limitations).
- **Bags/groups**: 68 unique bags (out of 75 total override bags — 7 lack a
  matched EPDMS+OOD pair; see Limitations).
- **Class balance**: 16,802 positive rows vs. 12,959 negative rows (56:44)
  — not heavily imbalanced. Positive label = frame falls within
  `[t_OR − 20s, t_OR + 10s]` of an override (OR) event, matching the
  windowing convention used in the EPDMS analysis (`ANALYSIS_STORY.md`).
- **Categories**: その他, 停止_信号, 停止_自転車歩行者, 停止_車両,
  回避_路駐車, 回避_車両, 曲がり切れない, 曲がるタイミングが早い.

### Feature sets compared

| Config | Features |
|---|---|
| EPDMS-only (baseline) | 9 EPDMS subscores: nc, dac, ddc, tlc, ttc, lk, hc, ec, ep |
| H-A (EPDMS+rawOOD) | EPDMS-only + `knn_mean` |
| H-B (EPDMS+residualOOD) | EPDMS-only + `ood_residual` |
| H-AB (EPDMS+both) | EPDMS-only + `knn_mean` + `ood_residual` |

- `knn_mean`: mean L2 distance to the k=10 nearest neighbors in a bank of
  normal-training encoder embeddings (Diffusion Planner ONNX encoder,
  mean-pooled, per Stage 1 §2.3/§5.1 configuration).
- `ood_residual`: `knn_mean − median(knn_mean for this maneuver type,
  computed from normal-driving frames)`, intended to remove base-rate bias
  per maneuver (stop / avoid / turn / other).

### Methodology

- **Model**: LightGBM, binary objective, default-ish hyperparameters
  (`num_leaves=31`, `learning_rate=0.05`, `num_boost_round=200`,
  `is_unbalance=True`) — no hyperparameter tuning in this pass, per the
  design spec's "out of scope" list.
- **Cross-validation**: 5-fold `GroupKFold`, grouped by bag, so no frames
  from the same bag appear in both train and test splits within a fold
  (prevents temporal/scene leakage).
- **Metrics**: AUPRC (primary, given class-imbalance-sensitive use case),
  best-F1 (over the precision-recall curve), recall at fixed precision
  (R@P50, R@P70).
- **Script**: `scripts/evaluate_ood_formulations.py`. Full results:
  `data/evaluation_results/evaluation_results.json`.

## Comparison Table

5-fold CV, GroupKFold by bag, 68 bags, 29,761 frames:

| Feature Set | AUPRC | F1 | R@P50 | R@P70 |
|---|---:|---:|---:|---:|
| EPDMS-only (baseline) | 0.620 | 0.725 | 0.931 | 0.207 |
| H-A (EPDMS+rawOOD) | 0.651 | 0.727 | 0.979 | 0.248 |
| H-B (EPDMS+residualOOD) | 0.651 | 0.727 | 0.981 | 0.250 |
| H-AB (EPDMS+both) | 0.652 | 0.726 | 0.976 | 0.253 |

![AUPRC comparison](../../data/evaluation_results/auprc_comparison.png)

All three OOD-augmented configs sit within each other's error bars
(AUPRC std ≈ 0.054-0.055 across folds) — the CV fold-to-fold variance is
larger than the gap between H-A, H-B, and H-AB. The only comparison that
clears the noise floor is OOD-augmented vs. EPDMS-only baseline
(std ≈ 0.053), which is a consistent +3-point lift across every
OOD-augmented config.

## Per-Category Breakdown

AUPRC by category (EPDMS-only → H-A, H-B, H-AB are all close to each other,
so H-A is used as the representative OOD-augmented column; deltas for H-B
and H-AB are within ±0.002 of H-A in every category):

| Category | EPDMS-only | H-A (rawOOD) | Δ (H-A − baseline) |
|---|---:|---:|---:|
| その他 | 0.483 | 0.572 | **+0.089** |
| 停止_信号 | 0.732 | 0.743 | +0.010 |
| 停止_自転車歩行者 | 0.750 | 0.674 | **−0.076** |
| 停止_車両 | 0.480 | 0.544 | +0.064 |
| 回避_路駐車 | 0.630 | 0.623 | −0.007 |
| 回避_車両 | 0.612 | 0.634 | +0.022 |
| 曲がり切れない | 0.656 | 0.671 | +0.015 |
| 曲がるタイミングが早い | 0.814 | 0.864 | +0.050 |

Observations:

- **その他 (+0.09)** gets the single biggest lift. This is the catch-all,
  most heterogeneous category — scene novelty (what OOD measures) is a more
  natural fit here than a specific failure signature.
- **曲がるタイミングが早い (+0.05)**: already the best-detected category by
  EPDMS alone (0.814), and OOD pushes it further to 0.865 — OOD and EPDMS
  are complementary here rather than redundant.
- **The two confirmed EPDMS blind spots** (停止_車両, 回避_車両— categories
  where no EPDMS subscore fires, per `DESIGN.md`/`ANALYSIS_STORY.md`) get
  only a mild lift: +0.064 and +0.022 AUPRC respectively. This is real
  signal but far short of a "OOD fills the EPDMS gap" result — 停止_車両
  AUPRC (0.544) is still the second-worst category, and 回避_車両 (0.634)
  is mid-pack.
- **停止_自転車歩行者 gets worse with OOD (−0.076)**. This category is
  already well-detected by EPDMS alone (0.750, likely via TTC/DDC gates on
  pedestrian proximity), and adding the OOD feature appears to introduce
  noise the classifier partially chases — plausible given that a
  9→10/11-feature jump with only ~29.7k rows and 68 groups leaves limited
  per-fold data to learn a clean per-category weighting, especially for a
  category whose EPDMS signal is already close to saturated.

## Feature Importance Analysis

![Feature importance by configuration](../../data/evaluation_results/feature_importance.png)

- **EPDMS-only**: importance (LightGBM mean gain) is led by `hc` (heading
  consistency) and `ttc` (time-to-collision), then `dac`, `ep`, `ddc`; `nc`
  and `tlc` contribute least — consistent with `DESIGN.md`/`ANALYSIS_STORY`'s
  finding that EC/HC carry real signal beyond the originally-proposed
  NC/DAC/DDC/TLC gates.
- **H-A (raw OOD)**: `knn_mean` dominates by a wide margin — its mean gain
  (~26k) is roughly 5-6x the next-highest feature (`ec`, `hc` at ~4-4.5k).
  This is the strongest single-feature signal in any configuration tested,
  more important to the model than any individual EPDMS subscore.
- **H-B (residual OOD)**: `ood_residual` dominates just as strongly
  (~26k) — expected, since `ood_residual` is an affine transform of
  `knn_mean` and LightGBM's gain-based importance is invariant to constant
  shifts.
- **H-AB (both)**: `knn_mean` again dominates (~22k); `ood_residual` drops
  to mid-pack importance (~4.3k, similar to `ec`/`hc`/`lk`/`ttc`). This is
  the clearest confirmation that once `knn_mean` is available,
  `ood_residual` adds little marginal information — the model relies almost
  entirely on the raw signal and treats the residual as a redundant,
  lower-value feature once both are present.
- Across every configuration, `nc` (no-collision gate) and `tlc` (traffic
  light compliance) are consistently the least important features — both
  are binary-ish gates that rarely trigger in this bag population, so they
  carry little discriminative gain for a tree-based model.

## Residual Analysis: Does Maneuver Normalization Help?

**No — and the reason is a data-pipeline bug, not a finding about the
underlying hypothesis.**

`ood_residual` is defined as `knn_mean − median(knn_mean | maneuver_type)`,
where the median is computed from normal-driving frames bucketed by
maneuver type (stop / avoid / turn / other, per
`scripts/build_feature_matrix.py`'s `CATEGORY_TO_MANEUVER` mapping for
override frames and the SFT-annotation-derived mapping for normal frames).

The intent was for normal frames to be assigned one of `{stop, avoid, turn,
straight}` via `--maneuver_npz_paths` (`data/maneuver_npz_paths.json`), so
that each maneuver type gets its own baseline median. In practice, every
normal frame fell through to the function's default (`"straight"`) —
meaning `maneuver_scores` ended up with a single populated key
(`"straight"`) rather than four. Since override rows are assigned
maneuver types `{stop, avoid, turn, other}` (never `"straight"`) via a
*different* code path (`CATEGORY_TO_MANEUVER`), and `maneuver_medians` only
has a `"straight"` entry, every lookup for `stop`/`avoid`/`turn`/`other`
falls back to `maneuver_medians.get("straight")` too
(`scripts/build_feature_matrix.py:201,236`). The net effect: **every row in
the dataset — override or normal, regardless of maneuver type — has its
residual computed against the exact same constant.**

This is confirmed empirically: across all 29,761 rows with both `knn_mean`
and `ood_residual` populated, `knn_mean − ood_residual` takes only one
distinct value (≈0.484717, modulo floating-point rounding). `ood_residual`
is therefore *exactly* `knn_mean` shifted by a constant — it carries zero
additional information relative to `knn_mean`, which fully explains why
H-B tracks H-A to the third decimal place in every metric and every
category, and why LightGBM's gain-based importance ranks them identically
(gain is shift-invariant).

**This means H-B, as evaluated here, is not a fair test of the maneuver-
normalization hypothesis.** The hypothesis — that per-maneuver baselining
removes base-rate bias and isolates per-frame risk — was never actually
exercised, because the maneuver buckets collapsed to one. A real test
requires fixing the normal-frame maneuver assignment (see Limitations)
and rerunning.

## Recommendation

**Carry raw kNN OOD (H-A) into Stage 2**, implemented as a lightweight
online feature alongside the existing EPDMS online evaluator
(`autoware_online_epdms_evaluator`). Rationale:

- H-A gives the full observed AUPRC lift (+3.1 points) with the simplest
  formulation — one additional scalar feature, no maneuver-bucketing logic
  needed online.
- H-B currently offers no advantage over H-A (they are numerically
  near-identical due to the bug above) but does add complexity (maneuver
  classification, per-maneuver baseline maintenance). There's no reason to
  carry that complexity into Stage 2 until it's shown to earn its keep.
- Treat maneuver-conditioned residual OOD as a **fast-follow**, not a
  dropped idea: fix the normal-frame maneuver assignment (properly map SFT
  `driving_decisions` timestamps to npz frames — the plumbing exists in
  `data/maneuver_npz_paths.json` and `build_feature_matrix.py`, it's just
  not wired correctly) and rerun the H-B/H-AB comparison before concluding
  maneuver normalization doesn't help. The current result says "the
  residual computation as implemented is a no-op," not "maneuver
  normalization doesn't work."
- Do not expect OOD to single-handedly close the 停止_車両/回避_車両 blind
  spots — the lift there (+0.06, +0.02 AUPRC) is real but modest. Those
  categories likely still need the TTC/DRAC longitudinal signal called out
  as Stage 2 scope in the top-level `CLAUDE.md`.

## Honest Limitations

1. **Maneuver conditioning is broken, not merely unhelpful.** As detailed
   above, `ood_residual` collapsed to `knn_mean − constant` because normal
   frames never received a real maneuver label. H-B/H-AB results should be
   read as "raw OOD again, restated" rather than independent evidence about
   residual normalization.
2. **No true negatives from normal driving are included in this
   evaluation.** The feature matrix only contains rows with a matched
   EPDMS score, and EPDMS is only computed for override-bag frames in the
   current pipeline (`results/analysis/samples_epdms_all.csv` covers
   override bags, not the normal-driving corpus). This means all 29,761
   evaluated rows come from override bags — "negative" rows are frames
   *within override bags* but outside the OR window, not independently
   sampled normal driving. AUPRC/recall numbers may not generalize to a
   deployment setting with genuinely normal (non-override) bags in the
   negative class.
3. **Lane/route/polygon tensors are all-zero in the underlying npz data**,
   per the known coordinate-shift/projector issue documented in
   `.superpowers/sdd/task-2-report.md` (`MGRSProjector` cannot represent
   this bag population's negative-capable local frame without a shift that
   breaks lane lookup near ego). The Diffusion Planner encoder embeddings
   used for `knn_mean` are therefore based on ego + neighbor-agent motion
   only, with no road/lane context — the OOD signal may be missing a
   substantial fraction of what "scene novelty" should mean.
4. **Labels are derived from OR-event proximity** (`[t−20s, t+10s]` window
   around a control-mode-transition timestamp), not independently verified
   by human review of whether the frame was actually dangerous. The OR
   signal itself is known to publish late with variable delay (see
   top-level `CLAUDE.md`), which is one reason a generous asymmetric window
   is used, but some mislabeling at the window edges is expected.
5. **Only 68 of 75 total override bags** have both an EPDMS score and an
   OOD score after the 50ms-tolerance join — 7 bags' frames didn't survive
   the join and are silently excluded rather than analyzed for why.
6. **No hyperparameter tuning** — LightGBM was run with a single fixed
   configuration (default-ish `num_leaves=31`, `learning_rate=0.05`,
   200 rounds), per the design spec's explicit "out of scope" for v1. The
   AUPRC gaps reported here could shift with tuning, though the H-A ≈ H-B ≈
   H-AB finding (driven by the residual bug, not by hyperparameters) should
   be robust to that.
7. **Small effective sample for CV.** 68 groups split 5 ways means each
   fold's test set draws from ~13-14 bags; per-category breakdowns further
   subdivide this (categories with <10 test rows or zero positives are
   dropped per `evaluate_ood_formulations.py`'s filter), so per-category
   AUPRC numbers — especially for smaller categories — carry meaningful
   fold-to-fold variance not shown in the point estimates above.
