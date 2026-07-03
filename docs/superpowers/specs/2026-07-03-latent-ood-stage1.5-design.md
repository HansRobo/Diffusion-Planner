# Latent OOD Stage 1.5 — Feature Formulation Evaluation

## Goal

Determine which formulation of the latent OOD signal best complements EPDMS
for override-scene detection, using a LightGBM classifier evaluated on
AUPRC and per-category F1.

Two hypotheses compete:

- **H-A (raw OOD):** The absolute kNN distance from the normal-training bank
  carries risk signal even without maneuver normalization. The classifier
  learns to weight it appropriately per context.
- **H-B (residual OOD):** Per-maneuver-type normalization
  (`score − median(scores for this maneuver type)`) removes the base-rate
  bias and isolates the per-frame risk component.

Both are evaluated as features alongside EPDMS subscores. The winner is
whichever formulation yields higher AUPRC lift over EPDMS-only baseline.

## Context — What Stage 1 Proved

The Stage 1 experiment (2026-07-03) established:

1. Override scenes score 1.38x higher OOD than normal training data
   (p < 10^-281), but this is partially base-rate bias.
2. **After maneuver-type control:**
   - 停止_車両 (stop for vehicle): 1.24x, p < 10^-60 — real signal
   - 曲がり切れない (can't complete turn): 1.14x, p < 10^-14 — real signal
   - 回避_車両 (vehicle avoidance): 1.01x, p = 0.17 — **no signal**
3. Per-frame temporal analysis (OOD vs time-relative-to-OR) shows no
   OR-aligned spike in absolute scores — the route/area offset dominates.
4. Residual analysis (score − rolling median) shows no significant
   per-frame spike either, but this was tested on only 3 bags.

## Data

### Override scenes (positive class)

- Source: `/mnt/nas/private_workspace/chenglin/ORScene_bags/` (174 bags, read-only)
- Already converted (Stage 1): 3 bags → 1,454 npz
- Need to convert: remaining 171 bags (estimate ~80k npz total)
- OR timestamps: `results/override_transitions.json` (98 events across bags)
- Categories: 停止_車両, 回避_車両, 曲がり切れない, 停止_信号, 停止_自転車歩行者,
  回避_路駐車, 曲がるタイミングが早い, その他

### Normal scenes (negative class)

- Source: sakurab:`/mnt/nvme/dataset/20260425_takanawa_full/x2_dev/2355_Takanawa_gateway_copied_from_Aisantec/`
  (501k npz, SSH key: `id_ed25519_sakuraDatacentric`)
- Maneuver annotations: `/home/chenglin/workspace/Diffusion-Planner-Meta-Repository/dataset/sft_filtering_json/x2_dev/`
  (7,745 annotated scenes with `driving_decisions` per scene, timestamps map to npz)
- Already pulled (Stage 1): 1,000 random normal + 800 maneuver-matched

### kNN Bank

- Built from 1,000 random normal training npz (Stage 1)
- For Stage 1.5: rebuild with larger sample (5k–10k) for tighter distance estimates
- Model: `/opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx`

### EPDMS features

- Existing per-frame EPDMS subscores: `results/analysis/samples_epdms_all.csv`
  (86,463 frames, 157 bags, columns: nc, dac, ddc, tlc, ttc, lk, hc, ec, ep, epdms)
- These are from the filtered-trajectory evaluator run, covering the same
  override bags

## Labeling

**Binary label (primary):** A frame is positive if it falls within a window
around an OR event. Window: [t − 20s, t + 10s], matching the EPDMS analysis
convention from ANALYSIS_STORY.md.

**Per-category label (secondary):** Same binary label but evaluated separately
per failure_mode category (停止_車両, 回避_車両, etc.) to understand where
each feature formulation helps.

## Feature Sets to Evaluate

### Baseline: EPDMS-only
Features: 9 EPDMS subscores (NC, DAC, DDC, TLC, TTC, LK, HC, EC, EP)

### H-A: EPDMS + raw OOD
Features: 9 EPDMS subscores + `knn_mean` (absolute OOD score)

### H-B: EPDMS + residual OOD
Features: 9 EPDMS subscores + `ood_residual` (score − median for maneuver type)

### H-AB: EPDMS + both OOD formulations
Features: 9 EPDMS subscores + `knn_mean` + `ood_residual`

## Evaluation

### Metrics
- AUPRC (primary — class imbalance expected)
- F1 at optimal threshold
- Recall at fixed precision (0.50, 0.70)
- Per-category breakdown of above

### Methodology
- 5-fold cross-validation, stratified by bag (no frames from the same bag in
  both train and test folds — prevents temporal leakage)
- LightGBM with default hyperparameters (no tuning in v1)
- Feature importance analysis (SHAP or built-in) to understand which features
  the classifier relies on

### Success criteria

**Minimum useful:**
- H-A or H-B improves AUPRC over EPDMS-only baseline by > 0.05
- The improvement is driven by categories where EPDMS is blind (停止_車両, 回避_車両)

**Strong:**
- H-B > H-A (maneuver normalization helps)
- Per-category analysis shows OOD feature has high importance for
  停止_車両/曲がり切れない but low importance for 回避_車両 (matching Stage 1 findings)

## Joining OOD scores with EPDMS

The OOD scores are per-npz (keyed by npz path with nanosecond timestamps
in companion JSONs). The EPDMS scores are per-frame (keyed by bag name +
ROS timestamp in seconds). Join by:

1. Parse bag name from npz path
2. Convert npz timestamp (nanoseconds) to seconds
3. Match to nearest EPDMS frame within 50ms tolerance

Frames without a match in either direction are dropped.

## Maneuver Type Assignment (for H-B)

For normal scenes: use SFT annotation JSONs — each scene has
`[start_time, end_time]` in nanoseconds and `driving_decisions` with
lateral/longitudinal categories.

For override scenes: assign based on `failure_mode` directory:
- 停止_車両, 停止_信号, 停止_自転車歩行者 → `stop`
- 回避_車両, 回避_路駐車 → `avoid`
- 曲がり切れない, 曲がるタイミングが早い → `turn`
- その他 → `other`

Map SFT decisions to the same groups:
- `Lead obstacle following`, `Stop for static constraints`, `Yield` → `stop`
- `Out-of-lane nudge`, `Lane change`, `In-lane nudge` → `avoid`
- `Turn (intersection/roundabout/U-turn)` → `turn`
- `Lane keeping & centering`, `Set speed tracking` → `straight`

Compute `median_ood[maneuver_type]` from the normal-scene scores, then
`ood_residual = knn_mean − median_ood[maneuver_type]`.

## Constraints

- `/mnt/nas` is read-only
- sakurab SSH: `id_ed25519_sakuraDatacentric`, batch mode works
- Check `df -h` before large operations; maintain > 50GB free
- All Python via `uv run` from DP repo root
- ONNX-only inference (no .pth)
- Branch: `feat/latent-ood-stage1` in Diffusion-Planner repo

## Implementation Outline

1. Scale up bag conversion (all 174 bags → npz)
2. Rebuild bank with larger normal sample (5k–10k npz)
3. Score all override + normal npz against bank
4. Assign maneuver types to all frames
5. Join OOD scores with EPDMS CSV
6. Build feature matrices for each hypothesis
7. Train/eval LightGBM with 5-fold CV
8. Per-category breakdown
9. Feature importance analysis
10. Write results + recommendation

## Out of Scope

- Hyperparameter tuning (v1 uses defaults)
- Stage 2 ROS integration (depends on this evaluation)
- Additional OOD formulations (e.g., Mahalanobis, ensemble uncertainty)
- Benign non-training bags (only training vs override compared)
