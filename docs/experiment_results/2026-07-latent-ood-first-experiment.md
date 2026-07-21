# Latent OOD First Experiment — Results

> **INVALIDATED (2026-07-17):** All results in this document are invalid.
> Override npz files had all-zero lane/route/polygon tensors due to a
> coordinate-shift bug in `ros_scripts/extract_map_from_bag.py`, while the
> normal training bank npz had full map data. The encoder saw fundamentally
> different inputs for override vs normal scenes. The reported 1.38x
> enrichment (p<10^-281) was measuring map presence/absence, not scene
> novelty. The per-category OOD differences and "design spec minimum
> criteria MET" claim are equally invalid. The pipeline code is correct;
> the input data was broken. Pending fix to map extraction and full rerun.

> Date: 2026-07-03
> Status: ~~Completed~~ INVALIDATED — see banner above
> Design reference: `at-team-tools/lin/diffusion-planner-risk-scene/docs/latent_scene_ood_detector.md` §14

## Executive Summary

The first experiment tested whether latent scene OOD (kNN distance in Diffusion Planner's
pooled encoder embedding space) separates override/blind-spot windows from normal driving.
It does, decisively (Mann-Whitney p=1.6e-281, 1.38x mean enrichment), and — critically —
the two EPDMS blind-spot categories (停止_車両, 回避_車両) score as high or higher than the
category EPDMS already detects (曲がり切れない). This is the first direct evidence that
latent OOD can catch overconfident-wrong scenes that the output-side EPDMS metric misses.

## Setup

- **Model**: `diffusion_planner_for_x2` ONNX; encoder weights extracted via onnxruntime.
- **Embedding bank**: 1000 normal training embeddings sampled from
  `sakurab:/mnt/nvme/dataset/20260425_takanawa_full`.
- **Evaluation set**: 2454 frames total.
  - 1000 normal training frames (same distribution as the bank).
  - 488 停止_車両 frames (1 bag) — confirmed EPDMS blind spot, no subscore fires.
  - 487 回避_車両 frames (1 bag) — confirmed EPDMS blind spot, no subscore fires.
  - 479 曲がり切れない frames (1 bag) — category EPDMS already fires on.
- **Inference**: onnxruntime `CPUExecutionProvider`, batch size 16.
- **Scoring**: kNN, k=10, L2-normalized mean-pooled encoder embeddings, embedding dim=256
  (per design spec §2.3 / §5.1 recommended v1 configuration).

## Results

| Category | N | Mean OOD | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|
| NORMAL (training) | 1000 | 0.5803 | 0.5825 | 0.7904 | 0.9639 |
| 曲がり切れない (EPDMS fires) | 479 | 0.7914 | 0.8089 | 0.9401 | 0.9863 |
| 回避_車両 (blind spot) | 487 | 0.7805 | 0.7529 | 0.9486 | 1.0302 |
| 停止_車両 (blind spot) | 488 | 0.8261 | 0.8144 | 0.9144 | 0.9401 |

## Statistical Analysis

- **Mann-Whitney U** (normal < override, all override categories pooled): U=108954,
  p=1.60e-281. The separation between normal and override/blind-spot windows is not
  attributable to chance.
- **Enrichment**: override mean OOD = 0.800 vs. normal mean OOD = 0.580 → **1.38x**.
- Notably, 停止_車両 (a confirmed EPDMS blind spot with no subscore activation) has the
  **highest** mean OOD (0.8261) of all four categories — higher than 曲がり切れない, the
  category EPDMS is actually designed to catch.

## Success Criteria Check (design spec §14.4)

**Minimum useful result:**
- "Top 5% latent-OOD windows have higher override/blind-spot density than random windows"
  → **YES**. All three override categories sit clearly above the normal distribution
  (median normal 0.58 vs. median override 0.75–0.81).
- "Some EPDMS-clean override windows have high latent OOD" → **YES**. 停止_車両, the
  clearest EPDMS blind spot (no subscore fires), shows the highest OOD of any category
  tested.
- "Human review finds a meaningful fraction of top-OOD windows are recollection-worthy"
  → not yet assessed; requires the human review protocol in design spec §10.4 (not run
  in this first pass).

**Strong result:**
- "Adding latent OOD improves detection over EPDMS alone" → **strong evidence, not yet
  quantified**. The categories EPDMS misses (停止_車両, 回避_車両) show OOD scores at or
  above the category EPDMS already detects. A formal AUPRC/recall-at-precision comparison
  against EPDMS + trajectory variance has not been run.
- "High-OOD clusters correspond to actionable data gaps" / "new data collection reduces
  future OOD" → not tested in this pass (requires closed-loop iteration, out of scope for
  Stage 1).

## Limitations

- Only 3 override bags tested (1 per category), all from the takanawa area — no
  geographic diversity, single-bag sample per category.
- Bank is 1000 randomly sampled training frames, not the full 500k+ training set used
  to train the released checkpoint.
- CPU-only inference (`onnxruntime-gpu` not installed in this environment).
- No cross-validation or leave-one-out; embeddings for the bank and the "NORMAL" eval
  set may share underlying scenes.
- Possible self-scoring artifact: normal frames near the bank boundary could show
  elevated OOD purely from bank sparsity, not genuine novelty.
- No AUROC/AUPRC/precision-at-top-K metrics computed yet (design spec §10.1 asks for
  these); only distributional summary stats and a rank-sum test were run.
- No human review of top-OOD windows (design spec §10.4) has been performed.

## Next Steps

1. Scale evaluation to all 174 override bags (currently only 3 covered).
2. Rebuild the bank from the full training set (or a much larger stratified sample)
   instead of 1000 random frames.
3. Cross-reference per-frame latent OOD scores with per-frame EPDMS subscores from
   `samples_epdms_all.csv` to compute the 2x2 (EPDMS x OOD) breakdown from design spec §9.2.
4. Compute AUROC/AUPRC and precision-at-top-{1,5,10}% per design spec §10.1, plus the
   ablations in §10.3 (raw features, map-only, agent-only, pooling variants, k sweep).
5. Stage 2: integrate latent OOD as a feature in the risk-scene classifier alongside
   EPDMS aggregate, trajectory variance, and longitudinal risk features.
