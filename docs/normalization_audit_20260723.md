# Normalization Audit (2026-07-23)

## Scope

This audit uses a reproducible random sample of 20,000 paths from the exact
Base80 training manifest:

`/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json`

The manifest contains 5,446,154 paths, all sampled files were NPZ version 2,
and all 20,000 files loaded successfully. The machine-readable report is
`artifacts/normalization_audit/normalization_audit_20260723.json`. The audit
uses the same geometry-only padding masks as the model; a valid ego origin is
not discarded as padding.

The report is diagnostic. It is not a replacement normalization contract and
must not be used to resume an existing checkpoint.

## Finding

The checked-in values are hand-selected range scales, largely inherited from
Tier IV DP and the public HDP implementation. They are not empirical train-set
mean/std estimates. That choice is not automatically wrong: preserving zero
padding, circular `(cos, sin)` channels, one-hot attributes, and physically
meaningful zero displacement is more important than making every channel have
unit sample variance.

The current `data_robust_v1` contract was rechecked on an independent 5,000-path
sample after installation. The machine-readable result is
`artifacts/normalization_audit/normalization_profile_v1_audit_20260723.json`;
all samples loaded successfully and the main continuous geometry groups are
within a bounded activation range (roughly 0.7--1.9 sample standard deviations).
The earlier 20,000-path report remains the pre-change comparison and should not
be read as the current profile's configured statistics.

However, the current scales are not equally well matched to the new vehicle-
parameter corpus:

| Contract | Sample mean | Sample std | Current mean/std | Assessment |
|---|---:|---:|---:|---|
| Ego delta `dx` | `0.537` | `0.449` | `0 / 0.5` | Good magnitude; zero mean is deliberate so a stopped action remains zero. |
| Ego delta `dy` | `-0.004` | `0.137` | `0 / 0.5` | Current scale is conservative and may underweight lateral corrections. |
| Ego current `vx` | `5.63` | `4.38` | `0 / 20` | Current input is compressed to roughly `0.28 +/- 0.22`. |
| Valid neighbor `x/y` | `7.96/-3.50` | `59.25/30.03` | `10/0`, `20/20` | Many far-field values exceed 3 normalized units; a global z-score would instead weaken near-field geometry. |
| Lane geometry `x/y` | `2.53/-4.16` | `67.31/46.64` | `10/0`, `20/20` | Far-field range dominates. |
| Lane tangent `dx/dy` | `0.038/0.011` | `2.50/1.79` | `0/20`, `0/20` | Current scale makes direction channels very small. |
| Lane boundaries | approximately `0` | `1.19-1.35` | `0/20` | Current scale makes border offsets very small. |
| Route goal `x/y` | `289.5/-27.6` | `826.8/933.0` | `10/20`, `0/20` | Current normalized mean/std are approximately `14/-1.4` and `41/47`; this is the largest mismatch. |
| Static objects | no valid rows in the sample | | | This is a data-coverage issue, not a normalization issue. |

The complete per-channel values, counts, min/max, and configured normalized
means/stds are in the JSON report.

A separate 10,000-path audit of the active validation manifest is stored at
`artifacts/normalization_audit/normalization_audit_valid_20260723.json`. Its
geometry is not identical to train (for example, route-goal `x` mean/std is
approximately `510/686 m` versus train `290/827 m`). This is another reason to
estimate any profile from train only and treat validation differences as a
generalization check, rather than silently pooling the two splits.

For context, a 10,000-path audit of the older filtered `20260623_full_sequence`
SFT manifest found the same broad pattern: valid-neighbor `x/y` std about
`63/30 m`, lane `x/y` std about `67/39 m`, and goal `x/y` std about `748/731 m`.
Therefore this is a long-standing range-vs-statistics design choice, not proof
that the new converter suddenly corrupted every coordinate.

## What should change

### Change in a future fresh-training experiment

1. **Use a versioned normalization profile.** Estimate statistics from the train
   manifest only, record the manifest SHA-256, NPZ version, schema, mask rules,
   sample count, and augmentation contract. Validation must never contribute to
   the profile.
2. **Keep semantic channels out of ordinary z-scoring.** Preserve `cos/sin`,
   traffic-light and line-type one-hot channels, agent-type one-hot channels,
   and zero-padding behavior. Give them identity scale (or a separately
   justified bounded transform).
3. **Keep the action mean at zero.** For velocity/delta actions, use a robust
   scale for `dx/dy` but keep `mean=0`; otherwise a stationary action becomes a
   nonzero latent and the stop behavior is harder to represent. Keep heading
   channels bounded rather than centering their highly imbalanced corpus mean.
4. **Separate geometry channels.** Do not use one `20 m` scale for lane tangent
   and left/right-boundary offsets. Test a per-channel robust scale with a
   lower bound, while preserving a near-field scale for `x/y`; using the global
   `67 m` lane std blindly would make the local road geometry almost invisible.
5. **Handle route goals as a separate contract.** The goal endpoint can be
   hundreds or thousands of metres away. A full-corpus mean/std is not a good
   fix. Compare a bounded route-relative goal representation or an explicitly
   clipped robust scale, and verify that the transformed goal still carries
   useful direction. This change affects the encoder/deployment input contract
   and therefore requires a fresh Base run.
6. **Give current proprioception its own scale.** The current `vx/vy`, yaw-rate,
   acceleration, and steering channels should be tested with physical robust
   scales rather than automatically inheriting a `20 m` position scale. Keep
   the action and current-state contracts internally consistent.

## What must not be done

- Do not replace every entry with `mean(data), std(data)`.
- Do not calculate statistics over validation or deployment data.
- Do not use categorical/padding zeros in a continuous statistic.
- Do not change `normalization.json` inside an active run. The original fixed-
  scale job 1406 was stopped before switching contracts; its epoch-7 checkpoint
  remains available as a baseline.
- Do not resume an old checkpoint after changing any normalization entry;
  strict resume correctly treats these values as part of the model contract.

## Selected Fresh-Training Profile

For the new from-scratch run, the default file has been changed to the
`data_robust_v1` profile. It uses the following policy rather than blind
sample z-scoring:

- position channels use bounded physical scales (`50/40 m` for broad map
  coordinates and `10/1 m` for recent ego history);
- lane tangent and left/right-boundary channels use approximately their
  measured metre scale (`3/2/1.5 m`);
- ego `dx/dy` keeps zero mean and uses `0.5/0.25 m` per step;
- current `vx` uses `5 m/s`, steering/yaw-rate use `0.2`, and categorical
  channels remain identity-scaled;
- route goals use a `500 m` bounded scale rather than the old `20 m` scale,
  avoiding four-dozen-unit activations for ordinary long routes.

The exact JSON is preserved at
`artifacts/normalization_audit/normalization_data_robust_v1.json`; the prior
contract is preserved at
`artifacts/normalization_audit/normalization_fixed_legacy_20260723.json`.
This is a reasoned first profile, not a claim that normalization can be proven
optimal without an A/B run. The new Base run is the required measurement of
whether it improves planning metrics.

## Recommended experiment order

The cancelled epoch-7 Base80 run remains the fixed-scale baseline. After the
new profile has a stable checkpoint, run small fixed-seed ablations on the same
train/validation manifests:

1. baseline fixed profile;
2. baseline plus per-channel lane tangent/boundary scales;
3. (2) plus a bounded route-goal representation;
4. (3) plus physical current-proprioception scales;
5. only then test an action `dy` robust-scale change.

Compare trajectory loss, DAC/PDMS, centerline deviation, road-border margin,
red-light behavior, and conditioning sensitivity. A lower normalized loss by
itself is not sufficient evidence. Any winning profile must be retrained from
the same Base initialization policy and then checked through validation and
ONNX export with the exact saved profile.
