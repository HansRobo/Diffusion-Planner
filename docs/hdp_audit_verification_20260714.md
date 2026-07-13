# HDP model/training audit verification (2026-07-14)

This note records the disposition of the model, SFT, RL, evaluation, and export findings
from the previous audit rounds and the follow-up review. ROS runtime code is intentionally
outside this audit.

## Fixed in this pass

| Area | Finding | Disposition |
| --- | --- | --- |
| Input conversion | Converting a zero-padded `(x,y,heading)` pose to `(x,y,cos,sin)` changed the sentinel into `(0,0,1,0)`. | Fixed. Training, validation, RL, visualization, and NPZ export paths preserve zero padding; stationary future targets still use the normal conversion. |
| Input ownership | Four-column heading conversion returned an alias that callers could mask in place. | Fixed. Conversion returns an owning tensor; the RL helper now clones sliced views too. |
| Lane geometry | Lane token position used a fixed midpoint even when that point was padding. | Fixed. Position and direction are pooled over valid points; fully empty lanes retain the existing masked fallback. |
| RL road border | Direct road-border reward enabled the same map availability mask used by the occupancy fallback, defeating `rl_occupancy_use_road_border=False`. | Fixed. Direct geometry availability and occupancy-source availability are separate; regression test covers the combination. |
| RL resume | New road-border optimization/evaluation weights and thresholds were absent from strict compatibility checks. | Fixed. Strict resume now checks both weights and both thresholds. |
| Standalone reward validation | `valid_predictor.py` could not set the road-border reward or its thresholds. | Fixed. CLI/config and remapping now expose the complete road-border family. |
| Numerical guards | A Python `assert` only checked NaN in the diffusion term and disappeared under `python -O`. | Fixed. Explicit `FloatingPointError` checks cover all returned loss terms. |
| DDP diagnostics | Loss reduction relied on identical key ordering without checking it. | Fixed. Epoch-level key count/digest agreement is checked before packing values. |
| Validation throughput | Multi-sample ADE/FDE inference computed and discarded turn logits for every candidate. | Fixed. The inference input sets `_skip_turn_indicator`. |
| Diffusion schedule | Inference beta constants were duplicated from the VP-SDE. | Fixed. `NoiseScheduleVP` receives the SDE beta endpoints through public properties. |
| Initialization | Route position embeddings used unit-scale random initialization unlike the other embeddings. | Fixed for new training. Existing checkpoints load unchanged; a fresh Base/SFT run is needed to measure the benefit. |
| Export coverage | ONNX trace dummy turn indicators never sampled input class 3. | Fixed. The trace now samples all four raw input classes. |
| API hygiene | DPM wrapper used mutable default dictionaries. | Fixed with `None` defaults. |

## Verified already fixed by earlier rounds

The current tree was rechecked for these items: all temporal HDP inputs use 80 action steps
(not 81) and no `delay`; replay reward JSON recursively converts non-finite values to JSON
`null`; RL `all` scope freezes the unused turn head; curved-lane masking uses centerline
geometry; resume samplers call `set_epoch` before each epoch; extra traffic-light masking is
occurrence/index based; equal-reward groups are discarded for every normalization mode;
`valid_group_fraction` is group based; lane heading NaN exposure was removed with the old
ego-frame proxy; pandas NaN resume fields are rejected; EMA and validation use the intended
checkpoint policy; ONNX and C++ HDP shapes are 80-step ego-only; and planner-metrics reward
primitives have regression coverage.

## Valid findings intentionally left as experiments or contract changes

These are real hypotheses, but changing them in place would invalidate the semantics of the
trained SFT/RL checkpoints or alter the fixed deployment contract:

- Turn-head samples currently use indices `0,10,...,70`; using `9,19,...,79` would include the
  eight-second endpoint but requires retraining the auxiliary head and re-baselining its ONNX
  output. The current trained policy is therefore left bit-compatible.
- No-decay parameter groups, generated-turn loss weighting by diffusion time, and the sparse
  neighbor-collision loss normalization are training-policy ablations, not silent correctness
  errors (their default coefficients/paths are documented and tested).
- Neighbor top-K selection and reducing the 31-frame encoder input to six frames change model
  shapes and data/deployment contracts; they require a separate Base/SFT benchmark.
- Drop-path naming is imprecise in a few local blocks but does not change the implemented
  dropout behavior. Renaming it would break configuration/checkpoint provenance.
- The rear-axle plus half-wheelbase ego-box center is the common Tier IV convention and is used
  consistently by SFT validation and RL geometry. Replacing it requires a vehicle-geometry
  calibration experiment.
- Strict weights-only initialization rejects missing route-AdaLN keys by design. Silently
  accepting them would create a partially initialized policy and is unsafe for this HDP-only
  branch.

## Verification

- Ruff: clean.
- Full tests: `446 passed, 15 skipped`.
- Full tests with `PYTHONWARNINGS=error`: `446 passed, 15 skipped`.
- Node02 direct road-border RL smoke completed with Slurm exit code 0. Node01 formal RL remains
  under monitoring and was not modified by this audit.
