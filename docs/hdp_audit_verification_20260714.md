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
| Optimizer portability | Supervised training could fail when fused AdamW was requested but unsupported by the active PyTorch/CUDA build. | Fixed. SFT and RL now run a one-element fused AdamW probe before training, fall back to standard AdamW on either construction/step capability failure, and persist the effective setting. |
| Reward configuration | Road-border occupancy fallback used the planner-metrics global wide threshold instead of the HDP reward configuration. | Fixed. The fallback now uses `rl_road_border_safe_m`, with a regression test for non-default thresholds. |
| Converter dtype | The direct ROS-bag converter wrote transform-derived `goal_pose` as NumPy `float64`. | Fixed. It is explicitly saved as `float32`, matching the rest of the NPZ schema. |
| RL resume metadata | Missing baseline sidecars were checked against the current CLI default, so automatic recovery could reject a source run created without baseline validation. | Fixed. Resume reads `rl_validate_before_training` from the source checkpoint metadata before deciding whether the sidecar is required. |
| Evaluation shape guards | `ego_is_comfortable` used Python `assert` for state/time shape checks. | Fixed. It now raises explicit `ValueError` under all optimization modes. |
| Dead SDE path | `subVPSDE_exp` was an unreferenced constructor that always raised `NotImplementedError`. | Removed. The HDP branch has one live VP-SDE implementation. |

## Verified already fixed by earlier rounds

The current tree was rechecked for these items: all temporal HDP inputs use 80 action steps
(not 81) and no `delay`; replay reward JSON recursively converts non-finite values to JSON
`null`; RL `all` scope freezes the unused turn head; curved-lane masking uses centerline
geometry; resume samplers call `set_epoch` before each epoch; traffic-light masking augmentation
has been removed; equal-reward groups are discarded for every normalization mode;
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

## Follow-up audit items (same-day second pass)

The follow-up pass rechecked every item in `code_review_findings_20260711.md` and
`hdp_code_review_20260712.md` against the current tree, rather than assuming the earlier
disposition table was still valid after later edits.

| Area | Finding | Disposition |
| --- | --- | --- |
| RL geometry | Empty lane/route tensors reached point-to-segment `amin` and zero-segment chunk sizing. | Fixed. Single-scene and batched distance helpers return `+inf` with the input shape; unavailable centerlines therefore produce an explicit lane-unavailable result instead of a crash. |
| Goal conditioning | An all-zero/missing goal was projected as an unmasked scene token. | Fixed. `GoalPoseEncoder` masks the zero sentinel and zeros its embedding; a valid origin pose with `(cos,sin)=(1,0)` remains active. |
| Augmentation dtype | Scene-frame rotation matrices were always `float32`, causing mixed-dtype matmul failures for double/other typed callers. | Fixed in both standard and bridge augmentation with `new_tensor`, preserving device and input dtype. |
| DDP diagnostics | Scalar-metric reduction still trusted rank-local key ordering even though loss reduction had a digest check. | Fixed. Key count/order digest is checked before packing scalar metrics. |
| DDP empty diagnostics | A rank-local empty metric dictionary could return before the collective while another rank packed non-empty metrics. | Fixed. Empty dictionaries participate in the digest collective whenever DDP is initialized; divergent empty/non-empty sets now raise coherently instead of hanging. |
| C++ benchmark | The benchmark's current HDP input map omitted the required `delay` vector; runtime transfer uses `.at("delay")`. The linked Autoware runtime may still allocate upstream joint-planner buffers. | Fixed defensively. It supplies a zero scalar delay and now refuses to run if the linked runtime dimensions are not `[1,1,80,4]`, preventing a false benchmark or undersized host copy. The external deployment runtime must be updated separately; deployment is intentionally outside this model/SFT audit. |
| Fused AdamW fallback | Unsupported fused construction can raise either `TypeError` or `RuntimeError` across PyTorch versions. | Fixed for RL launcher; both exception classes fall back before the effective `fused_optimizer` value is persisted to `args.json`. |
| RL selection | Legacy score reconstruction could mix reward/EPDMS/loss units. | Rechecked current code: new logs persist only finite full-evaluation reward scores; off-cadence rows are NaN and resume prefers the cumulative accepted score. No code change needed. |
| THW at a stop | A stopped ego's time headway is undefined and could otherwise look safe. | Rechecked current code: THW is combined with a clearance-based distance-headway score, and stopped-neighbor occupancy is independently fused with rear attenuation. This is an explicit shaping choice, not a silent zero-risk bug. |
| Profile cadence | Profiling is decided before a rollout while logging uses actual optimizer steps. | Confirmed as a profiling-only accounting trade-off when a rollout has zero valid groups; it cannot alter gradients, checkpoints, or reward values, so it remains unchanged. |
| Area output paths | Grouped closed-loop area names could contain path separators. | Rechecked current code: both video and per-area summary paths pass through `artifact_component`; no unsanitized path remains. |
| EMA/DDP | EMA could deep-copy a DDP wrapper and rank-0 evaluation could broadcast buffers. | Fixed earlier and rechecked: EMA is constructed from `ddp.get_model(...)` (unwrapped) for SFT and RL. |
| Resume memory | Full checkpoints could be materialized on every GPU during resume. | Fixed earlier and rechecked: `resume_model` loads with `map_location="cpu"` before copying state. |
| C++/ONNX shape | Legacy joint sampled shape and delay were still present in benchmark assumptions. | Fixed earlier plus the delay input fix above. Current temporal contract is ego-only `[B,1,80,4]`; no legacy split decoder is exported. |
| Empty/short map parity | Batched reward and scene-loop reward must agree when a scene has no usable map segments. | Added empty-centerline regression coverage; the existing curved-scene parity test remains passing. |
| Runtime guards | Live encoder/solver/validation guards used Python `assert`, which disappears under `python -O`. | Fixed. Public shape, DPM mode, classifier-guidance, solver-step, timestep, and validation-count checks now raise explicit exceptions. |

The earlier audit's design/ablation findings remain deliberately unchanged: turn-head endpoint
sampling (`0,10,...,70`), no-decay grouping, generated-turn diffusion-time weighting, top-K
neighbor selection, shortening the 31-frame input contract, exact reward shaping constants,
rear-axle box-center convention, and strict route-AdaLN checkpoint loading. Each would require
a fresh Base/SFT benchmark or would change a trained/deployment contract; none is a hidden
correctness failure in the default run.

## Final hygiene pass

Two additional audit details were checked after the second-pass table. The repository README
still described a vanilla-DP compatibility mode that the current HDP-only `Config` rejects; it
now states the temporal ego-only/velocity contract explicitly and points pure waypoint/joint
baselines to upstream `tier4-main`. `planner_metrics.geometry` also contained an unreachable
first polygon-builder definition shadowed by the vectorized implementation used by the live
reward path; the shadowed helper was renamed to remove the duplicate binding without changing
the exported implementation or numerical behavior. The full test, lint, compile, and whitespace
checks were rerun after these edits.

The strict-resume audit also found a reproducibility gap: reward and safety fields were
already protected, but the deterministic diffusion step count, stochastic minADE/minFDE
parameters, and EPDMS enable/source switches were not. These fields now participate in the
same compatibility check. A resumed RL run therefore cannot silently change validation
sampling or its policy-selection metric; fresh SFT-to-RL initialization remains weights-only
and is unaffected.

The exhaustive follow-up found two more real, bounded issues. Static-neighbor occupancy
classification was still reading `planner_metrics.RewardConfig` globals rather than the HDP
reward configuration, so a checkpoint could not reproduce a run after those library defaults
changed. The velocity and displacement thresholds are now explicit train/eval fields, mapped
into `HDPRewardConfig`, validated, logged, and covered by strict-resume tests; their defaults
remain `0.1 m/s` and `0.5 m`. The bridge augmentation path also used a single `2.75 m`
wheelbase for every vehicle. It now takes the per-scene `ego_shape[0]` when available and
retains `2.75 m` only as the non-dataset fallback, so steering-state augmentation is correct
for mixed vehicle fleets without changing the current default data.

The same bridge path had a separate floating-point time-axis hazard: `torch.arange` with a
stop such as `3 * 0.1` can produce one extra sample. Time vectors are now constructed from
integer sample counts, and a regression test covers the past, future, and full-scene lengths.

This is tracked as a separate bridge-only correctness item rather than folded into the
wheelbase change: it affects tensor lengths even when all vehicle geometry uses the default.

Finally, comparisons such as `value < 0` do not reject `NaN`; invalid reward weights or
thresholds could therefore reach the first rollout. The RL CLI and `HDPRewardConfig` now
reject non-finite reward/shaping values explicitly before model work begins.

The Slurm RL launcher now forwards the train/evaluation stopped-neighbor velocity and
displacement thresholds explicitly as well. This keeps queued jobs reproducible when those
fields are overridden through `HDP_RL_*` environment variables; the Python parser remains
the final finite-value/type validation boundary.

One further reproducibility omission was confirmed: `ego_history_dropout_rate` affected the
training distribution but was absent from strict resume compatibility. It is now checked like
the other training fields. These changes are configuration/geometry fixes, not reward-policy
retuning, so existing checkpoints remain loadable in weights-only SFT-to-RL initialization;
resuming a run with an older `args.json` intentionally fails loudly rather than mixing semantics.

The final configuration-boundary pass also rejected non-finite RL temperatures, learning-rate/
regularization values, `advantage_eps`, dropout probabilities, selection tolerances, and reward
weights before dataset/model work. `HDPRewardConfig` now enforces the same ordering and positivity
relations when constructed directly, so tests or future callers cannot bypass the CLI checks.

## Final boundary and numerical pass

The post-report pass deliberately checked direct Python APIs as well as command-line entrypoints.
Training and standalone-validation parsers now reject non-finite sampling scales, reward weights,
loss coefficients, augmentation values, closed-loop thresholds, and invalid generation counts.
Both normalizer classes reject missing paired statistics, shape mismatches, non-finite means, and
zero/negative standard deviations before any model is constructed. Standard and bridge
augmentation constructors apply the same finite/range checks, so unit tests and future callers
cannot bypass the CLI boundary. Reward-weight computation validates its group shape, floating
dtype, beta, and epsilon, and clamps the exponent below the dtype overflow boundary; rollout and
validation raise immediately on non-finite trajectories/rewards/logits. Validation metric and
per-scene JSON writers recursively map unavailable NaN/Inf values to JSON `null` and use strict
JSON output.

These changes are defensive only: current HDP defaults, checkpoint tensor shapes, reward formulas,
and the SFT/RL objective are unchanged.

## Verification

- Ruff: clean.
- Full tests after the final audit patch: `481 passed, 15 skipped`; the focused HDP/RL/augmentation suite is `211 passed`.
- Full tests with `PYTHONWARNINGS=error`: `481 passed, 15 skipped`.
- Node02 direct road-border RL smoke completed with Slurm exit code 0. Node01 formal RL remains
  under monitoring and was not modified by this audit.
- Ruff lint, `git diff --check`, and Python bytecode compilation pass. The repository-wide
  formatter still reports pre-existing formatting drift in several untouched historical files;
  no broad mechanical reformat was applied because it would obscure the model audit changes.
