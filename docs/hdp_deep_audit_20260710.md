# HDP Deep Code and Experiment Audit

Date: 2026-07-11 (Asia/Tokyo)

Branch: `feature/issue-219-hdp-rl` (GitHub issue #219)

This report records the post-fix state of the HDP branch. Local paper LaTeX and the
downloaded official implementations are the primary ground truth:

- `reference/hyper_diffusion_planner_paper/src/neurips_2026.tex`
- `reference/hyper_diffusion_planner_paper/src/code.tex`
- `reference/external/Hyper-Diffusion-Planner/HDP-nuplan`
- `reference/external/Hyper-Diffusion-Planner/HDP-navsim`

The old ROS node is not a model-quality constraint. Native ego-only output is retained;
the downstream Autoware consumer can be changed separately.

## Executive result

The HDP model and RL path are now internally coherent and suitable for controlled
Base -> SFT -> RL experiments. The implementation is faithful to the paper's action,
hybrid-loss, AWR, group-normalization, equal-group discard, and EMA-policy structure.
It is not described as a byte-for-byte reproduction because the released code and the
paper do not expose the in-house real-vehicle reward shaping constants or environment.

The recommended quality arm is native ego-only Base/SFT/RL:

```text
predicted_neighbor_num=0
use_velocity_representation=true
diffusion_model_type=x_start
diffusion_supervision_type=x_start
planning_hybrid_loss=0.01
hybrid_loss_window=10
```

Joint ego/neighbor action prediction remains available with
`predicted_neighbor_num=320` as a DP ablation. Both modes encode all 320 neighbor
histories and score safety against all logged neighbor futures.

## Reference variants are not identical

There is no single public configuration that matches every paper statement:

| Setting | Paper / real vehicle | Public nuPlan | Public NAVSIM |
| --- | ---: | ---: | ---: |
| Action | ego velocity | ego velocity | ego diff/velocity |
| Hybrid weight | table: 0.1 | 0.01 | 0.05 |
| Detach window | appendix: L-1 | 10 | config-driven |
| RL group | 32 | not released | 10 in config |
| RL rollout steps | real vehicle: 6 | inference noise 0.1 | 5 in config |
| EMA update | 0.05 | not released | disabled by default |
| Rollout storage | prior policy | not released | replay refresh every 10 epochs |

This branch uses the nuPlan-compatible SFT values (`omega=0.01`, `W=10`) and the
paper/real-vehicle-oriented RL values (group 32, beta 1, EMA update 0.05, six rollout
steps). Rollout temperature 0.5 follows the released NAVSIM RL path. These choices are
checkpointed and should be ablated rather than called universally official.

## Formula and representation audit

### Velocity action

The action is `(dx, dy, cos(yaw), sin(yaw))` per 0.1-second frame. Position channels
are first differences from the ego origin and are integrated with `cumsum` at inference.
The heading channels remain absolute cosine/sine values. This matches both released
implementations. Although the paper writes velocity multiplied by `dt`, the public code
stores frame displacement directly, so no second `dt` factor belongs in this branch.

Velocity and waypoint round-trip, normalization round-trip, and native ego-only tensor
shape tests pass.

### Hybrid loss

The diffusion target and loss are x-start in normalized velocity space. The waypoint
term integrates denormalized predicted displacement and compares its x/y to expert
waypoints. `_detached_integral` preserves the exact forward `cumsum` value and limits
each velocity gradient to the configured recent window. Its gradient pattern is covered
by an executable test.

The SFT and RL implementations both use sum over action coordinates followed by the
temporal mean, matching the public nuPlan loss scale. `planning_hybrid_loss=0.01`
therefore has the same relative convention in SFT and RL.

Supervised training uses the released HDP schedule: five-epoch linear warmup followed by
a fixed learning rate. The inherited Tier IV rule that reduced LR by 10x/100x over the
last ten epochs was removed because it consumes half of a 20-epoch Base run and is not
present in the paper or released HDP implementation.

For delay augmentation, known prefix predictions are clamped to the prefix target before
waypoint integration. Direct loss and future integrated loss therefore cannot send a
gradient through a prefix that inference constrains exactly.

### Goal and augmentation

`goal_pose` enters the scene encoder, so leaving it in the old frame during ego-centric
augmentation is incorrect. Both quintic and bridge augmentation now transform goal x/y
and heading/cos-sin with the same translation and rotation as lanes, route, and futures.
A 90-degree transform regression test passes.

## RL algorithm audit

The training step is:

1. Encode each scene once.
2. Sample 32 candidates from the EMA previous policy with six DPM-Solver steps.
3. Score logged-neighbor pseudo-closed-loop rollouts.
4. Normalize rewards within each scene group.
5. Discard non-finite and equal-reward groups.
6. Apply `exp(beta * normalized_reward)` to the HDP hybrid diffusion loss.
7. Update the live decoder, then update EMA with rate 0.05 (`timm decay=0.95`).

This is a fast one-batch policy-iteration design, closer to the paper equation than the
public NAVSIM replay implementation but different from NAVSIM's ten-epoch replay reuse.
The public replay path may improve sample reuse, while the current path avoids a very
large rollout cache and stale feature I/O. That tradeoff requires an experiment; it is
not hidden as implementation parity.

Decoder-only training is the default and matches the released parameter activation.
Frozen encoder dropout/drop-path is disabled. A fresh RL run initializes live weights
from the SFT EMA shadow, then initializes its rollout EMA from those same weights.
Validation, best-model selection, checkpointing, and ONNX export use RL EMA when enabled.

DDP loss scaling uses the global valid-sample count. Because DDP averages rank gradients,
each local numerator is multiplied by world size before dividing by the global count.
If every rank has zero valid groups, live forward, optimizer, AdamW weight decay, and EMA
are all skipped. This avoids both rank bias and policy drift without reward signal.

## Reward audit

Implemented paper structure:

- `risk = min(TTC, THW, occupancy)` across the horizon;
- SAT/exact OBB collision and continuous clearance-closing TTC;
- rear-impact attenuation of 0.3 rather than full active-collision penalty;
- vehicle-only leader association and time-gap, spacing, speed-match, comfort aspects;
- lane-center temporal score;
- expert off-lane and kinematic lane-change masking;
- total weights risk/follow/lane = 1.0/3.0/2.5.

The paper does not publish the numeric speed-adaptive curves. Current TTC, THW,
occupancy, gap, spacing, speed, and comfort thresholds are explicit local assumptions.
They are command-line configurable and saved in `args.json`.

Tier IV occupancy uses sources in this order: real static boxes, stopped logged agents,
then road-border clearance as a corpus fallback. Missing occupancy is neutral and source
coverage is logged; it does not raise merely to claim paper parity. The sampled 2026-06
corpus had 0% nonzero static-object coverage and 100% road-border coverage, so production
reward interpretation must account for that limitation.

Reward validity now uses the `(cos,sin)` pose marker after padding is zeroed. A neighbor
crossing exactly `(0,0)` therefore remains valid and cannot disappear from collision
scoring. Raw converter futures also use a contiguous-prefix mask, so an internal
`(0,0,yaw=0)` frame is retained while zero tail padding stays invalid. SFT, RL,
validation, and both augmentation paths share this mask. These failure modes have
regression tests.

## Data audit and no-migration policy

Structured audits found no missing/non-finite sampled NPZ files. Shapes match the current
constants, including polygons `(10,40,3)` and line strings `(60,20,4)`.

Converter commit `55eff4f` correctly makes future data start at `t+0.1s`. The existing
2026-06 corpus predates that fix. Every sampled short neighbor track duplicated the
current frame; full 80-frame tracks were already correct because the old fixed-size deque
evicted its seed.

No dataset migration is performed. With `align_legacy_neighbor_futures=true`, each
DataLoader worker detects only affected short tracks in the loaded in-memory dictionary,
uses `future[1:]`, and zero-pads the tail. The shared NPZ is never opened for writing and
is never renamed or replaced. The setting is saved in the checkpoint and standalone
validation inherits it. Regenerated datasets should disable the correction.

List counts from the audit:

| List | Count | Unique | Internal duplicates |
| --- | ---: | ---: | ---: |
| Node01 base | 5,092,382 | 5,092,382 | 0 |
| Node01 extra | 55,402 | 55,402 | 0 |
| Clean Base source | 9,081,354 | not recomputed | preserved |
| Filtered Base | 9,054,475 | not recomputed | preserved |
| Three-source right-turn extra | 52,870 | 52,870 | 0 |
| Validation | 53,008 | 53,008 | 0 |

Train/validation overlap is zero. Node01 base intersects its extra list in 33,913 paths;
with `extra*10`, those paths receive 11 total exposures while extra-only paths receive 10.
This is acceptable only if that slight extra emphasis is intended. The new clean Base
uses the supplied filtered list instead: 26,879 matching right-turn samples were removed,
then the complete, unique 53,185-sample three-source extra set is appended ten times.
The JT extra manifest originally contained 315 validation paths; those entries were removed
from the extra manifest without changing the Base list or any NPZ. The resulting 9,583,175
path exposures contain no train/validation overlap and do not retain an accidental eleventh copy.

## Validation and logging

The deterministic zero-noise trajectory is retained as a stable checkpoint proxy. The
paper Appendix explicitly generates six trajectories and reports minADE/minFDE; this
branch separately implements a seeded six-sample metric and does not call the
deterministic proxy the paper protocol. Six-sample evaluation reuses one scene encoding.
Evaluation shards no longer use duplicate padding, so non-divisible validation-set sizes
produce exact global metrics and prediction files are never written by two ranks.

EPDMS remains a Tier IV safety/quality diagnostic, not the RL reward and not official
closed-loop PDMScore. The obsolete `enable_pdms_eval` alias was removed so only
`enable_epdms_eval` remains.

The turn-indicator train/inference mismatch is removed. SFT evaluates the classifier on
both the expert trajectory and the detached model-generated x-start trajectory, then
uses their normalized weighted mean so the historical loss scale does not double.
Detaching prevents the auxiliary classifier from steering the planner trajectory merely
to simplify classification. The deployment-aligned generated branch owns the historical
accuracy key; expert, generated, change-only, all five per-class accuracies, and class
counts are logged separately. This is a Tier IV path rather than an HDP paper component.

RL step logs include reward mean/std/range, equal-group coverage, reward weights,
endpoint diversity, optimizer-step fraction, gradient norm, rollout/reward/update time,
scenes/s, candidates/s, and peak CUDA memory. Step metrics are rank-0 local; epoch means
are DDP-reduced. Checkpoints persist the real global step, W&B run ID, TSV history, and
best score.

## Performance changes

- Encoder output is cached once per scene and reused for all rollout candidates and the
  live decoder loss.
- Decoder-only expansion repeats current action-state tensors, not full 31-frame scene
  observations.
- Ego-only SFT skips heading conversion for the unused 320-agent future tensor and
  does not materialize that NPZ payload at all when neighbor-collision supervision is
  disabled. On a real cached sample this reduced loaded array bytes from 1.206 MB to
  0.899 MB and single-process materialization time by about 24%.
- Penalty losses inverse-normalize only the inputs they consume; road-border-only Base
  no longer allocates a denormalized 320-by-31 neighbor-history tensor every step.
- Full 320-neighbor logged futures remain scene-level instead of being duplicated 32x,
  removing roughly 100 MB of temporary data for eight local scenes.
- bf16 autocast covers model forward only; SDE/noising/loss math remains fp32.
- Fused AdamW, TF32, DDP static graph, persistent workers, pinned memory, and prefetch are
  enabled where supported.
- Extra-list weighting appends path references in memory instead of writing the previous
  roughly 798 MB combined manifest; giant W&B list artifacts are opt-in.
- SFT/RL checkpoint ONNX export defaults off because synchronous export makes every other
  rank idle at the next barrier. Strict standalone export remains available.
- Reward and update timing is sampled on the W&B logging cadence rather than synchronizing
  CUDA every step.
- RL marks only `decoder.dit` trainable and skips the unused turn-head forward, avoiding
  unsupervised DDP parameters and unnecessary classifier work.

Joint-action RL is supported but group-32 memory scales with 321 action rows. Use it as a
smaller-batch ablation; ego-only is the intended high-throughput real-vehicle arm.

## Checkpoint and runtime safety

Checkpoint writes use a same-directory temporary file and `os.replace`. Strict training
resume requires compatible model, optimizer, scheduler, epoch, architecture, action
shape, input normalizers, velocity normalizer, and EMA state when enabled. The check runs
before a same-directory `args.json` can be overwritten. Scheduler state represents the
next epoch, so resume does not repeat one learning rate.

`init_weights_path` is intentionally weights-only and may change joint/ego action shape.
`resume_model_path` may not. Empty loaders, invalid group sizes, incompatible diffusion
parameterization, invalid reward thresholds, and non-divisible global batch sizes fail
at startup.

## ONNX and deployment

HDP velocity deploys the full ONNX graph. The split raw decoder is skipped because an
external waypoint loop would misinterpret the ego velocity latent. Full output is already
decoded waypoint space:

```text
ego-only: prediction [B,1,80,4]
joint:    prediction [B,321,80,4]
turn indicator logits [B,5]
```

All full-graph inputs, including `delay`, have dynamic batch axes. CPU ONNX checker and
PyTorch/ORT parity passed for the joint artifact. Native ego-only EMA output was
`[1,1,80,4]` with prediction max/mean difference `1.38e-5 / 2.42e-6`; ORT batch 2 with
different per-row delays returned `[2,1,80,4]` and `[2,5]`. Training-time export is convenience-only and swallows failures. Release
must use `ros_scripts/torch2onnx.py`, which treats parity failure as failure.

The pinned Autoware consumer still assumes 321 output rows and publishes rows 1..320 as
predicted objects. The model is not padded or weakened for that contract. C++ postprocess
must be updated before ego-only deployment. TensorRT parser acceptance, engine build,
workspace, numerical parity, and 10 Hz p95 latency remain unverified on target hardware.

## Active experiment snapshot

Node01 SFT is still running on eight GPUs and was not interrupted. The process started
with the old joint action configuration, so it cannot observe this working-tree code.
Completed epoch 17 reached ego loss about 2.001, lateral error about 0.166, longitudinal
error about 1.265, and proxy EPDMS about 0.8844. It remains a useful joint-head candidate,
not the clean ego-only Base/SFT initialization recommended for RL.

The Node02 traffic-light-mask change was integrated without its giant generated manifest.
The current Dataset accepts the three source lists directly and masks only their in-memory
lane/route traffic-light fields, leaving the shared NPZ files untouched. Its signal
masking still needs protected/unprotected, red/green, stop-line, and non-right-turn
control slices before any model-quality conclusion.

## Verification status

- Full `diffusion_planner/tests`: 130 passed.
- Targeted ruff: passed.
- Python compileall: passed.
- Git whitespace check: passed.
- Actual legacy NPZ alignment/reward smoke: passed.
- Ego-only loss and ONNX shape/checker smoke: passed.
- Two-rank Gloo test with unequal local valid counts produced the exact global-mean
  gradient on both ranks.
- Joint checkpoint to ego-only weights-only load: passed.
- TensorRT: not available on this host; not tested.

The quintic augmentation constructor again accepts its historical defaults and both
`(x,y,yaw)` and `(x,y,cos,sin)` ego history formats. `scikit-learn` is now declared in
the workspace package metadata as well as `requirements.txt`, so the complete repository
test collection is runnable from the locked environment.

## Remaining experiments

1. Train a clean ego-only Base -> SFT checkpoint before RL. Do not use the running joint
   SFT as the only evidence for the ego-only policy.
2. Run RL smoke on a small controlled subset, then scale only after checking reward range,
   valid-group fraction, endpoint diversity, candidates/s, and peak memory.
3. Ablate `omega={0.01,0.05,0.1}`, `W={10,79}`, rollout temperature, and steps `{5,6,10}`.
4. Compare on-the-fly EMA policy iteration against a compact replay/reuse variant if
   rollout generation dominates wall time.
5. Evaluate fixed real-vehicle scenario slices and target TensorRT before release.
6. Ablate the generated/expert turn-indicator loss weights using the logged per-class and
   change-only metrics; do not select this head by KEEP-dominated overall accuracy alone.
