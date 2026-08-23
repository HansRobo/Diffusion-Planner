# DrivoR predictor head

`--predictor_head drivor` swaps Diffusion-Planner's DiT/diffusion decoder for
DrivoR's generate-then-score head. Encoder, dataset, augmentation, EMA and LR
schedule are unchanged; everything downstream of the encoder is DrivoR's, and the
output is **only an ego trajectory** — no neighbour prediction, no turn indicator.

- [Training](#training) — how to run it
- [Design notes](#design-notes) — what was ported and what it cost

## Training

Prerequisites: the repo venv (no new dependencies), and the two dataset lists the
diffusion path already uses — `--train_set_list` / `--valid_set_list` each point
at a JSON array of `.npz` paths. **No PDM cache.** DrivoR needs a `metric_cache`
for the map/agent/progress side; the oracle here recomputes it from the NPZ
tensors on the GPU, so there is nothing to build and nothing to invalidate.

Both lists must be **skip-filtered** — the loader does no filtering of its own.
Any frame the production filter would drop (sustained red-light stop, ego stuck,
GT collision, stale data) is otherwise trained and scored as if it were normal,
and those are the easy ones — ego stationary, nothing to hit — so an unfiltered
*valid* list inflates PDMS silently. Filter with `ros_scripts/filter_scene_list.py`;
audit an existing list by sampling `is_skipped` from each frame's `.json` sidecar.
A list's name is not evidence either way.

### 1. Check the port (20 s, CPU)

```bash
cd diffusion_planner
../.venv/bin/python -m pytest -q tests/test_drivor_*.py     # 103 passed
```

### 2. Smoke run (170 steps + one validation pass, 1 GPU)

Subsample hard, keep every other flag identical to the real run:

```bash
../.venv/bin/torchrun --nproc_per_node=1 train_predictor.py \
  --predictor_head drivor --exp_name drivor-smoke \
  --train_set_list "$TRAIN_LIST" --valid_set_list "$VALID_LIST" \
  --save_dir "$RUN" \
  --train_subsample_step 500 --valid_subsample_step 50 \
  --batch_size 64 --train_epochs 1 --save_utd 1 \
  --drivor_lr_schedule drivor --learning_rate 1e-4 \
  --drivor_lr_base_batch_size 64 --drivor_warmup_ratio 0.1 \
  --use_amp --amp_dtype bf16 --use_wandb False --closed_loop_npz_root ""
```

A finite `oracle_selected` and a `devkit panel:` line at the end mean head,
oracle, loss and metric adapter all agree on shapes.

### 3. Full run

Three numbers are *derived*; deriving them by hand is how runs get silently
misconfigured.

```bash
#!/bin/bash
set -euo pipefail
BATCH=${BATCH:-512}                  # GLOBAL batch, not per rank
EPOCHS=${EPOCHS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-2000}   # absolute steps, NOT a ratio
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:?read it off a previous run's tqdm total}
PY=../.venv/bin/python
TOTAL=$((STEPS_PER_EPOCH * EPOCHS))

RATIO=$($PY -c "print(f'{$WARMUP_STEPS/$TOTAL:.6f}')")                     # warmup
PEAK=$($PY -c "import math; print(f'{1e-4*math.sqrt($BATCH/64):.3e}')")    # sqrt rule
echo "batch $BATCH  epochs $EPOCHS ($TOTAL steps)  peak LR $PEAK  ratio $RATIO"

cd diffusion_planner
export DP_DIST_INIT_FILE="$RUN/dist_init"; rm -f "$DP_DIST_INIT_FILE"
export OMP_NUM_THREADS=8

exec ../.venv/bin/torchrun --nproc_per_node=8 train_predictor.py \
  --predictor_head drivor --exp_name drivor \
  --train_set_list "$TRAIN_LIST" --valid_set_list "$VALID_LIST" \
  --save_dir "$RUN" --train_epochs "$EPOCHS" --save_utd 1 \
  --train_subsample_step 1 --valid_subsample_step 8 \
  --batch_size "$BATCH" --num_workers 8 --prefetch_factor 4 \
  --drivor_num_poses 40 --drivor_pose_dt 0.1 \
  --drivor_lr_schedule drivor \
  --learning_rate 1e-4 --drivor_lr_base_batch_size 64 \
  --drivor_warmup_ratio "$RATIO" \
  --drivor_human_teacher_weight 0.3 \
  --use_amp --amp_dtype bf16 --compile_model --compile_mode default \
  --use_ema True --drivor_fused_ema True --drivor_guard_sync_every 1 \
  --use_wandb True --wandb_project_name "<entity>/<project>" \
  --closed_loop_npz_root ""
```

Batch 512 on 8xH100 is 43.8 GiB/device and ~2.1 steps/s; 1024 OOMs at step 0, so
512 is a ceiling, not a preference. Budget in *steps*, not epochs: ~450k steps is
~2.5 days, which is 40 epochs of a 10k-step corpus but 5 of a 90k-step one. Derive
`--train_epochs` from `STEPS_PER_EPOCH` rather than carrying a number across
datasets — it also sets the cosine's `T_max`. `DP_DIST_INIT_FILE` replaces the hardcoded
`/tmp/tmp_dist_init` rendezvous file. `train_run.py` is not a shortcut for this
path: it hardcodes `--train_epochs 80` and passes no `--predictor_head`.

### Defaults that are wrong for this head

Four belong to the diffusion path, one is DrivoR's at a different scale. Check
them in `args.json` after launch.

| flag | default | pass | why |
|---|---|---|---|
| `--predictor_head` | `diffusion` | `drivor` | otherwise none of this runs |
| `--drivor_lr_base_batch_size` | `0` | `64` | 0 disables the sqrt rule, making `--learning_rate` the literal peak |
| `--drivor_warmup_ratio` | `0.1` | derived | a fraction of *total* steps: 0.1 is DrivoR's ~3.9k-step ramp, but 42,548 steps here |
| `--use_amp --amp_dtype bf16` | off | on | every number in this doc was measured under bf16 + TF32 |
| `--compile_model` | off | on | ~1.6x on the step, lower peak memory |
| `--train_epochs` | `40` | what you will run | the cosine's `T_max` comes from it, so an aspirational value never decays |

Everything else is DrivoR-faithful: 64 proposals, `ref_num` 4, `scorer_ref_num` 4,
sub-score weights 0/5/5/2, `label_smoothing` 0.02, `logit_bound` 10, `grad_clip` 1.

### Reading the run

Two lines per epoch on rank 0:

```
Epoch 7/40  val ADE=… FDE=… oracle_selected=… oracle_best=… top1=…
devkit panel: PDMS=… NC=… DAC=… DDC=… TTC=… EP=… Comfort=…
```

`oracle_best - oracle_selected` is what better *selection* alone could still buy.
The second line is the devkit's own panel and is the number comparable to devkit
results. `perf/*` goes to W&B every `--drivor_log_every_n_steps`.

```
$RUN/args.json                  every flag as resolved -- check this first
$RUN/train_log.tsv              per-epoch metrics, no W&B required
$RUN/latest.pth                 per epoch; resume from here
$RUN/best_model/                best val/selection/oracle_selected so far
$RUN/epoch0007/                 every --save_utd epochs
```

Checkpoint selection tracks `val/selection/oracle_selected`, not a validation
loss: this head is trained to *choose*, and a lower regression loss over 64
proposals does not imply a better pick.

Resume with `--resume_model_path "$RUN/latest.pth"` (model, EMA, optimizer,
scheduler and epoch counter) plus optional `--wandb_run_id`. Pass the same
`--train_epochs` or the cosine changes shape mid-flight; `--save_utd` counts from
the resumed epoch.

### Deployment

The ROS node builds the model from the run's own `args.json`
(`diffusion_planner_node.py:77` → `utils/config.py`), so the deployed head
inherits the trained `--drivor_num_poses` and `--drivor_human_teacher_weight`
automatically — there is no separate inference-time setting. Keep the `args.json`
and the `.pth` together: `0.2` and `0.3` give identical `state_dict` shapes, so
pairing a checkpoint with another run's config silently changes which proposal is
selected, with no load error.

Inert on this path (they parse, they do nothing): closed-loop rollout validation,
temporal-stability / replan-consistency metrics, ONNX export. All three assume
the DiT sampler's inputs and the neighbour/turn-indicator outputs.

### Troubleshooting

| symptom | cause |
|---|---|
| OOM at step 0 | `--batch_size` is global. 512/8 GPUs = 43.8 GiB each; 1024 does not fit in 80 GiB. |
| LR near zero for whole epochs | `--drivor_warmup_ratio` carried over from a run with fewer total steps. |
| PDMS not comparable to devkit numbers | `--drivor_num_poses * --drivor_pose_dt` ≠ navsim's 4 s. |
| val PDMS implausibly high | the valid list was never skip-filtered ([above](#training)). |
| `Comfort` identically 0 | needs the simulator rollout, not finite differences ([below](#comfort-needs-the-simulator)). 0 on ~3-5 % of scenes is expected — the recorded-past prefix gates them. |
| rendezvous hangs at startup | stale `/tmp/tmp_dist_init`; set `DP_DIST_INIT_FILE` per run. |
| W&B goes offline despite `--use_wandb True` | the job cannot see the credential. `HOME` must point where the `.netrc` actually is — on a cluster, `/home` is per-node. |
| many guard skips | LR above the usable band. Measure it with `--drivor_lr_schedule probe` (geometric range test, prints `[lr-probe]`, then stops); DrivoR's own band does not transfer to this encoder. |
| `--compile_mode reduce-overhead` slower | expected. CUDA graphs measured 45 % slower: once the EMA's Python loop is gone the step is GPU-bound, not launch-bound. |

## Design notes

### Architecture

```
encoder tokens ──► 64 learned proposal queries
                      │  TransformerDecoder, ref_num=4 refinement stages
                      │  per-stage MLP head; proposals detached and re-embedded
                      ▼
                   64 candidate trajectories  [B, 64, T, 4]  (x, y, cos, sin)
                      │  scorer decoder (scorer_ref_num=4) + ego token
                      ▼
                   6 independent BCE logit heads (+ human-teacher head)
                      │  PDMS: NC x DAC x weighted-mean(DDC, TTC, EP, Comfort)
                      │  weights 0/5/5/2, plus 0.3 * sigma(human) as tie-break
                      ▼
                   argmax ──► the single ego trajectory
```

Losses (`model/module/drivor_loss.py`): WTA L1 over every refinement stage
(accumulated with `prev_weight`); six BCE heads supervised by PDM oracle labels
computed online for the model's *own* proposals (TTC's `2.0` sentinel masked,
NC/DDC through `three_to_two_classes`, labels smoothed 0.02, logits bounded by
`cap * tanh(raw / cap)`); human-teacher BCE against `1 / (1 + proposal_error)`.
Deliberately dropped from DrivoR: Hungarian agent loss, BEV semantic CE,
drivable-area raster, and `diversity_loss` (DrivoR runs `inter_weight=0`).

### Trajectory sampling

Three grids meet here; `utils/drivor_sampling.py` is the only place that converts
between them (21 tests).

| grid | sampling | fixed by |
|---|---|---|
| dataset `ego_agent_future` | 80 poses @ 0.1 s (8 s) | the NPZ shards |
| **head output** | **40 poses @ 0.1 s (4 s)** | `--drivor_num_poses` / `--drivor_pose_dt` |
| PDM scorer | 40 steps @ 0.1 s | `pdm_scoring/default_scoring_parameters.yaml` |

The **horizon** is not free. The devkit scores `num_poses: 40, interval_length:
0.1`, so a head emitting 8 s produces a PDMS incomparable to the devkit's — and
not merely rescaled, since two sub-scores are *easier* over 8 s (on the expert
trajectory: DAC 0.9629 vs 0.9766, DDC 0.9453 vs 0.9766). `--future_len 80` is the
diffusion decoder's setting and is unrelated.

The **density** inside those 4 s *is* free: the head is one-shot, so the pose
count is a linear layer's width, not a rollout length. At B=64 on one H100 the
model's forward+backward is 365 ms whether it emits 8, 40 or 80 poses; the full
step is 401 ms at 8 @ 0.5 s, **391 ms at 40 @ 0.1 s**, 399 ms at 80 @ 0.1 s.
Hence the default — navsim's horizon at the 10 Hz the dataset and controller
already use, so nothing is interpolated anywhere.

DrivoR upstream emits `num_poses: 8` at `t4_trajectory_dt_s: 0.5`;
`--drivor_num_poses 8 --drivor_pose_dt 0.5` reproduces it and still lines up,
because `upsample_poses` interpolates onto the scoring grid exactly as navsim's
`transform_trajectory` + `get_trajectory_as_array` do (linear x/y; unwrap →
linear → wrap on heading; anchored on the ego pose at t=0), checked to 1e-12
against a numpy transcription of nuPlan's `InterpolatedTrajectory`. Over 512
validation scenes both representations of the *expert* trajectory score PDMS
0.9680 on all six sub-scores, so the coarse grid is a control-side choice, not an
accuracy one.

### The PDM oracle

`utils/drivor_oracle.py` is a batched GPU re-implementation of the devkit's PDM
scorer over Diffusion-Planner's ego-centric NPZ tensors — no devkit `Scene`
objects, no CPU round-trip. 26 ms and 0.40 GiB at B=64, N=64, under 7 % of a
391 ms step. Every shape is static (no `.item()`, `torch.compile`-friendly), and
the candidate prefilters (`max_neighbours`, `max_border_segments`,
`max_route_segments`) are made provably conservative by a per-timestep guard ball
covering the whole proposal set plus the expert path: anything beyond
`guard_radius + ego_radius + other_radius` cannot touch a scored trajectory.
`tests/test_drivor_oracle.py` (47 tests) checks each metric against a scalar
reference.

The lineage is NAVSIM **v1** — the copy vendored in DrivoR and in the e2e devkit,
not `references/navsim` (v2). The two differ in the Savitzky-Golay windows the
comfort filters use (v1 passes `window_length=n_time` at four of the six sites;
v2 keeps the 8/15 defaults) and in `state_array_to_center_state_array`, which
exists only in v2. Both halves of this port are v1: full windows, and comfort
read off the rear-axle state while the *footprint* metrics (EP, DDC, ego-area,
collision polygons) use `BBCoordsIndex.CENTER`. v1's windows are the far stronger
low-pass — the noise level at which a recorded past stops being comfortable moves
from ~1-2 cm to ~10-30 cm.

### The simulator: what gets scored

`PDMScorer` never sees the proposal. It is handed the states
`PDMSimulator.simulate_proposals` produces — a `BatchLQRTracker` whose commands a
`BatchKinematicBicycleModel` integrates — so a trajectory the tracker cannot
follow is judged on where the vehicle actually ends up. Every metric reads that
rollout: NC, TTC, DAC, DDC, EP and comfort. Two consequences worth stating:

- **EP credits achieved progress.** A proposal that asks for 8 m/s from a
  standstill scores ~0.73, not 1.0, because the vehicle never covers the expert's
  arclength. On raw poses it was indistinguishable from a followable one.
- **The horizon is one row longer.** Row 0 of the rollout is the ego's own
  current state (NAVSIM scores `range(num_poses + 1)`), so the agent-box lists
  NC/TTC receive carry a frame 0 read off `neighbor_agents_past[:, :, -1]` — also
  the only frame with *recorded* agent velocities, which is what the
  stopped-track branch reads (NAVSIM takes it from `unique_objects[token]`, the
  box at first appearance, never from the frame being scored).

Comfort is where the rollout matters most. The tracker's first-order lags
(`accel_time_constant` 0.2 s, `steering_angle_time_constant` 0.05 s) remove
exactly the high-frequency component finite-differencing amplifies by
`1/dt² = 100`. Without it every proposal fails `lon_accel` and the label
collapses to a constant 0, killing both the comfort head's gradient and the
metric's share of the aggregate. On real proposals from an epoch-3 checkpoint:

| check | simulated | finite differences |
|---|---|---|
| lon_accel | 0.830 | 0.000 |
| lat_accel | 1.000 | 0.000 |
| mag_jerk | 0.960 | 0.000 |
| lon_jerk | 0.570 | 0.000 |
| yaw_accel | 1.000 | 0.141 |
| yaw_rate | 1.000 | 0.998 |

(`|smoothed acc_lon|` p50 goes 3.340 → 0.465 against bounds `+2.40 / -4.05`.)

Two implementations live in `planner_metrics/`: `pdm_simulator.py`, a 1:1 numpy
transliteration of NAVSIM (same loops, same `einsum` order, same `pinv`) kept as
ground truth; and `pdm_simulator_torch.py`, the batched fp64 GPU version the
oracle and devkit call. Three computations there are regrouped algebraically, not
approximated — the velocity/acceleration fit collapses to one constant matmul
(its normal matrix is pose-independent), the curvature fit's SVD-`pinv` becomes a
Cholesky solve (the system is SPD), and the 10-step lateral LQR product becomes
closed form (only one off-diagonal product is non-zero). fp64 because the fits are
ill-conditioned enough that fp32 changes the commands, and on an H100 fp64 costs
~2x fp32, i.e. nothing here. `torch.compile` takes B=2048 x T=80 from 80.2 ms to
14.7 ms. `test_pdm_simulator.py` makes "the regrouping is exact" a checked claim:
fast-vs-literal to <1e-8 on positions, compiled-vs-eager and CUDA-vs-CPU to
<1e-9, on trajectories carrying deliberate pose noise (a clean path would hide
the component the tracker filters).

Two NAVSIM properties are reproduced rather than fixed: `ACCELERATION_Y` is
identically zero in every simulated state, which makes the lateral-accel bound
structurally vacuous and collapses magnitude-jerk onto `|lon jerk|`; and
`PDMSimulator` never sets the tracker's `_wheel_base`, so the tracker tracks every
vehicle as a Pacifica (3.089 m). We keep that asymmetry and feed the motion model
the dataset's real wheel base from `ego_shape[0]` — `v * tan(steer) / wheelbase`
then reproduces the stored `yaw_rate` exactly, so this is not cosmetic.

**NC and TTC keep a per-track ledger.** NAVSIM does not judge each frame
independently. Tracks already touching the ego at t=0 (`collided_track_ids`) are
excluded for the whole horizon — otherwise the ego is found at fault against them
on every subsequent frame — and a track is *retired* the first time it touches
the ego without the ego being at fault, so a car that overtakes and is then
brushed from the side cannot be re-judged. TTC keeps its own independent ledger.
Both reduce to "the verdict at a track's first contact is the only one that
matters", which on the GPU is a `cummax` over the step axis rather than a
sequential scan; TTC folds the future-projection axis after the step axis to
preserve NAVSIM's `(time_idx, future_idx)` ordering. The tokens come from the
neighbour slot index, which DP keeps stable across frames.

**TTC scores the whole horizon.** NAVSIM extends its observation by 1 s
(`extend_observation_for_ttc`) precisely so the last `num_poses` rows keep a
projection target for the `[0, 3, 6, 9]`-step lookahead. Truncating the tail
instead — as this port used to — blinds TTC to collisions at the end of the
horizon, which are the ones a planner is most likely to produce.

**EP's denominator is the expert, not PDM-Closed.** NAVSIM normalises progress by
`max(raw_progress, pdm_progress)` (`train_pdm_scorer.py::_aggregate_scores`),
where `pdm_progress` is a *cached scalar*: at caching time the devkit runs the
rule-based PDM-Closed planner, simulates its trajectory and scores it, storing
the result in `MetricCache` alongside the scene (`train_cache_processor.py`). It
is the one entry in that cache that is not proposal-independent geometry, and it
is cacheable only because PDM-Closed does not depend on the model. We have no
such baseline: PDM-Closed needs nuPlan's map API — roadblock connectivity, speed
limits, an IDM policy — and the NPZ shards carry route-lane *geometry* only, so
the expert future plays the reference-proposal role instead
(`pdms_navsim.py::ego_progress_with_gate`). The structure is reproduced exactly —
`max(ref_progress, raw)`, the 5 m `progress_distance_threshold` gate and its
1.0/0.0 branches — only the reference's source differs. The denominator is shared
by every proposal in a scene, so EP stays a monotone (1.0-saturating) rescaling
and proposal *ranking* within a scene is untouched; what is not devkit-comparable
is the absolute EP value, and through it the weighted aggregate. Restoring
`pdm_progress` means re-running caching against the nuPlan source, not a change
in this port.

**`history_comfort` is not NAVSIM's `COMFORTABLE`.** The devkit's key is
load-bearing: `navsim_score.py::_history_comfort` finite-differences the ego's
*recorded past*, drops its last pose, concatenates it ahead of the simulated
future and requires all six bounds over the whole 30 + 41 rows.
NAVSIM's `COMFORTABLE` scores the rollout alone. The prefix is reconstructed at
training time from the NPZ's `ego_agent_past` — no dataset change, no cache. Two
consequences: the prefix is shared by every proposal in a scene, so a rough
recorded past gates the whole scene (4.7 % of 128 scenes, 3.3 % over a separate
300-scene draw; comfort mean 0.961 → 0.916, PDMS 0.877 → 0.870, oracle +2 %); and
under `StatePerturbation` the past is taken in the recorded frame, which is
correct rather than merely convenient — the bounds read body-frame accel channels
and the raw heading, so re-framing changes nothing except injecting the
perturbation as a step at the junction (accel channels agree to 4e-4 either way,
while the junction heading step grows 0.031 → 0.196 rad and `yaw_accel` falls
1.000 → 0.973).

### Metrics

`utils/drivor_metrics.py` reproduces DrivoR's taxonomy: `perf/*` per-step
diagnostics, `train/*` and `val/*` epoch aggregates, DrivoR's `_metric_name`
display table, and selection diagnostics (`oracle_selected`, `oracle_best`,
`oracle_gap`, `oracle_rank`, `top1_hit`, `top5_hit`). `utils/devkit_wandb.py` is a
**verbatim copy** of the devkit's `navsim/evaluation/wandb.py`, so the
`devkit/{pdms,nc,dac,ddc,tlc,ttc,ep,lk,comfort,ec}` panel is byte-for-byte the
devkit's own; it scores the *selected* trajectory alone, so `ego_progress` is
measured against the demonstration rather than the model's best proposal.

Two reporting details: soft-target BCE is `H(labels) + KL(labels || prediction)`
and only KL has a gradient, so with 0.02 smoothing the raw scorer loss can never
reach zero — `loss/learnable/*` reports the reducible KL remainder and
`loss/learnable/label_entropy_floor` the constant beneath it. And because the PDM
score is a product of bounded sub-scores it ties frequently, so the selection
metrics are tie-aware: any proposal matching the best label counts as a hit, and
`oracle_rank` ranks by label value rather than index.

### Performance

bf16 autocast, TF32 matmul/conv, `torch.compile` (~1.6x on the step), fused
AdamW, `persistent_workers` + `prefetch_factor`, `find_unused_parameters=False`
with `gradient_as_bucket_view=True`, and a step written to avoid host syncs:
per-step scalars accumulate on the GPU (one all-reduce and one readback per
epoch), gradient statistics only on the log cadence, and the divergence guard's
readback is issued *after* `backward()` is enqueued so it overlaps real work.

The encoder dominates: at B=64 uncompiled it is 94.6 % of peak memory and 83.6 %
of the step (44.2 GiB / 344 ms of a 46.8 GiB / 412 ms step); the head adds
2.1 GiB / 10 ms, the oracle plus loss 0.4 GiB / 58 ms. No packed sample cache is
needed — 8 workers per rank deliver 180-2200 samples/s against a GPU consuming
~100/s.

### Divergence guard

Non-finite loss, `|logit| > 15`, or `loss > 4x` the running EMA skips the
optimizer step. The flag is MAX-all-reduced exactly once per step, never inside a
conditional, so ranks skip together and NCCL cannot deadlock. More than 10 skips
in a 200-step window halves the LR; a separate slow logit-drift EMA has its own
cut with a 500-step cooldown.
