# DrivoR predictor head

`--predictor_head drivor` replaces Diffusion-Planner's DiT/diffusion decoder with
DrivoR's proposal-generate-then-score head. The encoder, dataset, augmentation,
EMA and LR schedule are unchanged; everything downstream of the encoder is
DrivoR's, and the head outputs **only an ego trajectory** — no neighbour
prediction, no turn indicator.

## What the head does

```
encoder tokens ──► 64 learned proposal queries
                      │  TransformerDecoder, ref_num=4 refinement stages
                      │  per-stage MLP trajectory head, detached proposals
                      │  re-embedded through pos_embed between stages
                      ▼
                   64 candidate trajectories  [B, 64, T, 4]  (x, y, cos, sin)
                      │  scorer decoder (scorer_ref_num=4) + ego token
                      ▼
                   6 independent BCE logit heads (+ human-teacher head)
                      │  PDMS aggregate: NC x DAC x weighted-mean(DDC, TTC, EP, Comfort)
                      │  weights 0 / 5 / 5 / 2, plus 0.2 * sigma(human) as tie-break
                      ▼
                   argmax ──► the single ego trajectory
```

Losses (`model/module/drivor_loss.py`):

- **WTA L1** over every refinement stage, accumulated with `prev_weight`.
- **Six BCE heads** supervised by PDM oracle labels computed online for the
  model's *own* proposals; TTC's `2.0` sentinel is masked, NC/DDC go through
  `three_to_two_classes`, labels are smoothed by 0.02, logits are bounded by
  `cap * tanh(raw / cap)`.
- **Human-teacher BCE** against `1 / (1 + proposal_error)`.

Deliberately dropped from DrivoR: Hungarian agent loss, BEV semantic CE,
drivable-area raster, and `diversity_loss` (DrivoR runs `inter_weight=0`).

## Output trajectory sampling

Three grids meet in this head, and `utils/drivor_sampling.py` is the only place
that converts between them:

| grid | sampling | fixed by |
|---|---|---|
| dataset `ego_agent_future` | 80 poses @ 0.1 s (8 s) | the NPZ shards |
| **head output** | **40 poses @ 0.1 s (4 s)** | `--drivor_num_poses` / `--drivor_pose_dt` |
| PDM scorer | 40 steps @ 0.1 s | `pdm_scoring/default_scoring_parameters.yaml` |

The **horizon** is not a free choice: the devkit scores `proposal_sampling:
num_poses: 40, interval_length: 0.1`, so a head emitting 8 s makes PDMS
incomparable to the devkit's — and not merely rescaled, since two sub-scores are
*easier* over 8 s (measured on the expert trajectory: DAC 0.9629 vs 0.9766, DDC
0.9453 vs 0.9766). Diffusion-Planner's `--future_len 80` is unrelated to this
head and is left alone; only the expert target is re-sampled.

The **density** inside those 4 s *is* free, because the head is one-shot: the
pose count is one linear layer's width, not a rollout length. Measured on one
H100 at batch 64:

| head output | scored | model f+b | oracle | step |
|---|---|---|---|---|
| 8 @ 0.5 s | 40 @ 0.1 s | 370.6 ms | 30.2 ms | 400.9 ms |
| **40 @ 0.1 s** | 40 @ 0.1 s | 364.5 ms | 26.1 ms | **390.6 ms** |
| 80 @ 0.1 s | 80 @ 0.1 s | 364.8 ms | 33.7 ms | 398.5 ms |

Hence the default: navsim's horizon at the 10 Hz the dataset and the downstream
controller already use, so nothing is interpolated anywhere — the expert target
is `slice(0, 40)` of the stored rows and the proposals reach the oracle
untouched. DrivoR upstream instead emits `num_poses: 8` (`drivoR.yaml`) at
`t4_trajectory_dt_s: 0.5` (`t4_training.yaml`); `--drivor_num_poses 8
--drivor_pose_dt 0.5` reproduces that and everything still lines up, because
`upsample_poses` then interpolates the proposals onto the scoring grid exactly
the way navsim's `transform_trajectory` + `get_trajectory_as_array` do (linear in
x/y, unwrap → linear → wrap on the heading, anchored on the ego pose at t=0,
checked against a numpy transcription of nuPlan's `InterpolatedTrajectory` to
1e-12). Over 512 validation scenes the two representations of the *expert*
trajectory score identically — PDMS 0.9680 on all six sub-scores — so the
coarse grid is a control-side choice, not an accuracy one.

`tests/test_drivor_sampling.py` (21 tests) covers both configurations.

## The PDM oracle

`utils/drivor_oracle.py` is a batched GPU re-implementation of the devkit's PDM
scorer, working directly on Diffusion-Planner's ego-centric NPZ tensors — no
devkit `Scene` objects, no CPU round-trip. 26 ms and 0.40 GiB at B=64, N=64 over
the scorer's 40 steps, i.e. under 7 % of a 391 ms training step.

Every shape is static (no `.item()`, `torch.compile`-friendly). Candidate
prefilters (`max_neighbours`, `max_border_segments`, `max_route_segments`) are
made *provably conservative* by a per-timestep guard ball that covers the whole
proposal set plus the expert path: anything farther than
`guard_radius + ego_radius + other_radius` cannot touch any scored trajectory.

`tests/test_drivor_oracle.py` (37 tests) checks each metric against a scalar
reference implementation.

### Comfort needs a simulator, not finite differences

Comfort is the one sub-score that cannot be read off the poses. NAVSIM's
`PDMScorer` never applies the comfort bounds to waypoints: it scores the states
`PDMSimulator.simulate_proposals` produces — a `BatchLQRTracker` computing
(acceleration, steering-rate) commands that a `BatchKinematicBicycleModel`
integrates. The model's first-order lags (`accel_time_constant = 0.2 s`,
`steering_angle_time_constant = 0.05 s`) remove exactly the high-frequency
component that finite-differencing amplifies by `1/dt**2 = 100`. Without the
rollout every proposal fails `lon_accel` and the label collapses to a constant 0,
which kills both the comfort head's gradient and the metric's share of the
aggregate. Measured on real proposals from an epoch-3 checkpoint:

| check | simulated | finite differences |
|---|---|---|
| lon_accel | 0.830 | 0.000 |
| lat_accel | 1.000 | 0.000 |
| mag_jerk | 0.960 | 0.000 |
| lon_jerk | 0.570 | 0.000 |
| yaw_accel | 1.000 | 0.141 |
| yaw_rate | 1.000 | 0.998 |

`|smoothed acc_lon|` p50 goes 3.340 -> 0.465 against bounds `+2.40 / -4.05`.

Two implementations live in `planner_metrics/`:

- `pdm_simulator.py` — a 1:1 numpy transliteration of NAVSIM: same loops, same
  `einsum` order, same `pinv`. This is the ground truth, and the honest answer to
  "is it really the same".
- `pdm_simulator_torch.py` — the batched fp64 GPU version the oracle and devkit
  actually call. Three computations are regrouped algebraically, not
  approximated: the velocity/acceleration fit collapses to one **constant**
  matmul (its normal matrix `A^T A = L^T L` is pose-independent), the curvature
  fit's SVD-`pinv` becomes a batched Cholesky solve (the system is SPD), and the
  10-step lateral LQR product becomes closed form (only one off-diagonal product
  of the per-step matrices is non-zero). fp64 because the fits are ill-conditioned
  enough that fp32 changes the commands — on an H100 fp64 is ~1:2 of fp32, so it
  costs nothing. `torch.compile` takes B=2048 x T=80 from 80.2 ms to 14.7 ms,
  ~2.7 % of a training step.

`planner_metrics/test_pdm_simulator.py` is what makes "the regrouping is exact" a checked
claim: fast-vs-literal agrees to <1e-8 on positions and <1e-7 overall,
compiled-vs-eager and CUDA-vs-CPU to <1e-9. The trajectories under test carry
deliberate pose noise — a clean path would hide the very component the tracker
filters.

Two NAVSIM properties are reproduced rather than "fixed":

- `ACCELERATION_Y` is identically zero in every simulated state
  (`_update_commands` writes `0.0`, `propagate_state` copies `ACCELERATION_2D`
  through), which makes the lateral-accel bound structurally vacuous and
  collapses magnitude-jerk onto `|lon jerk|`.
- `PDMSimulator` sets the motion model's vehicle from the ego but never the
  tracker's `_wheel_base`, so the tracker tracks every vehicle as a Pacifica
  (3.089 m). We keep that asymmetry, and feed the motion model the dataset's real
  wheel base from `ego_shape[0]` (this dataset's ego is a 10.7 m vehicle with a
  4.99 m wheel base — `v * tan(steer) / 4.99` reproduces the stored `yaw_rate`
  exactly, so this is not cosmetic).

### `history_comfort` is not NAVSIM's `COMFORTABLE`

The devkit's panel key is `history_comfort`, and the name is load-bearing:
`navsim_score.py::_history_comfort` finite-differences the ego's *recorded past*,
drops its last pose (the rollout's row 0 already is the current state),
concatenates it in front of the simulated future and requires all six bounds over
the whole 30 + 41 rows. NAVSIM's `COMFORTABLE` scores the rollout alone. The
prefix is reconstructed at training time from the NPZ's `ego_agent_past`
(31 rows x (x, y, heading) at dt = 0.1 s, oldest first, last row exactly
(0, 0, 0)) — no dataset change, no cache.

Two consequences worth knowing before reading the label:

- The prefix is **shared by every proposal in a scene**, so a rough recorded past
  is a scene-level gate: comfort is 0 for all 64 proposals no matter how smooth
  they are. Measured over 128 real scenes x 64 proposals, 4.7 % of scenes are
  gated this way (3.3 % over a separate 300-scene draw), comfort mean drops
  0.961 -> 0.916 and the PDMS aggregate 0.877 -> 0.870. Oracle cost is +2 %
  (104.0 -> 106.6 ms).
- Under `StatePerturbation` the past is taken in the frame the augmentation leaves
  it in — the recorded one, whose transform block is commented out. That is also
  the correct choice, not just the convenient one: the six bounds read the
  body-frame accel channels (invariant under a rigid transform, since they are
  differenced inside the segment and projected per row) and the raw heading, so
  re-framing into the perturbed frame changes nothing except to inject the
  perturbation as a step at the junction. Over 256 augmented samples the accel
  channels agree to 4e-4 either way, while the junction heading step grows from
  0.031 to 0.196 rad and `yaw_accel` falls 1.000 -> 0.973.

No PDM cache is required. The rollout reads only X, Y, HEADING, VELOCITY_X,
ACCELERATION_X, STEERING_ANGLE and ANGULAR_VELOCITY from the initial state, all
of which `ego_current_state` already carries; DrivoR's cache exists for the
map/agent/progress side, which this path recomputes from the NPZ tensors.

## Metrics

- `utils/drivor_metrics.py` reproduces DrivoR's metric taxonomy: `perf/*` live
  per-step diagnostics, `train/*` and `val/*` epoch aggregates, DrivoR's
  `_metric_name` display table, selection diagnostics (`oracle_selected`,
  `oracle_best`, `oracle_gap`, `oracle_rank`, `top1_hit`, `top5_hit`).
- `utils/devkit_wandb.py` is a **verbatim copy** of the e2e devkit's
  `navsim/evaluation/wandb.py`, so the `devkit/{pdms,nc,dac,ddc,tlc,ttc,ep,lk,
  comfort,ec}` panel is byte-for-byte the devkit's own. The panel scores the
  *selected* trajectory on its own, so `ego_progress` is measured against the
  demonstration rather than against the model's best proposal.

Checkpoint selection uses `val/selection/oracle_selected` — the PDM score of the
trajectory the scorer actually picks.

Two reporting details worth knowing when reading the panels:

- Soft-target BCE is `H(labels) + KL(labels || prediction)`, and only the KL term
  has a gradient. Since the labels are smoothed by 0.02, `H(labels) > 0` and the
  raw scorer loss can never reach zero. `loss/learnable/*` reports the KL
  remainder — the part training can actually reduce — and
  `loss/learnable/label_entropy_floor` reports the constant it sits on, so a
  flat-looking `loss/scorer/*` can be checked against its own floor.
- `top1_hit` compares against the oracle's argmax, and the PDM score is a product
  of bounded sub-scores that ties frequently (whole groups of proposals share the
  same label). The selection metrics are tie-aware: any proposal whose label
  equals the best label counts as a hit, and `oracle_rank` ranks by label value
  rather than by index, so ties do not turn into arbitrary misses.

## Accelerations

bf16 autocast (`--use_amp --amp_dtype bf16`), TF32 matmul/conv, `torch.compile`
(`--compile_model`, ~1.6x on the step and lower peak memory), fused AdamW,
`persistent_workers` + `prefetch_factor`, `find_unused_parameters=False` with
`gradient_as_bucket_view=True`, and a step written to avoid host syncs: per-step
scalars accumulate on the GPU (one all-reduce + one readback per epoch), gradient
statistics only on the log cadence, and the divergence guard's readback is issued
*after* `backward()` is enqueued so it overlaps with real work.

The encoder dominates: at B=64 uncompiled it is 94.6 % of peak memory and 83.6 %
of the step (44.2 GiB / 344 ms of a 46.8 GiB / 412 ms step), the head adds
2.1 GiB / 10 ms and the oracle plus loss 0.4 GiB / 58 ms. No packed
sample cache is needed: 8 dataloader workers per rank deliver 180-2200 samples/s
against a GPU that consumes ~100/s per rank.

## Divergence guard

Non-finite loss, `|logit| > 15`, or `loss > 4x` the running EMA skips the
optimizer step. The flag is MAX-all-reduced exactly once per step, never inside a
conditional, so ranks skip together and NCCL cannot deadlock. More than 10 skips
in a 200-step window halves the LR; a separate slow logit-drift EMA has its own
cut with a 500-step cooldown.

## Launching

```bash
cd diffusion_planner
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DP_DIST_INIT_FILE=$RUN/dist_init \
DRIVOR_WANDB_PATH=<entity>/<project> \
../.venv/bin/torchrun --nproc_per_node=8 --master_port=29650 train_predictor.py \
  --exp_name drivor \
  --predictor_head drivor \
  --train_set_list "$TRAIN_SET_LIST" \
  --valid_set_list "$VALID_SET_LIST" \
  --train_subsample_step 1 --valid_subsample_step 8 \
  --save_dir "$RUN" --train_epochs 6 --save_utd 1 \
  --batch_size 512 --num_workers 8 --prefetch_factor 4 \
  --drivor_lr_schedule drivor \
  --learning_rate 1e-4 --drivor_lr_base_batch_size 64 \
  --drivor_warmup_ratio 0.031337 \
  --use_amp --amp_dtype bf16 --compile_model --compile_mode default \
  --use_ema True --drivor_fused_ema True --drivor_guard_sync_every 1 \
  --drivor_human_teacher_weight 0.2 \
  --enable_replan_consistency_eval True \
  --use_wandb True
```

Global batch 512 is the largest that fits: it uses 43.8 GiB per device, so 1024
runs out of memory at step 0 on an 80 GiB card. The `sqrt` rule then puts the
peak LR at `1e-4 * sqrt(512/64) = 2.83e-4`. `--drivor_warmup_ratio` is a
fraction of *total* steps, so it must be re-derived whenever the batch or epoch
count changes if the ramp is to stay the same absolute length — 0.031337 is a
2,000-step ramp over 6 epochs at this batch.

`DP_DIST_INIT_FILE` overrides the hardcoded `/tmp/tmp_dist_init` rendezvous file
— needed whenever another user's stale file is in the way or two jobs run on the
same node.

Not available on this path: closed-loop rollout validation, temporal-stability /
replan-consistency metrics and ONNX export are diffusion-specific (their
wrappers assume the DiT sampler's inputs and the neighbour/turn-indicator
outputs). The flags parse but are inert.
