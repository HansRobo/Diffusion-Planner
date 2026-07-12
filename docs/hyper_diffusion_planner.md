# Hyper Diffusion Planner integration guide

This document defines how this branch should be used for HDP experiments in the Tier IV Diffusion Planner codebase.

## Decision: HDP-only branch

This branch has one model contract: ego-only HDP with normalized velocity actions,
x-start supervision, and 80 temporal action tokens. Vanilla waypoint and joint neighbor-action
heads are not retained as compatibility modes.

## Local ground truth references

Implementation decisions should be checked against local references first:

```text
reference/hyper_diffusion_planner_paper/src/
reference/external/Hyper-Diffusion-Planner/
```

Relevant paper-side item:

```text
reference/hyper_diffusion_planner_paper/src/code_rl.tex
```

Relevant official implementation-side items:

```text
reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py
reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/dp_vla/model/rl_utils.py
reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/config/agent/dp_vla_rl_agent.yaml
```

## Model changes relative to original Diffusion Planner

Original Diffusion Planner predicts all-agent future waypoints directly.

This HDP branch keeps the original scene encoder, turn-indicator path, and validation stack, but changes the ego planning target in HDP mode:

- Ego target is represented as a normalized velocity/action latent rather than a direct waypoint latent.
- The HDP ego prediction is converted back to waypoints through integration for trajectory loss and evaluation.
- Hybrid loss combines velocity-space diffusion supervision with waypoint-space reconstruction.
- The HDP action head is ego-only; all 320 neighbor histories remain scene context.
- Detailed route tokens remain in scene cross-attention, while an official-style ordered route
  encoder supplies a lightweight global AdaLN condition for every DiT block.
- Turn indicators and Tier IV validation metrics remain available.
- Turn-indicator SFT uses equal-weight expert and detached generated-trajectory supervision by default, removing the pure teacher-forcing train/inference mismatch without allowing classification gradients to distort the planned trajectory.

The intended HDP supervised setting is:

```text
use_velocity_representation=True
planning_hybrid_loss=0.01
hybrid_loss_window=10
diffusion_model_type=x_start
diffusion_supervision_type=x_start
diffusion_sample_steps=6
weight_decay=0.01
```

## Checkpoint rules

Use the right loading mode.

| Use case | Flag |
| --- | --- |
| Start SFT from base weights | `--init_weights_path` |
| Start RL from SFT weights | `--init_weights_path` |
| Continue the exact same interrupted run | `--resume_model_path` |
| Start HDP from a vanilla waypoint checkpoint intentionally | `--init_weights_path` only |

Do not use `--resume_model_path` to change representation or action shape. Strict resume requires
model, optimizer, scheduler, epoch, per-rank RNG state, and (when enabled) EMA state; it also
requires the same DDP world size and checks architecture plus exact normalization statistics
before the new run can overwrite `args.json`. Checkpoints use atomic replacement so interruption
cannot leave a partial `latest.pth`, and store the W&B run ID so automatic recovery continues the
same run.

The branch has one encoder implementation. It includes valid-point LineEncoder geometry from Tier IV PR
[#212](https://github.com/tier4/Diffusion-Planner/pull/212) and categorical turn-history
encoding from [#210](https://github.com/tier4/Diffusion-Planner/pull/210). There is no
legacy mode. #210 changes a weight shape, so old Base/SFT checkpoints fail loading instead
of silently reusing stale encoder features. Train the new ego-only Base from scratch.

## Data policy

Use fixed full-sequence lists for HDP experiments.

Reasons:

- HDP uses ego history and temporal behavior as part of the model design.
- Temporal stability metrics need consecutive frames.
- Mixed project/area lists can change the distribution and make comparisons unfair.

Single-frame lists can still run ordinary supervised training, but they are not sufficient for temporal-consistency evaluation.

The current 2026-06 Tier IV corpus was generated before converter commit `55eff4f` and duplicates the current frame in short neighbor futures. `align_legacy_neighbor_futures=true` detects those short tracks in `Dataset.__getitem__`, reads from the next frame, and zero-pads the tail in worker memory. It never writes, renames, or migrates the shared NPZ. Regenerated corpora should set the flag to `false`; standalone validation inherits the checkpoint setting unless explicitly overridden.

Oversampling accepts repeated `extra_train_set_list` flags and one shared
`extra_train_set_repeat` inside the Dataset. It concatenates the sources and appends
Python path references in memory instead of materializing a multi-hundred-MB combined
JSON. Optional extra-only traffic-light masking also happens in worker memory. Dataset-list
upload to W&B is opt-in through `WANDB_LOG_DATASET_ARTIFACT=1`.

## Supervised training stages

### Stage 1: HDP base

Base is trained from scratch with HDP flags enabled. Use full-sequence base train/valid lists. Use the same W&B project as the comparable DFP and quality-fix runs:

```text
Diffusion-Planner-Temporal
```

### Stage 2: HDP SFT

SFT starts from the base checkpoint with `--init_weights_path`. The SFT run must keep the same HDP representation flags as base.

Do not change from velocity representation to waypoint representation between base and SFT.

## HDP-RL path

The HDP paper's RL objective is reward-weighted RL-Hybrid. This branch keeps only that RL path, adapted to signals actually present in Tier IV NPZs.

The TeX algorithm computes a group-normalized reward and weights the hybrid loss with:

```text
exp(beta * normalized_reward)
```

The local default RL path now follows that objective:

```text
rl_reward_normalize=group
rl_reward_beta=0.5
rl_reward_w_risk=1.0
rl_reward_w_follow=3.0
rl_reward_w_lane=2.5
rl_reward_w_progress=3.0
rl_bc_weight=1.0
num_generations=32
rl_noise_scale=0.5
rl_rollout_steps=6
rl_updates_per_rollout=4
rl_ema_update_rate=0.05
rl_init_use_ema=true
rl_train_scope=decoder
```

The paper EMA update `0.05` is used at policy-iteration boundaries. The lower `beta`, explicit
progress term, and one-target-per-scene BC anchor are performance-oriented safeguards from real
Tier IV data audits; the unanchored configuration caused rapid progress and validation collapse.

Implementation notes:

- Zero-variance and non-finite reward groups are discarded; they do not become unweighted self-distillation samples.
- Reward scoring uses raw scene tensors before group expansion; only rollout/loss tensors are expanded to `B * num_generations`.
- Logged futures for all 320 neighbors stay scene-level and are not duplicated across the 32 candidates.
- Distributed training pads the shuffled index stream to a complete global batch instead of
  dropping the tail or compiling a second shape. Every source sample is used at least once and at
  most `global_batch_size - 1` randomly shuffled samples are repeated per epoch.
- EMA rollout sampling runs without autograd. The live global route condition is computed inside
  the DDP forward once per scene and only its 256-dimensional embedding is repeated per candidate.
- Rollout sampling uses a fixed temperature instead of a random per-row temperature range.
- Each sampled and scored group is reused for four independent diffusion-time updates while the
  rollout policy remains frozen for the whole epoch. This amortizes rollout and reward geometry;
  one update remains available as an ablation.
- RL starts from the SFT EMA shadow with `--init_weights_path` and `rl_init_use_ema=true`, while optimizer/scheduler/W&B state are fresh. Missing EMA weights produce an explicit live-weight fallback warning.
- `rl_train_scope=decoder` updates only the DiT trajectory policy and freezes the encoder plus the separate turn-indicator classifier. This matches the released decoder-policy intent without leaving an unsupervised Tier IV-only head in DDP.
- Encoder modules are kept in eval mode during decoder-only RL so frozen dropout/drop-path does not inject noise.
- The EMA shadow is the fixed previous policy required by the paper's replay-buffer objective.
  It is not changed between batches. At the epoch boundary, the live proposal is committed once
  with update rate `0.05` (`timm` decay `0.95`), live weights are synchronized to the accepted
  policy, and stale Adam moments are cleared before the next policy iteration.
- The boundary EMA update uses timm's foreach implementation to fuse the per-tensor interpolation
  kernels while preserving the existing `.ema` checkpoint state.
- SFT, RL, and standalone validation compile the encoder and decoder in place by default. The
  state dict stays unchanged, while RL's direct component calls and DPM validation use the same
  compiled modules. Use a persistent `TORCHINDUCTOR_CACHE_DIR` across restarts.
- The single reward path contains SAT collision, continuous TTC, THW, occupancy clearance, leader-conditioned following, lane-center scoring, lane-change/off-lane masking, and rear-end attenuation, using risk/follow/lane weights 1.0/3.0/2.5. Lane scoring uses the navigation route when it agrees with the logged expert trajectory and otherwise falls back to all visible lane centerlines.
- Occupancy automatically uses real static boxes, stopped-agent clearance, then road-border clearance as a corpus fallback. Missing sources are neutral and their coverage is logged.
- Scene encoding is computed once per candidate group. Decoder-only RL repeats only current action-state tensors, not the full 31-frame observation history.
- Full held-out stochastic-reward/EPDMS validation runs on `rl_full_eval_utd`; the deterministic proxy remains available each epoch. Reward validation uses fixed random candidates and logs every reward component and source-coverage diagnostic, so policy iterations are directly comparable.
- A fresh RL run validates and saves its source SFT policy before the first update. Best-checkpoint selection maximizes mean held-out reward, retains deterministic EPDMS as an independent diagnostic, rejects abnormal supervised validation-loss regressions, and requires a `0.0001` reward improvement before replacing the current best. Older runs without reward columns retain the EPDMS/loss fallback when resumed.
- Training stops after five consecutive full evaluations without a meaningful best-score improvement; set `rl_early_stop_patience=0` only for controlled ablations.
- Turn-indicator validation logs overall, change-only, and all five per-class accuracies plus class counts; the overall metric is computed from generated trajectories, never teacher-forced trajectories.
- SFT/RL ONNX export on every save is disabled by default because synchronous export stalls all other DDP ranks at the next barrier. Set `export_onnx_on_save=true` only when needed, or use the strict standalone converter.

### Ego-only action head

This branch has one action-head contract: 80 temporal ego tokens with
`predicted_neighbor_num=0`. The encoder still consumes all 320 configured neighbor histories,
and validation/RL safety scoring still uses every logged neighbor future. Neighbor trajectories
are scene context and reward inputs, not decoder prediction targets.

### What is faithful and what is DP-native

Faithful to HDP:

- Reward-weighted RL-Hybrid loss form.
- Group reward normalization.
- `exp(beta * normalized_reward)` weighting.
- Multi-reward risk/follow/lane weighting with the paper's 1.0/3.0/2.5 coefficients.
- Decoder-only RL fine-tuning by default.
- SFT checkpoint as RL initialization.
- Fixed rollout temperature.

DP-native adaptation:

- The released NAVSIM implementation uses NAVSIM PDM metric caches, Ray scoring, and a replay buffer; it does not expose the real-vehicle reward shaping implementation described in the paper.
- The public NAVSIM configuration uses group size 10, five rollout steps, a rollout epoch followed
  by nine replay-training epochs, and no active EMA by default. The paper table instead reports
  group size 32 and EMA update 0.05; the real-vehicle table reports six inference steps. This branch
  uses a memory-efficient streaming equivalent: a frozen previous policy supplies every rollout in
  an epoch, then one EMA policy commit occurs at the boundary.
- This branch runs on Tier IV NPZ data and computes the HDP reward directly from available geometry and map tensors.
- Exact NAVSIM PDM cache behavior is not assumed to exist in this repository.
- Tier IV line strings and polygons use valid-point centroid/direction positional geometry
  with masked consecutive diffs. Historical turn reports are one-hot encoded inside the graph;
  NPZ, ROS, and ONNX input tensors remain raw codes with unchanged shapes.

This means the branch is faithful at the objective and training-interface level, but not a byte-for-byte reproduction of the NAVSIM runtime environment.

## Validation and model selection

Use validation metrics consistently across base, SFT, and RL:

- `valid_loss/*` for supervised losses.
- `valid_epdms/*` for planning-quality proxy metrics.
- `valid_multisample/minADE`, `minFDE`, and their thresholded scores for stochastic open-loop diagnostics.
- Temporal metrics only when pair/full-sequence loading is available.

The six-trajectory count is not invented by this repository: the paper's Appendix "Open-Loop Metrics" explicitly says it generates six trajectories before computing minADE/minFDE. The seeded zero-noise trajectory remains a lower-variance checkpoint proxy; it is not mislabeled as that six-trajectory protocol.

When `amp_dtype=bf16`, validation inference uses the same bf16 autocast as training and RL
rollouts. Compare those metrics only with runs using the same precision; pre-temporal fp32
validation logs are not an exactly identical numerical baseline.

## ONNX export

HDP velocity checkpoints should be deployed with the full ONNX graph:

```text
diffusion_planner.onnx
```

The full graph runs the model's own sampler and decodes the HDP ego velocity latent back to
waypoint-space prediction before returning `prediction`.

The old waypoint split-decoder graph is not exported in this HDP-only branch: exposing the
velocity latent to an external waypoint denoising loop would be incorrect. Encoder and
turn-indicator diagnostic graphs remain available; production should consume the full graph.

Precision policy:

- Export uses float32 model execution.
- bf16 training autocast is not used during ONNX export.
- HDP velocity normalization is decoded inside the traced full graph.
- The standalone converter validates the full ONNX output against the PyTorch wrapper when
  `ros_scripts/torch2onnx.py` is used.
- Every full-graph input has a dynamic batch axis. Native output is `[B,1,80,4]`; the model has no delay-prefix or compatibility-row contract.

Smoke validation performed after the temporal-token conversion:

```text
result:
  ONNX checker         full / encoder / turn-indicator passed
  ORT provider         CPUExecutionProvider, dynamic batch 1 and 2 passed
  full prediction      [B, 1, 80, 4], finite
  turn indicator       [B, 5], finite
  split decoder        not part of the HDP export surface
```

The full graph takes `sampled_trajectories: [B,1,80,4]`. `prediction[:,0]` is the
integrated waypoint trajectory, not the normalized velocity latent. Scene encoder inputs,
including 320 neighbor histories, are unchanged.

For deployment or closed-loop handoff, record:

- Branch path.
- Commit hash if available.
- Run name.
- Checkpoint path.
- `args.json` path.
- W&B run URL.
- Data-list paths.
- Whether the checkpoint is base, SFT, or RL.

## Production caution

This branch is for HDP development and experiments. Before making a clean upstream PR:

- Remove local experiment-only scripts and notes.
- Keep reference paper/code files out of the PR unless requested.
- Split metric-only changes from model-training changes.
- Keep the PR description explicit about compatibility and checkpoint semantics.
