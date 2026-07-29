# Hyper Diffusion Planner integration guide

This document defines how this branch should be used for HDP experiments in the Tier IV Diffusion Planner codebase.

For the branch/PR state, the launcher map, the checkpoint rule, what was removed and why, and
the archive tags holding every unmerged experiment, read
[`hdp_final_state.md`](hdp_final_state.md) first. This page is the model contract.

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

This HDP branch keeps the Tier IV scene representation and validation stack, but changes the ego planning target in HDP mode:

- Ego target is represented as a normalized velocity/action latent rather than a direct waypoint latent.
- The HDP ego prediction is converted back to waypoints through integration for trajectory loss and evaluation.
- Hybrid loss combines velocity-space diffusion supervision with waypoint-space reconstruction.
- The HDP action head is ego-only; all 320 neighbor histories remain scene context.
- Detailed route tokens remain in scene cross-attention, while an official-style ordered route
  encoder supplies a lightweight global AdaLN condition for every DiT block.
- A detached three-state turn-intent head remains available, but turn-indicator
  history is not a policy input. Its complete design and data audit are in
  [HDP turn-indicator SFT design](hdp_turn_indicator_sft.md).

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
[#212](https://github.com/tier4/Diffusion-Planner/pull/212) and the polygon/line-string
type injection and geometry-only Mixer input from [#228](https://github.com/tier4/Diffusion-Planner/pull/228).
There is no legacy encoder mode. The SFT weights-only loader has one explicit
exception for Base80 migration: it removes the old #210 signal encoder and
reinitializes the old four-state head while loading every trajectory-policy tensor
strictly. Exact resume remains strict and never migrates architectures.

## Data policy

Use fixed full-sequence lists for HDP experiments.

Reasons:

- HDP uses ego history and temporal behavior as part of the model design.
- The current SFT encoder uses the newest 21 ego frames (current plus 20 past
  frames, approximately two seconds at 10 Hz) while retaining only the newest
  6 neighbor frames because older perception tracks are less reliable. The
  input tensor remains 31 frames wide and is left-padded after this selection.
- This is an encoder-contract change: old 6-frame checkpoints must not be resumed
  or used as the final 21-frame model. The shared NPZ files and manifests are not
  modified.
- Temporal stability metrics need consecutive frames.
- Mixed project/area lists can change the distribution and make comparisons unfair.

Single-frame lists can still run ordinary supervised training, but they are not sufficient for temporal-consistency evaluation.

The current regenerated Tier IV corpus already stores neighbor future index 0 at `t+0.1 s`; the default is `align_legacy_neighbor_futures=false`. The legacy correction is retained only as an explicit opt-in for pre-`55eff4f` manifests, where it detects unambiguous short tracks from their padded tail and verifies rare full-horizon cases against the next scene. Alignment happens in worker memory and never writes, renames, or migrates shared NPZ files. Standalone validation inherits the checkpoint setting unless explicitly overridden, so new runs serialize `false` and do not apply the old +1 shift.

Oversampling accepts repeated `extra_train_set_list` flags and one shared
`extra_train_set_repeat` inside the Dataset. It concatenates the sources and appends
Python path references in memory instead of materializing a multi-hundred-MB combined
JSON. Traffic-light features are kept unchanged. Dataset-list upload to W&B is opt-in through
`WANDB_LOG_DATASET_ARTIFACT=1`.

### `is_skipped` manifest filtering

Converter sidecars can mark gap-filling frames with `is_skipped: true`. The default
SFT/RL train input is the precomputed shared manifest
`/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train_sft_is_skipped_filtered.json`
(4,578,036 entries); SFT/RL pass `--filter_skipped False` for this file so every launch
avoids a multi-million-sidecar scan. The original `path_list_train_sft.json`, NPZs, and
sidecars remain unchanged. The full-sequence Base list was independently audited and
dropped zero entries, so the Base launcher also reads it directly. For an explicitly raw
or alternate list, set `--filter_skipped True` to create a run-local filtered manifest;
`--skip_filter_sidecar_root` handles corpora whose sidecars are stored separately and
`--skip_filter_workers` controls the one-time scan parallelism.

### Causal red-light manifest filtering

The current route tensor exposes a red signal but not the future signal transition. A
logged trajectory that waits and then starts therefore crosses an event the current
observation cannot predict. Causal red-light filtering is an optional experiment, not
part of the default SFT/RL data contract. The default launchers use the shared
`path_list_train_sft_is_skipped_filtered.json` directly and do not add causal extra
manifests or a runtime loss mask.

When explicitly enabled, the controlled-signal pre-pass writes new lists and metadata
without modifying NPZ files or the original manifests. It associates every signalized
route lanelet with the actual forward stop-line geometry, selects the nearest
unambiguous controlling signal, and removes only the red/stopped/future-moving
predicate. A nearer green signal therefore keeps valid green-start samples, downstream
red signals cannot trigger deletion, and full-horizon red-light holds remain supervised.
Conflicting or unavailable signal associations are retained conservatively.

## Supervised training stages

### Stage 1: HDP base

Base is trained from scratch with HDP flags enabled. Use full-sequence base train/valid lists. Use the same W&B project as the comparable DFP and quality-fix runs:

```text
Diffusion-Planner-Temporal
```

Base and ordinary policy SFT explicitly use `--supervised_training_stage policy`.
The trajectory model does not read historical turn indicators, and the auxiliary
head is frozen, skipped, and omitted from policy validation metrics until the later
head-only stage.

### Stage 2: staged HDP supervised fine-tuning

Fine-tuning starts from the stopped Base checkpoint with `--init_weights_path` and
keeps the same HDP representation and exact Base train/extra/validation manifests.

When starting from the current Base80 checkpoint, the signal-history token and old
head are migrated as described in [HDP turn-indicator SFT design](hdp_turn_indicator_sft.md).
This requires a new SFT run; do not resume an earlier SFT checkpoint.

The first phase uses `--supervised_training_stage policy`: the new intent head is
frozen and skipped while the trajectory policy adapts to removal of the signal
input. The second phase initializes from that phase's latest EMA and uses
`--supervised_training_stage turn_indicator`: the complete planner is frozen and
only the three-state head is trained on final DPM and expert trajectories. See
`diffusion_planner/slurm/run_hdp_staged_sft_node02.sbatch` for the hash-locked
protocol.

Do not change from velocity representation to waypoint representation between base and SFT.

## RL status

RL is experimental and intentionally separate from the settled Base/SFT ViT model contract. Its
reward design, policy-update schedule, and checkpoint-selection gates are documented in the
[HDP-RL experimental design](hdp_rl.md); changing them must not change the SFT model specification.

## Validation and model selection

Use validation metrics consistently across base, SFT, and RL:

- `valid_loss/*` for supervised losses.
- `valid_epdms/*` for planning-quality proxy metrics.
- `valid_multisample/minADE`, `minFDE`, and their thresholded scores for stochastic open-loop diagnostics.
- `valid_turn_indicator/*` for balanced three-state intent metrics; exact-frame
  change accuracy is not a supported selection metric.
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

Required smoke validation for each new SFT checkpoint:

```text
expected result:
  ONNX checker         full / encoder / turn-indicator passed
  ORT provider         CPUExecutionProvider, dynamic batch 1 and 2 passed
  full prediction      [B, 1, 80, 4], finite
  turn indicator       [B, 3], finite
  split decoder        not part of the HDP export surface
```

The full graph takes `sampled_trajectories: [B,1,80,4]` and does not take signal
history. `prediction[:,0]` is the
integrated waypoint trajectory, not the normalized velocity latent. Scene encoder input
shapes, including 320 neighbor histories, are unchanged; the effective ego-history
selection is recorded by `ego_history_frames` in `args.json`.

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
