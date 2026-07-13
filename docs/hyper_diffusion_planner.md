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

The current 2026-06 Tier IV corpus was generated before converter commit `55eff4f` and can duplicate the current frame in neighbor futures. `align_legacy_neighbor_futures=true` detects unambiguous short tracks from their padded tail. A rare track that disappears exactly at the 80-step boundary can look full; that case is shifted only when map-coordinate matching proves its second point is the next scene's current state. This preserves legitimate repeated poses. Alignment and tail padding happen in worker memory and never write, rename, or migrate shared NPZ files. Correctly regenerated corpora use NPZ format version 3, which marks neighbor futures as already starting at `t+0.1 s` and bypasses legacy alignment automatically. Standalone validation inherits the checkpoint setting unless explicitly overridden.

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

## RL status

RL is experimental and intentionally separate from the settled Base/SFT ViT model contract. Its
reward design, policy-update schedule, and checkpoint-selection gates are documented in the
[HDP-RL experimental design](hdp_rl.md); changing them must not change the SFT model specification.

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
