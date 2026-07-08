# Hyper Diffusion Planner integration guide

This document defines how this branch should be used for HDP experiments in the Tier IV Diffusion Planner codebase.

## Decision: HDP-specialized branch with vanilla compatibility

This branch should be treated as HDP-specialized.

It remains possible to run vanilla Diffusion Planner supervised training by disabling HDP flags, but the branch defaults, docs, and experiment workflow are HDP-oriented. This avoids ambiguity when comparing HDP, DFP, quality-fix DP, and original DP runs.

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

This HDP branch keeps the original scene encoder, neighbor prediction, turn-indicator path, and validation stack, but changes the ego planning target in HDP mode:

- Ego target can be represented as normalized velocity/action-like latent rather than direct waypoint latent.
- The HDP ego prediction is converted back to waypoints through integration for trajectory loss and evaluation.
- Hybrid loss combines velocity-space diffusion supervision with waypoint-space reconstruction.
- Neighbor futures are still trained through the original all-agent prediction path.
- Turn indicators and Tier IV validation metrics remain available.

The intended HDP supervised setting is:

```text
use_velocity_representation=True
planning_hybrid_loss=0.01
hybrid_loss_window=10
diffusion_model_type=x_start
diffusion_supervision_type=x_start
diffusion_sample_steps=10
```

## Checkpoint rules

Use the right loading mode.

| Use case | Flag |
| --- | --- |
| Start SFT from base weights | `--init_weights_path` |
| Start RL from SFT weights | `--init_weights_path` |
| Continue the exact same interrupted run | `--resume_model_path` |
| Start HDP from a vanilla waypoint checkpoint intentionally | `--init_weights_path` only |

Do not use `--resume_model_path` to change representation. The code checks representation compatibility to avoid silently interpreting waypoint latents as velocity latents.

## Data policy

Use fixed full-sequence lists for HDP experiments.

Reasons:

- HDP uses ego history and temporal behavior as part of the model design.
- Temporal stability metrics need consecutive frames.
- Mixed project/area lists can change the distribution and make comparisons unfair.

Single-frame lists can still run ordinary supervised training, but they are not sufficient for temporal-consistency evaluation.

## Supervised training stages

### Stage 1: HDP base

Base is trained from scratch with HDP flags enabled. Use full-sequence base train/valid lists. Use the same W&B project as the comparable DFP and quality-fix runs:

```text
Diffusion-Planner-Temporal
```

### Stage 2: HDP SFT

SFT starts from the base checkpoint with `--init_weights_path`. The SFT run must keep the same HDP representation flags as base.

Do not change from velocity representation to waypoint representation between base and SFT.

## Official HDP-RL path

The official HDP RL idea is reward-weighted RL-Hybrid. This branch keeps only that RL path.

The TeX algorithm computes a group-normalized reward and weights the hybrid loss with:

```text
exp(beta * normalized_reward)
```

The local default RL path now follows that objective:

```text
official_reward_normalize=group
official_reward_beta=1.0
rl_reward_w_risk=1.0
rl_reward_w_follow=3.0
rl_reward_w_lane=2.5
num_generations=32
rl_noise_scale=0.5
rl_train_scope=decoder
```

Implementation notes:

- Zero-variance finite reward groups are kept with neutral weight `exp(0)=1`, matching the reward-weighted hybrid formula and avoiding silent no-op epochs on saturated rewards.
- Reward scoring uses raw scene tensors before group expansion; only rollout/loss tensors are expanded to `B * num_generations`.
- Rollout sampling uses a fixed temperature instead of a random per-row temperature range.
- RL starts from SFT with `--init_weights_path`, so optimizer/scheduler/W&B state are fresh.
- `rl_train_scope=decoder` freezes non-decoder parameters, matching the official decoder fine-tuning style.
- Encoder modules are kept in eval mode during decoder-only RL so frozen dropout/drop-path does not inject noise.
- The reward backend is fixed to an NPZ-native multi-reward adaptation of the paper's risk/follow/lane setting, using the official weights 1.0/3.0/2.5.
- Best-checkpoint selection is based on validation EPDMS when available, falling back to negative ego validation loss.

### What is faithful and what is DP-native

Faithful to HDP:

- Reward-weighted RL-Hybrid loss form.
- Group reward normalization.
- `exp(beta * normalized_reward)` weighting.
- Multi-reward risk/follow/lane weighting with the official 1.0/3.0/2.5 coefficients.
- Decoder-only RL fine-tuning by default.
- SFT checkpoint as RL initialization.
- Fixed rollout temperature.

DP-native adaptation:

- The official NAVSIM implementation uses NAVSIM PDM metric caches, Ray scoring, and a replay buffer.
- This branch runs on Tier IV NPZ data and uses available DP scene tensors to build EPDMS-style risk, route/GT-following, and lane-keeping rewards.
- Exact NAVSIM PDM cache behavior is not assumed to exist in this repository.

This means the branch is faithful at the objective and training-interface level, but not a byte-for-byte reproduction of the NAVSIM runtime environment.

## Validation and model selection

Use validation metrics consistently across base, SFT, and RL:

- `valid_loss/*` for supervised losses.
- `valid_epdms/*` for planning-quality proxy metrics.
- Temporal metrics only when pair/full-sequence loading is available.

## ONNX export

HDP velocity checkpoints should be deployed with the full ONNX graph:

```text
diffusion_planner.onnx
```

The full graph runs the model's own sampler and decodes the HDP ego velocity latent back to
waypoint-space prediction before returning `prediction`.

The split decoder ONNX contract is only valid for vanilla waypoint-mode `x_start` checkpoints.
For HDP velocity checkpoints, exporting a split decoder would expose velocity-space ego latents
that an external waypoint denoising loop could misinterpret. Therefore the exporter skips the
split decoder for HDP and keeps the full graph as the deployable artifact.

Precision policy:

- Export uses float32 model execution.
- bf16 training autocast is not used during ONNX export.
- HDP velocity normalization is decoded inside the traced full graph.
- The standalone converter validates the full ONNX output against the PyTorch wrapper when
  `ros_scripts/torch2onnx.py` is used.

Smoke validation performed on this branch:

```text
checkpoint:
  outputs/hdp_velocity_hybrid_omega001_base60_fullseq_node01_8gpu_bf16_bs512_rerun3/epoch0010/best_model.pth

command:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python ros_scripts/torch2onnx.py \
    outputs/onnx_smoke_hdp_epoch0010 \
    --output-prefix diffusion_planner_hdp_smoke \
    --opset-version 20

result:
  full prediction      max diff = 4.3487548828125e-4, mean diff = 1.4785410712647717e-5
  turn indicator logit max diff = 8.58306884765625e-6, mean diff = 3.4928320928884204e-6
  encoder output       max diff = 4.082918167114258e-6, mean diff = 4.0447002902510576e-7
  ORT provider         CPUExecutionProvider
  split decoder        intentionally skipped for HDP velocity checkpoints
```

The checked full graph outputs:

```text
prediction: [B, 321, 80, 4]
turn_indicator_logit: [B, 5]
```

`prediction[:, 0]` is the HDP ego trajectory decoded back to waypoint space. It is not the
normalized velocity latent.

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
