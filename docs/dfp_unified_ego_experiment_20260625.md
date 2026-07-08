# DFP unified-ego experiment record - 2026-06-25 JST

This document records the current DFP integration experiment so the exact state is not lost.
It intentionally distinguishes proven evidence from hypotheses and next-step designs.

## Repository state

Worktree:

```text
/mnt/nvme/Diffusion-Planner-dfp-tier4-additive
```

Branch and base commit:

```text
branch: feat/dfp-additive-tier4-main
base commit: 0ae5b25d1428acfc18312e674f9b44123c263211
base commit title: Reproducer: one-pass collision-scene save + moving-neighbor augmentation tooling (#157)
```

Important: the experiment code is not represented by the base commit alone. The worktree is dirty.
Track both the base commit and the patch/file hashes below.

Dirty tracked files:

```text
M diffusion_planner/diffusion_planner/model/module/decoder.py
M diffusion_planner/diffusion_planner/train.py
M diffusion_planner/diffusion_planner/train_config.py
M diffusion_planner/diffusion_planner/train_epoch.py
M diffusion_planner/diffusion_planner/utils/ddp.py
M diffusion_planner/diffusion_planner/utils/unicycle_accel_curvature.py
M diffusion_planner/train_predictor.py
```

Important untracked files:

```text
diffusion_planner/diffusion_planner/model/module/dfp.py
diffusion_planner/run_baseline_sft_tier4main.sh
diffusion_planner/run_dfp_additive_lam03_sft.sh
diffusion_planner/run_dfp_additive_resume_epoch10.sh
diffusion_planner/run_dfp_additive_sft.sh
diffusion_planner/run_dfp_fusion_residual_lam03_scale05_sft.sh
diffusion_planner/run_dfp_fusion_residual_lam03_sft.sh
diffusion_planner/run_dfp_unified_ego_lam03_sft.sh
```

Tracked diff hash at record time:

```text
git diff sha256: 84ab34fb17db9124cbb8a2b94b4c8e802ca7733743ddad6299b0c5ece39212c5
```

Key file hashes at record time:

```text
fad8ed449b6d24045a9ecae5aa79dcb26af4d752ba09b0e107116cde91be8c5f  diffusion_planner/diffusion_planner/model/module/decoder.py
bff3bf057e1a2053eb530c995a4e2265f5465b6e415c6d274b19446d144444ed  diffusion_planner/diffusion_planner/model/module/dfp.py
132046b61fbdbf669486337a7e1911b0891ebf4ee7f1e2fe974de04cace50307  diffusion_planner/diffusion_planner/train.py
3c834809bf0d66ccd48bfad4eb9971146594267cbdeb30fd9572edc2371823a8  diffusion_planner/diffusion_planner/train_config.py
7209ad1c9d5df4d32bf73bac081a5b5401ba81c9f10bdaf5f7cb3184740f2034  diffusion_planner/diffusion_planner/train_epoch.py
a8eab3127bee67263edeeae19630cac38617663f648efab8282d483cb4b08301  diffusion_planner/train_predictor.py
0e8c3d5ebe9dc6b4591c3287743f11aea3e83973bfaaa43c5d3d9fb387717874  diffusion_planner/run_dfp_unified_ego_lam03_sft.sh
19eb0ea6db716ac53d12a620f8619232a28cd42f7b0ae0e78a5ec6827d16aa85  diffusion_planner/run_baseline_sft_tier4main.sh
```

## Data and initialization

Common data/config:

```text
train_set_list: /mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/_pathlists/path_list_train_rebuilt_step3.json
valid_set_list: /mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/_pathlists/path_list_valid_rebuilt_step3.json
normalization: /mnt/nvme/Diffusion-Planner-dfp/normalization_33d.json
init_weights_path: /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/checkpoints/base_sft/best_model.pth
init checkpoint epoch: 60
init checkpoint has DFP params: false
init checkpoint has turn params: true
```

The init checkpoint is the base-training epoch 60 checkpoint used as SFT start, not an SFT-final checkpoint.

## Active DFP run

Run:

```text
run name: dfp_unified_ego_lam03_tier4main_node02_8gpu_tf32
output: /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/outputs/dfp_unified_ego_lam03_tier4main_node02_8gpu_tf32
wandb project: advanced-technology-department/Diffusion-Planner-Temporal
node: node02
GPUs: 8
```

Command/script source:

```text
script: diffusion_planner/run_dfp_unified_ego_lam03_sft.sh
RUN_NAME default: dfp_unified_ego_lam03_tier4main_node02_8gpu_tf32
```

Key args:

```text
use_dfp_decoder: True
dfp_decoder_mode: unified_ego
dfp_use_inference: True
dfp_history_len: 20
dfp_chunk_len: 20
dfp_lambda_hist: 0.3
dfp_lambda_future: 0.3
dfp_lambda_current: 0.0
dfp_lambda_original_ego: 0.2
dfp_history_beta_a: 0.5
dfp_history_beta_b: 0.5
dfp_guidance_w: 0.2
dfp_guidance_beta: 2.0
dfp_sampler_steps: 10
batch_size: 512
learning_rate: 1e-4
warm_up_epoch: 5
gradient_accumulation_steps: 2
train_epochs: 80
stop policy for this experiment: stop after epoch 20 checkpoint
tf32: True
use_data_augment: True
augment_prob: 0.5
future_len: 80
time_len: 31
agent_num: 320
predicted_neighbor_num: 320
hidden_dim: 256
num_heads: 8
encoder_mixer_depth: 6
encoder_fusion_depth: 6
decoder_depth: 3
```

Status at 2026-06-25T02:22:30+09:00:

```text
DFP completed epoch 20 and was stopped by watcher after checkpoint was observed.
matched no-DFP baseline is running and had completed epoch 6.
stop20 report is pending until baseline reaches epoch 20.
```

DFP stop20 metrics:

| epoch | train | ego | neighbor | lat | lon |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.377286 | 2.633261 | 4.644602 | 0.294673 | 1.949021 |
| 2 | 0.145984 | 2.478082 | 4.802966 | 0.262337 | 1.793388 |
| 3 | 0.115843 | 2.446879 | 4.422278 | 0.259199 | 1.748491 |
| 4 | 0.104466 | 2.756251 | 4.247891 | 0.272606 | 1.806942 |
| 5 | 0.097550 | 2.485802 | 4.573757 | 0.225900 | 1.776826 |
| 6 | 0.091701 | 2.406174 | 4.999734 | 0.240285 | 1.689365 |
| 7 | 0.087934 | 2.290945 | 4.421613 | 0.213682 | 1.657224 |
| 8 | 0.085034 | 2.358116 | 4.984192 | 0.212933 | 1.677663 |
| 9 | 0.082671 | 2.148497 | 4.529927 | 0.236491 | 1.566430 |
| 10 | 0.080431 | 2.093993 | 4.454603 | 0.218809 | 1.556728 |
| 11 | 0.078725 | 2.080952 | 4.101002 | 0.214134 | 1.526397 |
| 12 | 0.077217 | 2.082172 | 4.833856 | 0.214360 | 1.544225 |
| 13 | 0.076288 | 2.167480 | 4.116881 | 0.218511 | 1.561458 |
| 14 | 0.075093 | 2.154116 | 4.035117 | 0.210854 | 1.568942 |
| 15 | 0.074143 | 2.017923 | 4.098047 | 0.239837 | 1.507572 |
| 16 | 0.073258 | 2.204737 | 4.371692 | 0.213241 | 1.592456 |
| 17 | 0.072684 | 1.961324 | 4.090093 | 0.198632 | 1.472875 |
| 18 | 0.071922 | 2.134301 | 4.022470 | 0.193309 | 1.550852 |
| 19 | 0.071435 | 2.013360 | 4.215456 | 0.223489 | 1.510653 |
| 20 | 0.070775 | 2.317632 | 3.916277 | 0.208086 | 1.625842 |

Best DFP points inside this run:

```text
best ego: epoch 17, 1.961324
best lat: epoch 18, 0.193309
best lon: epoch 17, 1.472875
best neighbor among completed epochs: epoch 20, 3.916277
```

Interpretation so far:

```text
DFP has strong positive ego/lon signal and later also strong lat/neighbor points.
The run is not monotonic; epoch 20 is not the best ego/lon checkpoint.
The final strict conclusion requires matched no-DFP baseline at epoch 20.
```

## Matched no-DFP baseline

Run:

```text
run name: baseline_sft80cfg_stop20_tier4main_node02_8gpu_tf32
output: /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/outputs/baseline_sft80cfg_stop20_tier4main_node02_8gpu_tf32
script: diffusion_planner/run_baseline_sft_tier4main.sh with RUN_NAME override
node: node02
GPUs: 8
```

Key args are intentionally matched to DFP except:

```text
use_dfp_decoder: False
dfp_use_inference: False
dfp_lambda_hist: 0.0
dfp_lambda_future: 0.0
dfp_lambda_current: 0.0
```

Baseline status at 2026-06-25T02:22:30+09:00:

```text
completed epoch: 6
still running on node02 8 GPUs
```

Baseline metrics available at record time:

| epoch | train | ego | neighbor | lat | lon |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.086435 | 3.856332 | 4.516604 | 0.246503 | 2.093706 |
| 2 | 0.066275 | 3.684780 | 4.358158 | 0.249551 | 2.045390 |
| 3 | 0.062025 | 3.579876 | 4.611333 | 0.250713 | 1.984138 |
| 4 | 0.060805 | 3.646439 | 4.810959 | 0.247521 | 2.032776 |
| 5 | 0.060814 | 3.859374 | 4.993103 | 0.241339 | 2.106930 |
| 6 | 0.059535 | 3.552315 | 4.402088 | 0.258769 | 2.045209 |

Early matched observations through baseline epoch 6:

```text
DFP is clearly better on ego and lon for epochs 1-6.
DFP is worse on lat for epochs 1-4 but better at epoch 5; epoch 6 lat remains slightly better for DFP than baseline.
DFP neighbor is worse at epochs 1-2, then better at epochs 3-5, and worse at epoch 6.
This is not a final result; wait for baseline epoch 20 and generated report.
```

## Current automatic experiment chain

Watchers/queues running on node02:

```text
1417757: stop DFP at epoch 20, launch matched no-DFP baseline, stop baseline at epoch 20
1419575: write /tmp/dfp_vs_baseline20_report.tsv after baseline epoch 20
1419921: after stop20 report, launch exact train_epochs=20 DFP/no-DFP pair
1421813: after both reports, write /tmp/dfp_experiment_summary.md
1422155: if conservative summary fails, launch neighbor-protected DFP fallback
```

Expected generated artifacts:

```text
/tmp/dfp_vs_baseline20_report.tsv
/tmp/dfp_vs_baseline_exact_sft20_report.tsv
/tmp/dfp_experiment_summary.md
/tmp/dfp_neighbor_protected_vs_baseline20_report.tsv, only if fallback is needed
```

## Current implementation summary

Current implementation is a hybrid DFP-on-DP integration, not a full paper reproduction and not just an auxiliary loss.

Original Diffusion-Planner:

```text
single all-agent future denoiser
predicts ego future and neighbor futures through original DiT decoder
turn indicator is computed from predicted ego trajectory plus pooled scene encoding
```

Paper DFP concept:

```text
ego-centric diffusion forcing over history/current/future chunks
explicitly trains temporal consistency from history to current to future
```

Current unified_ego integration:

```text
DFP branch predicts ego future from ego_agent_past/current/future chunks.
Original DP decoder still predicts neighbor futures.
In unified_ego mode, DFP ego future replaces the main ego output used for planner loss, validation, and inference.
Original ego output can remain as optional regularization via dfp_lambda_original_ego.
Turn indicator is recomputed from the DFP ego trajectory plus pooled scene encoding.
```

Why this hybrid was chosen:

```text
It preserves neighbor and turn-indicator behavior from original DP.
It allows reuse of the base epoch 60 checkpoint.
It avoids requiring a new all-agent DFP dataset/schema.
It provides a lower-risk path to test whether DFP improves ego planning before replacing the full decoder.
```

Cost of this hybrid:

```text
Two decoder stacks exist: original DP DiT plus DFPDiT.
Inference/training logic has multiple modes: additive, fusion, unified_ego.
Memory and code complexity are higher than necessary.
The model has two ego-producing mechanisms, even though unified_ego uses DFP as final ego.
```

## Deep design proposal: merge into one decoder

The current two-decoder structure is useful for proving the idea but is not ideal long term.
A single-decoder design should preserve original weights and minimize the need to rerun baselines.

Recommended next design: Shared-stack DFP adapter.

High-level design:

```text
Keep the original DP decoder transformer stack as the only decoder block stack.
Remove the separate DFPDiT block stack.
Add DFP-specific token embedding, time/noise embedding, chunk positional embedding, and output head.
Route both original all-agent denoising tokens and ego DFP chunk tokens through the same original decoder blocks.
Keep original neighbor output head unchanged.
Use DFP future output as ego output in unified_ego mode.
```

Token structure:

```text
agent tokens:
  original ego/neighbor future-denoising tokens used by DP

dfp tokens:
  ego history chunk token
  ego current chunk token
  ego future chunk tokens
```

Attention structure options:

```text
Option A, easiest:
  concatenate original agent tokens and DFP tokens into one sequence
  use full self-attention in the shared decoder stack
  preserve original output heads for all-agent prediction
  add DFP output head only for DFP tokens

Option B, safer for neighbor preservation:
  same shared stack, but mask or gate DFP-token influence into neighbor tokens initially
  allow agent-to-DFP attention, but start DFP-to-neighbor influence at zero
  gradually unfreeze/gate if needed

Recommended: Option B for first implementation.
```

Weight reuse strategy:

```text
Load the original base checkpoint exactly as before.
Initialize the shared transformer stack from original DP decoder weights.
Keep original neighbor path numerically unchanged at initialization.
Initialize DFP adapters and DFP head as new parameters.
Initialize DFP-to-agent residual/gate to zero so the model starts equivalent to original DP for neighbor and original ego.
Use DFP ego output for unified_ego after warm start or immediately with original ego regularizer.
```

Training strategy to avoid rerunning baseline:

```text
Use the same data, same init checkpoint, same schedule, same stop-at-20 protocol.
Compare against the already-running matched no-DFP baseline_sft80cfg_stop20.
Do not regenerate a baseline unless the data, optimizer schedule, batch size, validation set, or loss weights outside DFP change.
For the first single-decoder experiment, train only new DFP adapter/head/gates for a few epochs or use low LR for original stack.
Then unfreeze selected shared blocks if ego/lon improves without neighbor regression.
```

Why this should work:

```text
DFP only needs new temporal/noise/chunk conditioning for ego.
The original DP decoder already has strong scene-agent interaction weights.
Sharing the stack lets DFP reuse those learned scene/agent representations instead of learning a separate decoder from scratch.
Neighbor behavior is protected because the original neighbor head and original decoder stack remain initialized from the base checkpoint.
```

Main risks:

```text
DFP chunk tokens and original DP future tokens have different semantics.
If DFP tokens freely attend into neighbor tokens too early, neighbor loss may regress.
If DFP output immediately replaces ego without warmup, ego may be unstable before the adapter learns.
If original stack is fully unfrozen at full LR, the method may drift away from pretrained DP behavior.
```

Risk controls:

```text
zero-init DFP-to-agent gate
gradual DFP gate schedule
freeze original stack for initial epochs
low LR multiplier for original stack
keep original ego regularizer initially, e.g. dfp_lambda_original_ego=0.2
keep neighbor loss unchanged
keep turn indicator unchanged except using final DFP ego trajectory
```

Minimal implementation plan:

```text
1. Keep current hybrid implementation as baseline/proven reference.
2. Add new mode: dfp_decoder_mode=shared_stack_unified_ego.
3. Replace DFPDiT call with shared original decoder blocks over augmented token sequence.
4. Keep original all-agent output head for neighbor futures.
5. Add DFP chunk output head for DFP ego future.
6. Zero-init DFP-to-agent gate.
7. Reuse current DFP losses: history/current/future chunk x0 losses.
8. Reuse current planner/neighbor/turn losses unchanged.
9. Train from same init checkpoint and compare to current matched no-DFP baseline; no new baseline needed if only method changes.
```

Decision:

```text
The current two-decoder implementation is acceptable for proof-of-effect.
For production-quality simplification, implement shared_stack_unified_ego next.
Do not attempt full all-agent DFP replacement yet; it is higher risk and would likely require more data/model changes and a new baseline.
```
