# PDMS eval + step1 base train record - 2026-06-26

## Code changes

- Ported OnePlanner NAVSIM-formula PDMS proxy into `planner_metrics/pdms_navsim.py` and `planner_metrics/pdms_proxy.py`.
- Validation logs `valid_pdms/*` when `--enable_pdms_eval True`.
- DP-compatible NC/TTC uses `neighbor_agents_future` plus current neighbor shape from `neighbor_agents_past` to build `[x,y,z,w,l,h,yaw,vx,vy]` boxes.
- DAC uses existing DP road-border line detection: `line_strings[..., 3] > 0.5`.
- `train.py` logs PDMS metrics to W&B as `valid_pdms/*` and appends them to `train_log.tsv`.
- `train_predictor.py` / `train_config.py` add `--enable_pdms_eval`, `--pdms_eval_use_agent_boxes`, and `--pdms_eval_use_road_border`.
- `DiffusionPlannerData.__getitem__` converts unsigned integer ndarray fields to `int64`; this is required for full-sequence NPZ because `version` is `uint32` and PyTorch default collate cannot batch it.

## Data decision

Use the full-sequence data only after constraining it by the user-specified 2026-06-22 SFT/base datalist route whitelist. Do not use the top-level mixed full-sequence list directly.

- Correct train list: `/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_train_sft_fullseq_from_20260622_step3.json`
- Correct valid list: `/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_valid_sft_fullseq_from_20260622_step3.json`
- Summary: `/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/summary.json`
- Kept train scenes after sidecar JSON filtering: `2106827`.
- Top-level full-sequence train list has `24280101` scenes and mixes other projects/areas; it was rejected for this experiment.
- Launcher must use `SKIP_FILTER=False` with these pre-filtered artifact lists. The first attempted launch had `--skip_filter True`; it was stopped and fixed.

## Active training run

- Node: `node01`
- Run name: `dfp_shared_stack_step1_base_pdms_tier4main_node01_8gpu_tf32`
- W&B project: `advanced-technology-department/Diffusion-Planner-Temporal`
- W&B run: `https://wandb.ai/advanced-technology-department/Diffusion-Planner-Temporal/runs/mjged6mq`
- Launcher: `/mnt/nvme/Diffusion-Planner-dfp-shared-stack/diffusion_planner/run_dfp_shared_stack_step1_base_pdms.sh`
- Launch log: `/mnt/nvme/Diffusion-Planner-dfp-shared-stack/launch_step1_base_pdms_node01.log`
- Output dir: `/mnt/nvme/Diffusion-Planner-dfp-shared-stack/outputs/dfp_shared_stack_step1_base_pdms_tier4main_node01_8gpu_tf32`
- Train mode: base train from scratch, no init weights.
- GPUs: 8 cards on node01.
- Precision: TF32 enabled.
- Epochs: `80`.
- Batch size: `512`, gradient accumulation: `2`.
- DFP mode: `shared_stack_unified_ego`.
- PDMS eval: enabled, agent boxes enabled, road-border DAC enabled.
- Confirmed command line includes `--skip_filter False`.

## Runtime fixes applied on node01

- W&B login failed because script sets `HOME=/mnt/nvme/Diffusion-Planner-dfp/home`; copied node01 `/home/ubuntu/.netrc` to that HOME and set mode `600`.
- Full-sequence `version uint32` caused DataLoader collate failure; fixed dataset unsigned-int conversion and synced `dataset.py` to node01.
- Installed runtime dependencies earlier with uv where needed: `wandb==0.15.12`, `protobuf==3.20.3`, `scipy`, `shapely`, `setuptools==69.5.1`.

## Status snapshot

- Time: 2026-06-26 18:16 JST.
- Initial validation completed successfully.
- Training is active: about `320 / 8229` batches in epoch 1 at roughly `2.7 batch/s`.
- GPU memory during training is about `49 GB` per GPU.

## Gated shared-stack experiment - 2026-06-27

### Reasoning

The hard-replacement shared-stack run uses:

```text
ego_final = ego_dfp
```

The gated run should not start from a 50/50 mixture because that would inject an unproven original ego head too aggressively. It should also not start from a strict zero original contribution, because then the local gate head receives almost no useful gradient at the beginning. The chosen compromise is a near-hard-DFP residual gate:

```text
local_original_gate = sigmoid(gate_head([ego_original, ego_dfp, scene]))
alpha = sigmoid(alpha_logit), initialized from dfp_gate_alpha_init = 0.1
original_weight = alpha * local_original_gate
dfp_weight = 1 - original_weight
ego_final = dfp_weight * ego_dfp + original_weight * ego_original
```

With zero-initialized gate-head last layer and `dfp_gate_alpha_init=0.1`, the initial weights are approximately:

```text
original_weight ~= 0.05
dfp_weight ~= 0.95
```

This keeps the first behavior close to the already-working hard-DFP model while still letting the gate learn immediately.

### Logged gate metrics

Training step W&B metrics, emitted every `wandb_step_log_interval=50` optimizer steps:

```text
train_step/dfp_gate_mean
train_step/dfp_gate_std
train_step/dfp_gate_min
train_step/dfp_gate_max
train_step/dfp_original_gate_mean
train_step/dfp_original_gate_std
train_step/dfp_original_gate_min
train_step/dfp_original_gate_max
```

Validation metrics, emitted after each epoch to W&B and `train_log.tsv`:

```text
valid_gate/dfp_weight
valid_gate/dfp_weight_std_per_sample
valid_gate/dfp_weight_min_per_sample
valid_gate/dfp_weight_max_per_sample
valid_gate/original_weight
valid_gate/original_weight_std_per_sample
valid_gate/original_weight_min_per_sample
valid_gate/original_weight_max_per_sample
```

### Fair comparison anchor

Hard-replacement run stopped after epoch 8 metrics were written, then gated run was started from scratch with the same LR, total epochs, batch size, data lists, TF32, PDMS eval, and 8-GPU node01 setup.

Hard-replacement epoch 8:

```text
run = dfp_shared_stack_step1_base_pdms_tier4main_node01_8gpu_tf32
valid_loss_ego = 1.0498401446814425
valid_loss_neighbor = 21.405996691943432
valid_loss_ego_position_lat_loss = 0.1945439726114273
valid_loss_ego_position_lon_loss = 1.1502611637115479
valid_loss_ego_trajectory_consistency = 23.030189514160156
valid_pdms_total = 0.7762551307678223
valid_pdms_dac = 0.9787254333496094
valid_pdms_no_collision = 0.9874631762504578
valid_pdms_ttc = 0.9825244545936584
```

Gated run:

```text
run = dfp_shared_stack_gated_ego_alpha01_step1_base_pdms_tier4main_node01_8gpu_tf32
wandb = https://wandb.ai/advanced-technology-department/Diffusion-Planner-Temporal/runs/oixpfgke
mode = shared_stack_gated_ego
dfp_gate_alpha_init = 0.1
```

Primary comparison point is gated epoch 8 vs hard-replacement epoch 8. Gate interpretation:

```text
If original_weight stays near 0.05 and metrics improve, the small stabilizing residual is enough.
If original_weight grows and planning metrics improve, the gate is actively using the original ego head.
If original_weight grows but planning/PDMS/consistency worsen, the original ego head is harmful and hard DFP is preferable.
If original_weight collapses toward zero, the model itself rejects original ego and hard DFP is structurally cleaner.
```
