# Shared-stack DFP step3 SFT20 experiment log - 2026-06-25

## Purpose

Compare DFP architecture variants under the same old step3 SFT data setting before introducing Full Sequence data.

Decision: do not use Full Sequence for this comparison stage, because Full Sequence changes data distribution, valid coverage, and training speed. The clean order is:

1. Matched no-DFP baseline on old step3 SFT data.
2. Additive/unified_ego DFP on old step3 SFT data.
3. Shared-stack/unified_ego DFP on old step3 SFT data.
4. Only after selecting the best architecture, run the selected architecture on Full Sequence data.

## Active run

- Date: 2026-06-25 JST
- Node: node02
- GPUs: 8 x GPU, CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
- Run name: dfp_shared_stack_unified_ego_lam01_sft20_tier4main_node02_8gpu_tf32
- W&B project: advanced-technology-department/Diffusion-Planner-Temporal
- W&B run: https://wandb.ai/advanced-technology-department/Diffusion-Planner-Temporal/runs/2qvergoy
- Local output dir on node02: /mnt/nvme/Diffusion-Planner-dfp-shared-stack/outputs/dfp_shared_stack_unified_ego_lam01_sft20_tier4main_node02_8gpu_tf32
- Launcher log on node02: /mnt/nvme/Diffusion-Planner-dfp-shared-stack/launch_shared_stack_unified_ego_lam01_sft20.log

## Code/worktree

- Local source worktree: /mnt/nvme/Diffusion-Planner-dfp-shared-stack
- Synced to node02: /mnt/nvme/Diffusion-Planner-dfp-shared-stack
- Sync excluded: outputs/, wandb/, cache/, tmp/, .git/, __pycache__/
- Note: .git was intentionally not synced, so W&B may show git root errors. This does not affect training. Experiment identity is recorded in this file instead.

## Data

- Full Sequence: not used
- Train list: /mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/_pathlists/path_list_train_rebuilt_step3.json
- Valid list: /mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/_pathlists/path_list_valid_rebuilt_step3.json
- Dataset prepared at launch: 1,424,405 train samples
- skip_filter: False

## Initialization

- Init checkpoint: /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/checkpoints/base_sft/best_model.pth
- Load mode: weights-only init, strict=False compatible path
- Missing keys at launch: 21
- Unexpected keys at launch: 0
- Missing keys are expected new shared-stack DFP parameters, including dfp_shared_chunk_pos_embed, dfp_shared_preproj, dfp_shared_t_embedder, and dfp_shared_final_layer.

## Main architecture setting

- use_dfp_decoder: True
- dfp_decoder_mode: shared_stack_unified_ego
- dfp_use_inference: True
- dfp_history_len: 20
- dfp_chunk_len: 20
- dfp_lambda_hist: 0.1
- dfp_lambda_future: 0.1
- dfp_lambda_current: 0.0
- dfp_lambda_original_ego: 0.2
- dfp_history_beta_a: 0.5
- dfp_history_beta_b: 0.5
- dfp_guidance_w: 0.2
- dfp_guidance_beta: 2.0
- dfp_sampler_steps: 10

## Training parameters

- batch_size: 512
- learning_rate: 1e-4
- warm_up_epoch: 5
- gradient_accumulation_steps: 2
- train_epochs: 20
- save_utd: 10
- num_workers: 8
- ddp: True
- device: cuda
- tf32: True
- use_data_augment: True
- augment_prob: 0.5
- num_refine: 20
- ego_past_noise_std: 0.1
- use_smoothing_future_trajectory: True
- future_len: 80
- time_len: 31
- agent_num: 320
- predicted_neighbor_num: 320
- lane_num: 140
- route_num: 25
- static_objects_num: 5
- hidden_dim: 256
- num_heads: 8
- encoder_mixer_depth: 6
- encoder_fusion_depth: 6
- decoder_depth: 3

## Launch confirmation

Initial health check at 2026-06-25 12:29 JST:

- torch.distributed parent process: 1620666
- 8 worker processes active
- GPU memory at launch check: about 2.8-3.1 GB per GPU
- GPU utilization at launch check: rank GPUs mostly 100%, rank 0 briefly 0% during startup/logging
- W&B initialized successfully

## Comparison context

Existing matched old-step3 results to compare against:

- DFP additive/unified_ego SFT20 run: dfp_unified_ego_lam03_tier4main_node02_8gpu_tf32
- Matched no-DFP baseline SFT20 run: baseline_sft80cfg_stop20_tier4main_node02_8gpu_tf32
- Prior valid comparison at epoch20:
  - ego: DFP 2.3176318760 vs baseline 3.0082999139
  - neighbor: DFP 3.9162770875 vs baseline 4.2943136260
  - lat: DFP 0.2080857009 vs baseline 0.2365077138
  - lon: DFP 1.6258424520 vs baseline 1.7856726646

This shared-stack run should be compared to those using the same old step3 SFT train/valid lists before deciding whether to move the winning architecture to Full Sequence.
