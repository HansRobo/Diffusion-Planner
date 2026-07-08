# diffusion_planner entrypoints for the HDP branch

This directory contains the training, validation, and RL entrypoints used by the HDP branch.

## Main entrypoints

| Entrypoint | Purpose |
| --- | --- |
| `train_predictor.py` | Supervised base / SFT training. |
| `train_hdp_rl_predictor.py` | HDP-RL fine-tuning. Official HDP-RL fine-tuning with reward-weighted RL-Hybrid. |
| `valid_run.sh` | Validation wrapper for a saved checkpoint. |
| `train_run.sh` | Legacy convenience wrapper. Prefer explicit commands for HDP experiments. |

## HDP base training

Base training should start from scratch and use the same model/data settings as the comparison runs.

```bash
cd /mnt/nvme/Diffusion-Planner-hyper-diffusion-planner/diffusion_planner
source ../.venv/bin/activate

python3 -m torch.distributed.run --nproc_per_node=8 --master_port=<PORT> train_predictor.py \
  --exp_name <RUN_NAME> \
  --save_dir <OUTPUT_DIR> \
  --train_set_list <FULL_SEQUENCE_BASE_TRAIN_LIST> \
  --valid_set_list <FULL_SEQUENCE_BASE_VALID_LIST> \
  --train_subsample_step 1 \
  --normalization_file_path ./normalization.json \
  --batch_size 512 \
  --learning_rate 2e-4 \
  --warm_up_epoch 5 \
  --train_epochs 60 \
  --save_utd 10 \
  --future_len 80 \
  --time_len 31 \
  --agent_num 320 \
  --predicted_neighbor_num 320 \
  --diffusion_model_type x_start \
  --diffusion_supervision_type x_start \
  --diffusion_time_sample_method uniform \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --diffusion_sample_steps 10 \
  --enable_epdms_eval True \
  --enable_pdms_eval True \
  --use_wandb True \
  --wandb_project_name Diffusion-Planner-Temporal \
  --tf32 True \
  --amp_dtype bf16 \
  --fused_optimizer True \
  --ddp_static_graph True
```

## HDP SFT

SFT uses the same HDP architecture and starts from a base checkpoint with `--init_weights_path`.

```bash
python3 -m torch.distributed.run --nproc_per_node=8 --master_port=<PORT> train_predictor.py \
  --exp_name <SFT_RUN_NAME> \
  --save_dir <SFT_OUTPUT_DIR> \
  --train_set_list <FULL_SEQUENCE_SFT_TRAIN_LIST> \
  --valid_set_list <FULL_SEQUENCE_SFT_VALID_LIST> \
  --init_weights_path <HDP_BASE_CHECKPOINT> \
  --train_subsample_step 1 \
  --normalization_file_path ./normalization.json \
  --train_epochs 20 \
  --save_utd 10 \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --diffusion_sample_steps 10 \
  --enable_epdms_eval True \
  --enable_pdms_eval True \
  --use_wandb True \
  --wandb_project_name Diffusion-Planner-Temporal \
  --tf32 True \
  --amp_dtype bf16
```

Use `--resume_model_path` only for continuing the same interrupted run. Do not use it for base-to-SFT or SFT-to-RL transfer.

## Official HDP-RL

RL must start from an HDP SFT checkpoint. The only RL objective is reward-weighted HDP hybrid loss.

```bash
python3 -m torch.distributed.run --nproc_per_node=8 --master_port=<PORT> train_hdp_rl_predictor.py \
  --exp_name <RL_RUN_NAME> \
  --save_dir <RL_OUTPUT_DIR> \
  --train_set_list <FULL_SEQUENCE_SFT_TRAIN_LIST> \
  --valid_set_list <FULL_SEQUENCE_SFT_VALID_LIST> \
  --init_weights_path <HDP_SFT_CHECKPOINT> \
  --normalization_file_path ./normalization.json \
  --official_reward_normalize group \
  --official_reward_beta 1.0 \
  --rl_reward_w_risk 1.0 \
  --rl_reward_w_follow 3.0 \
  --rl_reward_w_lane 2.5 \
  --num_generations 32 \
  --rl_noise_scale 0.5 \
  --rl_train_scope decoder \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --diffusion_sample_steps 10 \
  --enable_epdms_eval True \
  --enable_pdms_eval True \
  --use_wandb True \
  --wandb_project_name Diffusion-Planner-Temporal \
  --tf32 True \
  --amp_dtype bf16
```

Important semantics:

- `--init_weights_path`: fresh RL run initialized from SFT weights only.
- `--resume_model_path`: strict resume of an interrupted RL run, including optimizer/scheduler state.
- `--rl_train_scope decoder`: official-style decoder fine-tuning.
- `--train_subsample_step` defaults to `1`; this branch optimizes for final model quality unless explicitly overridden.
- RL reward is fixed to the Tier IV NPZ adaptation of the official HDP multi-reward setting. It combines EPDMS-style risk, route/GT following, and lane-keeping subscores with the official 1.0/3.0/2.5 weights before group normalization.
- Legacy RL alternatives have been removed from this branch; HDP-RL has a single supported reward-weighted hybrid path.

## Vanilla DP compatibility mode

This branch can still run original DP-style supervised training by disabling HDP-specific options:

```bash
--use_velocity_representation False \
--planning_hybrid_loss 0.0 \
--diffusion_model_type x_start \
--diffusion_supervision_type x_start
```

This compatibility mode is useful for local comparison, but it is not the branch's primary contract. For clean baseline PRs or production vanilla-DP changes, use the upstream Tier IV main branch.

## HDP ONNX export

For HDP velocity checkpoints, deploy the full ONNX graph:

```text
diffusion_planner.onnx
```

The split decoder graph is intentionally skipped for HDP because its ego row is a velocity latent,
not a waypoint latent. The full graph runs sampling and decodes ego velocity back to waypoint-space
`prediction`.

Smoke-test conversion pattern:

```bash
CUDA_VISIBLE_DEVICES="" ../.venv/bin/python ../ros_scripts/torch2onnx.py \
  <CHECKPOINT_DIR_WITH_ARGS_JSON_AND_PTH> \
  --output-prefix diffusion_planner_hdp \
  --opset-version 20
```

Expected HDP outputs:

```text
prediction: [B, 321, 80, 4]
turn_indicator_logit: [B, 5]
```

The branch smoke test on `epoch0010/best_model.pth` passed PyTorch-vs-ORT validation with
`prediction` max diff `4.35e-4` and mean diff `1.48e-5`.
