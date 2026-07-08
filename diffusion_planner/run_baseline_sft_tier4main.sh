#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-baseline_sft_tier4main_node02_8gpu_tf32}
SAVE_DIR=${SAVE_DIR:-/mnt/nvme/Diffusion-Planner-dfp-tier4-additive/outputs/${RUN_NAME}}
INIT_WEIGHTS=${INIT_WEIGHTS:-/mnt/nvme/Diffusion-Planner-dfp-tier4-additive/checkpoints/base_sft/best_model.pth}
TRAIN_LIST=${TRAIN_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/_pathlists/path_list_train_rebuilt_step3.json}
VALID_LIST=${VALID_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/dfp_matched_20260622_step3/_pathlists/path_list_valid_rebuilt_step3.json}
NORMALIZATION=${NORMALIZATION:-/mnt/nvme/Diffusion-Planner-dfp/normalization_33d.json}
PORT=${PORT:-22350}

cd /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/diffusion_planner

export HOME=/mnt/nvme/Diffusion-Planner-dfp/home
export TMPDIR=/mnt/nvme/Diffusion-Planner-dfp-tier4-additive/tmp
export XDG_CACHE_HOME=/mnt/nvme/Diffusion-Planner-dfp-tier4-additive/cache
export WANDB_DIR="${SAVE_DIR}"
export WANDB_CACHE_DIR=/mnt/nvme/Diffusion-Planner-dfp/wandb_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_ENTITY=${WANDB_ENTITY:-advanced-technology-department}
export WANDB_PROJECT=${WANDB_PROJECT:-Diffusion-Planner-Temporal}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/nvme/Diffusion-Planner-dfp-tier4-additive:${PYTHONPATH:-}

mkdir -p "${SAVE_DIR}" /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/tmp /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/cache /mnt/nvme/Diffusion-Planner-dfp-tier4-additive/wandb_cache

exec /mnt/nvme/Diffusion-Planner-dfp/.venv/bin/python -m torch.distributed.run --nproc_per_node=8 --master_port="${PORT}" train_predictor.py \
  --exp_name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --train_set_list "${TRAIN_LIST}" \
  --valid_set_list "${VALID_LIST}" \
  --normalization_file_path "${NORMALIZATION}" \
  --init_weights_path "${INIT_WEIGHTS}" \
  --use_dfp_decoder False \
  --dfp_use_inference False \
  --dfp_history_len 20 \
  --dfp_chunk_len 20 \
  --dfp_lambda_hist 0.0 \
  --dfp_lambda_future 0.0 \
  --dfp_lambda_current 0.0 \
  --dfp_history_beta_a 0.5 \
  --dfp_history_beta_b 0.5 \
  --dfp_guidance_w 0.2 \
  --dfp_guidance_beta 2.0 \
  --dfp_sampler_steps 10 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --warm_up_epoch 5 \
  --gradient_accumulation_steps 2 \
  --train_epochs 80 \
  --save_utd 10 \
  --num_workers 8 \
  --skip_filter False \
  --ddp True \
  --device cuda \
  --tf32 True \
  --use_data_augment True \
  --augment_prob 0.5 \
  --num_refine 20 \
  --ego_past_noise_std 0.1 \
  --use_smoothing_future_trajectory True \
  --use_wandb True \
  --wandb_project_name Diffusion-Planner-Temporal \
  --wandb_step_log_interval 50 \
  --future_len 80 \
  --time_len 31 \
  --agent_num 320 \
  --predicted_neighbor_num 320 \
  --lane_num 140 \
  --lane_len 20 \
  --route_num 25 \
  --route_len 20 \
  --static_objects_num 5 \
  --hidden_dim 256 \
  --num_heads 8 \
  --encoder_mixer_depth 6 \
  --encoder_fusion_depth 6 \
  --decoder_depth 3 \
  --road_border_margin 0.25 \
  --road_border_n_interp 2 \
  --neighbor_collision_margin 0.25 \
  --port "${PORT}"
