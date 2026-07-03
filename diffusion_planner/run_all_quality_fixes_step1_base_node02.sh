#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-all_quality_fixes_step1_base_tier4main_node02_8gpu_tf32_20260703}
SAVE_DIR=${SAVE_DIR:-/mnt/nvme/Diffusion-Planner-all-quality-fixes/outputs/${RUN_NAME}}
TRAIN_LIST=${TRAIN_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train.json}
VALID_LIST=${VALID_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_valid_sft.json}
NORMALIZATION=${NORMALIZATION:-/mnt/nvme/Diffusion-Planner-dfp/normalization_33d.json}
PORT=${PORT:-22461}
TMPDIR=${TMPDIR:-/mnt/nvme/Diffusion-Planner-all-quality-fixes/tmp}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/mnt/nvme/Diffusion-Planner-all-quality-fixes/cache}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/mnt/nvme/Diffusion-Planner-dfp/wandb_cache}

mkdir -p "${SAVE_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${WANDB_CACHE_DIR}"

export HOME=/mnt/nvme/Diffusion-Planner-dfp/home
export TMPDIR
export XDG_CACHE_HOME
export WANDB_DIR="${SAVE_DIR}"
export WANDB_CACHE_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_ENTITY=advanced-technology-department
export WANDB_PROJECT=Diffusion-Planner-Temporal
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export PYTHONPATH=/mnt/nvme/Diffusion-Planner-all-quality-fixes/diffusion_planner:${PYTHONPATH:-}

cd /mnt/nvme/Diffusion-Planner-all-quality-fixes/diffusion_planner

exec /mnt/nvme/Diffusion-Planner-all-quality-fixes/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${PORT}" \
  train_predictor.py \
  --exp_name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --train_set_list "${TRAIN_LIST}" \
  --valid_set_list "${VALID_LIST}" \
  --train_subsample_step 1 \
  --normalization_file_path "${NORMALIZATION}" \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --warm_up_epoch 5 \
  --train_epochs 60 \
  --save_utd 10 \
  --num_workers 8 \
  --enable_epdms_eval True \
  --enable_pdms_eval True \
  --epdms_eval_use_agent_boxes True \
  --epdms_eval_use_road_border True \
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
  --port "${PORT}"
