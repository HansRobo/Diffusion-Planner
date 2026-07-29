#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=/mnt/nvme/wangbin/Diffusion-Planner-hyper-diffusion-planner/.venv/bin/python
MODEL_ROOT=/tmp/t4_v5_original_dp/model
CACHE_ROOT=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260716-184121_full_sequence_filtered_hdp_plain_mse_loader2_single_checkpoint/replay_buffer
OUTPUT_ROOT=${ROOT}/outputs/awr_t4_full_sequence_filtered
LOG_PATH=${LOG_PATH:-${OUTPUT_ROOT}/resume_20260717_hdp_plain_mse.log}
MASTER_PORT=${MASTER_PORT:-29641}

export PYTHONPATH=${ROOT}:${ROOT}/diffusion_planner:${PYTHONPATH:-}
export DP_NEIGHBOR_FUTURE_OFFSET=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

cd "${ROOT}"
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  -m rlvr.train_awr \
  --model_path "${MODEL_ROOT}/best_model.pth" \
  --args_path "${MODEL_ROOT}/args.json" \
  --train_npz_list "${ROOT}/outputs/data/path_list_train_sft_is_skipped_filtered.json" \
  --valid_npz_list "${ROOT}/outputs/data/path_list_valid_sft_balanced_is_skipped_filtered.json" \
  --config "${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_faithful.json" \
  --output_dir "${OUTPUT_ROOT}" \
  --exp_name full_sequence_filtered_hdp_plain_mse_replay_resume_single_checkpoint \
  --device cuda \
  --seed 3407 \
  --max_train_scenes 0 \
  --max_valid_scenes 2048 \
  --epochs 10 \
  --start_epoch 2 \
  --resume_replay_root "${CACHE_ROOT}" \
  --salvage_partial_replay \
  --K 10 \
  --eval_k 10 \
  --noise_scale_range 0.5 0.5 \
  --sample_steps 5 \
  --hdp_trajectory_augmentation \
  --hdp_trajectory_augmentation_std 0.5 \
  --trainable_scope dit \
  --learning_rate 1e-6 \
  --bc_weight 0 \
  --hdp_hybrid_loss_weight 0 \
  --awr_loss_type plain_mse \
  --reward_profile hdp_pdm \
  --reward_mode gate \
  --gradient_accumulation_scenes 1 \
  --scene_batch_size 192 \
  --save_rollouts 0 \
  --amp_dtype bf16 \
  --enable_compile \
  --compile_mode default \
  --no-compile_encoder \
  --compile_decoder \
  --skip_filter_manifest "${ROOT}/outputs/data/full_sequence_is_skipped_filter_manifest.json" \
  --no-skip_filtered_scenes \
  "$@" 2>&1 | tee -a "${LOG_PATH}"
