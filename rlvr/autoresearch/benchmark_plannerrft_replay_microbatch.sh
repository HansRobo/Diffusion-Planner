#!/usr/bin/env bash
set -euo pipefail

MICROBATCH=${1:?usage: benchmark_plannerrft_replay_microbatch.sh 96|192}
if [[ ${MICROBATCH} -ne 96 && ${MICROBATCH} -ne 192 ]]; then
  echo "microbatch must be 96 or 192" >&2
  exit 2
fi

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
MODEL=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth
ARGS=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
CACHE=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_anchored_positive_margin001_beta2_noanchor_overlay/replay_buffer
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_replay_microbatch_b${MICROBATCH}_benchmark
EFFECTIVE_STEPS=4
ACCUMULATION_STEPS=$(((192 + MICROBATCH - 1) / MICROBATCH))
REPLAY_UPDATES=$((EFFECTIVE_STEPS * ACCUMULATION_STEPS))

if [[ -e "${OUT}" ]]; then
  echo "benchmark output already exists: ${OUT}" >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec "${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
  -m rlvr.train_awr \
  --model_path "${MODEL}" \
  --args_path "${ARGS}" \
  --train_npz_list "${TRAIN}" \
  --valid_npz_list "${VALID}" \
  --config "${CONFIG}" \
  --output_dir "${OUT}" \
  --exp_name "plannerrft_replay_microbatch_b${MICROBATCH}_benchmark" \
  --device cuda \
  --seed 3407 \
  --start_epoch 2 \
  --epochs 2 \
  --resume_replay_root "${CACHE}" \
  --replay_sampling with_replacement \
  --replay_updates_per_epoch "${REPLAY_UPDATES}" \
  --awr_beta 2 \
  --positive_advantage_only \
  --positive_advantage_margin 0.01 \
  --behavior_anchor_weight 0 \
  --unsafe_behavior_anchor_weight 0 \
  --expert_anchor_weight 0.4 \
  --drop_all_zero_groups \
  --awr_loss_type plain_mse \
  --diffusion_t_range 0.001 0.2 \
  --trainable_scope output \
  --learning_rate 0.000001 \
  --ema_decay 0.95 \
  --ema_per_epoch \
  --ema_commit_live_policy \
  --max_train_scenes 153600 \
  --max_valid_scenes 8 \
  --train_selector_scenes 0 \
  --full_valid_interval 0 \
  --hard_valid_scenes 0 \
  --scene_batch_size "${MICROBATCH}" \
  --gradient_accumulation_scenes 192 \
  --scene_load_workers 4 \
  --eval_scene_load_workers 1 \
  --eval_scene_batch_size 8 \
  --save_rollouts 0 \
  --no-wandb \
  --no-skip_filtered_scenes
