#!/usr/bin/env bash
set -euo pipefail

# Causal mining ablation: keep all production rollout/reward settings fixed,
# but reserve candidate 0 for the zero-noise deployment behavior.  Reuse only
# the immutable scene encodings and strict scene order from the all-stochastic
# cache; trajectories, rewards, and weights are regenerated from scratch.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
MODEL=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth
ARGS=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
SHARED=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_enriched_unique153600_cycle_pilot/20260720-010747_plannerrft_enriched_unique153600_cycle_pilot/replay_buffer
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_anchored_unique153600_mine_pilot

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
  --exp_name plannerrft_anchored_unique153600_mine_pilot \
  --device cuda \
  --seed 3407 \
  --start_epoch 1 \
  --epochs 1 \
  --max_train_scenes 153600 \
  --max_valid_scenes 8 \
  --train_selector_scenes 8192 \
  --full_valid_interval 0 \
  --hard_valid_scenes 0 \
  --rollout_scene_batch_size 192 \
  --scene_batch_size 192 \
  --shared_replay_encoding_root "${SHARED}" \
  --shared_replay_order_epoch 1 \
  --rollout_scene_load_workers 1 \
  --rollout_prefetch_batches 1 \
  --scene_load_workers 4 \
  --eval_scene_load_workers 1 \
  --eval_scene_batch_size 512 \
  --save_rollouts 0 \
  --no-wandb \
  --no-skip_filtered_scenes
