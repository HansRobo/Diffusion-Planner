#!/usr/bin/env bash
set -euo pipefail

# Continue the exact anchored pilot transaction from epoch 2.  The live policy,
# deployable EMA, optimizer moments, cache, sampler, reward and selector are all
# restored; epochs 3--10 add eight shuffled exhaustive replay passes and retain
# the highest fixed-train-selector EMA checkpoint.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
SOURCE=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_anchored_unique153600_shuffle_replay_pilot/20260720-022422_plannerrft_anchored_unique153600_shuffle_replay_pilot
MODEL=${SOURCE}/epoch_002.pth
ARGS=${SOURCE}/model_args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
CACHE=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_anchored_unique153600_mine_pilot/20260720-021148_plannerrft_anchored_unique153600_mine_pilot/replay_buffer
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_anchored_unique153600_continue_e03_e10

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
  --use_policy_state \
  --resume_optimizer_state \
  --train_npz_list "${TRAIN}" \
  --valid_npz_list "${VALID}" \
  --config "${CONFIG}" \
  --output_dir "${OUT}" \
  --exp_name plannerrft_anchored_unique153600_continue_e03_e10 \
  --device cuda \
  --seed 3407 \
  --start_epoch 3 \
  --epochs 10 \
  --resume_replay_root "${CACHE}" \
  --replay_sampling shuffle \
  --max_train_scenes 153600 \
  --max_valid_scenes 8 \
  --train_selector_scenes 8192 \
  --full_valid_interval 0 \
  --hard_valid_scenes 0 \
  --scene_batch_size 96 \
  --gradient_accumulation_scenes 192 \
  --scene_load_workers 4 \
  --eval_scene_load_workers 1 \
  --eval_scene_batch_size 512 \
  --save_rollouts 0 \
  --no-wandb \
  --no-skip_filtered_scenes
