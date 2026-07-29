#!/usr/bin/env bash
set -euo pipefail

# One-pass projection ablation on the immutable anchored PlannerRFT cache.
# Required: EXP_NAME.  Optional environment overrides isolate the denoising
# timestep, Original-DP loss geometry and trainable decoder scope without
# changing candidates or rewards.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
MODEL=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth
ARGS=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
CACHE=${CACHE:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_anchored_positive_overlay_beta1/replay_buffer}
EXP_NAME=${EXP_NAME:?set EXP_NAME}
AWR_LOSS=${AWR_LOSS:-plain_mse}
TMIN=${TMIN:-0.001}
TMAX=${TMAX:-0.999}
TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-dit}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
BETA=${BETA:-1.0}
POSITIVE_ONLY=${POSITIVE_ONLY:-1}
POSITIVE_MARGIN=${POSITIVE_MARGIN:-0.0}
BEHAVIOR_ANCHOR_WEIGHT=${BEHAVIOR_ANCHOR_WEIGHT:-1.0}
UNSAFE_BEHAVIOR_ANCHOR_WEIGHT=${UNSAFE_BEHAVIOR_ANCHOR_WEIGHT:-1.0}
EXPERT_ANCHOR_WEIGHT=${EXPERT_ANCHOR_WEIGHT:-0.0}
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/${EXP_NAME}

POSITIVE_FLAGS=()
if [[ "${POSITIVE_ONLY}" == "1" ]]; then
  POSITIVE_FLAGS+=(
    --positive_advantage_only
    --positive_advantage_margin "${POSITIVE_MARGIN}"
    --behavior_anchor_weight "${BEHAVIOR_ANCHOR_WEIGHT}"
    --unsafe_behavior_anchor_weight "${UNSAFE_BEHAVIOR_ANCHOR_WEIGHT}"
  )
elif [[ "${POSITIVE_ONLY}" != "0" ]]; then
  echo "POSITIVE_ONLY must be 0 or 1, got ${POSITIVE_ONLY}" >&2
  exit 2
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
  --exp_name "${EXP_NAME}" \
  --device cuda \
  --seed 3407 \
  --start_epoch 2 \
  --epochs 2 \
  --resume_replay_root "${CACHE}" \
  --replay_sampling shuffle \
  --awr_beta "${BETA}" \
  "${POSITIVE_FLAGS[@]}" \
  --expert_anchor_weight "${EXPERT_ANCHOR_WEIGHT}" \
  --drop_all_zero_groups \
  --awr_loss_type "${AWR_LOSS}" \
  --diffusion_t_range "${TMIN}" "${TMAX}" \
  --trainable_scope "${TRAINABLE_SCOPE}" \
  --learning_rate "${LEARNING_RATE}" \
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
