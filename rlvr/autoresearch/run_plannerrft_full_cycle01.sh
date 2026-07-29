#!/usr/bin/env bash
set -euo pipefail

# Full-corpus cycle 1 for PlannerRFT-guided Original-DP AWR.
# Candidate mining remains group-relative because PlannerRFT's native/guided
# top-union is a candidate-generation contract.  A read-only overlay then
# applies the selected significant-positive AWR weights for replay.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
MODEL=${MODEL:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
ARGS=${ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
TRAIN=${TRAIN:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}
VALID=${VALID:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}
CONFIG=${CONFIG:-${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json}
MINE_PARENT=${MINE_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine}
OVERLAY_PARENT=${OVERLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_positive_overlay}
REPLAY_PARENT=${REPLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_replay_e02_e10}
EXPECTED_MODEL_SHA256=${EXPECTED_MODEL_SHA256:-db98405cf823ec5a9a82eed16cb19c5451d11020bf235b764eb4b0b3aa8855a0}
EXPECTED_TRAIN_LINES=5446155
EXPECTED_VALID_LINES=46263
EXPECTED_TRAIN_SCENES=5446154
WORLD_SIZE=8
ROLLOUT_SCENES_PER_RANK_BATCH=192
GLOBAL_ROLLOUT_BATCH=$((WORLD_SIZE * ROLLOUT_SCENES_PER_RANK_BATCH))
# DDP needs equal, static-shape rollout batches.  train_awr therefore repeats
# the first few entries of the deterministic shuffled order only in the final
# global batch.  Every source scene is still present exactly once before this
# explicitly accounted tail padding.
EXPECTED_MINED_GROUPS=$((
  (EXPECTED_TRAIN_SCENES + GLOBAL_ROLLOUT_BATCH - 1)
  / GLOBAL_ROLLOUT_BATCH * GLOBAL_ROLLOUT_BATCH
))
MIN_FREE_CACHE_KIB=${MIN_FREE_CACHE_KIB:-6442450944}

for path in "${ROOT}/.venv/bin/torchrun" "${MODEL}" "${ARGS}" "${TRAIN}" "${VALID}" "${CONFIG}"; do
  if [[ ! -r "${path}" || ! -s "${path}" ]]; then
    echo "required input is missing, empty, or unreadable: ${path}" >&2
    exit 1
  fi
done

if [[ $(wc -l < "${TRAIN}") -ne ${EXPECTED_TRAIN_LINES} ]]; then
  echo "unexpected train manifest line count" >&2
  exit 1
fi
if [[ $(wc -l < "${VALID}") -ne ${EXPECTED_VALID_LINES} ]]; then
  echo "unexpected validation manifest line count" >&2
  exit 1
fi
ACTUAL_MODEL_SHA256=$(sha256sum "${MODEL}" | awk '{print $1}')
if [[ "${ACTUAL_MODEL_SHA256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
  echo "incumbent checkpoint hash changed: ${ACTUAL_MODEL_SHA256}" >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 8 ]]; then
  echo "full cycle requires eight GPUs" >&2
  exit 1
fi
AVAILABLE_CACHE_KIB=$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')
if [[ ${AVAILABLE_CACHE_KIB} -lt ${MIN_FREE_CACHE_KIB} ]]; then
  echo "insufficient free NVMe: ${AVAILABLE_CACHE_KIB} KiB" >&2
  exit 1
fi
for output in "${MINE_PARENT}" "${OVERLAY_PARENT}" "${REPLAY_PARENT}"; do
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite existing formal output: ${output}" >&2
    exit 1
  fi
done

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "PlannerRFT full-cycle preflight passed"
  echo "train scenes: ${EXPECTED_TRAIN_SCENES}"
  echo "validation scenes: $((EXPECTED_VALID_LINES - 1))"
  echo "free NVMe KiB: ${AVAILABLE_CACHE_KIB}"
  exit 0
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-0}
export DP_DDP_TIMEOUT_MINUTES=${DP_DDP_TIMEOUT_MINUTES:-180}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "phase 1/2: mining ${EXPECTED_TRAIN_SCENES} guided candidate groups"
"${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
  -m rlvr.train_awr \
  --model_path "${MODEL}" \
  --args_path "${ARGS}" \
  --train_npz_list "${TRAIN}" \
  --valid_npz_list "${VALID}" \
  --config "${CONFIG}" \
  --output_dir "${MINE_PARENT}" \
  --exp_name plannerrft_full_cycle01_mine \
  --device cuda \
  --seed 3407 \
  --start_epoch 1 \
  --epochs 1 \
  --expert_anchor_weight 0.4 \
  --max_train_scenes 0 \
  --max_valid_scenes 8 \
  --eval_k 1 \
  --eval_sample_steps 10 \
  --train_selector_scenes 128 \
  --full_valid_interval 0 \
  --hard_valid_scenes 0 \
  --rollout_scene_batch_size 192 \
  --scene_batch_size 192 \
  --rollout_scene_load_workers 1 \
  --rollout_prefetch_batches 1 \
  --scene_load_workers 4 \
  --eval_scene_load_workers 1 \
  --eval_scene_batch_size 512 \
  --save_rollouts 0 \
  --no-wandb \
  --no-skip_filtered_scenes

MINE_RUN=$(find "${MINE_PARENT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
MINE_CACHE=${MINE_RUN}/replay_buffer
if [[ $(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l) -ne 8 ]]; then
  echo "full mine did not publish eight replay manifests" >&2
  exit 1
fi
if [[ $(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name expert_anchor_manifest.json | wc -l) -ne 8 ]]; then
  echo "full mine did not publish eight expert sidecars" >&2
  exit 1
fi
MINED_SCENES=$(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json -print0 \
  | xargs -0 jq -s 'map(.scene_count) | add')
if [[ ${MINED_SCENES} -ne ${EXPECTED_MINED_GROUPS} ]]; then
  echo "full mine group count mismatch: ${MINED_SCENES}; expected ${EXPECTED_MINED_GROUPS} including DDP tail padding" >&2
  exit 1
fi
jq -n \
  --argjson source_scene_count "${EXPECTED_TRAIN_SCENES}" \
  --argjson cached_group_count "${MINED_SCENES}" \
  --argjson global_rollout_batch "${GLOBAL_ROLLOUT_BATCH}" \
  '{
    version: 1,
    contract: "all_source_scenes_once_then_deterministic_prefix_padding",
    source_scene_count: $source_scene_count,
    cached_group_count: $cached_group_count,
    ddp_tail_padding_count: ($cached_group_count - $source_scene_count),
    global_rollout_batch: $global_rollout_batch
  }' > "${MINE_CACHE}/ddp_tail_padding_contract.json"

"${ROOT}/.venv/bin/python" -m \
  rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
  --source-replay "${MINE_CACHE}" \
  --output-replay "${OVERLAY_PARENT}/replay_buffer" \
  --beta 2 \
  --margin 0.01 \
  --behavior-anchor-weight 0 \
  --unsafe-behavior-anchor-weight 0

if [[ -z "${WANDB_MODE:-}" ]]; then
  WANDB_MODE=online
fi
export WANDB_MODE
RUN_NAME=${WANDB_RUN_NAME:-original-dp-awr-plannerrft-full-cycle01-e02-e10}

echo "phase 2/2: replay epochs 2-10 with one full validation pass each epoch"
exec "${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
  -m rlvr.train_awr \
  --model_path "${MODEL}" \
  --args_path "${ARGS}" \
  --train_npz_list "${TRAIN}" \
  --valid_npz_list "${VALID}" \
  --config "${CONFIG}" \
  --output_dir "${REPLAY_PARENT}" \
  --exp_name plannerrft_full_cycle01_replay_e02_e10 \
  --device cuda \
  --seed 3407 \
  --start_epoch 2 \
  --epochs 10 \
  --resume_replay_root "${OVERLAY_PARENT}/replay_buffer" \
  --replay_sampling with_replacement \
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
  --max_train_scenes 0 \
  --max_valid_scenes 0 \
  --eval_k 1 \
  --eval_sample_steps 10 \
  --rollback_rejected_epoch \
  --train_selector_scenes 65536 \
  --full_valid_interval 0 \
  --hard_valid_scenes 0 \
  --scene_batch_size 192 \
  --gradient_accumulation_scenes 192 \
  --scene_load_workers 4 \
  --eval_scene_load_workers 1 \
  --eval_scene_batch_size 512 \
  --save_rollouts 0 \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-original-dp-awr}" \
  --wandb_entity "${WANDB_ENTITY:-advanced-technology-department}" \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_tags original-dp awr plannerrft-guidance conditional-guidance \
    positive-margin001 beta2 expert04 epoch-boundary-alpha005 \
    full-sequence cycle01 \
  --no-skip_filtered_scenes
