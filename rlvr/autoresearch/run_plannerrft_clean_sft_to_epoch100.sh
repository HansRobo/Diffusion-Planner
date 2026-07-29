#!/usr/bin/env bash
set -Eeuo pipefail

# Clean-start full-corpus PlannerRFT/AWR campaign.  This deliberately does not
# resume any previous policy, replay cache, overlay, or compressed context.
# Existing campaigns remain available as separate comparison arms.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
TORCHRUN=${ROOT}/.venv/bin/torchrun
MODEL=/tmp/t4_v5_original_dp/model/best_model.pth
ARGS=/tmp/t4_v5_original_dp/model/args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_clean_sft.json
BASELINE_DP10=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/deployment_full10/source/scenes.json
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_clean_sft_offset0}
MINE_PARENT=${RUN_ROOT}_full_cycle01_mine
OVERLAY_PARENT=${RUN_ROOT}_full_cycle01_positive_overlay
REPLAY_PARENT=${RUN_ROOT}_full_cycle01_replay_e02_e10
CAMPAIGN_OUT=${RUN_ROOT}_full_cycles02_to10_e100
EXPECTED_MODEL_SHA256=4ffaeea21cd29904da73349eea642e1d28f8ddbf02be363b7386e3a9b8ebcc75

for path in "${MODEL}" "${ARGS}" "${TRAIN}" "${VALID}" "${CONFIG}" "${BASELINE_DP10}"; do
  [[ -s ${path} ]] || { echo "missing required clean-start input: ${path}" >&2; exit 1; }
done
[[ $(sha256sum "${MODEL}" | awk '{print $1}') == "${EXPECTED_MODEL_SHA256}" ]] \
  || { echo "clean SFT checkpoint hash mismatch" >&2; exit 1; }
[[ $(wc -l < "${TRAIN}") == 5446155 ]] || { echo "train manifest changed" >&2; exit 1; }
[[ $(wc -l < "${VALID}") == 46263 ]] || { echo "valid manifest changed" >&2; exit 1; }
[[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ge 8 ]] \
  || { echo "eight GPUs are required" >&2; exit 1; }
[[ $(jq -er '.awr.hdp_trajectory_augmentation == false and .awr.plannerrft_guided_exploration == true and .awr.original_dp_first_waypoint_gate_enabled == true and .plannerrft.reference_model_path == "/tmp/t4_v5_original_dp/model/best_model.pth"' "${CONFIG}") == true ]] \
  || { echo "clean-start config contract failed" >&2; exit 1; }
for path in "${MINE_PARENT}" "${OVERLAY_PARENT}" "${REPLAY_PARENT}" "${CAMPAIGN_OUT}"; do
  [[ ! -e ${path} ]] || { echo "refusing to overwrite existing clean-start output: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-0}
[[ ${DP_NEIGHBOR_FUTURE_OFFSET} =~ ^0$|^1$ ]] \
  || { echo "unsupported DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET}" >&2; exit 1; }
export DP_NEIGHBOR_FUTURE_OFFSET
export DP_DDP_TIMEOUT_MINUTES=${DP_DDP_TIMEOUT_MINUTES:-180}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export CONFIG REFERENCE REFERENCE_ARGS REWARD
REFERENCE=${MODEL}
REFERENCE_ARGS=${ARGS}
REWARD=${CONFIG}

mkdir -p "${RUN_ROOT}"
LAUNCH_LOG=${RUN_ROOT}/clean_start.launch.log
{
  echo "clean-start source=${MODEL}"
  echo "clean-start source_sha256=$(sha256sum "${MODEL}" | awk '{print $1}')"
  echo "clean-start config=${CONFIG}"
  echo "clean-start hdp_output_augmentation=false"
  echo "clean-start planner_rft_ramp_steps=20"
  echo "clean-start first_waypoint_gate=true"
  echo "clean-start neighbor_future_alignment_offset=${DP_NEIGHBOR_FUTURE_OFFSET}"
  echo "clean-start train_scenes=5446154 valid_scenes=46262 world_size=8"
} | tee "${LAUNCH_LOG}"

MODEL="${MODEL}" ARGS="${ARGS}" TRAIN="${TRAIN}" VALID="${VALID}" \
  CONFIG="${CONFIG}" MINE_PARENT="${MINE_PARENT}" \
  OVERLAY_PARENT="${OVERLAY_PARENT}" REPLAY_PARENT="${REPLAY_PARENT}" \
  EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256}" \
  bash "${ROOT}/rlvr/autoresearch/run_plannerrft_full_cycle01.sh" \
  >> "${LAUNCH_LOG}" 2>&1

MINE_RUN=$(find "${MINE_PARENT}" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -n 1)
[[ -s "${MINE_RUN}/final_summary.json" ]] || { echo "clean cycle-1 mine did not complete" | tee -a "${LAUNCH_LOG}" >&2; exit 1; }
jq -e '
  .awr.hdp_trajectory_augmentation == false and
  .awr.plannerrft_guided_exploration == true and
  .awr.original_dp_first_waypoint_gate_enabled == true
' "${MINE_RUN}/effective_config.json" >/dev/null \
  || { echo "clean cycle-1 mine effective config is not clean" | tee -a "${LAUNCH_LOG}" >&2; exit 1; }
jq -e --arg model "${MODEL}" --arg hash "${EXPECTED_MODEL_SHA256}" '
  .model_source == $model and .staged_model_sha256 == $hash
' "${MINE_RUN}/provenance.json" >/dev/null \
  || { echo "clean cycle-1 mine provenance does not point to pristine SFT" | tee -a "${LAUNCH_LOG}" >&2; exit 1; }

CONFIG="${CONFIG}" BASELINE_DP10="${BASELINE_DP10}" \
  CYCLE1_MINE_RUN="${MINE_RUN}" \
  CYCLE1_REPLAY_PARENT="${REPLAY_PARENT}" \
  CYCLE1_SOURCE_CHECKPOINT="${MODEL}" \
  COMPRESSED_CONTEXT_ROOT="${MINE_RUN}/replay_context_zstd_v1" \
  OUT="${CAMPAIGN_OUT}" TARGET_EPOCH=100 \
  REFERENCE="${MODEL}" REFERENCE_ARGS="${ARGS}" REWARD="${CONFIG}" \
  bash "${ROOT}/rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh" \
  >> "${LAUNCH_LOG}" 2>&1

echo "clean-start campaign completed: ${CAMPAIGN_OUT}" | tee -a "${LAUNCH_LOG}"
