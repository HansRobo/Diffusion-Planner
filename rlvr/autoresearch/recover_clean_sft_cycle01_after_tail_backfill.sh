#!/usr/bin/env bash
# Crash-recovery owner for the clean-SFT offset-0 PlannerRFT/AWR campaign.
set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main

# The valid manifest is group-readable by ubuntu.  Re-enter with the current
# supplementary groups when this watcher was launched from an old shell.
if ! id -Gn | tr ' ' '\n' | grep -qx 'ubuntu'; then
  if getent group ubuntu | awk -F: -v user="${USER:-$(id -un)}" '
      { n = split($4, members, ","); for (i = 1; i <= n; ++i) if (members[i] == user) found = 1 }
      END { exit(found ? 0 : 1) }
    '; then
    script_path=$(readlink -f "${BASH_SOURCE[0]}")
    printf -v command_q '%q ' "${script_path}" "$@"
    exec sg ubuntu -c "exec ${command_q}"
  fi
  echo "wang is a member of ubuntu but the current shell lacks the group; sg ubuntu re-exec failed" >&2
  exit 1
fi

PYTHON=${ROOT}/.venv/bin/python
CONTINUE=${ROOT}/rlvr/autoresearch/continue_plannerrft_full_cycle01_after_mine.sh
FULL=${ROOT}/rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh
TRAIN=${TRAIN:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}
VALID=${VALID:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}
CONFIG=${CONFIG:-${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_clean_sft.json}
MODEL=${MODEL:-/tmp/t4_v5_original_dp/model/best_model.pth}
MINE_RUN=${MINE_RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_clean_sft_offset0_full_cycle01_mine/20260723-002534_plannerrft_full_cycle01_mine}
BACKFILL_PID=${BACKFILL_PID:-4150266}
OVERLAY_PARENT=${OVERLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_clean_sft_offset0_repair_cycle01_positive_overlay}
REPLAY_PARENT=${REPLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_clean_sft_offset0_repair_cycle01_replay_e02_e10}
CAMPAIGN_OUT=${CAMPAIGN_OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_clean_sft_offset0_repair_full_cycles02_to10_e100}
BASELINE_DP10=${BASELINE_DP10:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/deployment_full10/source/scenes.json}
MINE_PARENT=$(dirname "${MINE_RUN}")
CACHE=${MINE_RUN}/replay_buffer
PLAN=${MINE_RUN}/tail_repair_20260723
LOG=${PLAN}/recovery_supervisor.log
EXPECTED_RANK_GROUPS=680832
EXPECTED_TOTAL_GROUPS=5446656

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-0}
export DP_DDP_TIMEOUT_MINUTES=${DP_DDP_TIMEOUT_MINUTES:-180}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "${PLAN}"
LOG_LOCK=${PLAN}/recovery_supervisor.lock
exec 9>"${LOG_LOCK}"
flock -n 9 || exit 1

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG}" >&2
}

die() {
  log "FATAL: $*"
  exit 1
}

[[ ${DP_NEIGHBOR_FUTURE_OFFSET} == 0 ]] || die "recovery requires offset 0"
[[ -d ${CACHE} && -d ${PLAN} && -s ${MODEL} && -s ${CONFIG} ]] \
  || die "recovery input is missing"

if [[ ${BACKFILL_PID} =~ ^[0-9]+$ ]]; then
  while kill -0 "${BACKFILL_PID}" 2>/dev/null; do
    log "waiting for tail backfill pid=${BACKFILL_PID}"
    sleep 30
  done
fi

for _ in $(seq 1 60); do
  if ! pgrep -f 'backfill_awr_replay_tail_distributed' >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
if pgrep -f 'backfill_awr_replay_tail_distributed' >/dev/null 2>&1; then
  die "tail backfill is still active"
fi

log "validating strict cache after tail backfill"
total=0
for rank in $(seq 0 7); do
  printf -v rank_dir '%s/rank_%04d' "${CACHE}" "${rank}"
  manifest=${rank_dir}/manifest.json
  expert=${rank_dir}/expert_anchor_manifest.json
  [[ -s ${manifest} && -s ${expert} ]] \
    || die "rank ${rank} strict manifest missing"
  count=$(jq -er '.scene_count' "${manifest}")
  [[ ${count} -eq ${EXPECTED_RANK_GROUPS} ]] \
    || die "rank ${rank}: count=${count}, expected=${EXPECTED_RANK_GROUPS}"
  jq -e '
    .version >= 3 and .group_size == 10 and .future_len == 80 and
    .decoder_context_cached == true and
    .decoder_context_schema == "canonical_loader_aligned_v1" and
    (.arrays.scene_encoding | type == "string") and
    (.arrays.context_ego_current_state | type == "string") and
    (.arrays.context_neighbor_agents_past | type == "string") and
    (.arrays.context_neighbor_agents_future | type == "string")
  ' "${manifest}" >/dev/null || die "rank ${rank}: context contract failed"
  jq -e --argjson n "${count}" '
    .schema == "logged_ego_future_hard_gate_v1" and .scene_count == $n
  ' "${expert}" >/dev/null || die "rank ${rank}: expert contract failed"
  total=$((total + count))
done
[[ ${total} -eq ${EXPECTED_TOTAL_GROUPS} ]] \
  || die "strict cache total=${total}, expected=${EXPECTED_TOTAL_GROUPS}"

if [[ ! -s ${MINE_RUN}/decoder_context_attachment.json ]]; then
  log "attaching inline decoder-context metadata"
  "${PYTHON}" -m rlvr.autoresearch.tools.attach_replay_decoder_context \
    --replay-root "${CACHE}" --expected-world-size 8 >> "${LOG}" 2>&1
fi

jq -n \
  --argjson source_scene_count 5446154 \
  --argjson cached_group_count "${EXPECTED_TOTAL_GROUPS}" \
  --argjson global_rollout_batch 1536 \
  '{version:1, contract:"all_source_scenes_once_then_deterministic_prefix_padding",
    source_scene_count:$source_scene_count, cached_group_count:$cached_group_count,
    ddp_tail_padding_count:($cached_group_count-$source_scene_count),
    global_rollout_batch:$global_rollout_batch}' \
  > "${CACHE}/ddp_tail_padding_contract.json"

if [[ ! -s ${OVERLAY_PARENT}/overlay_manifest.json ]]; then
  log "building positive-anchor overlay"
  "${PYTHON}" -m rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
    --source-replay "${CACHE}" --output-replay "${OVERLAY_PARENT}/replay_buffer" \
    --beta 2 --margin 0.01 --behavior-anchor-weight 0 \
    --unsafe-behavior-anchor-weight 0 >> "${LOG}" 2>&1
fi
if [[ ! -s ${OVERLAY_PARENT}/target_geometry.json ]]; then
  log "auditing overlay target geometry"
  "${PYTHON}" -m rlvr.autoresearch.tools.analyze_replay_target_geometry \
    --replay-root "${OVERLAY_PARENT}/replay_buffer" \
    --output "${OVERLAY_PARENT}/target_geometry.json" --expert-weight 0.4 \
    >> "${LOG}" 2>&1
fi
jq -e --argjson groups "${EXPECTED_TOTAL_GROUPS}" '
  .source_complete == true and .prefix_groups == $groups and
  .group_size == 10 and .future_len == 80 and
  .contract.weight_source == "stored replay weights" and
  .parameters.expert_weight == 0.4 and
  .active_group_fraction > 0 and .active_candidate_target_reward_gain > 0
' "${OVERLAY_PARENT}/target_geometry.json" >/dev/null \
  || die "overlay target geometry audit failed"

if [[ ! -s ${REPLAY_PARENT}/final_summary.json ]]; then
  log "continuing Cycle-1 replay epochs 2-10"
  MODEL="${MODEL}" ARGS="${MINE_RUN}/model_args.json" TRAIN="${TRAIN}" VALID="${VALID}" \
    CONFIG="${CONFIG}" MINE_PARENT="${MINE_PARENT}" \
    OVERLAY_PARENT="${OVERLAY_PARENT}" REPLAY_PARENT="${REPLAY_PARENT}" \
    WANDB_MODE="${WANDB_MODE:-online}" DP_NEIGHBOR_FUTURE_OFFSET=0 \
    DP_DDP_TIMEOUT_MINUTES="${DP_DDP_TIMEOUT_MINUTES}" \
    bash "${CONTINUE}" >> "${LOG}" 2>&1
else
  log "Cycle-1 replay already committed"
fi
[[ -s ${REPLAY_PARENT}/final_summary.json && -s ${REPLAY_PARENT}/best_train.pth ]] \
  || die "Cycle-1 replay checkpoint is missing"

if [[ ! -s ${CAMPAIGN_OUT}/cycle_state.jsonl ]] || \
   ! rg -q '"cycle": 10' "${CAMPAIGN_OUT}/cycle_state.jsonl" 2>/dev/null; then
  log "starting autonomous cycles 2-10 through epoch 100"
  CONFIG="${CONFIG}" BASELINE_DP10="${BASELINE_DP10}" \
    CYCLE1_MINE_RUN="${MINE_RUN}" CYCLE1_REPLAY_PARENT="${REPLAY_PARENT}" \
    CYCLE1_SOURCE_CHECKPOINT="${MODEL}" \
    COMPRESSED_CONTEXT_ROOT="${MINE_RUN}/replay_context_zstd_v1" \
    OUT="${CAMPAIGN_OUT}" TARGET_EPOCH=100 DP_NEIGHBOR_FUTURE_OFFSET=0 \
    DP_DDP_TIMEOUT_MINUTES="${DP_DDP_TIMEOUT_MINUTES}" \
    bash "${FULL}" >> "${LOG}" 2>&1
else
  log "epoch-100 campaign already committed"
fi
log "clean-SFT offset-0 campaign recovery completed"
