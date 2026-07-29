#!/usr/bin/env bash
# From-zero PlannerRFT/AWR campaign with the corrected Original-DP
# first-waypoint gate (tangent displacement floor) and the overlay that
# respects mining-time zero weights.  Every stage is idempotent: a committed
# artifact is reused, an interrupted stage restarts from its immutable input,
# so the autonomous owner can re-enter this script after any interruption.
#
# Stage layout (all under one campaign root):
#   ${RUN_ROOT}/cycle01_mine              full-corpus guided candidate mining
#   ${RUN_ROOT}/cycle01_positive_overlay  positive-advantage replay weights
#   ${RUN_ROOT}/cycle01_replay_e02_e10    replay epochs 2-10
#   ${RUN_ROOT}/campaign_e100             cycles 2-10 supervisor output

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
TORCHRUN=${ROOT}/.venv/bin/torchrun
MODEL=/tmp/t4_v5_original_dp/model/best_model.pth
ARGS=/tmp/t4_v5_original_dp/model/args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_clean_sft.json
BASELINE_DP10=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/deployment_full10/source/scenes.json
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_gatefix_clean}
MINE_PARENT=${RUN_ROOT}/cycle01_mine
OVERLAY_PARENT=${RUN_ROOT}/cycle01_positive_overlay
REPLAY_PARENT=${RUN_ROOT}/cycle01_replay_e02_e10
CAMPAIGN_OUT=${RUN_ROOT}/campaign_e100
SFT_STAGE=${RUN_ROOT}/sft_checkpoint
SUPERVISOR=${ROOT}/rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh
HEALTH_MONITOR=${ROOT}/rlvr/autoresearch/monitor_plannerrft_epoch100.sh
EXPECTED_MODEL_SHA256=4ffaeea21cd29904da73349eea642e1d28f8ddbf02be363b7386e3a9b8ebcc75
EXPECTED_SELECTOR_SHA256=49fd1863a3c1d54db59440da426e3475df67ccc4edd8da6ddde7e32945f3620c
EXPECTED_VALID_SHA256=91a1c8ef4004c7074f024495d13416456a4dbe2f0320f0b1a43282659d435dff
EXPECTED_RANK_GROUPS=680832
EXPECTED_PADDED_GROUPS=5446656
TARGET_EPOCH=${TARGET_EPOCH:-100}
MIN_FREE_CACHE_KIB=${MIN_FREE_CACHE_KIB:-6442450944}
LOCK=${RUN_ROOT}/continuation.lock

mkdir -p "${RUN_ROOT}"
exec 9>"${LOCK}"
flock -n 9 || exit 0

STATUS=${RUN_ROOT}/gatefix_launch.log
log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATUS}" >&2
}
die() {
  log "FATAL: $*"
  exit 1
}

# --- durable SFT checkpoint: /tmp does not survive a reboot ------------------
if [[ ! -s ${MODEL} && -s ${SFT_STAGE}/best_model.pth ]]; then
  mkdir -p "$(dirname "${MODEL}")"
  cp "${SFT_STAGE}/best_model.pth" "${MODEL}"
  cp "${SFT_STAGE}/args.json" "${ARGS}"
  log "restored pristine SFT checkpoint from ${SFT_STAGE}"
fi
[[ -s ${MODEL} && -s ${ARGS} ]] || die "pristine SFT checkpoint missing: ${MODEL}"
[[ $(sha256sum "${MODEL}" | awk '{print $1}') == "${EXPECTED_MODEL_SHA256}" ]] \
  || die "pristine SFT checkpoint hash changed"
if [[ ! -s ${SFT_STAGE}/best_model.pth ]]; then
  mkdir -p "${SFT_STAGE}"
  cp "${MODEL}" "${SFT_STAGE}/best_model.pth"
  cp "${ARGS}" "${SFT_STAGE}/args.json"
  log "staged durable SFT checkpoint copy under ${SFT_STAGE}"
fi

# --- preflight ---------------------------------------------------------------
for path in "${TRAIN}" "${VALID}" "${CONFIG}" "${BASELINE_DP10}"; do
  [[ -r ${path} && -s ${path} ]] \
    || die "unreadable required input ${path} (dataset lists need the ubuntu group; launch via 'sg ubuntu')"
done
[[ $(wc -l < "${TRAIN}") == 5446155 ]] || die "train manifest changed"
[[ $(wc -l < "${VALID}") == 46263 ]] || die "valid manifest changed"
jq -er '
  .awr.hdp_trajectory_augmentation == false and
  .awr.plannerrft_guided_exploration == true and
  .awr.deterministic_first == true and
  .awr.original_dp_first_waypoint_gate_enabled == true and
  .awr.original_dp_first_waypoint_gate_tangent_min_step_m == 0.05 and
  .plannerrft.reference_model_path == "/tmp/t4_v5_original_dp/model/best_model.pth"
' "${CONFIG}" >/dev/null || die "gate-fixed config contract failed"
grep -q "original_dp_first_waypoint_gate_tangent_min_step_m" "${ROOT}/rlvr/awr.py" \
  || die "checkout does not contain the tangent-floor gate fix"
grep -q "respect_source_zero_weights" \
  "${ROOT}/rlvr/autoresearch/tools/build_positive_anchor_replay_overlay.py" \
  || die "checkout does not contain the source-weight-respecting overlay"
[[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ge 8 ]] \
  || die "eight GPUs are required"

# Never start while another AWR trainer owns the GPUs; wait for a healthy
# orphan stage to finish and then resume from whatever it committed.
while pgrep -f 'rlvr\.train_awr' >/dev/null; do
  log "another rlvr.train_awr process is alive; waiting 120 s before resuming"
  sleep 120
done

available=$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')
(( available >= MIN_FREE_CACHE_KIB )) \
  || die "only ${available} KiB free; reserve is ${MIN_FREE_CACHE_KIB} KiB"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=0
export DP_DDP_TIMEOUT_MINUTES=${DP_DDP_TIMEOUT_MINUTES:-180}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Unprotected-right-turn oversampling, on by default for every mine from here
# on.  Set EXTRA_TRAIN_REPEAT=0 to train on the un-augmented list.
RIGHT_TURN_ARTIFACTS=${RIGHT_TURN_ARTIFACTS:-/mnt/nvme/wangbin/Diffusion-Planner-t4-main/artifacts/right_turn_remapped_20260707}
EXTRA_TRAIN_REPEAT=${EXTRA_TRAIN_REPEAT:-10}
EXTRA_TRAIN_LISTS=(
  "${RIGHT_TURN_ARTIFACTS}/path_list_train_unprotected_right_turn_is_skipped_filtered.json"
  "${RIGHT_TURN_ARTIFACTS}/path_list_unprotected_right_turn_xx1_is_skipped_filtered.json"
  "${RIGHT_TURN_ARTIFACTS}/path_list_unprotected_right_turn_xx1_psim_is_skipped_filtered.json"
)
EXTRA_TRAIN_ARGS=()
EXTRA_TRAIN_APPENDED=0
if (( EXTRA_TRAIN_REPEAT > 0 )); then
  for extra_list in "${EXTRA_TRAIN_LISTS[@]}"; do
    [[ -s ${extra_list} ]] || die "missing extra train list: ${extra_list}"
    EXTRA_TRAIN_ARGS+=(--extra_train_set_list "${extra_list}")
    EXTRA_TRAIN_APPENDED=$(( EXTRA_TRAIN_APPENDED
      + $(jq 'length' "${extra_list}") * EXTRA_TRAIN_REPEAT ))
  done
  EXTRA_TRAIN_ARGS+=(--extra_train_set_repeat "${EXTRA_TRAIN_REPEAT}")
fi

mine_cache_shape_complete() {
  local run=$1 cache=$1/replay_buffer total=0 rank rank_dir count
  [[ -d ${cache} ]] || return 1
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${cache}" "${rank}"
    [[ -s ${rank_dir}/manifest.json && -s ${rank_dir}/expert_anchor_manifest.json ]] \
      || return 1
    count=$(jq -er '.scene_count' "${rank_dir}/manifest.json") || return 1
    [[ ${count} -eq ${EXPECTED_RANK_GROUPS} ]] || return 1
    total=$((total + count))
  done
  [[ ${total} -eq ${EXPECTED_PADDED_GROUPS} ]]
}

latest_completed_mine_run() {
  local run
  [[ -d ${MINE_PARENT} ]] || return 0
  while IFS= read -r run; do
    if mine_cache_shape_complete "${run}"; then
      printf '%s\n' "${run}"
      return 0
    fi
  done < <(find "${MINE_PARENT}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
    -print 2>/dev/null | sort -r)
}

# --- stage 1: cycle-1 full-corpus mine ---------------------------------------
MINE_RUN=$(latest_completed_mine_run)
if [[ -z ${MINE_RUN} ]]; then
  mkdir -p "${MINE_PARENT}"
  if [[ -s ${RUN_ROOT}/cycle01_mine.log ]]; then
    mkdir -p "${RUN_ROOT}/interrupted_attempts"
    mv "${RUN_ROOT}/cycle01_mine.log" \
      "${RUN_ROOT}/interrupted_attempts/cycle01_mine_$(date '+%Y%m%d-%H%M%S').log"
  fi
  log "stage 1: mining 5,446,154 scene groups from the pristine SFT checkpoint"
  if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    log "stage 1: oversampling right-turn extras (repeat=${EXTRA_TRAIN_REPEAT}, +${EXTRA_TRAIN_APPENDED} entries)"
  fi
  "${TORCHRUN}" --standalone --nproc_per_node=8 -m rlvr.train_awr \
    --model_path "${MODEL}" --args_path "${ARGS}" \
    --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
    ${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"} \
    --config "${CONFIG}" --output_dir "${MINE_PARENT}" \
    --exp_name plannerrft_full_cycle01_mine \
    --device cuda --seed 3407 --start_epoch 1 --epochs 1 \
    --expert_anchor_weight 0.4 --max_train_scenes 0 --max_valid_scenes 8 \
    --eval_k 1 --eval_sample_steps 10 \
    --train_selector_scenes 128 --full_valid_interval 0 --hard_valid_scenes 0 \
    --rollout_scene_batch_size 192 --scene_batch_size 192 \
    --rollout_scene_load_workers 1 --rollout_prefetch_batches 1 \
    --scene_load_workers 4 --eval_scene_load_workers 1 \
    --eval_scene_batch_size 512 --save_rollouts 0 --no-wandb \
    --no-skip_filtered_scenes > "${RUN_ROOT}/cycle01_mine.log" 2>&1
  MINE_RUN=$(latest_completed_mine_run)
fi
[[ -n ${MINE_RUN} ]] || die "cycle-1 mine did not commit a complete strict cache"
MINE_CACHE=${MINE_RUN}/replay_buffer
log "stage 1 committed: ${MINE_RUN}"

jq -er '
  .awr.original_dp_first_waypoint_gate_enabled == true and
  .awr.original_dp_first_waypoint_gate_tangent_min_step_m == 0.05 and
  .awr.hdp_trajectory_augmentation == false and
  .awr.plannerrft_guided_exploration == true
' "${MINE_RUN}/effective_config.json" >/dev/null \
  || die "cycle-1 mine ran without the gate fix; do not reuse this cache"
jq -er --arg model "${MODEL}" --arg hash "${EXPECTED_MODEL_SHA256}" '
  .model_source == $model and .staged_model_sha256 == $hash and
  .neighbor_future_alignment_offset == 0
' "${MINE_RUN}/provenance.json" >/dev/null \
  || die "cycle-1 mine provenance does not point at the pristine SFT checkpoint"

if [[ ! -s ${MINE_RUN}/decoder_context_attachment.json ]]; then
  log "attaching inline decoder-context metadata"
  "${PYTHON}" -m rlvr.autoresearch.tools.attach_replay_decoder_context \
    --replay-root "${MINE_CACHE}" --expected-world-size 8 \
    >> "${STATUS}" 2>&1
fi
if [[ ! -s ${MINE_CACHE}/ddp_tail_padding_contract.json ]]; then
  jq -n '{version:1,
    contract:"all_source_scenes_once_then_deterministic_prefix_padding",
    source_scene_count:5446154, cached_group_count:5446656,
    ddp_tail_padding_count:502, global_rollout_batch:1536}' \
    > "${MINE_CACHE}/ddp_tail_padding_contract.json"
fi

# --- stage 2: positive-advantage overlay --------------------------------------
if [[ ! -s ${OVERLAY_PARENT}/overlay_manifest.json ]]; then
  [[ ! -e ${OVERLAY_PARENT}/replay_buffer ]] \
    || die "uncommitted overlay exists: ${OVERLAY_PARENT}/replay_buffer"
  log "stage 2: building positive-advantage overlay"
  "${PYTHON}" -m rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
    --source-replay "${MINE_CACHE}" \
    --output-replay "${OVERLAY_PARENT}/replay_buffer" \
    --beta 2 --margin 0.01 \
    --behavior-anchor-weight 0 --unsafe-behavior-anchor-weight 0 \
    > "${RUN_ROOT}/cycle01_overlay.log" 2>&1
fi
jq -er --arg source "$(readlink -f "${MINE_CACHE}")" '
  .source_replay == $source and
  .groups == 5446656 and
  .expert_anchor_sidecar == true and
  .parameters.beta == 2 and .parameters.margin == 0.01 and
  .parameters.behavior_anchor_weight == 0 and
  .parameters.respect_source_zero_weights == true
' "${OVERLAY_PARENT}/overlay_manifest.json" >/dev/null \
  || die "cycle-1 overlay contract differs"
log "stage 2 committed: masked candidates=$(jq -r '.source_zero_masked_candidates' "${OVERLAY_PARENT}/overlay_manifest.json") active targets=$(jq -r '.active_targets' "${OVERLAY_PARENT}/overlay_manifest.json")"

# --- stage 3: cycle-1 replay epochs 2-10 --------------------------------------
latest_completed_replay() {
  [[ -d ${REPLAY_PARENT} ]] || return 0
  find "${REPLAY_PARENT}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
    -exec test -s '{}/final_summary.json' ';' -print 2>/dev/null | sort | tail -n 1
}
REPLAY_RUN=$(latest_completed_replay)
if [[ -z ${REPLAY_RUN} ]]; then
  mkdir -p "${REPLAY_PARENT}"
  if [[ -s ${RUN_ROOT}/cycle01_replay.log ]]; then
    mkdir -p "${RUN_ROOT}/interrupted_attempts"
    mv "${RUN_ROOT}/cycle01_replay.log" \
      "${RUN_ROOT}/interrupted_attempts/cycle01_replay_$(date '+%Y%m%d-%H%M%S').log"
  fi
  log "stage 3: replaying epochs 2-10 with full per-epoch validation"
  "${TORCHRUN}" --standalone --nproc_per_node=8 -m rlvr.train_awr \
    --model_path "${MODEL}" --args_path "${ARGS}" \
    --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
    --config "${CONFIG}" --output_dir "${REPLAY_PARENT}" \
    --exp_name plannerrft_full_cycle01_replay_e02_e10 \
    --device cuda --seed 3407 --start_epoch 2 --epochs 10 \
    --resume_replay_root "${OVERLAY_PARENT}/replay_buffer" \
    --replay_sampling with_replacement \
    --awr_beta 2 --positive_advantage_only --positive_advantage_margin 0.01 \
    --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
    --expert_anchor_weight 0.4 --expert_anchor_active_groups_only \
    --awr_candidate_loss_horizon 40 --drop_all_zero_groups \
    --awr_loss_type plain_mse --diffusion_t_range 0.001 0.2 \
    --trainable_scope output --learning_rate 0.000001 \
    --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy \
    --max_train_scenes 0 --max_valid_scenes 0 \
    --eval_k 1 --eval_sample_steps 10 --train_selector_scenes 65536 \
    --expected_train_selector_sha256 "${EXPECTED_SELECTOR_SHA256}" \
    --expected_valid_sha256 "${EXPECTED_VALID_SHA256}" \
    --full_valid_interval 0 --hard_valid_scenes 0 \
    --scene_batch_size 192 --gradient_accumulation_scenes 192 \
    --scene_load_workers 4 --eval_scene_load_workers 1 \
    --eval_scene_batch_size 512 --save_rollouts 0 \
    --wandb --wandb_project original-dp-awr \
    --wandb_entity advanced-technology-department \
    --wandb_run_name original-dp-awr-gatefix-cycle01-e02-e10 \
    --wandb_mode "${WANDB_MODE:-online}" \
    --wandb_tags original-dp awr plannerrft-guidance conditional-guidance \
      positive-margin001 beta2 expert04 epoch-boundary-alpha005 \
      first-waypoint-gate-tangent-floor full-sequence cycle01 global-e02-e10 \
    --no-skip_filtered_scenes > "${RUN_ROOT}/cycle01_replay.log" 2>&1
  REPLAY_RUN=$(latest_completed_replay)
fi
[[ -n ${REPLAY_RUN} && -s ${REPLAY_RUN}/final_summary.json ]] \
  || die "cycle-1 replay did not complete"
log "stage 3 committed: ${REPLAY_RUN}"

# --- stages 4+: cycles 2-10 under the standard supervisor ---------------------
mkdir -p "${CAMPAIGN_OUT}"
if [[ -x ${HEALTH_MONITOR} ]]; then
  setsid env CAMPAIGN="${CAMPAIGN_OUT}" INTERVAL=60 \
    TARGET_EPOCH="${TARGET_EPOCH}" \
    bash "${HEALTH_MONITOR}" \
    > "${CAMPAIGN_OUT}/health_monitor.launch.log" 2>&1 < /dev/null &
fi
log "handing over to the epoch-${TARGET_EPOCH} supervisor"
exec env \
  CONFIG="${CONFIG}" TRAIN="${TRAIN}" VALID="${VALID}" \
  BASELINE_DP10="${BASELINE_DP10}" \
  CYCLE1_MINE_RUN="${MINE_RUN}" \
  CYCLE1_REPLAY_PARENT="${REPLAY_PARENT}" \
  CYCLE1_FIRST=2 CYCLE1_LAST=10 \
  CYCLE1_SOURCE_CHECKPOINT="${MODEL}" \
  COMPRESSED_CONTEXT_ROOT="${MINE_RUN}/replay_context_zstd_v1" \
  AWR_CANDIDATE_LOSS_HORIZON=40 \
  EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=1 \
  DP_NEIGHBOR_FUTURE_OFFSET=0 \
  TARGET_EPOCH="${TARGET_EPOCH}" \
  OUT="${CAMPAIGN_OUT}" \
  bash "${SUPERVISOR}"
