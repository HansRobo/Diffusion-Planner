#!/usr/bin/env bash
# PlannerRFT/AWR campaign continuing from LAST WEEK's audited-good checkpoint
# (e004_a0p05), which improved reward *and* every safety subscore over the SFT
# baseline.  The from-pristine-SFT restart that replaced it traded safety for
# reward (collisions +15% relative at P(improved)=0.003) and never fixed the
# standstill steering jitter, because a 5 cm tangent floor sat above the 3.9 cm
# measured stop-turn first step and switched the safeguard off exactly where it
# was needed.  This run carries the implied-steer gate, the overlay that
# respects mining-time zero weights, right-turn oversampling, and a hard
# non-regression veto on every cycle commit.  Every stage is idempotent: a committed
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
MODEL=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth
ARGS=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/model_args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_clean_sft.json
BASELINE_DP10=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/deployment_full10/source/scenes.json
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_jitterfix_e100}
MINE_PARENT=${RUN_ROOT}/cycle01_mine
OVERLAY_PARENT=${RUN_ROOT}/cycle01_positive_overlay
REPLAY_PARENT=${RUN_ROOT}/cycle01_replay_e02_e10
CAMPAIGN_OUT=${RUN_ROOT}/campaign_e100
# Durable copy of the pristine SFT, used only to restore the guided-exploration
# reference model after a reboot wipes /tmp.  Staged by the earlier campaign;
# keep pointing at it rather than duplicating 233 MB per run root.
SFT_STAGE=${SFT_STAGE:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_gatefix_clean/sft_checkpoint}
SUPERVISOR=${ROOT}/rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh
HEALTH_MONITOR=${ROOT}/rlvr/autoresearch/monitor_plannerrft_epoch100.sh
# Derived, never hardcoded: right-turn oversampling changes the scene count, and
# a stale literal here fails the cache-shape check hours into a mine.  The
# rollout pads the corpus up to a whole number of global batches (8 ranks x 192).
GLOBAL_ROLLOUT_BATCH=1536
EXPECTED_SOURCE_SCENES=0
EXPECTED_PADDED_GROUPS=0
EXPECTED_RANK_GROUPS=0
DDP_TAIL_PADDING_COUNT=0
TARGET_EPOCH=${TARGET_EPOCH:-100}
MIN_FREE_CACHE_KIB=${MIN_FREE_CACHE_KIB:-6442450944}
LOCK=${RUN_ROOT}/continuation.lock

mkdir -p "${RUN_ROOT}"
exec 9>"${LOCK}"
flock -n 9 || exit 0

STATUS=${RUN_ROOT}/jitterfix_launch.log
log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATUS}" >&2
}
die() {
  log "FATAL: $*"
  exit 1
}

# --- start checkpoint --------------------------------------------------------
# Lives under outputs/, so it survives a reboot on its own.  The guided-
# exploration *reference* model is still the pristine SFT (see CONFIG), which
# does live in /tmp, so verify that separately.
[[ -s ${MODEL} && -s ${ARGS} ]] || die "start checkpoint missing: ${MODEL}"
REFERENCE_MODEL=$(jq -r '.plannerrft.reference_model_path' "${CONFIG}")
if [[ ! -s ${REFERENCE_MODEL} && -s ${SFT_STAGE}/best_model.pth ]]; then
  mkdir -p "$(dirname "${REFERENCE_MODEL}")"
  cp "${SFT_STAGE}/best_model.pth" "${REFERENCE_MODEL}"
  cp "${SFT_STAGE}/args.json" "$(dirname "${REFERENCE_MODEL}")/args.json"
  log "restored the SFT reference model from ${SFT_STAGE}"
fi
[[ -s ${REFERENCE_MODEL} ]] || die "SFT reference model missing: ${REFERENCE_MODEL}"

# --- preflight ---------------------------------------------------------------
for path in "${TRAIN}" "${VALID}" "${CONFIG}" "${BASELINE_DP10}"; do
  [[ -r ${path} && -s ${path} ]] \
    || die "unreadable required input ${path} (dataset lists need the ubuntu group; launch via 'sg ubuntu')"
done
[[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ge 8 ]] \
  || die "eight GPUs are required"

# Never start while another AWR trainer owns the GPUs; wait for a healthy
# orphan stage to finish and then resume from whatever it committed.
while pgrep -f 'rlvr\.train_awr' >/dev/null; do
  log "another rlvr.train_awr process is alive; waiting 120 s before resuming"
  sleep 120
done

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
RIGHT_TURN_ARTIFACTS=${RIGHT_TURN_ARTIFACTS:-/mnt/nvme/wangbin/Diffusion-Planner-hyper-diffusion-planner/artifacts/right_turn_is_skipped_filtered_20260716}
EXTRA_TRAIN_REPEAT=${EXTRA_TRAIN_REPEAT:-10}
# Retention anchor on the deterministic trajectory.  0 (the previous value) left
# 81.7% of scenes contributing nothing to the loss, so the policy drifted freely
# exactly where the rare collision events live; see rlvr/campaign_contract.py.
BEHAVIOR_ANCHOR_WEIGHT=${BEHAVIOR_ANCHOR_WEIGHT:-$(
  "${PYTHON}" -c 'from rlvr.campaign_contract import BEHAVIOR_ANCHOR_WEIGHT as w; print(f"{w:g}")'
)}
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
    extra_count=$(jq 'length' "${extra_list}")
    # A remapped list can legitimately end up empty: the 20260623 x2_dev scenes
    # have no counterpart in the 20260707 corpus.  Skip it — the trainer rejects
    # an empty list, and `-s` cannot tell "[]" from real content.
    if (( extra_count == 0 )); then
      log "extra train list contributes 0 scenes, skipping: ${extra_list##*/}"
      continue
    fi
    EXTRA_TRAIN_ARGS+=(--extra_train_set_list "${extra_list}")
    EXTRA_TRAIN_APPENDED=$(( EXTRA_TRAIN_APPENDED + extra_count * EXTRA_TRAIN_REPEAT ))
  done
  if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    EXTRA_TRAIN_ARGS+=(--extra_train_set_repeat "${EXTRA_TRAIN_REPEAT}")
  fi
fi

EXPECTED_SOURCE_SCENES=$(( $(jq 'length' "${TRAIN}") + EXTRA_TRAIN_APPENDED ))
EXPECTED_PADDED_GROUPS=$((
  (EXPECTED_SOURCE_SCENES + GLOBAL_ROLLOUT_BATCH - 1)
  / GLOBAL_ROLLOUT_BATCH * GLOBAL_ROLLOUT_BATCH ))
EXPECTED_RANK_GROUPS=$(( EXPECTED_PADDED_GROUPS / 8 ))
DDP_TAIL_PADDING_COUNT=$(( EXPECTED_PADDED_GROUPS - EXPECTED_SOURCE_SCENES ))
log "corpus: ${EXPECTED_SOURCE_SCENES} scenes (base + ${EXTRA_TRAIN_APPENDED} oversampled) -> ${EXPECTED_PADDED_GROUPS} groups, ${EXPECTED_RANK_GROUPS}/rank, ${DDP_TAIL_PADDING_COUNT} padded"
export AWR_EXPECTED_SOURCE_SCENES=${EXPECTED_SOURCE_SCENES}
export AWR_EXPECTED_PADDED_GROUPS=${EXPECTED_PADDED_GROUPS}
export AWR_EXPECTED_RANK_GROUPS=${EXPECTED_RANK_GROUPS}

mine_cache_shape_complete() {
  local run=$1 cache=$1/replay_buffer total=0 rank rank_dir count first_count=''
  [[ -d ${cache} ]] || return 1
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${cache}" "${rank}"
    [[ -s ${rank_dir}/manifest.json && -s ${rank_dir}/expert_anchor_manifest.json ]] \
      || return 1
    count=$(jq -er '.scene_count' "${rank_dir}/manifest.json") || return 1
    if [[ -z ${first_count} ]]; then
      first_count=${count}
    elif [[ ${count} -ne ${first_count} ]]; then
      return 1
    fi
    total=$((total + count))
  done
  [[ ${total} -gt 0 ]]
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
  # The multi-TiB reserve only matters when a full-corpus cache still has to be
  # written.  Checking it unconditionally at startup would abort every *resume*
  # once the cache itself has consumed the disk — which is exactly the state the
  # campaign is in right after stage 1 commits.
  available=$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')
  (( available >= MIN_FREE_CACHE_KIB )) \
    || die "only ${available} KiB free; a fresh cycle-1 mine reserves ${MIN_FREE_CACHE_KIB} KiB"
  mkdir -p "${MINE_PARENT}"
  if [[ -s ${RUN_ROOT}/cycle01_mine.log ]]; then
    mkdir -p "${RUN_ROOT}/interrupted_attempts"
    mv "${RUN_ROOT}/cycle01_mine.log" \
      "${RUN_ROOT}/interrupted_attempts/cycle01_mine_$(date '+%Y%m%d-%H%M%S').log"
  fi
  log "stage 1: mining ${EXPECTED_SOURCE_SCENES} scene groups from last week's audited-good checkpoint"
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


if [[ ! -s ${MINE_RUN}/decoder_context_attachment.json ]]; then
  log "attaching inline decoder-context metadata"
  "${PYTHON}" -m rlvr.autoresearch.tools.attach_replay_decoder_context \
    --replay-root "${MINE_CACHE}" --expected-world-size 8 \
    >> "${STATUS}" 2>&1
fi
if [[ ! -s ${MINE_CACHE}/ddp_tail_padding_contract.json ]]; then
  jq -n \
    --argjson src "${EXPECTED_SOURCE_SCENES}" \
    --argjson groups "${EXPECTED_PADDED_GROUPS}" \
    --argjson pad "${DDP_TAIL_PADDING_COUNT}" \
    --argjson batch "${GLOBAL_ROLLOUT_BATCH}" \
    '{version:1,
    contract:"all_source_scenes_once_then_deterministic_prefix_padding",
    source_scene_count:$src, cached_group_count:$groups,
    ddp_tail_padding_count:$pad, global_rollout_batch:$batch}' \
    > "${MINE_CACHE}/ddp_tail_padding_contract.json"
fi

# --- stage 2: positive-advantage overlay --------------------------------------
# beta=1, not 2.  The paired beta probe (same cache, same seed, same updates,
# 65,536-scene fixed selector) ranked beta=2 WORST of {0.5, 1, 2} on both reward
# delta (-1.94e-4 vs -1.14e-4) and kinematic violations (+9.2e-5 vs 0), and both
# earlier campaigns' probes selected 1.  Cycle 1 nonetheless hardcoded 2 and runs
# before the probe — i.e. the cycle with the largest policy movement used the
# weighting the probe then rejected.  beta=1 also matches the clean public NAVSIM
# release per docs/awr_zero_risk_improvement_audit_20260718.md.
if [[ ! -s ${OVERLAY_PARENT}/overlay_manifest.json ]]; then
  [[ ! -e ${OVERLAY_PARENT}/replay_buffer ]] \
    || die "uncommitted overlay exists: ${OVERLAY_PARENT}/replay_buffer"
  log "stage 2: building positive-advantage overlay"
  "${PYTHON}" -m rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
    --source-replay "${MINE_CACHE}" \
    --output-replay "${OVERLAY_PARENT}/replay_buffer" \
    --beta 1 --margin 0.01 \
    --behavior-anchor-weight "${BEHAVIOR_ANCHOR_WEIGHT}" \
    --unsafe-behavior-anchor-weight "${BEHAVIOR_ANCHOR_WEIGHT}" \
    > "${RUN_ROOT}/cycle01_overlay.log" 2>&1
fi
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
    --awr_beta 1 --positive_advantage_only --positive_advantage_margin 0.01 \
    --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
    --expert_anchor_weight 0.4 --expert_anchor_active_groups_only \
    --awr_candidate_loss_horizon 40 --drop_all_zero_groups \
    --awr_loss_type plain_mse --diffusion_t_range 0.001 0.2 \
    --trainable_scope output --learning_rate 0.000001 \
    --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy \
    --max_train_scenes 0 --max_valid_scenes 0 \
    --eval_k 1 --eval_sample_steps 10 --train_selector_scenes 65536 \
    --full_valid_interval 0 --hard_valid_scenes 0 \
    --scene_batch_size 192 --gradient_accumulation_scenes 192 \
    --scene_load_workers 4 --eval_scene_load_workers 1 \
    --eval_scene_batch_size 512 --save_rollouts 0 \
    --wandb --wandb_project original-dp-awr \
    --wandb_entity advanced-technology-department \
    --wandb_run_name original-dp-awr-gatefix-cycle01-e02-e10 \
    --wandb_mode "${WANDB_MODE:-online}" \
    --wandb_tags original-dp awr plannerrft-guidance conditional-guidance \
      positive-margin001 beta1 expert04 epoch-boundary-alpha005 \
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
  AWR_EXPECTED_SOURCE_SCENES="${EXPECTED_SOURCE_SCENES}" \
  AWR_EXPECTED_PADDED_GROUPS="${EXPECTED_PADDED_GROUPS}" \
  AWR_EXPECTED_RANK_GROUPS="${EXPECTED_RANK_GROUPS}" \
  RIGHT_TURN_ARTIFACTS="${RIGHT_TURN_ARTIFACTS}" \
  EXTRA_TRAIN_REPEAT="${EXTRA_TRAIN_REPEAT}" \
  EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=1 \
  DP_NEIGHBOR_FUTURE_OFFSET=0 \
  TARGET_EPOCH="${TARGET_EPOCH}" \
  OUT="${CAMPAIGN_OUT}" \
  bash "${SUPERVISOR}"
