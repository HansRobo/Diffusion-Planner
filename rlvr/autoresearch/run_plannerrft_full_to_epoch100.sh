#!/usr/bin/env bash
# Continue the formal Original-DP PlannerRFT/AWR experiment from the already
# running Cycle 1 through a configurable global epoch target. One process owns
# every later cycle;
# every cycle refreshes trajectories with the latest selected checkpoint,
# reuses only policy-independent Cycle-1 artifacts, replays epochs N+1..N+9,
# and retains the starting incumbent when no train-selector improvement exists.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
TORCHRUN=${ROOT}/.venv/bin/torchrun
TRAIN=${TRAIN:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}
VALID=${VALID:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}
CONFIG=${CONFIG:-${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json}
TARGET_EPOCH=${TARGET_EPOCH:-100}
[[ ${TARGET_EPOCH} =~ ^[1-9][0-9]*$ ]] \
  || { echo "TARGET_EPOCH must be a positive integer" >&2; exit 2; }
(( TARGET_EPOCH >= 20 && TARGET_EPOCH % 10 == 0 )) \
  || { echo "TARGET_EPOCH must be a multiple of 10 and at least 20" >&2; exit 2; }
TOTAL_CYCLES=$((TARGET_EPOCH / 10))

CYCLE1_MINE_RUN=${CYCLE1_MINE_RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine}
CYCLE1_REPLAY_PARENT=${CYCLE1_REPLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_replay_e02_e10}
CYCLE1_FIRST=${CYCLE1_FIRST:-2}
CYCLE1_LAST=${CYCLE1_LAST:-10}
CYCLE1_SOURCE_CHECKPOINT=${CYCLE1_SOURCE_CHECKPOINT:-}
SOURCE_CACHE=${CYCLE1_MINE_RUN}/replay_buffer
COMPRESSED_CONTEXT_ROOT=${COMPRESSED_CONTEXT_ROOT:-${CYCLE1_MINE_RUN}/replay_context_zstd_v1}
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycles02_to${TOTAL_CYCLES}_e${TARGET_EPOCH}}
STATE=${OUT}/cycle_state.jsonl
STATUS=${OUT}/status.log
LOCK=${OUT}/supervisor.lock
BETA_SENSITIVITY_SCRIPT=${ROOT}/rlvr/autoresearch/run_plannerrft_replay_beta_sensitivity.sh
BETA_SENSITIVITY_OUT=${OUT}/beta_sensitivity
BETA_SELECTION=${BETA_SENSITIVITY_OUT}/selection.json
CANDIDATE_SENSITIVITY_SCRIPT=${ROOT}/rlvr/autoresearch/run_plannerrft_candidate_sensitivity.sh
CANDIDATE_SENSITIVITY_OUT=${OUT}/candidate_sensitivity
CANDIDATE_SELECTION=${CANDIDATE_SENSITIVITY_OUT}/selection.json
SOLVER_STEP_SCRIPT=${ROOT}/rlvr/autoresearch/run_plannerrft_solver_step_sensitivity.sh
SOLVER_STEP_OUT=${OUT}/solver_step_sensitivity
SOLVER_STEP_SELECTION=${SOLVER_STEP_OUT}/paired_5_vs_10.json
LINE_SEARCH_SCRIPT=${ROOT}/rlvr/autoresearch/run_conditional_train_selector_line_search.sh
EVAL_LOADER_BENCHMARK_SCRIPT=${ROOT}/rlvr/autoresearch/benchmark_eval_scene_load_workers.sh
EVAL_LOADER_BENCHMARK_OUT=${OUT}/eval_loader_benchmark
# Default to the audited/probe-selected beta=1 (the probe ranked 2 worst on both
# reward and kinematic violations). After cycle 1 completes, a paired bounded
# same-cache probe sets this value for every later cycle.
REPLAY_BETA=${REPLAY_BETA:-1.0}
# Retention anchor on the deterministic trajectory.  0 left 81.7% of scenes
# contributing nothing to the loss, so the policy drifted freely exactly where
# the rare collision events live; see rlvr/campaign_contract.py.
# The validated run used 0.0.  My 0.25 was introduced on a diagnosis that this
# comparison refutes: that run degraded collision by +14% across its cycle-5
# epochs with anchor=0.0 and still improved at cycle level, so the anchor was
# never the cause.  Back to 0.0; it can be tested later as a single variable.
BEHAVIOR_ANCHOR_WEIGHT=${BEHAVIOR_ANCHOR_WEIGHT:-0.0}
AWR_CANDIDATE_LOSS_HORIZON=${AWR_CANDIDATE_LOSS_HORIZON:-40}
EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=${EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY:-1}
# 80.55% of mined groups have no candidate that beats the deployed deterministic
# output, so AWR trains nothing on them and replay epochs are nearly inert
# (+0.00024 selection reward over 9 epochs in the last full cycle).  The logged
# human trajectory for those same scenes is already cached by the mine, and on
# the expert_safe ones it scores above the deployed output by >0.01 in 42.52% of
# cases -- mean +0.029, about twice the within-group candidate headroom.  Setting
# a margin here makes the overlay keep expert_safe only where the human actually
# wins, which turns those groups into behaviour-cloning targets with no extra
# mining and lifts trainable coverage 19.45% -> 52.31%.  Only 0.46% of groups are
# dead, safe, and expert-worse, so the margin costs almost nothing to respect.
# It also hands the gating entirely to the overlay, hence the unrestricted anchor.
# Empty disables it, and that is the default: this cycle must isolate the reward
# fixes.  The overlay rebuilds from the mine cache in minutes, so enabling it
# later costs no mine time.
EXPERT_IMPROVES_MARGIN=${EXPERT_IMPROVES_MARGIN:-}
OVERLAY_EXPERT_ARGS=()
if [[ -n ${EXPERT_IMPROVES_MARGIN} ]]; then
  OVERLAY_EXPERT_ARGS+=(--expert-improves-margin "${EXPERT_IMPROVES_MARGIN}")
  EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=0
fi
# run_plannerrft_jitterfix_to_epoch100.sh, which owns cycle 1 and hands over to
# this script, does NOT carry this knob: it was mid-mine when the knob landed and
# editing a running bash script corrupts its read offset.  So cycle 1 is always
# the un-gated baseline, which is what isolates the reward fixes anyway.  Say so
# rather than letting a set margin look like it applied to the whole campaign.
# Aligned to plannerrft_prefix_full_cycles02_to10_e100, the only run that
# improved across cycles for 41 epochs (0.93233 -> 0.93297, no cycle
# regressing).  It held these four values for its entire length.
PLANNERRFT_REFERENCE_MODE=${PLANNERRFT_REFERENCE_MODE:-stochastic}
PLANNERRFT_REFERENCE_NOISE_SCALE=${PLANNERRFT_REFERENCE_NOISE_SCALE:-0.5}
PLANNERRFT_LONGITUDINAL_MODE=candidate_stretch
PLANNERRFT_LAMBDA_LAT=${PLANNERRFT_LAMBDA_LAT:-1.5}
ROLLOUT_SAMPLE_STEPS=${ROLLOUT_SAMPLE_STEPS:-10}
# Row order and decoded float32 bytes are invariant to this thread count.
# On the production 192-scene row shape, 16 workers measured 0.161 s versus
# 0.254 s at 4 workers; 8 ranks x 16 remains below this node's 224 logical CPUs.
REPLAY_SCENE_LOAD_WORKERS=${REPLAY_SCENE_LOAD_WORKERS:-16}
[[ ${REPLAY_SCENE_LOAD_WORKERS} == 4 || ${REPLAY_SCENE_LOAD_WORKERS} == 8 || ${REPLAY_SCENE_LOAD_WORKERS} == 16 ]] \
  || { echo "invalid REPLAY_SCENE_LOAD_WORKERS=${REPLAY_SCENE_LOAD_WORKERS}" >&2; exit 2; }
EVAL_SCENE_LOAD_WORKERS=1

# Unprotected-right-turn scenes are oversampled into every mine stage by
# default.  Appending them changes the canonical per-rank scene order, which
# the shared-encoding prefix contract in train_awr.py rejects, so a mine that
# carries extras must build its own cache instead of reusing SOURCE_CACHE.
# Set EXTRA_TRAIN_REPEAT=0 to fall back to the un-augmented training set.
# Oversampling is applied as replay weight, not repeated mining: mine each scene
# once and multiply its overlay weights by EXTRA_TRAIN_WEIGHT.  See the
# entrypoint for the coverage trade-off this buys the mine time with.
RIGHT_TURN_ARTIFACTS=${RIGHT_TURN_ARTIFACTS:-/mnt/nvme/wangbin/Diffusion-Planner-hyper-diffusion-planner/artifacts/right_turn_is_skipped_filtered_20260716}
EXTRA_TRAIN_REPEAT=${EXTRA_TRAIN_REPEAT:-1}
EXTRA_TRAIN_WEIGHT=${EXTRA_TRAIN_WEIGHT:-10}
EXTRA_TRAIN_LISTS=(
  "${RIGHT_TURN_ARTIFACTS}/path_list_train_unprotected_right_turn_is_skipped_filtered.json"
  "${RIGHT_TURN_ARTIFACTS}/path_list_unprotected_right_turn_xx1_is_skipped_filtered.json"
  "${RIGHT_TURN_ARTIFACTS}/path_list_unprotected_right_turn_xx1_psim_is_skipped_filtered.json"
)
EXTRA_TRAIN_ARGS=()
OVERSAMPLE_ARGS=()
if (( EXTRA_TRAIN_REPEAT > 0 )); then
  for extra_list in "${EXTRA_TRAIN_LISTS[@]}"; do
    [[ -s ${extra_list} ]] || { echo "missing extra train list: ${extra_list}" >&2; exit 2; }
    # Skip legitimately-empty remapped lists; the trainer rejects them and `-s`
    # cannot distinguish "[]" from real content.  Must match the entrypoint's
    # filtering exactly, or the shared-cache provenance check would disagree.
    (( $(jq 'length' "${extra_list}") > 0 )) || continue
    EXTRA_TRAIN_ARGS+=(--extra_train_set_list "${extra_list}")
    OVERSAMPLE_ARGS+=(--oversample-list "${extra_list}")
  done
  if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    EXTRA_TRAIN_ARGS+=(--extra_train_set_repeat "${EXTRA_TRAIN_REPEAT}")
    OVERSAMPLE_ARGS+=(--oversample-weight "${EXTRA_TRAIN_WEIGHT}")
  fi
fi

# The entrypoint derives these from the actual corpus (base list + right-turn
# oversampling) and exports them; fall back to the un-augmented literals so a
# standalone invocation still validates.
BASELINE_DP10=${BASELINE_DP10:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/deployment_full10/e004_a0p05/scenes.json}
# Keep the same fail-closed reserve as the formal Cycle-1 launcher.  Later
# refreshes symlink the multi-terabyte policy-independent encoding/context,
# but a stale contract or future schema change must never be allowed to fill
# the local NVMe before the strict writer can publish its manifests.
MIN_FREE_CACHE_KIB=${MIN_FREE_CACHE_KIB:-6442450944}

mkdir -p "${OUT}"
exec 9>"${LOCK}"
flock -n 9 || {
  echo "another campaign supervisor owns ${LOCK}" >&2
  exit 1
}
printf '%s\n' "$$" > "${OUT}/supervisor.pid"

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CONFIG
NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-1}
[[ ${NEIGHBOR_FUTURE_OFFSET} =~ ^0$|^1$ ]] \
  || { echo "unsupported DP_NEIGHBOR_FUTURE_OFFSET=${NEIGHBOR_FUTURE_OFFSET}" >&2; exit 2; }
export DP_NEIGHBOR_FUTURE_OFFSET=${NEIGHBOR_FUTURE_OFFSET}
export DP_DDP_TIMEOUT_MINUTES=${DP_DDP_TIMEOUT_MINUTES:-180}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATUS}" >&2
}

die() {
  log "FATAL: $*"
  exit 1
}

if [[ -n ${EXPERT_IMPROVES_MARGIN} ]]; then
  log "expert-improves gate ON at margin ${EXPERT_IMPROVES_MARGIN}: overlays keep" \
      "expert_safe only where the logged human beats the deterministic output," \
      "and the trainer anchors on all safe groups, not just AWR-active ones." \
      "Cycle 1 was built without it, so compare cycle >=2 against cycle 1."
fi

require_nvme_reserve() {
  local phase=$1 available
  available=$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')
  [[ ${available} =~ ^[0-9]+$ ]] \
    || die "${phase}: could not read available NVMe capacity"
  (( available >= MIN_FREE_CACHE_KIB )) \
    || die "${phase}: only ${available} KiB free; required reserve is ${MIN_FREE_CACHE_KIB} KiB"
  log "${phase}: NVMe reserve check passed (${available} KiB free)"
}

latest_completed_run() {
  local parent=$1
  if [[ ! -d ${parent} ]]; then
    return 0
  fi
  find "${parent}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
    -exec test -s '{}/final_summary.json' ';' -print 2>/dev/null \
    | sort | tail -n 1
}

mine_cache_shape_complete() {
  # A mine stage produces the replay cache before the trainer writes its
  # run-level summaries.  The cache itself is the durable, policy-independent
  # input to the next replay stage, so a complete strict cache must survive a
  # late SIGTERM/SIGABRT during summary finalization.
  local run=$1 cache=${run}/replay_buffer
  [[ -d ${cache} ]] || return 1
  local total=0 rank rank_dir manifest expert count first_count=''
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${cache}" "${rank}"
    manifest=${rank_dir}/manifest.json
    expert=${rank_dir}/expert_anchor_manifest.json
    [[ -s ${manifest} && -s ${expert} ]] || return 1
    if ! count=$(jq -er '.scene_count' "${manifest}"); then
      return 1
    fi
    # Completeness is derived from the artifact itself: every rank must carry the
    # same scene_count and the arrays the loader dereferences.  No external
    # constant is involved, so changing the corpus (e.g. right-turn
    # oversampling) cannot invalidate a perfectly good cache — the failure mode
    # that stalled this campaign repeatedly.
    jq -e '
      .decoder_context_cached == true and
      (.arrays.scene_encoding | type == "string") and
      (.arrays.context_ego_current_state | type == "string") and
      (.arrays.context_neighbor_agents_past | type == "string") and
      (.arrays.context_neighbor_agents_future | type == "string")
    ' "${manifest}" >/dev/null || return 1
    jq -e '.scene_count > 0' "${expert}" >/dev/null || return 1
    if [[ -z ${first_count} ]]; then
      first_count=${count}
    elif [[ ${count} -ne ${first_count} ]]; then
      return 1   # ranks disagree -> partial cache
    fi
    total=$((total + count))
  done
  [[ ${total} -gt 0 ]]
}

latest_completed_mine_run() {
  # Unlike replay runs, mine runs are allowed to be committed by a strict
  # cache-only marker.  This makes a completed cache recoverable even when a
  # process dies after the last rank manifest and before final_summary.json.
  local parent=$1 run
  [[ -d ${parent} ]] || return 0
  while IFS= read -r run; do
    if mine_cache_shape_complete "${run}"; then
      printf '%s\n' "${run}"
      return 0
    fi
  done < <(
    find "${parent}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
      -print 2>/dev/null | sort -r
  )
}

reusable_mine_run_for_checkpoint() {
  # Return an already-committed mine cache that was rolled from exactly this
  # policy, or nothing.  Matching is on the checkpoint's content hash recorded in
  # the cache's own provenance, so a renamed or copied checkpoint still matches
  # and a genuinely different policy never does.
  local checkpoint=$1 want run provenance
  [[ -s ${checkpoint} ]] || return 0
  want=$(sha256sum "${checkpoint}" | awk '{print $1}')
  while IFS= read -r run; do
    provenance=${run}/provenance.json
    [[ -s ${provenance} ]] || continue
    [[ $(jq -r '.staged_model_sha256 // empty' "${provenance}") == "${want}" ]] \
      || continue
    mine_cache_shape_complete "${run}" || continue
    printf '%s\n' "${run}"
    return 0
  done < <(
    find "${CYCLE1_MINE_RUN%/*}" "${OUT}" -mindepth 1 -maxdepth 3 -type d \
      -name '20*_plannerrft_full_cycle*_mine*' -print 2>/dev/null | sort -r
  )
}

run_from_log() {
  local path=$1
  sed -n 's/^Run directory: //p' "${path}" | tail -n 1
}

archive_interrupted_stage_log() {
  # A stage log without a strict final_summary is not a committed result.  It
  # may be left behind by a node/torchrun interruption, so preserve it for
  # diagnosis and allow the deterministic stage to restart from its immutable
  # cycle input.  Repeated algorithm/data failures are still bounded by the
  # outer autonomous owner and fail closed after three no-progress attempts.
  local path=$1 label=$2 archive stamp
  [[ -e ${path} ]] || return 0
  stamp=$(date '+%Y%m%d-%H%M%S')
  archive=${OUT}/interrupted_attempts
  mkdir -p "${archive}"
  mv "${path}" "${archive}/${label}_${stamp}.log"
  log "Archived interrupted ${label} log; restarting from committed cycle input"
}

checkpoint_sha() {
  sha256sum "$1" | awk '{print $1}'
}

validate_cycle1_source_cache() {
  [[ -d ${SOURCE_CACHE} ]] || die "missing Cycle-1 source cache ${SOURCE_CACHE}"
  [[ -s ${CYCLE1_MINE_RUN}/decoder_context_attachment.json ]] \
    || die "Cycle-1 source has no committed decoder-context attachment"
  [[ -s ${CYCLE1_MINE_RUN}/provenance.json ]] || die "Cycle-1 source has no provenance"

  local total=0 rank rank_dir manifest expert count
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${SOURCE_CACHE}" "${rank}"
    manifest=${rank_dir}/manifest.json
    expert=${rank_dir}/expert_anchor_manifest.json
    [[ -s ${manifest} && -s ${expert} ]] \
      || die "Cycle-1 source rank ${rank} has no strict replay/expert manifest"
    count=$(jq -er '.scene_count' "${manifest}")
    total=$((total + count))
  done
}

validate_compressed_context() {
  local rank rank_dir manifest
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${COMPRESSED_CONTEXT_ROOT}" "${rank}"
    manifest=${rank_dir}/manifest.json
    [[ -s ${manifest} && -s ${rank_dir}/context_rows.zst && \
       -s ${rank_dir}/context_row_offsets.npy ]] || return 1
    jq -e \
      --argjson rank "${rank}" \
      '
      .schema == "awr_replay_context_zstd_rows_v1" and
      .rank == $rank and .scene_count > 0 and
      .codec == "zstd" and .frame_checksum == true and
      .one_frame_per_scene == true and
      .raw_row_bytes > 0 and .compressed_size_bytes > 0 and
      (.expected_paths_sha256 | length) == 64 and
      (.fields | map(.name) | sort) ==
        (["ego_current_state", "expert_rewards", "expert_safe",
          "expert_trajectories", "neighbor_agents_future",
          "neighbor_agents_past", "scene_encoding"] | sort)
    ' "${manifest}" >/dev/null || return 1
  done
}

ensure_compressed_context() {
  if validate_compressed_context; then
    log "Lossless compressed replay context already complete: ${COMPRESSED_CONTEXT_ROOT}"
    return
  fi
  require_nvme_reserve "pre-compressed-context build"
  mkdir -p "${COMPRESSED_CONTEXT_ROOT}/logs"
  log "Building eight lossless row-addressable replay-context sidecars"
  local rank pid failed=0
  local -a pids=()
  for rank in $(seq 0 7); do
    ionice -c2 -n7 nice -n 10 "${PYTHON}" \
      -m rlvr.autoresearch.tools.build_compressed_replay_context \
      --source-replay "${SOURCE_CACHE}" \
      --output-root "${COMPRESSED_CONTEXT_ROOT}" \
      --rank "${rank}" --compression-level 1 --progress-interval 10000 \
      > "${COMPRESSED_CONTEXT_ROOT}/logs/rank_$(printf '%04d' "${rank}").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  [[ ${failed} -eq 0 ]] || die "lossless compressed-context build failed"
  validate_compressed_context \
    || die "lossless compressed-context manifests are incomplete"
  log "Lossless compressed replay context complete: ${COMPRESSED_CONTEXT_ROOT}"
}

validate_mine_run() {
  local run=$1 cycle=$2 epoch=$3
  local cache=${run}/replay_buffer
  local mine_summary_complete=1
  if [[ ! -s ${run}/final_summary.json || \
        ! -s ${run}/train_epoch_$(printf '%03d' "${epoch}")/summary.json ]]; then
    mine_summary_complete=0
    mine_cache_shape_complete "${run}" \
      || die "Cycle ${cycle} mine is missing both summaries and a complete strict cache"
    log "Cycle ${cycle}: accepting complete cache-only mine artifact; run summaries were interrupted after cache close"
  fi

  # These three contracts assert that a *freshly rolled* later-cycle mine
  # symlinked cycle 1's encoding rather than copying 3 TB.  When the cache being
  # validated IS cycle 1's — the policy was unchanged so we reused it outright —
  # the question is vacuous and the contract files do not exist.
  if [[ $(readlink -f "${run}") == $(readlink -f "${CYCLE1_MINE_RUN}") ]]; then
    log "Cycle ${cycle}: validating cycle-1's own cache; encoding-reuse contracts do not apply"
  elif [[ ${cycle} -ge 2 || ${CYCLE1_FIRST} -ge 3 ]]; then
    jq -e --arg root "${SOURCE_CACHE}" '
      .enabled == true and .root == $root and .order_epoch == 1
    ' "${run}/shared_replay_encoding_contract.json" >/dev/null \
      || die "Cycle ${cycle} did not reuse the exact Cycle-1 encoding"
    jq -e --arg root "${SOURCE_CACHE}" '
      .enabled == true and .root == $root and .order_epoch == 1 and
      .decoder_context_schema == "canonical_loader_aligned_v1"
    ' "${run}/shared_replay_context_contract.json" >/dev/null \
      || die "Cycle ${cycle} did not reuse the exact Cycle-1 decoder context"
    jq -e --arg root "${SOURCE_CACHE}" '
      .enabled == true and .root == $root and .order_epoch == 1 and
      .expert_anchor_schema == "logged_ego_future_hard_gate_v1"
    ' "${run}/shared_replay_expert_anchor_contract.json" >/dev/null \
      || die "Cycle ${cycle} did not reuse the exact Cycle-1 expert sidecar"
  fi

  local total=0 rank rank_dir manifest expert count encoding
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${cache}" "${rank}"
    manifest=${rank_dir}/manifest.json
    expert=${rank_dir}/expert_anchor_manifest.json
    [[ -s ${manifest} && -s ${expert} ]] \
      || die "Cycle ${cycle} rank ${rank} has no strict manifests"
    count=$(jq -er '.scene_count' "${manifest}")
    # Same reasoning as the contracts above: when this *is* cycle 1's cache the
    # arrays are the originals, not symlinks into it.
    if [[ ${cycle} -ge 2 && $(readlink -f "${run}") != $(readlink -f "${CYCLE1_MINE_RUN}") ]]; then
      encoding=${rank_dir}/$(jq -er '.arrays.scene_encoding' "${manifest}")
      [[ -L ${encoding} ]] || die "Cycle ${cycle} rank ${rank} copied the 3-TB encoding"
      [[ $(readlink -f "${encoding}") == "$(readlink -f "${SOURCE_CACHE}/rank_$(printf '%04d' "${rank}")/scene_encoding.npy")" ]] \
        || die "Cycle ${cycle} rank ${rank} encoding symlink points elsewhere"
      local key context_path
      for key in ego_current_state neighbor_agents_past neighbor_agents_future; do
        context_path=${rank_dir}/context_${key}.npy
        [[ -L ${context_path} ]] \
          || die "Cycle ${cycle} rank ${rank} copied context_${key}"
        [[ $(readlink -f "${context_path}") == "$(readlink -f "${SOURCE_CACHE}/rank_$(printf '%04d' "${rank}")/context_${key}.npy")" ]] \
          || die "Cycle ${cycle} rank ${rank} context_${key} points elsewhere"
      done
    fi
    total=$((total + count))
  done
  if [[ ${mine_summary_complete} -eq 0 && ! -s ${run}/mine_cache_commit.json ]]; then
    local marker=${run}/mine_cache_commit.json
    local marker_tmp=${marker}.tmp.$$
    jq -n \
      --arg timestamp "$(date --iso-8601=seconds)" \
      --arg cycle "${cycle}" \
      --arg epoch "${epoch}" \
      --arg cache "$(readlink -f "${cache}")" \
      --argjson groups "${total:-0}" \
      '{schema:"plannerrft_mine_cache_commit_v1", status:"cache_complete_summary_interrupted",
        timestamp:$timestamp, cycle:($cycle|tonumber), start_epoch:($epoch|tonumber),
        replay_buffer:$cache, padded_groups:$groups, strict_manifests:true}' \
      > "${marker_tmp}"
    mv "${marker_tmp}" "${marker}"
  fi
  printf '%s\n' "${cache}"
}

build_or_validate_overlay() {
  local cycle=$1 source=$2 output_parent=$3
  local output=${output_parent}/replay_buffer manifest=${output_parent}/overlay_manifest.json
  local geometry=${output_parent}/target_geometry.json
  if [[ ! -s ${manifest} ]]; then
    [[ ! -e ${output} ]] || die "Cycle ${cycle} has an uncommitted overlay ${output}"
    "${PYTHON}" -m rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
      --source-replay "${source}" --output-replay "${output}" \
      --beta "${REPLAY_BETA}" --margin 0.01 \
      --behavior-anchor-weight "${BEHAVIOR_ANCHOR_WEIGHT}" \
      --unsafe-behavior-anchor-weight "${BEHAVIOR_ANCHOR_WEIGHT}" \
      ${OVERSAMPLE_ARGS[@]+"${OVERSAMPLE_ARGS[@]}"} \
      ${OVERLAY_EXPERT_ARGS[@]+"${OVERLAY_EXPERT_ARGS[@]}"} \
      > "${output_parent}.log" 2>&1
  fi
  if [[ ! -s ${geometry} ]]; then
    "${PYTHON}" -m rlvr.autoresearch.tools.analyze_replay_target_geometry \
      --replay-root "${output}" --output "${geometry}" \
      --expert-weight 0.4 \
      > "${output_parent}/target_geometry.log" 2>&1
  fi
  log "Cycle ${cycle}: target audit active=$(jq -r '.active_group_fraction' "${geometry}") reward_gain=$(jq -r '.active_candidate_target_reward_gain' "${geometry}") path_delta=$(jq -r '.active_candidate_path_length_delta_m' "${geometry}")"
  printf '%s\n' "${output}"
}

preflight_replay_source() {
  local cycle=$1 replay=$2
  local replay_config
  replay_config=$(dirname "${replay}")/effective_config.json
  [[ -s ${replay_config} ]] \
    || die "Cycle ${cycle} replay source has no effective config"

  # Open the exact reader that train_awr will use before paying for model
  # startup and two full distributed baseline evaluations.  This validates the
  # real scene-path bytes, overlay attachment hash, compressed-context hash,
  # shapes and memmap visibility on all ranks.
  observed_rank_groups=$(jq -er '.scene_count' "${replay}/rank_0000/manifest.json")
  "${PYTHON}" - "${replay}" "${COMPRESSED_CONTEXT_ROOT}" "${observed_rank_groups}" <<'PY'
import hashlib
import json
import pathlib
import sys

from rlvr.train_awr import _DiskReplayReader

replay = pathlib.Path(sys.argv[1]).resolve()
compressed = pathlib.Path(sys.argv[2]).resolve()
for rank in range(8):
    rank_name = f"rank_{rank:04d}"
    rank_dir = replay / rank_name
    manifest = json.loads((rank_dir / "manifest.json").read_text())
    paths_path = rank_dir / manifest.get("paths", "scene_paths.jsonl")
    paths_sha256 = hashlib.sha256(paths_path.read_bytes()).hexdigest()
    attached_sha256 = str(
        manifest.get("decoder_context_attachment", {}).get(
            "expected_paths_sha256", ""
        )
    )
    compressed_manifest = json.loads(
        (compressed / rank_name / "manifest.json").read_text()
    )
    compressed_sha256 = str(compressed_manifest["expected_paths_sha256"])
    if not paths_sha256 == attached_sha256 == compressed_sha256:
        raise RuntimeError(
            f"{rank_name}: replay/attachment/compressed scene order differs: "
            f"{paths_sha256} / {attached_sha256} / {compressed_sha256}"
        )
    reader = _DiskReplayReader(
        replay,
        rank,
        positive_advantage_only=True,
        deterministic_first=True,
        compressed_context_root=compressed,
        compressed_context_workers=1,
    )
    if reader.scene_count != int(sys.argv[3]) or reader.group_size != 10:
        raise RuntimeError(
            f"{rank_name}: unexpected replay shape "
            f"{reader.scene_count} x {reader.group_size}"
        )
    assert reader.compressed_context is not None
    reader.compressed_context.close()
PY
  log "Cycle ${cycle}: 8/8 replay/compressed-context readers passed preflight"
}

validate_replay_run() {
  local run=$1 cycle=$2 first=$3 last=$4 source_checkpoint=$5 replay_root=${6:-}
  [[ -s ${run}/final_summary.json && -s ${run}/best_train.pth ]] \
    || die "Cycle ${cycle} replay lacks final summary/best_train"
  local actual_sha reported_sha source_sha staged_sha best start delta
  actual_sha=$(checkpoint_sha "${run}/best_train.pth")
  reported_sha=$(jq -er '.best_train_checkpoint_sha256' "${run}/final_summary.json")
  source_sha=$(checkpoint_sha "${source_checkpoint}")
  staged_sha=$(jq -er '.staged_model_sha256' "${run}/provenance.json")
  [[ ${source_sha} == "${staged_sha}" ]] \
    || die "Cycle ${cycle} replay did not start from prior selected checkpoint"
  "${PYTHON}" - "${run}/final_summary.json" "${source_checkpoint}" \
    "${run}/best_train.pth" <<'PY'
import json
import pathlib
import sys

import torch

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
if bool(summary["best_train_is_starting_incumbent"]):
    source = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    selected = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
    source_state = source.get("ema_state_dict", source.get("model", source))
    for name in ("model", "ema_state_dict"):
        target = selected[name]
        assert set(source_state) == set(target)
        assert all(torch.equal(source_state[key], target[key]) for key in source_state)
    assert not selected.get("optimizer_state_dict", {}).get("state", {})
PY
  best=$(jq -er '.best_train_reward' "${run}/final_summary.json")
  start=$(jq -er '.starting_train_selection_reward' "${run}/final_summary.json")
  delta=$(jq -er '.best_train_reward_delta_from_start' "${run}/final_summary.json")
  "${PYTHON}" - "${best}" "${start}" "${delta}" <<'PY'
import math, sys
best, start, delta = map(float, sys.argv[1:])
assert all(math.isfinite(x) for x in (best, start, delta))
assert best + 1e-12 >= start
assert math.isclose(best - start, delta, rel_tol=0.0, abs_tol=1e-12)
PY
  # The retention anchor is checked for internal consistency, not against the
  # current constant: a finished replay must stay valid when the constant is
  # re-pinned, or every past cycle fails validation the moment we tune it.  A
  # mismatch is a note, not a failure — new replays get the configured value.
  run_anchor=$(jq -er '.awr.behavior_anchor_weight' "${run}/effective_config.json")
  if [[ ${run_anchor} != "${BEHAVIOR_ANCHOR_WEIGHT}" ]]; then
    log "note: cycle ${cycle} replay used behaviour anchor ${run_anchor}; configured is ${BEHAVIOR_ANCHOR_WEIGHT}"
  fi
  if [[ ${cycle} -ge 2 ]]; then
    [[ -n ${replay_root} ]] \
      || die "Cycle ${cycle} replay validation lacks the expected replay root"
  fi

  local epoch train_summary eval_summary selector_summary
  for epoch in $(seq "${first}" "${last}"); do
    printf -v train_summary '%s/train_epoch_%03d/summary.json' "${run}" "${epoch}"
    printf -v eval_summary '%s/eval_epoch_%03d/summary.json' "${run}" "${epoch}"
    printf -v selector_summary '%s/train_selector_epoch_%03d/summary.json' "${run}" "${epoch}"
    [[ -s ${train_summary} && -s ${eval_summary} && -s ${selector_summary} ]] \
      || die "Cycle ${cycle} epoch ${epoch} lacks train/eval/selector summary"
    "${PYTHON}" - "${train_summary}" <<'PY'
import json, math, pathlib, sys
s = json.loads(pathlib.Path(sys.argv[1]).read_text())
proposal = float(s["policy_proposal_relative_l2"])
accepted = float(s["policy_accepted_relative_l2"])
assert proposal > 0.0
assert math.isclose(accepted / proposal, 0.05, rel_tol=1e-6, abs_tol=1e-9)
assert math.isclose(float(s["ema_policy_commit_rate"]), 0.05, abs_tol=1e-12)
assert float(s["optimizer_state_reset_after_policy_commit"]) == 1.0
assert float(s["policy_transaction_candidate"]) == 1.0
promoted = float(s["policy_transaction_accepted"])
rolled_back = float(s["policy_transaction_rolled_back"])
continued = float(s["policy_training_continued_from_proposal"])
assert promoted in (0.0, 1.0)
assert rolled_back == 0.0
assert continued == 1.0
delta = float(s["train_selection_reward_delta_vs_best"])
assert (delta > 0.0) if promoted else (delta <= 0.0)
PY
  done
  "${PYTHON}" - "${run}/best_train.pth" <<'PY'
import pathlib, sys, torch
p = torch.load(pathlib.Path(sys.argv[1]), map_location="cpu", weights_only=False)
assert isinstance(p, dict) and "model" in p and "ema_state_dict" in p
assert isinstance(p.get("optimizer_state_dict"), dict)
assert p["optimizer_state_dict"].get("state", {}) == {}
PY
}

select_cycle_policy() {
  local run=$1 cycle=$2 first=$3 last=$4 source_checkpoint=$5
  [[ -x ${LINE_SEARCH_SCRIPT} ]] \
    || die "missing conditional line-search runner ${LINE_SEARCH_SCRIPT}"
  local selection selected reported_sha actual_sha selected_reward start_reward
  selection=$(
    "${LINE_SEARCH_SCRIPT}" \
      "${run}" "${source_checkpoint}" "${first}" "${last}" "${cycle}" \
      | tail -n 1
  )
  [[ -s ${selection} ]] \
    || die "Cycle ${cycle} policy selection produced no result"
  selected=$(jq -er '.selected.checkpoint' "${selection}")
  [[ -s ${selected} ]] || die "Cycle ${cycle} selected checkpoint is missing"
  reported_sha=$(jq -er '.selected.checkpoint_sha256' "${selection}")
  actual_sha=$(checkpoint_sha "${selected}")
  selected_reward=$(jq -er '.selected.mean_det_reward' "${selection}")
  start_reward=$(jq -er '.starting_train_selection_reward' "${run}/final_summary.json")
  "${PYTHON}" - "${selected_reward}" "${start_reward}" <<'PY'
import math
import sys

selected, start = map(float, sys.argv[1:])
assert math.isfinite(selected) and math.isfinite(start)
assert selected + 1e-12 >= start
PY
  log "Cycle ${cycle}: propagated policy kind=$(jq -er '.selected.kind' "${selection}") reward=${selected_reward} delta=$(jq -er '.selected.delta_vs_cycle_start' "${selection}")"
  printf '%s\n' "${selection}"
}

deployment_audit() {
  local run=$1 cycle=$2 previous_scenes=$3 checkpoint=$4 selection=$5
  local output=${run}/deployment_full10/selected_policy
  local manifest=${output}/manifest.json
  local selected_epoch selected_sha manifest_model
  selected_epoch=$(jq -er '.selected.epoch' "${selection}")
  selected_sha=$(jq -er '.selected.checkpoint_sha256' "${selection}")
  if [[ ! -s ${manifest} ]]; then
    mkdir -p "${output}"
    "${TORCHRUN}" --standalone --nproc_per_node=8 \
      "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --model_path "${checkpoint}" --args_path "${run}/model_args.json" \
      --scenes "${VALID}" --config "${run}/effective_config.json" \
      --output_dir "${output}" --label "cycle$(printf '%02d' "${cycle}")_selected_policy_dp10" \
      --batch_size 512 --scene_load_workers "${EVAL_SCENE_LOAD_WORKERS}" --sample_steps 10 \
      --amp_dtype bf16 --compile_mode default \
      > "${run}/deployment_full10/eval.log" 2>&1
    "${PYTHON}" "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --aggregate --model_path "${checkpoint}" --args_path "${run}/model_args.json" \
      --scenes "${VALID}" --config "${run}/effective_config.json" \
      --output_dir "${output}" --label "cycle$(printf '%02d' "${cycle}")_selected_policy_dp10" \
      >> "${run}/deployment_full10/eval.log" 2>&1
  fi
  manifest_model=$(jq -er '.model' "${manifest}")
  [[ $(readlink -f "${manifest_model}") == "$(readlink -f "${checkpoint}")" ]] \
    || die "Cycle ${cycle} deployment manifest points at another checkpoint"
  "${PYTHON}" -m rlvr.autoresearch.tools.full_paired_eval_report \
    --baseline "${BASELINE_DP10}" --candidate "${output}/scenes.json" \
    --epoch "${selected_epoch}" \
    --bootstrap 10000 --seed 3407 \
    --output "${run}/deployment_full10/paired_vs_original_incumbent.json" \
    >> "${run}/deployment_full10/eval.log" 2>&1
  if [[ -s ${previous_scenes} ]]; then
    "${PYTHON}" -m rlvr.autoresearch.tools.full_paired_eval_report \
      --baseline "${previous_scenes}" --candidate "${output}/scenes.json" \
      --epoch "${selected_epoch}" \
      --bootstrap 10000 --seed 3407 \
      --output "${run}/deployment_full10/paired_vs_previous_cycle.json" \
      >> "${run}/deployment_full10/eval.log" 2>&1
  fi
  printf '%s\n' "${output}/scenes.json"
}

# Turn the full-46k paired audit from a report into a gate.  Cycles 1-4 all
# committed on the 128-scene selector aggregate alone while the audit showed
# collisions up ~15% relative at P(improved)=0.003.  On veto we fall back to the
# cycle's starting incumbent, so a cycle's worst case is "no change" rather than
# a regression — and the campaign keeps running unattended either way.
enforce_non_regression() {
  local replay=$1 cycle=$2 selection=$3 incumbent=$4
  local epoch verdict out override
  epoch=$(jq -er '.selected.epoch' "${selection}")
  out=${replay}/deployment_full10/non_regression_verdict.json
  if "${PYTHON}" -m rlvr.autoresearch.tools.enforce_cycle_non_regression \
      --paired-audit "${replay}/deployment_full10/paired_vs_original_incumbent.json" \
      --epoch "${epoch}" \
      --candidate-scenes "${replay}/eval_epoch_$(printf '%03d' "${epoch}")/scenes.json" \
      --baseline-scenes "${replay}/eval_epoch_000/scenes.json" \
      --output "${out}" >> "${replay}/deployment_full10/eval.log" 2>&1; then
    printf '%s\n' "${selection}"
    return 0
  fi
  verdict=$(jq -r '.failed_checks | join(",")' "${out}" 2>/dev/null || echo "unknown")
  log "Cycle ${cycle}: NON-REGRESSION VETO (${verdict}); keeping the incumbent checkpoint"
  override=${replay}/deployment_full10/selection_after_veto.json
  "${PYTHON}" - "${selection}" "${incumbent}" "${out}" "${override}" <<'PY'
import hashlib, json, pathlib, sys
selection_path, incumbent, verdict_path, override_path = sys.argv[1:]
selection = json.loads(pathlib.Path(selection_path).read_text())
incumbent_path = pathlib.Path(incumbent)
selection["selected"] = dict(selection["selected"])
selection["selected"].update(
    {
        "checkpoint": str(incumbent_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(incumbent_path.read_bytes()).hexdigest(),
        "kind": "non_regression_veto_incumbent",
        "delta_vs_cycle_start": 0.0,
    }
)
selection["non_regression_verdict"] = json.loads(pathlib.Path(verdict_path).read_text())
selection["superseded_selection"] = selection_path
pathlib.Path(override_path).write_text(json.dumps(selection, indent=2, sort_keys=True))
PY
  printf '%s\n' "${override}"
}

record_cycle() {
  local cycle=$1 mine=$2 replay=$3 selection=$4 deployment=$5
  "${PYTHON}" - "${STATE}" "${cycle}" "${mine}" "${replay}" "${selection}" "${deployment}" <<'PY'
import datetime, hashlib, json, pathlib, sys
state, cycle, mine, replay, selection_path, deployment = sys.argv[1:]
summary = json.loads((pathlib.Path(replay) / "final_summary.json").read_text())
selection = json.loads(pathlib.Path(selection_path).read_text())
selected = selection["selected"]
checkpoint = pathlib.Path(selected["checkpoint"])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected["checkpoint_sha256"]
row = {
    "time": datetime.datetime.now().astimezone().isoformat(),
    "cycle": int(cycle),
    "mine_run": str(pathlib.Path(mine).resolve()),
    "replay_run": str(pathlib.Path(replay).resolve()),
    "replay_best_train_epoch": int(summary["best_train_epoch"]),
    "replay_best_train_reward": float(summary["best_train_reward"]),
    "replay_best_train_is_starting_incumbent": bool(summary["best_train_is_starting_incumbent"]),
    "selection": str(pathlib.Path(selection_path).resolve()),
    "selection_kind": selected["kind"],
    "best_train_epoch": int(selected["epoch"]),
    "best_train_reward": float(selected["mean_det_reward"]),
    "best_train_reward_delta_from_start": float(selected["delta_vs_cycle_start"]),
    "best_train_is_starting_incumbent": float(selected["delta_vs_cycle_start"]) <= 0.0,
    "best_train_checkpoint_sha256": selected["checkpoint_sha256"],
    "selected_checkpoint": str(checkpoint.resolve()),
    "deployment_scenes": str(pathlib.Path(deployment).resolve()),
}
path = pathlib.Path(state)
rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
rows = [old for old in rows if int(old["cycle"]) != int(cycle)] + [row]
rows.sort(key=lambda item: int(item["cycle"]))
path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
PY
}

wait_for_cycle1() {
  local run last_notice=0 now
  while true; do
    run=$(latest_completed_run "${CYCLE1_REPLAY_PARENT}")
    if [[ -n ${run} ]]; then
      printf '%s\n' "${run}"
      return 0
    fi
    # Cycle 1 has an audited phase boundary between strict cache close and
    # torchrun replay: context attachment, positive-overlay construction and
    # target-geometry validation run under the continuation owner.  Treat
    # that owner as live work; otherwise the campaign supervisor can
    # incorrectly fail during a healthy CPU-only transition window.
    if ! pgrep -f 'plannerrft_full_cycle01_(mine|replay_e02_e10)|watch_plannerrft_full_cycle01_transition|continue_plannerrft_full_cycle01_after_mine' >/dev/null; then
      die "Cycle 1 stopped without a completed replay run"
    fi
    now=$(date +%s)
    if (( now - last_notice >= 600 )); then
      log "waiting for Cycle 1 replay epoch 10 to complete"
      last_notice=${now}
    fi
    sleep 60
  done
}

run_mine() {
  local cycle=$1 epoch=$2 checkpoint=$3 args=$4 parent=$5 log_path=$6
  require_nvme_reserve "Cycle ${cycle} pre-mine"
  mkdir -p "${parent}"
  log "Cycle ${cycle}: mining global epoch ${epoch} from $(checkpoint_sha "${checkpoint}")"
  # The cycle-1 cache was mined with the same oversampling, so the canonical
  # scene order matches and it can be reused; train_awr.py verifies that against
  # the cache's provenance and refuses on any mismatch.
  local -a shared_cache_args=(
    --shared_replay_encoding_root "${SOURCE_CACHE}"
    --shared_replay_context_root "${SOURCE_CACHE}"
    --shared_replay_expert_anchor_root "${SOURCE_CACHE}"
  )
  "${TORCHRUN}" --standalone --nproc_per_node=8 -m rlvr.train_awr \
    --model_path "${checkpoint}" --args_path "${args}" \
    --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
    ${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"} \
    --config "${CONFIG}" --output_dir "${parent}" \
    --exp_name "plannerrft_full_cycle$(printf '%02d' "${cycle}")_mine_e${epoch}" \
    --device cuda --seed 3407 --start_epoch "${epoch}" --epochs "${epoch}" \
    --expert_anchor_weight 0.4 --max_train_scenes 0 --max_valid_scenes 8 \
    --plannerrft_reference_mode "${PLANNERRFT_REFERENCE_MODE}" \
    --plannerrft_reference_noise_scale "${PLANNERRFT_REFERENCE_NOISE_SCALE}" \
    --plannerrft_longitudinal_mode "${PLANNERRFT_LONGITUDINAL_MODE}" \
    --plannerrft_lambda_lat "${PLANNERRFT_LAMBDA_LAT}" \
    --sample_steps "${ROLLOUT_SAMPLE_STEPS}" \
    --eval_k 1 --eval_sample_steps 10 \
    --train_selector_scenes 128 --full_valid_interval 0 --hard_valid_scenes 0 \
    ${shared_cache_args[@]+"${shared_cache_args[@]}"} \
    --shared_replay_order_epoch 1 \
    --rollout_scene_batch_size 192 --scene_batch_size 192 \
    --rollout_scene_load_workers 1 --rollout_prefetch_batches 1 \
    --scene_load_workers 4 --eval_scene_load_workers "${EVAL_SCENE_LOAD_WORKERS}" \
    --eval_scene_batch_size 512 --save_rollouts 0 \
    --wandb --wandb_project original-dp-awr \
    --wandb_entity advanced-technology-department \
    --wandb_run_name "original-dp-awr-plannerrft-cycle$(printf '%02d' "${cycle}")-mine-e${epoch}" \
    --wandb_mode "${WANDB_MODE:-online}" \
    --wandb_tags original-dp awr plannerrft-guidance mine-stage \
      full-sequence "cycle$(printf '%02d' "${cycle}")" "global-e${epoch}" \
    --no-skip_filtered_scenes > "${log_path}" 2>&1
  local run
  run=$(run_from_log "${log_path}")
  [[ -n ${run} && -d ${run} ]] || die "Cycle ${cycle} mine log has no run directory"
  printf '%s\n' "${run}"
}

run_replay() {
  local cycle=$1 first=$2 last=$3 checkpoint=$4 args=$5 overlay=$6 parent=$7 log_path=$8
  local expert_scope_arg=--expert_anchor_active_groups_only
  if [[ ${EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY} -ne 1 ]]; then
    expert_scope_arg=--no-expert_anchor_active_groups_only
  fi
  preflight_replay_source "${cycle}" "${overlay}"
  require_nvme_reserve "Cycle ${cycle} pre-replay"
  mkdir -p "${parent}"
  log "Cycle ${cycle}: replaying global epochs ${first}-${last}"
  WANDB_MODE=${WANDB_MODE:-online} "${TORCHRUN}" --standalone --nproc_per_node=8 \
    -m rlvr.train_awr --model_path "${checkpoint}" --args_path "${args}" \
    --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
    --config "${CONFIG}" --output_dir "${parent}" \
    --exp_name "plannerrft_full_cycle$(printf '%02d' "${cycle}")_replay_e${first}_e${last}" \
    --device cuda --seed 3407 --start_epoch "${first}" --epochs "${last}" \
    --resume_replay_root "${overlay}" --replay_sampling with_replacement \
    --compressed_replay_context_root "${COMPRESSED_CONTEXT_ROOT}" \
    --plannerrft_reference_mode "${PLANNERRFT_REFERENCE_MODE}" \
    --plannerrft_reference_noise_scale "${PLANNERRFT_REFERENCE_NOISE_SCALE}" \
    --plannerrft_longitudinal_mode "${PLANNERRFT_LONGITUDINAL_MODE}" \
    --plannerrft_lambda_lat "${PLANNERRFT_LAMBDA_LAT}" \
    --sample_steps "${ROLLOUT_SAMPLE_STEPS}" \
    --awr_beta "${REPLAY_BETA}" --positive_advantage_only --positive_advantage_margin 0.01 \
    --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
    --expert_anchor_weight 0.4 \
    --awr_candidate_loss_horizon "${AWR_CANDIDATE_LOSS_HORIZON}" \
    "${expert_scope_arg}" \
    --drop_all_zero_groups \
    --awr_loss_type plain_mse --diffusion_t_range 0.001 0.2 \
    --trainable_scope output --learning_rate 0.000001 \
    --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy \
    --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 \
    --eval_k 1 --eval_sample_steps 10 \
    --full_valid_interval 0 --hard_valid_scenes 0 \
    --scene_batch_size 192 --gradient_accumulation_scenes 192 \
    --scene_load_workers "${REPLAY_SCENE_LOAD_WORKERS}" \
    --eval_scene_load_workers "${EVAL_SCENE_LOAD_WORKERS}" \
    --eval_scene_batch_size 512 --save_rollouts 0 \
    --wandb --wandb_project original-dp-awr \
    --wandb_entity advanced-technology-department \
    --wandb_run_name "original-dp-awr-plannerrft-cycle$(printf '%02d' "${cycle}")-e${first}-e${last}" \
    --wandb_mode "${WANDB_MODE:-online}" \
    --wandb_tags original-dp awr plannerrft-guidance conditional-guidance \
      positive-margin001 "beta${REPLAY_BETA}" expert04 scored-prefix40 active-group-expert \
      epoch-boundary-alpha005 cumulative-replay \
      full-sequence "cycle$(printf '%02d' "${cycle}")" "global-e${first}-e${last}" \
    --no-skip_filtered_scenes > "${log_path}" 2>&1
  local run
  run=$(run_from_log "${log_path}")
  [[ -n ${run} && -d ${run} ]] || die "Cycle ${cycle} replay log has no run directory"
  printf '%s\n' "${run}"
}

main() {
  [[ -s ${BASELINE_DP10} ]] || die "missing fixed incumbent DP10 baseline"
  local cycle1_replay
  cycle1_replay=$(wait_for_cycle1)
  validate_cycle1_source_cache
  local cycle1_source cycle1_selection cycle1_checkpoint cycle1_deployment
  cycle1_source=${CYCLE1_SOURCE_CHECKPOINT}
  if [[ -z ${cycle1_source} ]]; then
    cycle1_source=$(jq -er '.staged_model' "${CYCLE1_MINE_RUN}/provenance.json")
  fi
  [[ -s ${cycle1_source} ]] || die "Cycle 1 source checkpoint is missing"
  validate_replay_run "${cycle1_replay}" 1 "${CYCLE1_FIRST}" "${CYCLE1_LAST}" \
    "${cycle1_source}"
  cycle1_selection=$(select_cycle_policy "${cycle1_replay}" 1 \
    "${CYCLE1_FIRST}" "${CYCLE1_LAST}" "${cycle1_source}")
  cycle1_checkpoint=$(jq -er '.selected.checkpoint' "${cycle1_selection}")

  [[ -x ${EVAL_LOADER_BENCHMARK_SCRIPT} ]] \
    || die "missing eval-loader benchmark ${EVAL_LOADER_BENCHMARK_SCRIPT}"
  log "Benchmarking deterministic full-validation NPZ loader concurrency"
  "${EVAL_LOADER_BENCHMARK_SCRIPT}" \
    "${cycle1_checkpoint}" "${cycle1_replay}/model_args.json" \
    "${cycle1_replay}/effective_config.json" "${VALID}" \
    "${EVAL_LOADER_BENCHMARK_OUT}" >> "${STATUS}" 2>&1
  EVAL_SCENE_LOAD_WORKERS=$(jq -er '.selected_workers' \
    "${EVAL_LOADER_BENCHMARK_OUT}/selection.json")
  [[ ${EVAL_SCENE_LOAD_WORKERS} == 1 || ${EVAL_SCENE_LOAD_WORKERS} == 4 || ${EVAL_SCENE_LOAD_WORKERS} == 8 ]] \
    || die "invalid selected eval loader workers ${EVAL_SCENE_LOAD_WORKERS}"
  export EVAL_SCENE_LOAD_WORKERS
  log "Selected eval_scene_load_workers=${EVAL_SCENE_LOAD_WORKERS}; speedup=$(jq -er '.speedup_vs_worker1' "${EVAL_LOADER_BENCHMARK_OUT}/selection.json")"

  cycle1_deployment=$(deployment_audit "${cycle1_replay}" 1 "" "${cycle1_checkpoint}" "${cycle1_selection}")
  # Cycle 1 was previously exempt from the non-regression gate — it committed on
  # the train selector alone, exactly like cycles 2-10 used to.  Gate it too.
  cycle1_selection=$(enforce_non_regression "${cycle1_replay}" 1 "${cycle1_selection}" "${cycle1_source}")
  cycle1_checkpoint=$(jq -er '.selected.checkpoint' "${cycle1_selection}")
  record_cycle 1 "${CYCLE1_MINE_RUN}" "${cycle1_replay}" "${cycle1_selection}" "${cycle1_deployment}"
  log "Cycle 1 complete and verified: ${cycle1_replay}"

  ensure_compressed_context

  [[ -x ${BETA_SENSITIVITY_SCRIPT} ]] \
    || die "missing beta-sensitivity runner ${BETA_SENSITIVITY_SCRIPT}"
  log "Running paired same-cache beta sensitivity before Cycle 2"
  MODEL="${cycle1_checkpoint}" ARGS="${cycle1_replay}/model_args.json" \
    MINE_RUN="${CYCLE1_MINE_RUN}" OUT="${BETA_SENSITIVITY_OUT}" \
    COMPRESSED_REPLAY_CONTEXT_ROOT="${COMPRESSED_CONTEXT_ROOT}" \
    EVAL_SCENE_LOAD_WORKERS="${EVAL_SCENE_LOAD_WORKERS}" \
    REPLAY_SCENE_LOAD_WORKERS="${REPLAY_SCENE_LOAD_WORKERS}" \
    "${BETA_SENSITIVITY_SCRIPT}" >> "${STATUS}" 2>&1
  [[ -s ${BETA_SELECTION} ]] || die "beta sensitivity produced no selection"
  # Probe reports, does not override.  The 41-epoch run that actually improved
  # (plannerrft_prefix_full_cycles02_to10_e100: 0.93233 -> 0.93297 over five
  # cycles, no cycle regressing) held beta=1.0 for its entire length.  The probe
  # picks on a short paired delta whose three arms all came out negative and
  # within tolerance of each other, so it is choosing noise; a 41-epoch empirical
  # result outranks that.  Set PROBE_MAY_OVERRIDE=1 to restore probe control.
  if [[ ${PROBE_MAY_OVERRIDE:-0} == 1 ]]; then
    REPLAY_BETA=$(jq -er '.selected_beta' "${BETA_SELECTION}")
  else
    log "beta probe suggested $(jq -r '.selected_beta' "${BETA_SELECTION}"); keeping validated ${REPLAY_BETA}"
  fi
  [[ ${REPLAY_BETA} == 0.5 || ${REPLAY_BETA} == 1 || ${REPLAY_BETA} == 2 ]] \
    || die "invalid selected replay beta ${REPLAY_BETA}"
  log "Cycles 2--${TOTAL_CYCLES} selected replay beta=${REPLAY_BETA}; reason=$(jq -er '.selection_reason' "${BETA_SELECTION}")"

  [[ -x ${SOLVER_STEP_SCRIPT} ]] \
    || die "missing solver-step sensitivity runner ${SOLVER_STEP_SCRIPT}"
  log "Running paired 5-vs-10-step candidate solver sensitivity before Cycle 2"
  POLICY="${cycle1_checkpoint}" \
    POLICY_ARGS="${cycle1_replay}/model_args.json" \
    OUT="${SOLVER_STEP_OUT}" \
    "${SOLVER_STEP_SCRIPT}" >> "${STATUS}" 2>&1
  [[ -s ${SOLVER_STEP_SELECTION} ]] \
    || die "solver-step sensitivity produced no selection"
  ROLLOUT_SAMPLE_STEPS=$(jq -er '.selected_sample_steps' \
    "${SOLVER_STEP_SELECTION}")
  [[ ${ROLLOUT_SAMPLE_STEPS} == 5 || ${ROLLOUT_SAMPLE_STEPS} == 10 ]] \
    || die "invalid selected rollout sample steps ${ROLLOUT_SAMPLE_STEPS}"
  log "Cycles 2--${TOTAL_CYCLES} selected training-time sample_steps=${ROLLOUT_SAMPLE_STEPS}; reason=$(jq -er '.selection_reason' "${SOLVER_STEP_SELECTION}")"

  [[ -x ${CANDIDATE_SENSITIVITY_SCRIPT} ]] \
    || die "missing candidate-sensitivity runner ${CANDIDATE_SENSITIVITY_SCRIPT}"
  log "Running paired PlannerRFT candidate-generation sensitivity before Cycle 2"
    POLICY="${cycle1_checkpoint}" \
    POLICY_ARGS="${cycle1_replay}/model_args.json" \
    SAMPLE_STEPS="${ROLLOUT_SAMPLE_STEPS}" \
    OUT="${CANDIDATE_SENSITIVITY_OUT}" \
    "${CANDIDATE_SENSITIVITY_SCRIPT}" >> "${STATUS}" 2>&1
  [[ -s ${CANDIDATE_SELECTION} ]] \
    || die "candidate sensitivity produced no selection"
  # Same reasoning.  The validated run generated candidates with
  # reference_mode=stochastic and reference_noise_scale=0.5 throughout.  The
  # probe selected zero_noise/0.0, which removes the only source of behavioural
  # diversity between candidates — and the measured median headroom (0.00366) is
  # already below the within-group candidate std (0.00479), so making candidates
  # *more* homogeneous removes learnable signal rather than adding it.
  if [[ ${PROBE_MAY_OVERRIDE:-0} == 1 ]]; then
    PLANNERRFT_REFERENCE_MODE=$(jq -er '.selected_reference_mode' "${CANDIDATE_SELECTION}")
    PLANNERRFT_REFERENCE_NOISE_SCALE=$(jq -er '.selected_reference_noise_scale' "${CANDIDATE_SELECTION}")
    PLANNERRFT_LONGITUDINAL_MODE=$(jq -er '.selected_longitudinal_mode' "${CANDIDATE_SELECTION}")
    PLANNERRFT_LAMBDA_LAT=$(jq -er '.selected_lambda_lat_m' "${CANDIDATE_SELECTION}")
  else
    log "candidate probe suggested $(jq -r '.selected_reference_mode' "${CANDIDATE_SELECTION}")/$(jq -r '.selected_longitudinal_mode' "${CANDIDATE_SELECTION}") noise=$(jq -r '.selected_reference_noise_scale' "${CANDIDATE_SELECTION}"); keeping validated ${PLANNERRFT_REFERENCE_MODE}/${PLANNERRFT_LONGITUDINAL_MODE} noise=${PLANNERRFT_REFERENCE_NOISE_SCALE}"
  fi
  [[ $(jq -er '.selected_sample_steps' "${CANDIDATE_SELECTION}") == ${ROLLOUT_SAMPLE_STEPS} ]] \
    || die "candidate sensitivity used a different diffusion solver"
  [[ ${PLANNERRFT_REFERENCE_MODE} == zero_noise || ${PLANNERRFT_REFERENCE_MODE} == stochastic ]] \
    || die "invalid selected reference mode ${PLANNERRFT_REFERENCE_MODE}"
  [[ ${PLANNERRFT_LONGITUDINAL_MODE} == candidate_stretch || ${PLANNERRFT_LONGITUDINAL_MODE} == reference_speed_push ]] \
    || die "invalid selected longitudinal mode ${PLANNERRFT_LONGITUDINAL_MODE}"
  [[ ${PLANNERRFT_LAMBDA_LAT} == 1 || ${PLANNERRFT_LAMBDA_LAT} == 1.0 || ${PLANNERRFT_LAMBDA_LAT} == 1.5 || ${PLANNERRFT_LAMBDA_LAT} == 2.5 ]] \
    || die "invalid selected lateral guidance target ${PLANNERRFT_LAMBDA_LAT}"
  log "Cycles 2--${TOTAL_CYCLES} candidate mode=${PLANNERRFT_REFERENCE_MODE}/${PLANNERRFT_LONGITUDINAL_MODE}, lambda_lat=${PLANNERRFT_LAMBDA_LAT}m; reason=$(jq -er '.selection_reason' "${CANDIDATE_SELECTION}")"

  local previous_replay=${cycle1_replay}
  local previous_checkpoint=${cycle1_checkpoint}
  local previous_deployment=${cycle1_deployment}
  local cycle mine_epoch replay_first replay_last checkpoint args selection selected_checkpoint
  local mine_parent replay_parent overlay_parent mine_log replay_log mine_run cache overlay replay_run
  for cycle in $(seq 2 "${TOTAL_CYCLES}"); do
    mine_epoch=$((10 * cycle - 9))
    replay_first=$((mine_epoch + 1))
    replay_last=$((10 * cycle))
    checkpoint=${previous_checkpoint}
    args=${previous_replay}/model_args.json
    [[ -s ${checkpoint} && -s ${args} ]] || die "Cycle ${cycle} has no prior selected checkpoint/args"
    mine_parent=${OUT}/cycle$(printf '%02d' "${cycle}")_mine_e${mine_epoch}
    replay_parent=${OUT}/cycle$(printf '%02d' "${cycle}")_replay_e${replay_first}_e${replay_last}
    overlay_parent=${OUT}/cycle$(printf '%02d' "${cycle}")_positive_overlay
    mine_log=${OUT}/cycle$(printf '%02d' "${cycle}")_mine.log
    replay_log=${OUT}/cycle$(printf '%02d' "${cycle}")_replay.log

    mine_run=$(latest_completed_mine_run "${mine_parent}")
    if [[ -z ${mine_run} ]]; then
      # A mine cache is a function of (policy, corpus).  When the previous cycle
      # kept its incumbent — a non-regression veto, or no epoch beating the
      # selector — the policy is byte-identical to the one that produced the
      # previous cache, so re-rolling it costs ~10 GPU-hours and yields a
      # statistically identical sample.  Reuse it instead.
      mine_run=$(reusable_mine_run_for_checkpoint "${checkpoint}")
      if [[ -n ${mine_run} ]]; then
        log "Cycle ${cycle}: policy unchanged since $(basename "${mine_run}"); reusing that mine cache instead of re-rolling"
      else
        archive_interrupted_stage_log "${mine_log}" \
          "cycle$(printf '%02d' "${cycle}")_mine"
        mine_run=$(run_mine "${cycle}" "${mine_epoch}" "${checkpoint}" "${args}" "${mine_parent}" "${mine_log}")
      fi
    fi
    cache=$(validate_mine_run "${mine_run}" "${cycle}" "${mine_epoch}")
    overlay=$(build_or_validate_overlay "${cycle}" "${cache}" "${overlay_parent}")

    replay_run=$(latest_completed_run "${replay_parent}")
    if [[ -z ${replay_run} ]]; then
      archive_interrupted_stage_log "${replay_log}" \
        "cycle$(printf '%02d' "${cycle}")_replay"
      replay_run=$(run_replay "${cycle}" "${replay_first}" "${replay_last}" \
        "${checkpoint}" "${args}" "${overlay}" "${replay_parent}" "${replay_log}")
    fi
    validate_replay_run "${replay_run}" "${cycle}" "${replay_first}" "${replay_last}" \
      "${checkpoint}" "${overlay}"
    selection=$(select_cycle_policy "${replay_run}" "${cycle}" "${replay_first}" "${replay_last}" "${checkpoint}")
    selected_checkpoint=$(jq -er '.selected.checkpoint' "${selection}")
    previous_deployment=$(deployment_audit "${replay_run}" "${cycle}" "${previous_deployment}" "${selected_checkpoint}" "${selection}")
    selection=$(enforce_non_regression "${replay_run}" "${cycle}" "${selection}" "${checkpoint}")
    selected_checkpoint=$(jq -er '.selected.checkpoint' "${selection}")
    record_cycle "${cycle}" "${mine_run}" "${replay_run}" "${selection}" "${previous_deployment}"
    previous_replay=${replay_run}
    previous_checkpoint=${selected_checkpoint}
    log "Cycle ${cycle} verified; selected epoch=$(jq -er '.selected.epoch' "${selection}") reward=$(jq -er '.selected.mean_det_reward' "${selection}") kind=$(jq -er '.selected.kind' "${selection}")"
  done

  "${PYTHON}" - "${STATE}" "${OUT}/epoch${TARGET_EPOCH}_completion.json" \
    "${TOTAL_CYCLES}" "${TARGET_EPOCH}" <<'PY'
import json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line]
total_cycles = int(sys.argv[3])
target_epoch = int(sys.argv[4])
latest = {int(row["cycle"]): row for row in rows}
assert set(latest) >= set(range(1, total_cycles + 1))
final = latest[total_cycles]
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "status": "target_epoch_training_and_per_cycle_deployment_audit_complete",
    "target_epoch": target_epoch,
    "total_cycles": total_cycles,
    "cycles": [latest[index] for index in range(1, total_cycles + 1)],
    "final_selected_checkpoint": final["selected_checkpoint"],
    "final_deployment_scenes": final["deployment_scenes"],
}, indent=2, sort_keys=True) + "\n")
PY
  log "PlannerRFT/AWR chain reached global epoch ${TARGET_EPOCH} with strict completion artifact"
}

trap 'status=$?; log "supervisor exit status=${status} line=${LINENO}"; exit ${status}' ERR
main "$@"
