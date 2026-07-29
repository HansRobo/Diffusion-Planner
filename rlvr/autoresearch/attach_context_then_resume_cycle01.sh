#!/usr/bin/env bash
# Wait for both independent full-corpus products and commit the decoder-context
# metadata.  The mining launcher predates the current replay contract, so it
# must not execute the commands that Bash loaded when mining started.  After
# the miner exits naturally, this owner retires only that stale parent shell
# and execs the current idempotent continuation under a transition lock.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUN=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine
CACHE=${RUN}/replay_buffer
OVERLAY_PARENT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_positive_overlay
LAUNCHER_PID=${LAUNCHER_PID:-2890300}
LOG=${RUN}/decoder_context_transition.log
EXPECTED=5446656
CONTEXT_RESTARTS=${RUN}/context_build_restarts
CONTEXT_PID=${RUN}/context_build.pid
CONTEXT_LOG=${RUN}/context_build.log
TRANSITION_LOCK=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_transition.lock
CONTINUATION=${ROOT}/rlvr/autoresearch/continue_plannerrft_full_cycle01_after_mine.sh

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=1
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "${RUN}/attach_context_watcher.pid"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG}" >&2
}

restart_context_builder() {
  local attempts=0
  if [[ -s ${CONTEXT_RESTARTS} ]]; then
    attempts=$(<"${CONTEXT_RESTARTS}")
  fi
  [[ ${attempts} =~ ^[0-9]+$ ]] || attempts=0
  if (( attempts >= 3 )); then
    log "decoder-context builder failed ${attempts} times; keeping launcher stopped"
    exit 1
  fi
  attempts=$((attempts + 1))
  printf '%s\n' "${attempts}" > "${CONTEXT_RESTARTS}"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    setsid "${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
      -m rlvr.autoresearch.tools.build_replay_decoder_context \
      --replay-root "${CACHE}" --args-path "${RUN}/model_args.json" \
      --batch-size 640 --scene-load-workers 4 --flush-interval-batches 16 \
      >> "${CONTEXT_LOG}" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${CONTEXT_PID}"
  log "restarted resumable decoder-context builder attempt ${attempts}/3 pid=$!"
}

audit_target_geometry() {
  local overlay=${OVERLAY_PARENT}/replay_buffer
  local output=${OVERLAY_PARENT}/target_geometry.json
  if [[ ! -s ${output} ]]; then
    "${ROOT}/.venv/bin/python" -m \
      rlvr.autoresearch.tools.analyze_replay_target_geometry \
      --replay-root "${overlay}" --output "${output}" \
      --expert-weight 0.4 \
      > "${OVERLAY_PARENT}/target_geometry.log" 2>&1
  fi
  jq -e --argjson groups "${EXPECTED}" '
    .source_complete == true and .prefix_groups == $groups and
    .group_size == 10 and .future_len == 80 and
    .contract.weight_source == "stored replay weights" and
    .parameters.expert_weight == 0.4 and
    .active_group_fraction > 0 and
    .active_candidate_target_reward_gain > 0
  ' "${output}" >/dev/null
  log "Cycle-1 target audit committed: active=$(jq -r '.active_group_fraction' "${output}") reward_gain=$(jq -r '.active_candidate_target_reward_gain' "${output}") path_delta=$(jq -r '.active_candidate_path_length_delta_m' "${output}")"
}

while true; do
  replay_manifests=$(find "${CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json 2>/dev/null | wc -l)
  expert_manifests=$(find "${CACHE}" -mindepth 2 -maxdepth 2 -name expert_anchor_manifest.json 2>/dev/null | wc -l)
  context_manifests=$(find "${CACHE}" -mindepth 2 -maxdepth 2 -name context_build_manifest.json 2>/dev/null | wc -l)
  if [[ ${replay_manifests} -eq 8 && ${expert_manifests} -eq 8 && ${context_manifests} -eq 8 ]]; then
    break
  fi
  if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    log "launcher ${LAUNCHER_PID} disappeared before context attachment"
    exit 1
  fi
  if [[ ${replay_manifests} -lt 8 ]] && \
     ! pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_mine' >/dev/null; then
    log "candidate miner disappeared before strict replay close; releasing launcher to propagate failure"
    kill -CONT "${LAUNCHER_PID}"
    exit 1
  fi
  if [[ ${context_manifests} -lt 8 ]] && \
     ! pgrep -f 'rlvr.autoresearch.tools.build_replay_decoder_context.*plannerrft_full_cycle01_mine' >/dev/null; then
    restart_context_builder
  fi
  sleep 30
done

# The parent shell was stopped while its torchrun child continued mining. Keep
# this idempotent in case the watcher is restarted during the close boundary.
kill -STOP "${LAUNCHER_PID}"
log "candidate, expert and context manifests are complete; validating attachment"

"${ROOT}/.venv/bin/python" -m \
  rlvr.autoresearch.tools.attach_replay_decoder_context \
  --replay-root "${CACHE}" --expected-world-size 8 \
  >> "${LOG}" 2>&1

total=0
for rank in $(seq 0 7); do
  printf -v rank_dir '%s/rank_%04d' "${CACHE}" "${rank}"
  count=$(jq -er '.scene_count' "${rank_dir}/manifest.json")
  total=$((total + count))
  jq -e '
    .version >= 3 and .decoder_context_cached == true and
    .decoder_context_schema == "canonical_loader_aligned_v1" and
    (.arrays.context_ego_current_state | type == "string") and
    (.arrays.context_neighbor_agents_past | type == "string") and
    (.arrays.context_neighbor_agents_future | type == "string")
  ' "${rank_dir}/manifest.json" >/dev/null
done
[[ ${total} -eq ${EXPECTED} ]] || {
  log "attached replay total ${total} != ${EXPECTED}"
  exit 1
}

log "decoder context committed for ${total} groups; waiting for miner shutdown"
for _ in $(seq 1 120); do
  if ! pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_mine' >/dev/null; then
    break
  fi
  sleep 5
done
if pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_mine' >/dev/null; then
  log "miner did not exit after strict cache close; keeping stale launcher stopped"
  exit 1
fi

# SIGKILL is intentionally scoped to the stopped Bash parent only.  At this
# point torchrun and all workers have exited and all strict manifests exist.
# Never signal the process group: unrelated wrappers share that group.
if kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
  kill -KILL "${LAUNCHER_PID}"
  log "retired stale Cycle-1 launcher pid=${LAUNCHER_PID}"
fi

exec 9>"${TRANSITION_LOCK}"
if ! flock -n 9; then
  log "another transition owner already holds ${TRANSITION_LOCK}"
  exit 0
fi
log "starting current Cycle-1 continuation contract"
exec bash "${CONTINUATION}"
