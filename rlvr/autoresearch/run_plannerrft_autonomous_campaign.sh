#!/usr/bin/env bash
# Keep the formal PlannerRFT/AWR campaign advancing without per-epoch human
# decisions. The trainer and cycle supervisor own policy selection; this outer
# process only resumes committed artifacts after a process-level interruption.
# Repeated deterministic failures are recorded as a high-severity event and
# retried after a long cooldown. The immutable artifact checks still fail
# closed inside the supervisor; the outer owner itself must remain alive so a
# transient filesystem/process problem cannot leave eight GPUs idle overnight.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
ENTRYPOINT=${ENTRYPOINT:-${ROOT}/rlvr/autoresearch/run_plannerrft_prefix_to_epoch100.sh}
# Optional extra liveness probe: campaigns whose entrypoint owns long stages
# before the epoch-100 supervisor registers its PID (e.g. a from-scratch
# Cycle-1 mine) declare a command-line pattern here so a restarted owner does
# not treat that healthy stage as an absent supervisor.
ALIVE_PATTERN=${ALIVE_PATTERN:-}
# Durable-progress signature root; campaigns that commit Cycle-1 artifacts
# outside CAMPAIGN_OUT point this at the common parent.
PROGRESS_ROOT=${PROGRESS_ROOT:-}
TARGET_EPOCH=${TARGET_EPOCH:-100}
[[ ${TARGET_EPOCH} =~ ^[1-9][0-9]*$ ]] \
  || { echo "TARGET_EPOCH must be a positive integer" >&2; exit 2; }
(( TARGET_EPOCH >= 20 && TARGET_EPOCH % 10 == 0 )) \
  || { echo "TARGET_EPOCH must be a multiple of 10 and at least 20" >&2; exit 2; }
TOTAL_CYCLES=$((TARGET_EPOCH / 10))
CAMPAIGN_OUT=${CAMPAIGN_OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycles02_to${TOTAL_CYCLES}_e${TARGET_EPOCH}}
COMPLETION=${CAMPAIGN_OUT}/epoch${TARGET_EPOCH}_completion.json
LOCK=${CAMPAIGN_OUT}/autonomous_campaign.lock
PID_FILE=${CAMPAIGN_OUT}/autonomous_campaign.pid
LOG=${CAMPAIGN_OUT}/autonomous_campaign.log
POLL_INTERVAL=${POLL_INTERVAL:-60}
RETRY_DELAY=${RETRY_DELAY:-10}
MAX_STAGNANT_RESTARTS=${MAX_STAGNANT_RESTARTS:-3}
STAGNANT_FAILURE_COOLDOWN=${STAGNANT_FAILURE_COOLDOWN:-300}
FAILURE_STATE=${CAMPAIGN_OUT}/autonomous_failure_state.json

[[ ${POLL_INTERVAL} =~ ^[1-9][0-9]*$ && ${RETRY_DELAY} =~ ^[1-9][0-9]*$ && \
   ${STAGNANT_FAILURE_COOLDOWN} =~ ^[1-9][0-9]*$ ]] \
  || { echo "poll/retry intervals must be positive integers" >&2; exit 2; }
[[ ${MAX_STAGNANT_RESTARTS} =~ ^[1-9][0-9]*$ ]] \
  || { echo "MAX_STAGNANT_RESTARTS must be positive" >&2; exit 2; }
[[ -x ${ENTRYPOINT} ]] || { echo "missing entrypoint ${ENTRYPOINT}" >&2; exit 1; }

mkdir -p "${CAMPAIGN_OUT}"
exec 9>"${LOCK}"
flock -n 9 || exit 0
printf '%s\n' "$$" > "${PID_FILE}"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${LOG}" >&2
}

supervisor_alive() {
  local pid
  if [[ -s ${CAMPAIGN_OUT}/supervisor.pid ]]; then
    pid=$(<"${CAMPAIGN_OUT}/supervisor.pid")
    if [[ ${pid} =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null \
      && ps -p "${pid}" -o args= 2>/dev/null \
        | grep -Fq 'run_plannerrft_full_to_epoch100.sh'; then
      return 0
    fi
  fi
  [[ -n ${ALIVE_PATTERN} ]] && pgrep -f "${ALIVE_PATTERN}" >/dev/null
}

committed_signature() {
  # Logs and PID files are deliberately excluded: only durable experiment
  # artifacts count as progress for the bounded auto-restart policy.
  find "${PROGRESS_ROOT:-${CAMPAIGN_OUT}}" -xdev -type f \
    \( -name final_summary.json -o -name selection.json \
       -o -name manifest.json -o -name 'epoch*_completion.json' \) \
    -printf '%p\t%s\t%T@\n' 2>/dev/null \
    | sort | sha256sum | awk '{print $1}'
}

stagnant_restarts=0
previous_failed_signature=''
log "autonomous owner active target_epoch=${TARGET_EPOCH} campaign=${CAMPAIGN_OUT}"

while [[ ! -s ${COMPLETION} ]]; do
  if supervisor_alive; then
    sleep "${POLL_INTERVAL}"
    continue
  fi

  # A critical state describes the previous bounded retry burst. Preserve it
  # as history, but remove the live marker when a fresh retry actually starts;
  # otherwise health monitoring cannot distinguish cooldown from recovery.
  if [[ -s ${FAILURE_STATE} ]]; then
    failure_history=${CAMPAIGN_OUT}/autonomous_failure_history
    mkdir -p "${failure_history}"
    mv "${FAILURE_STATE}" \
      "${failure_history}/retrying_$(date '+%Y%m%d-%H%M%S')_$$.json"
    log "archived prior critical failure state before a fresh retry"
  fi

  before=$(committed_signature)
  attempt_log=${CAMPAIGN_OUT}/autonomous_attempt_$(date '+%Y%m%d-%H%M%S').log
  log "supervisor absent before completion; resuming from committed artifacts"
  set +e
  CAMPAIGN_OUT="${CAMPAIGN_OUT}" TARGET_EPOCH="${TARGET_EPOCH}" \
    bash "${ENTRYPOINT}" >> "${attempt_log}" 2>&1
  status=$?
  set -e

  [[ -s ${COMPLETION} ]] && break
  after=$(committed_signature)
  if [[ ${after} == "${before}" && ${after} == "${previous_failed_signature}" ]]; then
    stagnant_restarts=$((stagnant_restarts + 1))
  elif [[ ${after} == "${before}" ]]; then
    stagnant_restarts=1
  else
    stagnant_restarts=0
  fi
  previous_failed_signature=${after}
  log "supervisor exited status=${status}; stagnant_restarts=${stagnant_restarts}/${MAX_STAGNANT_RESTARTS}; log=${attempt_log}"
  if (( stagnant_restarts >= MAX_STAGNANT_RESTARTS )); then
    tmp=${FAILURE_STATE}.tmp.$$
    jq -n \
      --arg timestamp "$(date --iso-8601=seconds)" \
      --arg attempt_log "${attempt_log}" \
      --arg signature "${after}" \
      --argjson status "${status}" \
      --argjson attempts "${stagnant_restarts}" \
      --argjson cooldown "${STAGNANT_FAILURE_COOLDOWN}" \
      '{timestamp:$timestamp, severity:"critical", status:$status,
        stagnant_restarts:$attempts, committed_signature:$signature,
        attempt_log:$attempt_log, retry_cooldown_seconds:$cooldown}' \
      > "${tmp}"
    mv "${tmp}" "${FAILURE_STATE}"
    log "repeated supervisor exits produced no new committed artifact; critical failure recorded after ${stagnant_restarts} stagnant exits; retrying after ${STAGNANT_FAILURE_COOLDOWN}s instead of abandoning the campaign"
    stagnant_restarts=0
    sleep "${STAGNANT_FAILURE_COOLDOWN}"
    continue
  fi
  sleep "${RETRY_DELAY}"
done

log "strict completion detected: ${COMPLETION}"
