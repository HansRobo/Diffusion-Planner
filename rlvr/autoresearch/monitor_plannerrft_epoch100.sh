#!/usr/bin/env bash
# Low-noise health watchdog for the formal PlannerRFT/AWR campaign.
#
# The trainer/supervisor own experiment state and checkpoint selection. This
# watchdog does not poll the model from an agent and does not emit per-batch or
# per-epoch reports. It keeps one replace-in-place health snapshot, appends only
# state-change events, and recovers a process that is alive but has stopped
# producing runtime progress for a conservative timeout.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
CAMPAIGN=${CAMPAIGN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycles02_to10_e100}
TARGET_EPOCH=${TARGET_EPOCH:-100}
[[ ${TARGET_EPOCH} =~ ^[1-9][0-9]*$ ]] \
  || { echo "TARGET_EPOCH must be a positive integer" >&2; exit 2; }

COMPLETION=${CAMPAIGN}/epoch${TARGET_EPOCH}_completion.json
LATEST=${CAMPAIGN}/health_monitor_latest.json
EVENTS=${CAMPAIGN}/health_events.jsonl
LOCK=${CAMPAIGN}/health_monitor.lock
PID_FILE=${CAMPAIGN}/health_monitor.pid
AUTONOMOUS_OWNER=${ROOT}/rlvr/autoresearch/run_plannerrft_autonomous_campaign.sh
RECOVERY_LOG=${CAMPAIGN}/health_monitor_recovery.log

# Five-minute local checks are enough because training writes progress every
# few seconds. A 30-minute no-progress window is deliberately much longer than
# a normal epoch (~12 minutes), avoiding false recovery during evaluation or
# cycle transitions while still preventing an overnight silent hang.
INTERVAL=${INTERVAL:-300}
STALE_AFTER=${STALE_AFTER:-1800}
RECOVERY_COOLDOWN=${RECOVERY_COOLDOWN:-1800}
MIN_DISK_AVAILABLE_KIB=${MIN_DISK_AVAILABLE_KIB:-536870912}
ONCE=${ONCE:-0}

for value in "${INTERVAL}" "${STALE_AFTER}" "${RECOVERY_COOLDOWN}" \
  "${MIN_DISK_AVAILABLE_KIB}"; do
  [[ ${value} =~ ^[1-9][0-9]*$ ]] \
    || { echo "watchdog intervals and disk floor must be positive integers" >&2; exit 2; }
done
[[ ${ONCE} == 0 || ${ONCE} == 1 ]] \
  || { echo "ONCE must be 0 or 1" >&2; exit 2; }

mkdir -p "${CAMPAIGN}"
exec 9>"${LOCK}"
flock -n 9 || exit 0
printf '%s\n' "$$" > "${PID_FILE}"

emit_event() {
  local severity=$1 kind=$2 message=$3 details=${4:-'{}'}
  jq -cn \
    --arg timestamp "$(date --iso-8601=seconds)" \
    --arg severity "${severity}" \
    --arg kind "${kind}" \
    --arg message "${message}" \
    --argjson details "${details}" \
    '{timestamp:$timestamp,severity:$severity,kind:$kind,message:$message,details:$details}' \
    >> "${EVENTS}"
}

valid_process_from_pid_file() {
  local pid_file=$1 expected=$2 pid
  [[ -s ${pid_file} ]] || return 1
  pid=$(<"${pid_file}")
  [[ ${pid} =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  ps -p "${pid}" -o args= 2>/dev/null | grep -Fq "${expected}"
}

latest_runtime_progress() {
  # Exclude the watchdog/owner logs themselves: only the active supervisor,
  # trainer, and W&B local stream count as experiment progress.
  find "${CAMPAIGN}" -xdev -type f \
    \( -name '*.log' -o -name '*.wandb' -o -name 'metrics.jsonl' \) \
    ! -name 'health_*' ! -name 'autonomous_*' \
    -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 1
}

latest_completed_epoch() {
  find "${CAMPAIGN}" -xdev -type f \
    -path '*/train_selector_epoch_*/summary.json' -printf '%h\n' 2>/dev/null \
    | sed -n 's#.*train_selector_epoch_0*\([0-9][0-9]*\)$#\1#p' \
    | sort -n | tail -n 1
}

latest_best_checkpoint() {
  find "${CAMPAIGN}" -xdev -type f -name best_train.pth \
    -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 1
}

process_is_stopped() {
  # A SIGSTOPped process answers kill -0, so "alive" is not the same as
  # "working".  Freezing the campaign to hand its GPUs to an A/B arm is a
  # deliberate state and must not be mistaken for a stall.
  local pid=$1 state
  [[ ${pid} =~ ^[1-9][0-9]*$ ]] || return 1
  state=$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')
  [[ ${state} == T* ]]
}

terminate_stalled_work() {
  # Scope the kill to the supervisor's own process group.  An unscoped
  # `pgrep -f torchrun` is machine-wide: on 2026-07-28 it SIGTERMed a
  # hand-launched A/B arm that lives in the same experiment tree but runs in
  # its own session, costing 13 minutes of training.  A path scope cannot
  # separate the two; the supervisor launches every torchrun inline (no
  # setsid), so its work is exactly its process group.
  local supervisor_pid=${1:-} pids=() pgid=''
  [[ ${supervisor_pid} =~ ^[1-9][0-9]*$ ]] || return
  pgid=$(ps -o pgid= -p "${supervisor_pid}" 2>/dev/null | tr -d ' ')
  if [[ ${pgid} =~ ^[1-9][0-9]*$ ]]; then
    mapfile -t pids < <(pgrep -g "${pgid}" -f '(^|/)torchrun ' || true)
    if (( ${#pids[@]} > 0 )); then
      kill -TERM "${pids[@]}" 2>/dev/null || true
      return
    fi
  fi
  kill -TERM "${supervisor_pid}" 2>/dev/null || true
}

previous_health=unknown
previous_completed_epoch=-1
previous_best_sha=''
last_recovery_unix=0
if [[ -s ${LATEST} ]]; then
  previous_health=$(jq -r '.health // "unknown"' "${LATEST}" 2>/dev/null || echo unknown)
  previous_completed_epoch=$(jq -r '.progress.latest_completed_epoch // -1' "${LATEST}" 2>/dev/null || echo -1)
  previous_best_sha=$(jq -r '.progress.best_checkpoint_sha256 // ""' "${LATEST}" 2>/dev/null || true)
  last_recovery_unix=$(jq -r '.recovery.last_attempt_unix // 0' "${LATEST}" 2>/dev/null || echo 0)
fi

while true; do
  now_unix=$(date +%s)
  timestamp=$(date --iso-8601=seconds)

  owner_alive=0
  valid_process_from_pid_file "${CAMPAIGN}/autonomous_campaign.pid" \
    run_plannerrft_autonomous_campaign.sh && owner_alive=1

  supervisor_alive=0
  supervisor_pid=''
  if valid_process_from_pid_file "${CAMPAIGN}/supervisor.pid" \
      run_plannerrft_full_to_epoch100.sh; then
    supervisor_alive=1
    supervisor_pid=$(<"${CAMPAIGN}/supervisor.pid")
  fi

  torchrun_count=$(pgrep -fc '(^|/)torchrun ' || true)
  awr_worker_count=$(pgrep -fc '(^|[ /])-m rlvr\.train_awr( |$)' || true)
  owner_failure_active=0
  [[ -s ${CAMPAIGN}/autonomous_failure_state.json ]] && owner_failure_active=1

  progress_record=$(latest_runtime_progress || true)
  progress_mtime=0
  progress_path=''
  if [[ -n ${progress_record} ]]; then
    progress_mtime=${progress_record%%$'\t'*}
    progress_mtime=${progress_mtime%.*}
    progress_path=${progress_record#*$'\t'}
  fi
  if (( progress_mtime > 0 )); then
    progress_age=$((now_unix - progress_mtime))
  else
    progress_age=${STALE_AFTER}
  fi

  disk_available_kib=$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')
  gpu_json=$(nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{for(i=1;i<=4;i++)gsub(/ /,"",$i); printf "%s{\"index\":%s,\"utilization_pct\":%s,\"memory_mib\":%s,\"power_w\":%s}",sep,$1,$2,$3,$4;sep=","}' \
    || true)

  completed_epoch=$(latest_completed_epoch || true)
  completed_epoch=${completed_epoch:--1}

  best_record=$(latest_best_checkpoint || true)
  best_path=''
  best_mtime=0
  best_sha=${previous_best_sha}
  if [[ -n ${best_record} ]]; then
    best_mtime=${best_record%%$'\t'*}
    best_mtime=${best_mtime%.*}
    best_path=${best_record#*$'\t'}
    previous_best_mtime=0
    [[ -s ${LATEST} ]] && previous_best_mtime=$(jq -r '.progress.best_checkpoint_mtime // 0' "${LATEST}" 2>/dev/null || echo 0)
    if [[ ${best_mtime} != "${previous_best_mtime}" || -z ${best_sha} ]]; then
      best_sha=$(sha256sum "${best_path}" | awk '{print $1}')
    fi
  fi

  health=healthy
  reason=running
  if (( disk_available_kib < MIN_DISK_AVAILABLE_KIB )); then
    health=critical
    reason=low_disk
  elif process_is_stopped "${supervisor_pid}"; then
    # Frozen on purpose: progress cannot advance and never will until someone
    # sends SIGCONT, so every stall test below would be true forever and
    # recovery would fire once per cooldown.  Suspend recovery instead.
    health=paused
    reason=supervisor_deliberately_stopped
  elif [[ ! -s ${COMPLETION} && ${owner_alive} -eq 0 && ${supervisor_alive} -eq 0 ]]; then
    health=critical
    reason=owner_and_supervisor_absent
  elif [[ ! -s ${COMPLETION} && ${owner_failure_active} -eq 1 && ${supervisor_alive} -eq 0 ]]; then
    health=critical
    reason=owner_repeated_failure
  elif [[ ! -s ${COMPLETION} && ${supervisor_alive} -eq 1 && ${progress_age} -ge ${STALE_AFTER} ]]; then
    health=critical
    reason=runtime_progress_stalled
  fi

  if [[ ${health} != "${previous_health}" ]]; then
    details=$(jq -cn \
      --arg reason "${reason}" --arg progress_path "${progress_path}" \
      --argjson progress_age "${progress_age}" \
      --argjson owner_alive "${owner_alive}" \
      --argjson supervisor_alive "${supervisor_alive}" \
      --argjson owner_failure_active "${owner_failure_active}" \
      --argjson torchrun_count "${torchrun_count}" \
      --argjson awr_worker_count "${awr_worker_count}" \
      --argjson disk_available_kib "${disk_available_kib}" \
      '{reason:$reason,progress_path:$progress_path,progress_age_seconds:$progress_age,
        owner_alive:($owner_alive==1),supervisor_alive:($supervisor_alive==1),
        owner_failure_active:($owner_failure_active==1),
        torchrun_count:$torchrun_count,awr_worker_count:$awr_worker_count,
        disk_available_kib:$disk_available_kib}')
    if [[ ${health} == healthy ]]; then
      emit_event info health_recovered "campaign health is normal" "${details}"
    elif [[ ${health} == paused ]]; then
      emit_event info campaign_paused \
        "supervisor is stopped; automatic recovery suspended" "${details}"
    else
      emit_event critical "${reason}" "campaign health requires automatic recovery" "${details}"
    fi
  fi

  if (( completed_epoch >= 0 && completed_epoch != previous_completed_epoch && completed_epoch % 10 == 0 )); then
    emit_event info ten_epoch_milestone \
      "completed global epoch ${completed_epoch}" \
      "$(jq -cn --argjson epoch "${completed_epoch}" '{epoch:$epoch}')"
  fi
  if [[ -n ${previous_best_sha} && -n ${best_sha} && ${best_sha} != "${previous_best_sha}" ]]; then
    emit_event info new_best_checkpoint "new train-selector best checkpoint" \
      "$(jq -cn --arg path "${best_path}" --arg sha256 "${best_sha}" \
        --argjson latest_completed_epoch "${completed_epoch}" \
        '{path:$path,sha256:$sha256,latest_completed_epoch:$latest_completed_epoch}')"
  fi

  if [[ ${health} == critical && ${reason} != low_disk && \
        ${reason} != owner_repeated_failure && \
        $((now_unix - last_recovery_unix)) -ge ${RECOVERY_COOLDOWN} ]]; then
    last_recovery_unix=${now_unix}
    if [[ ${reason} == owner_and_supervisor_absent ]]; then
      setsid -f env TARGET_EPOCH="${TARGET_EPOCH}" CAMPAIGN_OUT="${CAMPAIGN}" \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        bash "${AUTONOMOUS_OWNER}" \
        >> "${CAMPAIGN}/autonomous_campaign.launch.log" 2>&1 < /dev/null
      action=restarted_owner
    else
      terminate_stalled_work "${supervisor_pid}"
      action=terminated_stalled_stage_for_owner_recovery
    fi
    printf '%s %s reason=%s progress_age=%s\n' \
      "${timestamp}" "${action}" "${reason}" "${progress_age}" >> "${RECOVERY_LOG}"
    emit_event warning automatic_recovery "${action}" \
      "$(jq -cn --arg reason "${reason}" --arg action "${action}" \
        '{reason:$reason,action:$action}')"
  fi

  tmp=${LATEST}.tmp.$$
  jq -cn \
    --arg timestamp "${timestamp}" --arg health "${health}" --arg reason "${reason}" \
    --arg progress_path "${progress_path}" --arg best_path "${best_path}" \
    --arg best_sha "${best_sha}" \
    --argjson target_epoch "${TARGET_EPOCH}" \
    --argjson owner_alive "${owner_alive}" --argjson supervisor_alive "${supervisor_alive}" \
    --argjson owner_failure_active "${owner_failure_active}" \
    --argjson torchrun_count "${torchrun_count}" --argjson awr_worker_count "${awr_worker_count}" \
    --argjson progress_age "${progress_age}" --argjson stale_after "${STALE_AFTER}" \
    --argjson completed_epoch "${completed_epoch}" --argjson best_mtime "${best_mtime}" \
    --argjson disk_available_kib "${disk_available_kib}" \
    --argjson min_disk_available_kib "${MIN_DISK_AVAILABLE_KIB}" \
    --argjson last_recovery_unix "${last_recovery_unix}" --argjson gpus "[${gpu_json}]" \
    '{timestamp:$timestamp,health:$health,reason:$reason,target_epoch:$target_epoch,
      processes:{owner_alive:($owner_alive==1),supervisor_alive:($supervisor_alive==1),
        owner_failure_active:($owner_failure_active==1),torchrun_count:$torchrun_count,
        awr_worker_count:$awr_worker_count},
      progress:{latest_runtime_path:$progress_path,age_seconds:$progress_age,
        stale_after_seconds:$stale_after,latest_completed_epoch:$completed_epoch,
        best_checkpoint_path:$best_path,best_checkpoint_mtime:$best_mtime,
        best_checkpoint_sha256:$best_sha},
      storage:{available_kib:$disk_available_kib,critical_floor_kib:$min_disk_available_kib},
      recovery:{last_attempt_unix:$last_recovery_unix},gpus:$gpus}' > "${tmp}"
  mv "${tmp}" "${LATEST}"

  previous_health=${health}
  previous_completed_epoch=${completed_epoch}
  previous_best_sha=${best_sha}

  [[ -s ${COMPLETION} || ${ONCE} == 1 ]] && exit 0
  sleep "${INTERVAL}"
done
