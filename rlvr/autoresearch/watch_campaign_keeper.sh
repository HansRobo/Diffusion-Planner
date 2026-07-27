#!/usr/bin/env bash
# Guard the keeper, not the trainers.  The keeper already restarts a dead
# supervisor; nothing restarts the keeper, and it has silently exited twice
# (once holding a stale flock, once when its supervisor was killed out from
# under it).  This loop is the outermost layer.
#
# Every pattern here is bracketed so this script's own command line -- and any
# shell that merely mentions the filename -- cannot match.  Unbracketed patterns
# cost 13 minutes of a dead campaign that looked alive on 2026-07-27.
#
# Counting must never produce an empty string: `pgrep -c ... || echo 0` emits
# "0\n0" when pgrep both prints 0 and exits non-zero, and the arithmetic that
# consumes it then dies with a syntax error, taking the watchdog with it.  Use
# `|| true` and a numeric default instead.

set -uo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LAUNCHER=${ROOT}/rlvr/autoresearch/launch_jitterfix_campaign.sh
INTERVAL=${INTERVAL:-300}
LOG=${RUN_ROOT}/keeper_watchdog.log
KEEPER_PAT='run_plannerrft_autonomous_campaig[n]\.sh'

mkdir -p "${RUN_ROOT}"
exec 8>"${RUN_ROOT}/keeper_watchdog.lock"
flock -n 8 || { echo "another watchdog holds the lock" >&2; exit 0; }

log() { printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "${LOG}"; }

count() {
  local n
  n=$(pgrep -cf "$1" 2>/dev/null || true)
  [[ ${n} =~ ^[0-9]+$ ]] || n=0
  printf '%s' "${n}"
}

log "watchdog armed pid=$$ run_root=${RUN_ROOT} interval=${INTERVAL}s"
while true; do
  keepers=$(count "${KEEPER_PAT}")
  if (( keepers == 0 )); then
    log "keeper absent; relaunching"
    RUN_ROOT="${RUN_ROOT}" setsid nohup bash "${LAUNCHER}" \
      >> "${RUN_ROOT}/keeper_watchdog_relaunch.log" 2>&1 < /dev/null || \
      log "relaunch invocation failed status=$?"
    sleep 60
    keepers=$(count "${KEEPER_PAT}")
    log "post-relaunch keepers=${keepers}"
  fi
  sleep "${INTERVAL}"
done
