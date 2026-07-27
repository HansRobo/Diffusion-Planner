#!/usr/bin/env bash
# Launch (or relaunch) the jitterfix campaign keeper.
#
# Exists as a *file* for the same reason launch_wandb_heartbeat.sh does: the
# keeper's ALIVE_PATTERN and the operator's pkill patterns are regexes over full
# command lines, so an interactive shell that merely *mentions* the entrypoint
# filename will (a) be killed by its own pkill and (b) make the keeper's
# supervisor_alive check return true, so it polls forever instead of launching.
# Both happened on 2026-07-26/27.  Keeping the names inside a script file means
# the caller's command line stays clean.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_jitterfix_v2_e100}
ENTRYPOINT=${ROOT}/rlvr/autoresearch/run_plannerrft_jitterfix_to_epoch100.sh
KEEPER=${ROOT}/rlvr/autoresearch/run_plannerrft_autonomous_campaign.sh
SUPERVISOR_NAME=run_plannerrft_full_to_epoch100
ENTRY_NAME=run_plannerrft_jitterfix_to_epoch100

[[ -x ${ENTRYPOINT} || -f ${ENTRYPOINT} ]] || { echo "missing entrypoint ${ENTRYPOINT}" >&2; exit 2; }
[[ -f ${KEEPER} ]] || { echo "missing keeper ${KEEPER}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/campaign_e100"

# Stop any previous keeper.  Bracket a character so this script's own command
# line cannot match.
pkill -f "run_plannerrft_autonomous_campaig[n]\.sh" 2>/dev/null || true
sleep 3

setsid env \
  RUN_ROOT="${RUN_ROOT}" \
  ENTRYPOINT="${ENTRYPOINT}" \
  ALIVE_PATTERN="${ENTRY_NAME}[.]sh|${SUPERVISOR_NAME}[.]sh" \
  CAMPAIGN_OUT="${RUN_ROOT}/campaign_e100" \
  PROGRESS_ROOT="${RUN_ROOT}/campaign_e100" \
  TARGET_EPOCH="${TARGET_EPOCH:-100}" \
  nohup sg ubuntu -c "bash ${KEEPER}" \
  >> "${RUN_ROOT}/keeper.launch.log" 2>&1 < /dev/null &
disown

sleep 8
if pgrep -f "run_plannerrft_autonomous_campaig[n]\.sh" > /dev/null; then
  echo "keeper launched; log: ${RUN_ROOT}/keeper.launch.log"
else
  echo "keeper failed to start; see ${RUN_ROOT}/keeper.launch.log" >&2
  exit 1
fi
