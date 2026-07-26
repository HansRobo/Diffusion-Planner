#!/usr/bin/env bash
# (Re)start the W&B campaign heartbeat under a keep-alive loop.  Called as a
# file so the daemon filename never appears in an interactive shell's command
# line (a raw pkill/pgrep there can match the caller itself or trip the
# sensitivity probes' competing-process guards).

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_jitterfix_e100}
RUN_NAME=${RUN_NAME:-jitterfix-campaign-e100-status}
CAMPAIGN=${RUN_ROOT}/campaign_e100
DAEMON=${ROOT}/rlvr/autoresearch/tools/campaign_wandb_heartbeat.py
LOG=${CAMPAIGN}/wandb_heartbeat.log
KEEPALIVE=${CAMPAIGN}/wandb_heartbeat_keepalive.sh

pkill -f "wandb_heartbeat_keepalive" 2>/dev/null || true
pkill -f "campaign_wandb_heartbeat.py" 2>/dev/null || true
sleep 2

cat > "${KEEPALIVE}" <<EOF
#!/usr/bin/env bash
while true; do
  nice -n 10 env PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}" WANDB_MODE=online \\
    "${ROOT}/.venv/bin/python" "${DAEMON}" \\
    --run-root "${RUN_ROOT}" --campaign "${CAMPAIGN}" \\
    --target-epoch 100 --interval 60 --run-name "${RUN_NAME}" >> "${LOG}" 2>&1
  status=\$?
  if [[ \${status} -eq 0 ]]; then exit 0; fi
  echo "\$(date '+%F %T') heartbeat exited status=\${status}; restarting in 30 s" >> "${LOG}"
  sleep 30
done
EOF
chmod +x "${KEEPALIVE}"

setsid nohup "${KEEPALIVE}" > /dev/null 2>&1 < /dev/null &
disown
sleep 8
if pgrep -f "campaign_wandb_heartbeat.py" > /dev/null; then
  echo "heartbeat daemon running; log: ${LOG}"
  tail -n 1 "${LOG}"
else
  echo "heartbeat failed to start; see ${LOG}" >&2
  exit 1
fi
