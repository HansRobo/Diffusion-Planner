#!/usr/bin/env bash
# Decide the formal Cycle-1 branch only after complete paired epoch-3 evidence.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUN=${RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_replay_e02_e10/20260720-151500_plannerrft_full_cycle01_replay_e02_e10}
AUDIT=${RUN}/epoch_003_paired_audit.json
TRAIN_SUMMARY=${RUN}/train_epoch_003/summary.json
EVAL_SUMMARY=${RUN}/eval_epoch_003/summary.json
SELECTOR_SUMMARY=${RUN}/train_selector_epoch_003/summary.json
CHECKPOINT=${RUN}/epoch_003.pth
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_scored_prefix_active_expert_ablation
LOG=${RUN}/epoch_003_automatic_decision.log
LOCK=${RUN}/.epoch_003_automatic_decision.lock
ABLATION=${ROOT}/rlvr/autoresearch/run_plannerrft_scored_prefix_active_expert_ablation.sh

mkdir "${LOCK}" 2>/dev/null || exit 0
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

while [[ ! -s ${AUDIT} ]]; do
  if ! pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10' >/dev/null; then
    printf '%s formal training exited before epoch-3 paired audit\n' \
      "$(date --iso-8601=seconds)" >> "${LOG}"
    exit 1
  fi
  sleep 30
done

for required in "${TRAIN_SUMMARY}" "${EVAL_SUMMARY}" \
  "${SELECTOR_SUMMARY}" "${CHECKPOINT}"; do
  [[ -s ${required} ]] || {
    printf '%s paired audit appeared before required artifact %s\n' \
      "$(date --iso-8601=seconds)" "${required}" >> "${LOG}"
    exit 1
  }
done

decision=$(
  "${ROOT}/.venv/bin/python" - "${AUDIT}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
full = payload["full_validation"]["metrics"]["det_reward"]
selector = payload["fixed_train_selector"]["metrics"]["det_reward"]
full_ci_high = float(full["improvement_ci95"][1])
selector_ci_high = float(selector["improvement_ci95"][1])
if full_ci_high < 0.0 and selector_ci_high < 0.0:
    print("stop_and_ablate")
elif float(selector["improvement"]) > 0.0:
    print("continue_positive_primary")
else:
    print("continue_ambiguous")
PY
)

printf '%s decision=%s audit=%s\n' \
  "$(date --iso-8601=seconds)" "${decision}" "${AUDIT}" >> "${LOG}"
if [[ ${decision} != stop_and_ablate ]]; then
  exit 0
fi

[[ -x ${ABLATION} ]] || {
  printf '%s missing executable ablation runner %s\n' \
    "$(date --iso-8601=seconds)" "${ABLATION}" >> "${LOG}"
  exit 1
}

torchrun_pid=$(pgrep -fo 'torchrun.*--exp_name plannerrft_full_cycle01_replay_e02_e10' || true)
[[ -n ${torchrun_pid} ]] || {
  printf '%s formal torchrun already absent; not launching automatically\n' \
    "$(date --iso-8601=seconds)" >> "${LOG}"
  exit 1
}
pgid=$(ps -o pgid= -p "${torchrun_pid}" | tr -d ' ')
[[ ${pgid} =~ ^[1-9][0-9]*$ ]] || {
  printf '%s invalid formal process group for pid %s\n' \
    "$(date --iso-8601=seconds)" "${torchrun_pid}" >> "${LOG}"
  exit 1
}

printf '%s terminating significantly negative formal PGID=%s after safe epoch-3 close\n' \
  "$(date --iso-8601=seconds)" "${pgid}" >> "${LOG}"
kill -TERM -- "-${pgid}"

for _ in $(seq 1 120); do
  pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10' >/dev/null \
    || break
  sleep 5
done
if pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10' >/dev/null; then
  printf '%s formal ranks did not stop within 10 minutes; fail closed\n' \
    "$(date --iso-8601=seconds)" >> "${LOG}"
  exit 1
fi

mkdir -p "${OUT}"
launch_log=${OUT}/automatic_epoch3_transition.log
printf '%s launching score-aligned active-expert ablation log=%s\n' \
  "$(date --iso-8601=seconds)" "${launch_log}" >> "${LOG}"
setsid sg ubuntu -c "cd ${ROOT} && exec bash ${ABLATION}" \
  > "${launch_log}" 2>&1 < /dev/null &
printf '%s ablation launcher pid=%s\n' \
  "$(date --iso-8601=seconds)" "$!" >> "${LOG}"
