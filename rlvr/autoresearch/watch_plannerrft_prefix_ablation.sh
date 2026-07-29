#!/usr/bin/env bash
# Monitor the matched scored-prefix/active-expert ablation, publish paired
# evidence, and start the already-prepared solver-step diagnosis only when both
# selectors statistically reject the ablation.  This watcher never promotes or
# replaces a checkpoint by itself.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_scored_prefix_active_expert_ablation}
LAUNCH_PATTERN='run_plannerrft_scored_prefix_active_expert_ablation.sh'
TRAIN_PATTERN='rlvr.train_awr.*plannerrft_scored_prefix40_active_expert_full_e02'
SOLVER=${ROOT}/rlvr/autoresearch/run_plannerrft_solver_step_sensitivity.sh
EXTENSION=${ROOT}/rlvr/autoresearch/run_plannerrft_prefix_to_epoch100.sh
WATCHER=${ROOT}/rlvr/autoresearch/watch_plannerrft_paired_audit.sh
LOG=${OUT}/automatic_paired_monitor.log
LOCK=${OUT}/.automatic_paired_monitor.lock

mkdir -p "${OUT}"
if ! mkdir "${LOCK}" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

printf '%s waiting for ablation run publication\n' \
  "$(date --iso-8601=seconds)" >> "${LOG}"

run=''
while [[ -z ${run} ]]; do
  run=$(find "${OUT}" -mindepth 2 -maxdepth 2 -type f \
    -name effective_config.json -printf '%h\n' 2>/dev/null \
    | sort | tail -n 1)
  if [[ -z ${run} ]] \
    && ! pgrep -f "${LAUNCH_PATTERN}|${TRAIN_PATTERN}|build_compressed_replay_context" \
      >/dev/null; then
    printf '%s ablation launcher ended before run publication\n' \
      "$(date --iso-8601=seconds)" >> "${LOG}"
    exit 1
  fi
  [[ -n ${run} ]] || sleep 30
done

printf '%s monitoring run=%s\n' "$(date --iso-8601=seconds)" "${run}" \
  >> "${LOG}"

while [[ ! -s ${run}/eval_epoch_002/scenes.json \
      || ! -s ${run}/train_selector_epoch_002/scenes.json ]]; do
  if ! pgrep -f "${LAUNCH_PATTERN}|${TRAIN_PATTERN}" >/dev/null; then
    printf '%s ablation stopped before epoch-2 paired inputs closed\n' \
      "$(date --iso-8601=seconds)" >> "${LOG}"
    exit 1
  fi
  sleep 30
done

RUN="${run}" EPOCH=2 bash "${WATCHER}" >> "${LOG}" 2>&1
audit=${run}/epoch_002_paired_audit.json
[[ -s ${audit} ]] || {
  printf '%s paired audit was not published\n' "$(date --iso-8601=seconds)" \
    >> "${LOG}"
  exit 1
}

read -r selector_mean selector_hi full_hi < <(
  "${PYTHON}" - "${audit}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
selector = payload["fixed_train_selector"]["metrics"]["det_reward"]
full = payload["full_validation"]["metrics"]["det_reward"]
print(
    float(selector["improvement"]),
    float(selector["improvement_ci95"][1]),
    float(full["improvement_ci95"][1]),
)
PY
)

decision=$("${PYTHON}" - "${selector_mean}" "${selector_hi}" "${full_hi}" <<'PY'
import sys

mean, selector_hi, full_hi = map(float, sys.argv[1:])
if mean > 0:
    print("positive_train_selector_start_extension")
elif selector_hi < 0 and full_hi < 0:
    print("rejected_start_solver_step_diagnosis")
else:
    print("ambiguous_start_solver_step_diagnosis")
PY
)

printf '%s decision=%s audit=%s selector_mean=%s selector_ci_hi=%s full_ci_hi=%s\n' \
  "$(date --iso-8601=seconds)" "${decision}" "${audit}" \
  "${selector_mean}" "${selector_hi}" "${full_hi}" >> "${LOG}"

if [[ ${decision} == positive_train_selector_start_extension ]]; then
  while pgrep -f "${LAUNCH_PATTERN}|${TRAIN_PATTERN}" >/dev/null; do
    sleep 10
  done
  extension_log=${OUT}/automatic_epoch100_extension.log
  setsid sg ubuntu -c "cd ${ROOT} && exec bash ${EXTENSION}" \
    > "${extension_log}" 2>&1 < /dev/null &
  printf '%s launched positive-recipe epoch-100 continuation pid=%s log=%s\n' \
    "$(date --iso-8601=seconds)" "$!" "${extension_log}" >> "${LOG}"
elif [[ ${decision} == rejected_start_solver_step_diagnosis \
    || ${decision} == ambiguous_start_solver_step_diagnosis ]]; then
  while pgrep -f "${LAUNCH_PATTERN}|${TRAIN_PATTERN}" >/dev/null; do
    sleep 10
  done
  solver_log=${OUT}/automatic_solver_step_sensitivity.log
  setsid sg ubuntu -c "cd ${ROOT} && exec bash ${SOLVER}" \
    > "${solver_log}" 2>&1 < /dev/null &
  printf '%s launched solver-step diagnosis pid=%s log=%s\n' \
    "$(date --iso-8601=seconds)" "$!" "${solver_log}" >> "${LOG}"
fi
