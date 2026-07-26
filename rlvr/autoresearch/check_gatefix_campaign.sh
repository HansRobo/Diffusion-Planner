#!/usr/bin/env bash
# One-command status verdict for the gate-fixed clean campaign.
# Prints the current phase, its progress, and an explicit STUCK/OK verdict —
# including the CPU-only phases where idle GPUs are expected and healthy.

set -uo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_gatefix_clean}
CAMPAIGN=${RUN_ROOT}/campaign_e100
TARGET_EPOCH=${TARGET_EPOCH:-100}

say() { printf '%s\n' "$*"; }

if [[ -s ${CAMPAIGN}/epoch${TARGET_EPOCH}_completion.json ]]; then
  say "VERDICT: COMPLETE — epoch ${TARGET_EPOCH} reached."
  jq -r '.final_selected_checkpoint' "${CAMPAIGN}/epoch${TARGET_EPOCH}_completion.json"
  exit 0
fi

keeper=$(pgrep -fc "run_plannerrft_autonomous_campaign[.]sh" || true)
entry=$(pgrep -fc "gatefix_clean_to_epoch100[.]sh" || true)
super=$(pgrep -fc "run_plannerrft_full_to_epoch100[.]sh" || true)
trainers=$(pgrep -fc "[-]m rlvr[.]train_awr" || true)
compressors=$(pgrep -fc "build_compressed_replay_contex[t]" || true)
evals=$(pgrep -fc "eval_awr_full_distribute[d]" || true)

say "processes: keeper=${keeper} entrypoint=${entry} supervisor=${super} trainers=${trainers} compressors=${compressors} eval=${evals}"

if [[ -s ${CAMPAIGN}/health_monitor_latest.json ]]; then
  say "health monitor: $(jq -r '"\(.health) (\(.reason)), progress age \(.progress.age_seconds)s / stale after \(.progress.stale_after_seconds)s"' "${CAMPAIGN}/health_monitor_latest.json")"
fi

phase="unknown"
detail=""
if (( trainers > 0 )); then
  latest_log=$(ls -t "${RUN_ROOT}"/cycle01_*.log "${CAMPAIGN}"/cycle*_{mine,replay}.log 2>/dev/null | head -n 1)
  phase="GPU training/mining"
  detail=$(grep -vE "convert_frame|Warning" "${latest_log}" 2>/dev/null | tail -n 1 | cut -c1-160)
  say "phase: ${phase} (${latest_log##*/})"
  say "latest: ${detail}"
elif (( compressors > 0 )); then
  phase="compressed-context build (CPU-only; idle GPUs are EXPECTED here)"
  say "phase: ${phase}"
  for log in "${RUN_ROOT}"/cycle01_mine/20*/replay_context_zstd_v1/logs/rank_0000.log; do
    [[ -s ${log} ]] && say "progress: $(tail -n 1 "${log}")"
  done
elif (( evals > 0 )); then
  say "phase: distributed evaluation / deployment audit (GPUs busy soon if not already)"
elif (( super > 0 || entry > 0 )); then
  say "phase: supervisor transition (validation/selection/sensitivity probes; short CPU-only gaps are normal)"
  say "latest status: $(tail -n 1 "${CAMPAIGN}/status.log" 2>/dev/null)"
fi

completed=$(wc -l < "${CAMPAIGN}/cycle_state.jsonl" 2>/dev/null || echo 0)
say "completed cycles: ${completed}/10"
[[ -s ${CAMPAIGN}/cycle_state.jsonl ]] && tail -n 1 "${CAMPAIGN}/cycle_state.jsonl" \
  | jq -r '"latest cycle \(.cycle): reward=\(.best_train_reward) delta=\(.best_train_reward_delta_from_start) kind=\(.selection_kind)"'

if (( keeper == 0 && super == 0 && entry == 0 )); then
  say "VERDICT: STUCK — no keeper, entrypoint, or supervisor process is alive."
  say "last attempt log:"
  ls -t "${CAMPAIGN}"/autonomous_attempt_*.log 2>/dev/null | head -n 1 | xargs tail -n 5 2>/dev/null
  exit 2
fi

if [[ -s ${CAMPAIGN}/health_monitor_latest.json ]] \
  && [[ $(jq -r '.health' "${CAMPAIGN}/health_monitor_latest.json") != healthy ]]; then
  say "VERDICT: DEGRADED — health monitor reports a problem (see above); the keeper should recover it, re-check in 10 minutes."
  exit 3
fi

say "VERDICT: OK — campaign is alive and progressing."
