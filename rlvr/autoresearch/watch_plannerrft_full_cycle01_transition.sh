#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
MINE_PARENT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine
TRANSITION_LOCK=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_transition.lock

# Wait for the immutable cache commit.  A transient process failure cannot
# satisfy this condition because manifests are published only by strict close.
while true; do
  MINE_RUN=$(find "${MINE_PARENT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
  if [[ -n "${MINE_RUN}" ]] && \
     [[ $(find "${MINE_RUN}/replay_buffer" -mindepth 2 -maxdepth 2 -name manifest.json 2>/dev/null | wc -l) -eq 8 ]] && \
     [[ $(find "${MINE_RUN}/replay_buffer" -mindepth 2 -maxdepth 2 -name expert_anchor_manifest.json 2>/dev/null | wc -l) -eq 8 ]] && \
     [[ $(find "${MINE_RUN}/replay_buffer" -mindepth 2 -maxdepth 2 -name context_build_manifest.json 2>/dev/null | wc -l) -eq 8 ]]; then
    break
  fi
  sleep 30
done
printf '%s\n' "$$" > "${MINE_RUN}/transition_watcher.pid"

# Give the original launcher sole priority.  If it enters replay, there is
# nothing to recover.  If it exits at the boundary, take over exactly once.
while pgrep -f 'run_plannerrft_full_cycle01.sh' >/dev/null; do
  if pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10' >/dev/null; then
    exit 0
  fi
  sleep 10
done
if pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10' >/dev/null; then
  exit 0
fi

# The context-attachment owner normally starts replay.  This fallback is kept
# for crash recovery, but the lock guarantees that only one owner can build
# the overlay and create the formal replay output.
exec 9>"${TRANSITION_LOCK}"
flock -n 9 || exit 0
exec bash "${ROOT}/rlvr/autoresearch/continue_plannerrft_full_cycle01_after_mine.sh"
