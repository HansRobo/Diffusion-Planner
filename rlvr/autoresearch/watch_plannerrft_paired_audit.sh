#!/usr/bin/env bash
# Produce scene-paired Cycle-1 evidence as soon as one epoch closes.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
RUN=${RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_replay_e02_e10/20260720-151500_plannerrft_full_cycle01_replay_e02_e10}
EPOCH=${EPOCH:?set EPOCH to the replay epoch to audit}
# The watcher is reused by later replay ranges whose experiment names differ
# from the original Cycle-1 run.  Keep the historical default, but let the
# launcher provide the exact active process signature instead of silently
# declaring a healthy continuation dead.
PROCESS_PATTERN=${PROCESS_PATTERN:-rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10}
PADDED_EPOCH=$(printf '%03d' "${EPOCH}")
LOG=${RUN}/epoch_${PADDED_EPOCH}_paired_audit_watcher.log
EVAL_BASE=${RUN}/eval_epoch_000/scenes.json
EVAL_CANDIDATE=${RUN}/eval_epoch_${PADDED_EPOCH}/scenes.json
SELECTOR_BASE=${RUN}/train_selector_epoch_000/scenes.json
SELECTOR_CANDIDATE=${RUN}/train_selector_epoch_${PADDED_EPOCH}/scenes.json
EVAL_REPORT=${RUN}/eval_epoch_${PADDED_EPOCH}/paired_vs_epoch000.json
SELECTOR_REPORT=${RUN}/train_selector_epoch_${PADDED_EPOCH}/paired_vs_epoch000.json
SUMMARY=${RUN}/epoch_${PADDED_EPOCH}_paired_audit.json

export PYTHONPATH=${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

if [[ -s ${SUMMARY} ]]; then
  exit 0
fi

while [[ ! -s ${EVAL_CANDIDATE} || ! -s ${SELECTOR_CANDIDATE} ]]; do
  if ! pgrep -f -- "${PROCESS_PATTERN}" >/dev/null; then
    printf '%s training stopped before epoch-%s paired inputs closed\n' \
      "$(date --iso-8601=seconds)" "${EPOCH}" >> "${LOG}"
    exit 1
  fi
  sleep 30
done

nice -n 10 "${PYTHON}" -m rlvr.autoresearch.tools.full_paired_eval_report \
  --baseline "${EVAL_BASE}" --candidate "${EVAL_CANDIDATE}" \
  --epoch "${EPOCH}" --bootstrap 10000 --seed 3407 \
  --evidence_scope full_filtered_validation_46262 \
  --output "${EVAL_REPORT}" >> "${LOG}" 2>&1

nice -n 10 "${PYTHON}" -m rlvr.autoresearch.tools.full_paired_eval_report \
  --baseline "${SELECTOR_BASE}" --candidate "${SELECTOR_CANDIDATE}" \
  --epoch "${EPOCH}" --bootstrap 10000 --seed 3407 \
  --evidence_scope fixed_train_selector_65536 \
  --output "${SELECTOR_REPORT}" >> "${LOG}" 2>&1

"${PYTHON}" - "${EPOCH}" "${EVAL_REPORT}" "${SELECTOR_REPORT}" "${SUMMARY}" <<'PY'
import datetime
import json
import pathlib
import sys

epoch = int(sys.argv[1])
eval_path, selector_path, output_path = map(pathlib.Path, sys.argv[2:])


def compact(path):
    payload = json.loads(path.read_text())
    result = payload["epochs"][str(epoch)]
    metrics = result["metrics"]
    selected = {}
    for name in (
        "det_reward",
        "det_progress",
        "path_len",
        "ade",
        "fde",
        "det_collision",
        "det_rb_crossing",
        "det_kinematic_violation",
        "det_lane_crossing",
        "det_zero_reward",
        "det_smoothness",
    ):
        if name not in metrics:
            continue
        row = dict(metrics[name])
        if name in {
            "det_collision",
            "det_rb_crossing",
            "det_kinematic_violation",
            "det_lane_crossing",
            "det_zero_reward",
        }:
            count = int(row["paired_scene_count"])
            row["recovered_scene_count"] = round(
                float(row["improved_scene_fraction"]) * count
            )
            row["new_event_scene_count"] = round(
                float(row["regressed_scene_fraction"]) * count
            )
            row["net_event_scene_delta"] = (
                row["new_event_scene_count"] - row["recovered_scene_count"]
            )
        selected[name] = row
    return {
        "scene_count": int(result["scene_count"]),
        "bootstrap_samples": int(result["bootstrap_samples"]),
        "metrics": selected,
        "source_report": str(path.resolve()),
    }


payload = {
    "version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "epoch": epoch,
    "interpretation": (
        "paired evidence only; checkpoint promotion remains fixed train-selector "
        "mean deterministic reward"
    ),
    "full_validation": compact(eval_path),
    "fixed_train_selector": compact(selector_path),
}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
print(output_path)
PY
