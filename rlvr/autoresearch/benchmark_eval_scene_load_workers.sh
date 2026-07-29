#!/usr/bin/env bash
# Select validation NPZ-loader concurrency without changing model semantics.
# Every arm evaluates the exact same selected EMA checkpoint on the complete
# 46,262-scene validation list. Adoption requires byte-identical scene rows.

set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MODEL ARGS CONFIG SCENES OUT" >&2
  exit 2
fi

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
TORCHRUN=${ROOT}/.venv/bin/torchrun
PYTHON=${ROOT}/.venv/bin/python
MODEL=$(realpath -e "$1")
ARGS=$(realpath -e "$2")
CONFIG=$(realpath -e "$3")
SCENES=$(realpath -e "$4")
OUT=$5
RESULT=${OUT}/selection.json
EXPECTED_SCENES=46262
MODEL_SHA256=$(sha256sum "${MODEL}" | awk '{print $1}')
WORKERS=(1 4 8)

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

die() {
  echo "eval-loader benchmark: $*" >&2
  exit 1
}

if [[ -s ${RESULT} ]]; then
  jq -e --arg model "${MODEL}" --arg sha "${MODEL_SHA256}" '
    .version == 1 and .model == $model and .model_sha256 == $sha and
    (.selected_workers == 1 or .selected_workers == 4 or .selected_workers == 8) and
    .scene_count == 46262 and
    (.scene_outputs_byte_identical == true or .selected_workers == 1)
  ' "${RESULT}" >/dev/null || die "existing selection differs"
  echo "${RESULT}"
  exit 0
fi

mkdir -p "${OUT}"
for workers in "${WORKERS[@]}"; do
  arm=${OUT}/workers_${workers}
  elapsed=${arm}/elapsed_seconds.txt
  if [[ ! -s ${arm}/manifest.json ]]; then
    [[ ! -e ${arm} ]] || die "incomplete arm ${arm}"
    mkdir -p "${arm}"
    /usr/bin/time -f '%e' -o "${elapsed}" \
      "${TORCHRUN}" --standalone --nproc_per_node=8 \
        "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
        --model_path "${MODEL}" --args_path "${ARGS}" \
        --scenes "${SCENES}" --config "${CONFIG}" \
        --output_dir "${arm}" --label "eval_loader_workers_${workers}" \
        --batch_size 512 --scene_load_workers "${workers}" --sample_steps 10 \
        --amp_dtype bf16 --compile_mode default \
        > "${arm}/eval.log" 2>&1
    "${PYTHON}" "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --aggregate --model_path "${MODEL}" --args_path "${ARGS}" \
      --scenes "${SCENES}" --config "${CONFIG}" \
      --output_dir "${arm}" --label "eval_loader_workers_${workers}" \
      >> "${arm}/eval.log" 2>&1
  fi
  [[ -s ${elapsed} && -s ${arm}/scenes.json ]] \
    || die "arm ${workers} lacks timing or scenes"
  jq -e --argjson expected "${EXPECTED_SCENES}" '
    .scene_contract_verified == true and
    .actual_scene_count == $expected and .expected_scene_count == $expected and
    .sample_steps == "10" and .candidate_count == "1" and
    .noise_scale == "0.0" and .checkpoint_payload == "ema_state_dict"
  ' "${arm}/manifest.json" >/dev/null || die "arm ${workers} contract differs"
done

"${PYTHON}" - "${OUT}" "${RESULT}" "${MODEL}" "${MODEL_SHA256}" \
  "${ARGS}" "${CONFIG}" "${SCENES}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
result = pathlib.Path(sys.argv[2])
model = pathlib.Path(sys.argv[3]).resolve()
model_sha = sys.argv[4]
args = pathlib.Path(sys.argv[5]).resolve()
config = pathlib.Path(sys.argv[6]).resolve()
scenes = pathlib.Path(sys.argv[7]).resolve()

arms = {}
reference_sha = None
identical = True
for workers in (1, 4, 8):
    arm = root / f"workers_{workers}"
    scene_path = arm / "scenes.json"
    digest = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if reference_sha is None:
        reference_sha = digest
    identical = identical and digest == reference_sha
    elapsed = float((arm / "elapsed_seconds.txt").read_text().strip())
    if not elapsed > 0:
        raise ValueError(f"invalid elapsed time for worker={workers}: {elapsed}")
    arms[str(workers)] = {
        "elapsed_seconds": elapsed,
        "scenes_sha256": digest,
        "manifest": str((arm / "manifest.json").resolve()),
    }

fastest = min((1, 4, 8), key=lambda value: arms[str(value)]["elapsed_seconds"])
baseline = arms["1"]["elapsed_seconds"]
fastest_elapsed = arms[str(fastest)]["elapsed_seconds"]
# Avoid spending more CPU for a timing tie. A faster arm must save at least 3%
# of end-to-end full-evaluation wall time; otherwise preserve worker=1. Any
# cross-process numerical difference also falls back to worker=1 instead of
# blocking the epoch-100 campaign.
selected = fastest if identical and fastest_elapsed <= 0.97 * baseline else 1
payload = {
    "version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "contract": "paired_full_validation_same_checkpoint_byte_identical_rows",
    "model": str(model),
    "model_sha256": model_sha,
    "args": str(args),
    "config": str(config),
    "scenes": str(scenes),
    "scene_count": 46262,
    "scene_outputs_byte_identical": identical,
    "selected_workers": selected,
    "speedup_vs_worker1": baseline / arms[str(selected)]["elapsed_seconds"],
    "arms": arms,
}
result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(result)
PY

echo "${RESULT}"
