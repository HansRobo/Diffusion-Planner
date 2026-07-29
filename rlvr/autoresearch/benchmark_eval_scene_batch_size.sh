#!/usr/bin/env bash
# Benchmark full-validation packing without changing model/reward semantics.
#
# H100 BF16/Inductor inference is not byte deterministic. Two independent
# batch-512 full evaluations establish the natural repeat envelope; a larger
# batch is adopted only when it is at least 3% faster and its aggregate
# reward/quality/event changes remain inside three times that envelope.

set -Eeuo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 MODEL ARGS CONFIG SCENES REFERENCE_DIR REPEAT_DIR OUT" >&2
  exit 2
fi

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
TORCHRUN=${ROOT}/.venv/bin/torchrun
PYTHON=${ROOT}/.venv/bin/python
EVALUATOR=${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py
MODEL=$(realpath -e "$1")
ARGS=$(realpath -e "$2")
CONFIG=$(realpath -e "$3")
SCENES=$(realpath -e "$4")
REFERENCE_DIR=$(realpath -e "$5")
REPEAT_DIR=$(realpath -e "$6")
OUT=$7
RESULT=${OUT}/selection.json
BATCHES=(${BATCHES:-1024 2048 4096})
EXPECTED_SCENES=46262
MODEL_SHA256=$(sha256sum "${MODEL}" | awk '{print $1}')

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

die() {
  echo "eval-batch benchmark: $*" >&2
  exit 1
}

validate_eval_dir() {
  local directory=$1
  [[ -s ${directory}/manifest.json && -s ${directory}/summary.json && \
     -s ${directory}/scenes.json ]] || return 1
  jq -e --arg model "${MODEL}" --argjson expected "${EXPECTED_SCENES}" '
    .scene_contract_verified == true and .model == $model and
    .actual_scene_count == $expected and .expected_scene_count == $expected and
    .sample_steps == "10" and .candidate_count == "1" and
    .noise_scale == "0.0" and .checkpoint_payload == "ema_state_dict"
  ' "${directory}/manifest.json" >/dev/null
}

[[ -f ${EVALUATOR} && -x ${TORCHRUN} ]] || die "missing evaluator or torchrun"
validate_eval_dir "${REFERENCE_DIR}" || die "invalid batch-512 reference"
validate_eval_dir "${REPEAT_DIR}" || die "invalid batch-512 repeat"
mkdir -p "${OUT}"

for batch in "${BATCHES[@]}"; do
  [[ ${batch} =~ ^[1-9][0-9]*$ ]] || die "invalid batch ${batch}"
  arm=${OUT}/batch_${batch}
  elapsed=${arm}/elapsed_seconds.txt
  if validate_eval_dir "${arm}" && [[ -s ${elapsed} ]]; then
    continue
  fi
  if [[ -e ${arm} ]]; then
    stamp=$(date '+%Y%m%d-%H%M%S')
    mkdir -p "${OUT}/interrupted_attempts"
    mv "${arm}" "${OUT}/interrupted_attempts/batch_${batch}_${stamp}"
  fi
  mkdir -p "${arm}"
  set +e
  /usr/bin/time -f '%e' -o "${elapsed}" \
    "${TORCHRUN}" --standalone --nproc_per_node=8 \
      "${EVALUATOR}" \
      --model_path "${MODEL}" --args_path "${ARGS}" \
      --scenes "${SCENES}" --config "${CONFIG}" \
      --output_dir "${arm}" --label "eval_batch_${batch}_workers_1" \
      --batch_size "${batch}" --scene_load_workers 1 --sample_steps 10 \
      --amp_dtype bf16 --compile_mode default \
      > "${arm}/eval.log" 2>&1
  status=$?
  set -e
  if (( status != 0 )); then
    printf '%s\n' "${status}" > "${arm}/failed_exit_status.txt"
    continue
  fi
  "${PYTHON}" "${EVALUATOR}" --aggregate \
    --model_path "${MODEL}" --args_path "${ARGS}" \
    --scenes "${SCENES}" --config "${CONFIG}" \
    --output_dir "${arm}" --label "eval_batch_${batch}_workers_1" \
    >> "${arm}/eval.log" 2>&1
  validate_eval_dir "${arm}" || die "batch ${batch} produced an invalid result"
done

"${PYTHON}" - "${RESULT}" "${MODEL}" "${MODEL_SHA256}" "${ARGS}" \
  "${CONFIG}" "${SCENES}" "${REFERENCE_DIR}" "${REPEAT_DIR}" \
  "${OUT}" "${BATCHES[@]}" <<'PY'
import datetime
import json
import pathlib
import sys

result = pathlib.Path(sys.argv[1])
model = pathlib.Path(sys.argv[2]).resolve()
model_sha = sys.argv[3]
args = pathlib.Path(sys.argv[4]).resolve()
config = pathlib.Path(sys.argv[5]).resolve()
scenes = pathlib.Path(sys.argv[6]).resolve()
reference_dir = pathlib.Path(sys.argv[7]).resolve()
repeat_dir = pathlib.Path(sys.argv[8]).resolve()
root = pathlib.Path(sys.argv[9]).resolve()
batches = [int(value) for value in sys.argv[10:]]

metric_floors = {
    "mean_det_reward": 1.0e-4,
    "mean_ade": 1.0e-3,
    "mean_fde": 1.0e-3,
}
event_metrics = (
    "mean_det_collision",
    "mean_det_rb_crossing",
    "mean_det_kinematic_violation",
    "mean_det_lane_crossing",
)

reference = json.loads((reference_dir / "summary.json").read_text())
repeat = json.loads((repeat_dir / "summary.json").read_text())
scene_count = int(reference["scene_count"])
assert scene_count == int(repeat["scene_count"]) == 46262
ref_paths = [row["scene_path"] for row in json.loads((reference_dir / "scenes.json").read_text())]
repeat_paths = [row["scene_path"] for row in json.loads((repeat_dir / "scenes.json").read_text())]
assert ref_paths == repeat_paths

envelope = {}
for key, floor in metric_floors.items():
    envelope[key] = max(
        floor, 3.0 * abs(float(reference[key]) - float(repeat[key]))
    )
for key in event_metrics:
    repeat_count_delta = abs(
        round(float(reference[key]) * scene_count)
        - round(float(repeat[key]) * scene_count)
    )
    envelope[key] = max(5, 3 * repeat_count_delta) / scene_count

elapsed_path = reference_dir / "elapsed_seconds.txt"
if not elapsed_path.is_file():
    raise FileNotFoundError(elapsed_path)
baseline_elapsed = float(elapsed_path.read_text().strip())
arms = {
    "512": {
        "status": "reference",
        "elapsed_seconds": baseline_elapsed,
        "equivalent_to_repeat_envelope": True,
        "summary": str((reference_dir / "summary.json").resolve()),
    }
}
eligible = [(baseline_elapsed, 512)]
for batch in batches:
    arm = root / f"batch_{batch}"
    failed = arm / "failed_exit_status.txt"
    if failed.is_file():
        arms[str(batch)] = {
            "status": "failed",
            "exit_status": int(failed.read_text().strip()),
            "log": str((arm / "eval.log").resolve()),
        }
        continue
    summary_path = arm / "summary.json"
    rows_path = arm / "scenes.json"
    elapsed_path = arm / "elapsed_seconds.txt"
    if not (summary_path.is_file() and rows_path.is_file() and elapsed_path.is_file()):
        arms[str(batch)] = {"status": "incomplete"}
        continue
    summary = json.loads(summary_path.read_text())
    paths = [row["scene_path"] for row in json.loads(rows_path.read_text())]
    if paths != ref_paths:
        raise RuntimeError(f"batch {batch}: scene order differs")
    deltas = {key: float(summary[key]) - float(reference[key]) for key in envelope}
    passed = all(abs(deltas[key]) <= envelope[key] for key in envelope)
    elapsed = float(elapsed_path.read_text().strip())
    arms[str(batch)] = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "speedup_vs_batch512": baseline_elapsed / elapsed,
        "equivalent_to_repeat_envelope": passed,
        "metric_deltas_vs_reference": deltas,
        "summary": str(summary_path.resolve()),
    }
    if passed:
        eligible.append((elapsed, batch))

fastest_elapsed, fastest_batch = min(eligible)
selected_batch = (
    fastest_batch if fastest_elapsed <= 0.97 * baseline_elapsed else 512
)
payload = {
    "version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "contract": "full_validation_same_checkpoint_batch_packing_with_repeat_noise_envelope",
    "model": str(model),
    "model_sha256": model_sha,
    "args": str(args),
    "config": str(config),
    "scenes": str(scenes),
    "scene_count": scene_count,
    "reference_batch": 512,
    "reference_repeat": str(repeat_dir),
    "natural_repeat_envelope": envelope,
    "minimum_speedup_for_adoption": 1.0 / 0.97,
    "selected_batch_size": selected_batch,
    "selected_speedup_vs_batch512": (
        baseline_elapsed / arms[str(selected_batch)]["elapsed_seconds"]
    ),
    "arms": arms,
}
result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(result)
PY

echo "${RESULT}"
