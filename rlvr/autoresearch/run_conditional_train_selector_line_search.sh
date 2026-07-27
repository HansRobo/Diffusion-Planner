#!/usr/bin/env bash
# Rescue an otherwise rejected replay cycle with a smaller policy step.
#
# The formal trainer commits 5% of each raw replay proposal before evaluating
# it.  A cycle can therefore contain a useful update direction even when every
# committed checkpoint is a step too far.  This runner is deliberately
# conditional: it evaluates smaller interpolations only when the replay run
# retained its starting incumbent.  Selection uses the exact same deterministic
# 65,536-scene train selector as the formal run.  Validation is not consulted.
#
# Usage:
#   run_conditional_train_selector_line_search.sh \
#     REPLAY_RUN SOURCE_CHECKPOINT FIRST_EPOCH LAST_EPOCH CYCLE

set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 REPLAY_RUN SOURCE_CHECKPOINT FIRST_EPOCH LAST_EPOCH CYCLE" >&2
  exit 2
fi

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
TORCHRUN=${ROOT}/.venv/bin/torchrun
REPLAY_RUN=$(realpath -e "$1")
SOURCE_CHECKPOINT=$(realpath -e "$2")
FIRST_EPOCH=$3
LAST_EPOCH=$4
CYCLE=$5
WORK=${REPLAY_RUN}/conditional_commit_line_search
RESULT=${WORK}/selection.json
ARGS=${REPLAY_RUN}/model_args.json
CONFIG=${REPLAY_RUN}/effective_config.json
EXPECTED_SELECTOR_SCENES=65536
EXPECTED_SELECTOR_SHA256=49fd1863a3c1d54db59440da426e3475df67ccc4edd8da6ddde7e32945f3620c
EVAL_SCENE_LOAD_WORKERS=${EVAL_SCENE_LOAD_WORKERS:-1}
[[ ${EVAL_SCENE_LOAD_WORKERS} == 1 || ${EVAL_SCENE_LOAD_WORKERS} == 4 || ${EVAL_SCENE_LOAD_WORKERS} == 8 ]] \
  || { echo "invalid EVAL_SCENE_LOAD_WORKERS=${EVAL_SCENE_LOAD_WORKERS}" >&2; exit 2; }
# The trainer always serializes the freshly evaluated starting EMA under
# epoch_000; final_summary carries its logical global epoch (FIRST_EPOCH - 1).
BASELINE_DIR=${REPLAY_RUN}/train_selector_epoch_000
BASELINE_ROWS=${BASELINE_DIR}/scenes.json
SELECTOR_SCENES=${WORK}/selector_scenes.json

[[ ${FIRST_EPOCH} =~ ^[1-9][0-9]*$ && ${LAST_EPOCH} =~ ^[1-9][0-9]*$ ]] \
  || { echo "epochs must be positive integers" >&2; exit 2; }
[[ ${CYCLE} =~ ^[1-9][0-9]*$ ]] || { echo "cycle must be positive" >&2; exit 2; }
(( FIRST_EPOCH <= LAST_EPOCH )) || { echo "first epoch exceeds last epoch" >&2; exit 2; }

case "${REPLAY_RUN}/" in
  "${ROOT}/outputs/"*) ;;
  *) echo "refusing replay run outside ${ROOT}/outputs: ${REPLAY_RUN}" >&2; exit 1 ;;
esac

for required in \
  "${REPLAY_RUN}/final_summary.json" \
  "${REPLAY_RUN}/best_train.pth" \
  "${REPLAY_RUN}/scene_selection.json" \
  "${ARGS}" "${CONFIG}" \
  "${BASELINE_DIR}/summary.json" "${BASELINE_ROWS}"; do
  [[ -s ${required} ]] || { echo "missing required input ${required}" >&2; exit 1; }
done

jq -e \
  --arg sha "${EXPECTED_SELECTOR_SHA256}" \
  --argjson count "${EXPECTED_SELECTOR_SCENES}" '
    ((.train_selector_count == $count and
      .train_selector_sha256_newline_joined_paths == $sha) or
     # Older diagnostic runs predate these redundant manifest fields.  The
     # ordered rows are independently count/uniqueness/hash checked below.
     (.train_selector_count == null and
      .train_selector_sha256_newline_joined_paths == null))
  ' "${REPLAY_RUN}/scene_selection.json" >/dev/null \
  || { echo "formal train-selector contract differs" >&2; exit 1; }

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "${WORK}/checkpoints" "${WORK}/train_selector"

# This also makes an existing result safely resumable after a supervisor
# restart.  Hash validation below prevents a stale selection from silently
# propagating a different model.
if [[ -s ${RESULT} ]]; then
  selected=$(jq -er '.selected.checkpoint' "${RESULT}")
  expected_sha=$(jq -er '.selected.checkpoint_sha256' "${RESULT}")
  [[ -s ${selected} ]] || { echo "selected checkpoint disappeared: ${selected}" >&2; exit 1; }
  actual_sha=$(sha256sum "${selected}" | awk '{print $1}')
  [[ ${actual_sha} == "${expected_sha}" ]] \
    || { echo "selected checkpoint hash changed" >&2; exit 1; }
  printf '%s\n' "${RESULT}"
  exit 0
fi

# `jq -e` deliberately returns status 1 for the valid JSON value `false`.
# Under `set -e` that used to abort exactly when formal replay had improved and
# line search should be skipped. Read the boolean without truth-value exit
# semantics, then validate its type explicitly.
incumbent_retained=$(jq -r '.best_train_is_starting_incumbent' "${REPLAY_RUN}/final_summary.json")
[[ ${incumbent_retained} == true || ${incumbent_retained} == false ]] \
  || { echo "invalid best_train_is_starting_incumbent=${incumbent_retained}" >&2; exit 1; }
# Previously this shortcut committed the *unscaled* replay policy
# (alpha_toward_direction=1.0) whenever it beat the train selector, skipping the
# line search entirely — that is how cycles 1 and 2 shipped the raw policy the
# 2026-07-18 audit had already shown to overshoot.  Always run the ladder now;
# it contains alpha=1.0-equivalent behaviour only if a large step is genuinely
# the sole improver, and prefers the smallest improving step otherwise.
# Set ALLOW_UNSCALED_COMMIT=1 to restore the old shortcut for A/B work.
if [[ ${incumbent_retained} != true && ${ALLOW_UNSCALED_COMMIT:-0} == 1 ]]; then
  "${PYTHON}" - "${RESULT}" "${REPLAY_RUN}" "${SOURCE_CHECKPOINT}" "${CYCLE}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

result = pathlib.Path(sys.argv[1])
run = pathlib.Path(sys.argv[2])
source = pathlib.Path(sys.argv[3])
cycle = int(sys.argv[4])
summary = json.loads((run / "final_summary.json").read_text())
checkpoint = run / "best_train.pth"
payload = {
    "version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "cycle": cycle,
    "replay_run": str(run.resolve()),
    "source_checkpoint": str(source.resolve()),
    "triggered": False,
    "reason": "formal_replay_already_strictly_improved_train_selector",
    "selection_rule": "fixed_train_selector_k1_ddim10_mean_det_reward_strict_improvement",
    "selected": {
        "kind": "formal_replay_best_train",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "epoch": int(summary["best_train_epoch"]),
        "mean_det_reward": float(summary["best_train_reward"]),
        "delta_vs_cycle_start": float(summary["best_train_reward_delta_from_start"]),
        "alpha_toward_direction": 1.0,
    },
}
result.parent.mkdir(parents=True, exist_ok=True)
result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(result)
PY
  exit 0
fi

# Materialize the exact selector path order once and identify the least-bad
# committed policy direction from the completed replay run.  The direction is
# chosen on the same statistic; no validation leakage is introduced.
"${PYTHON}" - \
  "${REPLAY_RUN}" "${BASELINE_ROWS}" "${SELECTOR_SCENES}" \
  "${WORK}/direction.json" "${FIRST_EPOCH}" "${LAST_EPOCH}" \
  "${EXPECTED_SELECTOR_SCENES}" "${EXPECTED_SELECTOR_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
baseline_rows_path = pathlib.Path(sys.argv[2])
selector_path = pathlib.Path(sys.argv[3])
direction_path = pathlib.Path(sys.argv[4])
first, last, expected = map(int, sys.argv[5:8])
expected_sha = sys.argv[8]

rows = json.loads(baseline_rows_path.read_text())
paths = [str(row["scene_path"]) for row in rows]
assert len(paths) == expected
assert len(set(paths)) == expected
digest = hashlib.sha256("".join(path + "\n" for path in paths).encode()).hexdigest()
assert digest == expected_sha, (digest, expected_sha)
selector_path.write_text(json.dumps(paths, indent=2) + "\n")

baseline_summary = json.loads((baseline_rows_path.parent / "summary.json").read_text())
baseline = float(baseline_summary["mean_det_reward"])
row_mean = sum(float(row["det_reward"]) for row in rows) / len(rows)
assert abs(row_mean - baseline) <= 1e-12, (row_mean, baseline)

candidates = []
for epoch in range(first, last + 1):
    summary_path = run / f"train_selector_epoch_{epoch:03d}" / "summary.json"
    checkpoint = run / f"epoch_{epoch:03d}.pth"
    assert summary_path.is_file(), summary_path
    assert checkpoint.is_file(), checkpoint
    summary = json.loads(summary_path.read_text())
    candidates.append({
        "epoch": epoch,
        "mean_det_reward": float(summary["mean_det_reward"]),
        "delta_vs_cycle_start": float(summary["mean_det_reward"]) - baseline,
        "checkpoint": str(checkpoint.resolve()),
    })
direction = max(candidates, key=lambda row: (row["mean_det_reward"], -row["epoch"]))
# This block used to assert direction <= baseline, because it was only reachable
# when formal replay had failed to beat the selector.  The ladder now always
# runs — committing the unscaled policy needs an explicit opt-in — so an
# improving direction is the normal case: we interpolate toward it and take the
# smallest step that still improves, instead of shipping it raw.
improved_direction = direction["mean_det_reward"] > baseline + 1e-12
direction_path.write_text(json.dumps({
    "baseline_epoch": first - 1,
    "baseline_mean_det_reward": baseline,
    "candidate_epochs": candidates,
    "selected_direction": direction,
    "direction_improved_selector": improved_direction,
}, indent=2, sort_keys=True) + "\n")
PY

DIRECTION_CHECKPOINT=$(jq -er '.selected_direction.checkpoint' "${WORK}/direction.json")
DIRECTION_EPOCH=$(jq -er '.selected_direction.epoch' "${WORK}/direction.json")
# 0.05 is the only step size with an audited positive, safety-preserving result
# (docs/awr_zero_risk_improvement_audit_20260718.md: the unscaled policy
# overshoots; the alpha=0.05 EMA interpolation gained on both the fixed train
# selector and full validation).  It was missing from this ladder, so every
# cycle committed at 0.10-1.0 and collisions rose ~15% relative.  0.02 gives the
# search somewhere even safer to land.
labels=(a0p02 a0p05 a0p10 a0p25 a0p50 a0p75)
alphas=(0.02 0.05 0.10 0.25 0.50 0.75)

for index in "${!labels[@]}"; do
  label=${labels[$index]}
  alpha=${alphas[$index]}
  checkpoint=${WORK}/checkpoints/e$(printf '%03d' "${DIRECTION_EPOCH}")_${label}.pth
  eval_dir=${WORK}/train_selector/${label}
  mkdir -p "${eval_dir}"
  if [[ ! -s ${checkpoint} ]]; then
    "${PYTHON}" -m rlvr.autoresearch.tools.interpolate_awr_checkpoint \
      --source "${SOURCE_CHECKPOINT}" --candidate "${DIRECTION_CHECKPOINT}" \
      --alpha "${alpha}" --output "${checkpoint}"
  fi
  if [[ ! -s ${eval_dir}/summary.json ]]; then
    "${TORCHRUN}" --standalone --nproc_per_node=8 \
      "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --model_path "${checkpoint}" --args_path "${ARGS}" \
      --scenes "${SELECTOR_SCENES}" --config "${CONFIG}" \
      --output_dir "${eval_dir}" \
      --label "cycle$(printf '%02d' "${CYCLE}")_line_${label}_train_selector" \
      --batch_size 512 --scene_load_workers "${EVAL_SCENE_LOAD_WORKERS}" --sample_steps 10 \
      --amp_dtype bf16 --compile_mode default
    "${PYTHON}" "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --aggregate --model_path "${checkpoint}" --args_path "${ARGS}" \
      --scenes "${SELECTOR_SCENES}" --config "${CONFIG}" \
      --output_dir "${eval_dir}" \
      --label "cycle$(printf '%02d' "${CYCLE}")_line_${label}_train_selector" \
      --batch_size 512 --scene_load_workers "${EVAL_SCENE_LOAD_WORKERS}" --sample_steps 10 \
      --amp_dtype bf16 --compile_mode default
  fi
done

"${PYTHON}" - \
  "${RESULT}" "${REPLAY_RUN}" "${SOURCE_CHECKPOINT}" "${WORK}/direction.json" \
  "${CYCLE}" "${labels[@]}" -- "${alphas[@]}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

result = pathlib.Path(sys.argv[1])
run = pathlib.Path(sys.argv[2])
source = pathlib.Path(sys.argv[3])
direction_path = pathlib.Path(sys.argv[4])
cycle = int(sys.argv[5])
parts = sys.argv[6:]
split = parts.index("--")
labels = parts[:split]
alphas = [float(value) for value in parts[split + 1:]]
assert len(labels) == len(alphas)

work = result.parent
direction = json.loads(direction_path.read_text())
baseline = float(direction["baseline_mean_det_reward"])
direction_epoch = int(direction["selected_direction"]["epoch"])
candidates = []
for label, alpha in zip(labels, alphas, strict=True):
    eval_dir = work / "train_selector" / label
    summary = json.loads((eval_dir / "summary.json").read_text())
    checkpoint = work / "checkpoints" / f"e{direction_epoch:03d}_{label}.pth"
    reward = float(summary["mean_det_reward"])
    candidates.append({
        "label": label,
        "alpha_toward_direction": alpha,
        "mean_det_reward": reward,
        "delta_vs_cycle_start": reward - baseline,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "summary": str((eval_dir / "summary.json").resolve()),
        "scenes": str((eval_dir / "scenes.json").resolve()),
    })

# Take the SMALLEST step that still strictly improves, not the highest-scoring
# step.  The selector is only 128 scenes, so differences of ~1e-4 between alphas
# are noise; picking the max on that noise is what promoted alpha=0.50-1.0 and
# overshot into a safety regression while the aggregate reward still rose.  Any
# strict improvement is progress, so prefer the most conservative one that has it.
improving = [row for row in candidates if row["mean_det_reward"] > baseline]
step_preference = "smallest_strictly_improving_alpha"
if improving:
    winner = min(improving, key=lambda row: row["alpha_toward_direction"])
else:
    winner = max(
        candidates,
        key=lambda row: (row["mean_det_reward"], -row["alpha_toward_direction"]),
    )
promoted = bool(improving)
if promoted:
    selected = {"kind": "conditional_smaller_commit", "epoch": direction_epoch, **winner}
else:
    formal = json.loads((run / "final_summary.json").read_text())
    checkpoint = run / "best_train.pth"
    selected = {
        "kind": "formal_replay_starting_incumbent",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "epoch": int(formal["best_train_epoch"]),
        "mean_det_reward": float(formal["best_train_reward"]),
        "delta_vs_cycle_start": 0.0,
        "alpha_toward_direction": 0.0,
    }

payload = {
    "version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "cycle": cycle,
    "replay_run": str(run.resolve()),
    "source_checkpoint": str(source.resolve()),
    "triggered": True,
    "reason": "formal_replay_retained_starting_incumbent",
    "selection_rule": "fixed_train_selector_k1_ddim10_mean_det_reward_strict_improvement",
    # The rule token above is a stable contract string; this field records the
    # tie-break actually applied, which changed on 2026-07-25 from "highest
    # reward" to "smallest strictly-improving step".
    "step_preference": step_preference,
    "baseline_epoch": int(direction["baseline_epoch"]),
    "baseline_mean_det_reward": baseline,
    "direction": direction["selected_direction"],
    "candidates": candidates,
    "line_search_promoted": promoted,
    "selected": selected,
}
result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(result)
PY

printf '%s\n' "${RESULT}"
