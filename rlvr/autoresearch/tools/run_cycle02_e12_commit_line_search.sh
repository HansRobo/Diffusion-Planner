#!/usr/bin/env bash
set -euo pipefail

# Cheap trust-region line search along the already-computed epoch-12 proposal.
# The epoch-12 checkpoint is a 5% interpolation of the raw proposal. These
# alpha values interpolate from the incumbent toward that accepted checkpoint,
# corresponding to raw-proposal rates 0.5%, 1.25%, 2.5%, and 3.75%.
# Selection is fixed train-set K=1 mean_det_reward; full validation runs only
# if a candidate strictly exceeds the incumbent.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
SOURCE_RUN=${OUT}/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100
SOURCE=${SOURCE_RUN}/best_model_awr_retained_e004_a0p05.pth
E12_RUN=${OUT}/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache
CANDIDATE=${E12_RUN}/epoch_012.pth
ARGS=${E12_RUN}/model_args.json
CONFIG=${E12_RUN}/effective_config.json
WORK=${E12_RUN}/commit_rate_line_search
SELECTOR_SCENES=${WORK}/selector_scenes.json
VALID_SCENES=${SOURCE_RUN}/scene_selection_valid.json
BASELINE_SELECTOR=${E12_RUN}/train_selector_epoch_000/scenes.json
BASELINE_VALID=${SOURCE_RUN}/deployment_full10/e004_a0p05/scenes.json

cd "${ROOT}"
mkdir -p "${WORK}/checkpoints"

pgrep -af 'rlvr.train_awr|eval_awr_full_distributed.py' > "${WORK}/active_gpu_jobs.txt" || true
if [[ -s ${WORK}/active_gpu_jobs.txt ]]; then
  echo "refusing line search while another AWR train/eval job is active:" >&2
  cat "${WORK}/active_gpu_jobs.txt" >&2
  exit 1
fi

"${ROOT}/.venv/bin/python" - "${BASELINE_SELECTOR}" "${SELECTOR_SCENES}" <<'PY'
import json
import pathlib
import sys

rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
paths = [str(row["scene_path"]) for row in rows]
assert len(paths) == 65_536
assert len(set(paths)) == len(paths)
pathlib.Path(sys.argv[2]).write_text(json.dumps(paths, indent=2) + "\n")
PY

labels=(a0p10 a0p25 a0p50 a0p75)
alphas=(0.10 0.25 0.50 0.75)
raw_rates=(0.005 0.0125 0.025 0.0375)

for i in "${!labels[@]}"; do
  label=${labels[$i]}
  alpha=${alphas[$i]}
  checkpoint=${WORK}/checkpoints/e12_${label}.pth
  eval_dir=${WORK}/train_selector/${label}
  mkdir -p "${eval_dir}"
  if [[ ! -s ${checkpoint} ]]; then
    PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
      -m rlvr.autoresearch.tools.interpolate_awr_checkpoint \
      --source "${SOURCE}" --candidate "${CANDIDATE}" \
      --alpha "${alpha}" --output "${checkpoint}"
  fi
  if [[ ! -s ${eval_dir}/summary.json ]]; then
    sg ubuntu -c "cd '${ROOT}' && env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 '${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py' --model_path '${checkpoint}' --args_path '${ARGS}' --scenes '${SELECTOR_SCENES}' --config '${CONFIG}' --output_dir '${eval_dir}' --label cycle02_e12_${label}_train_selector --batch_size 512 --scene_load_workers 1 --sample_steps 10 --amp_dtype bf16 --compile_mode default"
    PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
      "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --model_path "${checkpoint}" --args_path "${ARGS}" \
      --scenes "${SELECTOR_SCENES}" --config "${CONFIG}" \
      --output_dir "${eval_dir}" --label "cycle02_e12_${label}_train_selector" \
      --batch_size 512 --scene_load_workers 1 --sample_steps 10 \
      --amp_dtype bf16 --compile_mode default --aggregate
  fi
done

"${ROOT}/.venv/bin/python" - "${WORK}" "${SOURCE}" "${BASELINE_SELECTOR}" \
  "${labels[@]}" -- "${alphas[@]}" -- "${raw_rates[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

work = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
baseline_rows = json.loads(pathlib.Path(sys.argv[3]).read_text())
parts = sys.argv[4:]
sep1 = parts.index("--")
sep2 = parts.index("--", sep1 + 1)
labels = parts[:sep1]
alphas = [float(value) for value in parts[sep1 + 1 : sep2]]
raw_rates = [float(value) for value in parts[sep2 + 1 :]]
assert len(labels) == len(alphas) == len(raw_rates)

baseline = sum(float(row["det_reward"]) for row in baseline_rows) / len(baseline_rows)
candidates = []
for label, alpha, raw_rate in zip(labels, alphas, raw_rates, strict=True):
    summary = json.loads((work / "train_selector" / label / "summary.json").read_text())
    checkpoint = work / "checkpoints" / f"e12_{label}.pth"
    paired_report = work / "train_selector" / label / "paired_vs_incumbent.json"
    candidates.append(
        {
            "label": label,
            "alpha_toward_e12_accepted": alpha,
            "effective_raw_proposal_rate": raw_rate,
            "mean_det_reward": float(summary["mean_det_reward"]),
            "delta_vs_incumbent": float(summary["mean_det_reward"]) - baseline,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "summary": str((work / "train_selector" / label / "summary.json").resolve()),
            "paired_report": str(paired_report.resolve()) if paired_report.is_file() else None,
        }
    )
winner = max(candidates, key=lambda row: row["mean_det_reward"])
winner["paired_report"] = str(
    (work / "train_selector" / winner["label"] / "paired_vs_incumbent.json").resolve()
)
promoted = winner if winner["mean_det_reward"] > baseline else None
selection = {
    "selection_rule": "fixed_train_selector_k1_mean_det_reward_strict_improvement",
    "incumbent_mean_det_reward": baseline,
    "incumbent_checkpoint": str(source.resolve()),
    "candidates": candidates,
    "best_line_search_candidate": winner,
    "promoted_candidate": promoted,
}
(work / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
print(json.dumps(selection, indent=2))
PY

winner=$(jq -r '.promoted_candidate.checkpoint // empty' "${WORK}/selection.json")
winner_label=$(jq -r '.promoted_candidate.label // empty' "${WORK}/selection.json")
best_line_label=$(jq -r '.best_line_search_candidate.label' "${WORK}/selection.json")
best_line_dir=${WORK}/train_selector/${best_line_label}

# The exact full-selector mean above is the selection statistic.  A 10,000-sample
# paired bootstrap is diagnostic only, so run it once for the best line-search
# point instead of four times on candidates that cannot be selected.
if [[ ! -s ${best_line_dir}/paired_vs_incumbent.json ]]; then
  PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
    -m rlvr.autoresearch.tools.full_paired_eval_report \
    --baseline "${BASELINE_SELECTOR}" \
    --candidate "${best_line_dir}/scenes.json" \
    --epoch 12 --bootstrap 10000 --seed 3407 \
    --evidence_scope fixed_train_selector_k1 \
    --metrics det_reward \
    --output "${best_line_dir}/paired_vs_incumbent.json"
fi

if [[ -n ${winner} ]]; then
  valid_dir=${WORK}/full_validation/${winner_label}
  mkdir -p "${valid_dir}"
  sg ubuntu -c "cd '${ROOT}' && env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 '${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py' --model_path '${winner}' --args_path '${ARGS}' --scenes '${VALID_SCENES}' --config '${CONFIG}' --output_dir '${valid_dir}' --label cycle02_e12_${winner_label}_full_validation --batch_size 512 --scene_load_workers 1 --sample_steps 10 --amp_dtype bf16 --compile_mode default"
  PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
    "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
    --model_path "${winner}" --args_path "${ARGS}" \
    --scenes "${VALID_SCENES}" --config "${CONFIG}" \
    --output_dir "${valid_dir}" --label "cycle02_e12_${winner_label}_full_validation" \
      --batch_size 512 --scene_load_workers 1 --sample_steps 10 \
    --amp_dtype bf16 --compile_mode default --aggregate
  PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
    -m rlvr.autoresearch.tools.full_paired_eval_report \
    --baseline "${BASELINE_VALID}" --candidate "${valid_dir}/scenes.json" \
    --epoch 12 --bootstrap 10000 --seed 3407 \
    --output "${valid_dir}/paired_vs_incumbent.json"
fi

echo "line search complete: ${WORK}/selection.json"
