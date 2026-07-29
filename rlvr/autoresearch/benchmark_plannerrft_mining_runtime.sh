#!/usr/bin/env bash
# Benchmark the real PlannerRFT collect-only path on a canonical prefix of the
# full T4 corpus.  Shared policy-independent context, K=10 reward scoring,
# five-step guided sampling and strict replay publication all remain enabled.

set -Eeuo pipefail

LABEL=${1:?usage: benchmark_plannerrft_mining_runtime.sh LABEL BATCH [PREFETCH] [OMP_THREADS]}
BATCH=${2:?missing rollout batch size}
PREFETCH=${3:-1}
OMP_THREADS=${4:-1}
for value in "${BATCH}" "${PREFETCH}" "${OMP_THREADS}"; do
  [[ ${value} =~ ^[0-9]+$ ]] || {
    echo "numeric arguments must be non-negative integers" >&2
    exit 2
  }
done
(( BATCH > 0 && OMP_THREADS > 0 )) || {
  echo "batch and OMP threads must be positive" >&2
  exit 2
}

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
CAMPAIGN=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycles02_to10_e100
SOURCE_RUN=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycle01_replay_e03_e10/20260720-183510_plannerrft_prefix_full_cycle01_replay_e03_e10
SOURCE_MINE=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine
MODEL=${SOURCE_RUN}/best_train.pth
ARGS=${SOURCE_RUN}/model_args.json
SOURCE_CACHE=${SOURCE_MINE}/replay_buffer
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
OUT=${CAMPAIGN}/acceleration_benchmark/mining_runtime/${LABEL}
LOG=${OUT}/mine.log
BATCHES=${MINING_BENCHMARK_BATCHES_PER_RANK:-32}

[[ -s ${MODEL} && -s ${ARGS} && -d ${SOURCE_CACHE} ]] || {
  echo "formal mining benchmark inputs are incomplete" >&2
  exit 1
}
if find "${OUT}" -mindepth 1 -maxdepth 2 -type f -name final_summary.json -print -quit 2>/dev/null | grep -q .; then
  echo "completed mining benchmark arm already exists: ${OUT}" >&2
  exit 1
fi
mkdir -p "${OUT}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=${OMP_THREADS}
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

/usr/bin/time -f '%e' -o "${OUT}/wall_seconds.txt" \
  "${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
  -m rlvr.train_awr \
  --model_path "${MODEL}" --args_path "${ARGS}" \
  --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
  --config "${CONFIG}" --output_dir "${OUT}" \
  --exp_name "mining_runtime_${LABEL}_b${BATCH}_n${BATCHES}" \
  --device cuda --seed 3407 --start_epoch 11 --epochs 11 \
  --expert_anchor_weight 0.4 --max_train_scenes 0 --max_valid_scenes 8 \
  --plannerrft_reference_mode zero_noise \
  --plannerrft_reference_noise_scale 0.5 \
  --plannerrft_longitudinal_mode candidate_stretch \
  --plannerrft_lambda_lat 1.0 --sample_steps 5 \
  --eval_k 1 --eval_sample_steps 10 --train_selector_scenes 0 \
  --full_valid_interval 0 --hard_valid_scenes 0 \
  --shared_replay_encoding_root "${SOURCE_CACHE}" \
  --shared_replay_context_root "${SOURCE_CACHE}" \
  --shared_replay_expert_anchor_root "${SOURCE_CACHE}" \
  --shared_replay_order_epoch 1 \
  --rollout_scene_batch_size "${BATCH}" --scene_batch_size "${BATCH}" \
  --benchmark_rollout_batches_per_rank "${BATCHES}" \
  --rollout_scene_load_workers 1 --rollout_prefetch_batches "${PREFETCH}" \
  --scene_load_workers 4 --eval_scene_load_workers 1 \
  --eval_prefetch_batches 0 --eval_scene_batch_size 8 \
  --save_rollouts 0 --no-wandb --no-skip_filtered_scenes \
  > "${LOG}" 2>&1

RUN=$(sed -n 's/^Run directory: //p' "${LOG}" | tail -n 1)
SUMMARY=${RUN}/train_epoch_011/summary.json
[[ -n ${RUN} && -s ${RUN}/final_summary.json && -s ${SUMMARY} ]] || {
  echo "mining benchmark arm did not commit a complete strict cache" >&2
  exit 1
}
"${ROOT}/.venv/bin/python" - "${OUT}/result.json" "${RUN}" \
  "${LABEL}" "${BATCH}" "${PREFETCH}" "${OMP_THREADS}" "${BATCHES}" \
  "$(<"${OUT}/wall_seconds.txt")" <<'PY'
import datetime
import json
import pathlib
import sys

out, run, label = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
batch, prefetch, omp, batches = map(int, sys.argv[4:8])
wall = float(sys.argv[8])
summary = json.loads((run / "train_epoch_011" / "summary.json").read_text())
rank_manifest = json.loads(
    (run / "replay_buffer" / "rank_0000" / "manifest.json").read_text()
)
payload = {
    "version": 1,
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "label": label,
    "run": str(run.resolve()),
    "rollout_scene_batch_size": batch,
    "rollout_prefetch_batches": prefetch,
    "omp_num_threads": omp,
    "batches_per_rank": batches,
    "rank_scene_count": int(rank_manifest["scene_count"]),
    "global_scene_count": int(summary["scene_count"]),
    "train_elapsed_seconds": float(summary["elapsed_sec"]),
    "scenes_per_second": float(summary["scenes_per_sec"]),
    "wall_seconds": wall,
    "gpu_peak_allocated_gib_rank0": float(summary["gpu_peak_allocated_gib_rank0"]),
    "gpu_peak_reserved_gib_rank0": float(summary["gpu_peak_reserved_gib_rank0"]),
    "strict_manifest": str(
        (run / "replay_buffer" / "rank_0000" / "manifest.json").resolve()
    ),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
