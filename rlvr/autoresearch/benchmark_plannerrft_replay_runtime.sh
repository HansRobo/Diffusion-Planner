#!/usr/bin/env bash
# Benchmark one result-preserving replay execution configuration on the exact
# formal Cycle-1 cache.  The short run keeps production batch/RNG/objective
# semantics but omits the expensive selector because only train throughput is
# being measured.  Every arm starts from the same immutable checkpoint.

set -Eeuo pipefail

LABEL=${1:?usage: benchmark_plannerrft_replay_runtime.sh LABEL BUCKET WORKERS PREFETCH [OMP_THREADS] [COMPILE_MODE]}
BUCKET=${2:?missing active-target bucket size}
WORKERS=${3:?missing replay context worker count}
PREFETCH=${4:?missing replay prefetch depth}
OMP_THREADS=${5:-1}
COMPILE_MODE=${6:-default}

for value in "${BUCKET}" "${WORKERS}" "${PREFETCH}" "${OMP_THREADS}"; do
  [[ ${value} =~ ^[0-9]+$ ]] || {
    echo "all numeric arguments must be non-negative integers" >&2
    exit 2
  }
done
(( BUCKET > 0 && WORKERS > 0 && OMP_THREADS > 0 )) || {
  echo "bucket, workers and OMP threads must be positive" >&2
  exit 2
}

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
CAMPAIGN=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycles02_to10_e100
SOURCE_RUN=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycle01_replay_e03_e10/20260720-183510_plannerrft_prefix_full_cycle01_replay_e03_e10
MODEL=${SOURCE_RUN}/best_train.pth
ARGS=${SOURCE_RUN}/model_args.json
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
CACHE=${CAMPAIGN}/beta_sensitivity/beta0.5_overlay/replay_buffer
COMPRESSED=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/replay_context_zstd_v1
OUT=${CAMPAIGN}/acceleration_benchmark/replay_runtime/${LABEL}
LOG=${OUT}/train.log
UPDATES=${REPLAY_BENCHMARK_UPDATES:-512}

[[ -s ${MODEL} && -s ${ARGS} && -d ${CACHE} && -d ${COMPRESSED} ]] || {
  echo "formal replay benchmark inputs are incomplete" >&2
  exit 1
}
if find "${OUT}" -mindepth 1 -maxdepth 2 -type f -name final_summary.json -print -quit 2>/dev/null | grep -q .; then
  echo "completed benchmark arm already exists: ${OUT}" >&2
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
  --exp_name "replay_runtime_${LABEL}_u${UPDATES}" \
  --device cuda --seed 3407 --start_epoch 2 --epochs 2 \
  --resume_replay_root "${CACHE}" --replay_sampling with_replacement \
  --compressed_replay_context_root "${COMPRESSED}" \
  --replay_updates_per_epoch "${UPDATES}" \
  --awr_beta 0.5 --positive_advantage_only \
  --positive_advantage_margin 0.01 \
  --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
  --expert_anchor_weight 0.4 --expert_anchor_active_groups_only \
  --awr_candidate_loss_horizon 40 --drop_all_zero_groups \
  --awr_loss_type plain_mse --diffusion_t_range 0.001 0.2 \
  --trainable_scope output --learning_rate 0.000001 \
  --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy \
  --max_train_scenes 192 --max_valid_scenes 8 --train_selector_scenes 0 \
  --full_valid_interval 0 --hard_valid_scenes 0 \
  --eval_k 1 --eval_sample_steps 10 \
  --scene_batch_size 192 --gradient_accumulation_scenes 192 \
  --active_target_bucket_size "${BUCKET}" \
  --compile_mode "${COMPILE_MODE}" \
  --scene_load_workers "${WORKERS}" \
  --replay_prefetch_batches "${PREFETCH}" \
  --eval_scene_load_workers 1 --eval_prefetch_batches 0 \
  --eval_scene_batch_size 8 --save_rollouts 0 --no-wandb \
  --no-skip_filtered_scenes > "${LOG}" 2>&1

RUN=$(sed -n 's/^Run directory: //p' "${LOG}" | tail -n 1)
[[ -n ${RUN} && -s ${RUN}/final_summary.json && -s ${RUN}/train_epoch_002/summary.json ]] || {
  echo "benchmark arm did not commit a complete run" >&2
  exit 1
}
"${ROOT}/.venv/bin/python" - "${OUT}/result.json" "${RUN}" \
  "${LABEL}" "${BUCKET}" "${WORKERS}" "${PREFETCH}" "${OMP_THREADS}" \
  "${COMPILE_MODE}" "${UPDATES}" "$(<"${OUT}/wall_seconds.txt")" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

out, run, label = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
bucket, workers, prefetch, omp = map(int, sys.argv[4:8])
compile_mode = sys.argv[8]
updates = int(sys.argv[9])
wall = float(sys.argv[10])
summary = json.loads((run / "train_epoch_002" / "summary.json").read_text())
checkpoint = run / "epoch_002.pth"
if not checkpoint.is_file():
    checkpoint = run / "best_train.pth"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
payload = {
    "version": 1,
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "label": label,
    "run": str(run.resolve()),
    "active_target_bucket_size": bucket,
    "scene_load_workers": workers,
    "replay_prefetch_batches": prefetch,
    "omp_num_threads": omp,
    "compile_mode": compile_mode,
    "replay_updates": updates,
    "train_elapsed_seconds": float(summary["elapsed_sec"]),
    "updates_per_second": updates / float(summary["elapsed_sec"]),
    "scenes_per_second": float(summary["scenes_per_sec"]),
    "wall_seconds": wall,
    "mean_loss": float(summary["mean_loss"]),
    "mean_active_target_count": float(summary["mean_active_target_count"]),
    "mean_computed_target_count": float(summary["mean_computed_target_count"]),
    "gpu_peak_allocated_gib_rank0": float(summary["gpu_peak_allocated_gib_rank0"]),
    "gpu_peak_reserved_gib_rank0": float(summary["gpu_peak_reserved_gib_rank0"]),
    "checkpoint": str(checkpoint.resolve()) if checkpoint.is_file() else None,
    "checkpoint_sha256": digest,
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
