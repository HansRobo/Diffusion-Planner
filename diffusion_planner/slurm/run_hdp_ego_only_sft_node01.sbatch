#!/usr/bin/env bash
#SBATCH --job-name=hdp-route-sft20
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1900G
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output=/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner/outputs/hdp_ego_only_route_adaln_sft20_filteredbase_afterentry_tlsvisible_allveh_x10_adamw_nodecay_node01_8gpu_bf16_bs512_from_base20/slurm-%j.out
#SBATCH --error=/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner/outputs/hdp_ego_only_route_adaln_sft20_filteredbase_afterentry_tlsvisible_allveh_x10_adamw_nodecay_node01_8gpu_bf16_bs512_from_base20/slurm-%j.err

set -euo pipefail

REPO=/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner
SOURCE_REPO=${HDP_SOURCE_REPO:?submit with an immutable HDP_SOURCE_REPO worktree}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with HDP_EXPECTED_COMMIT}
RUN_NAME=hdp_ego_only_route_adaln_sft20_filteredbase_afterentry_tlsvisible_allveh_x10_adamw_nodecay_node01_8gpu_bf16_bs512_from_base20
SAVE_DIR=${REPO}/outputs/${RUN_NAME}
BASE_RUN=${REPO}/outputs/hdp_ego_only_route_adaln_base20_filtered_afterentry_tlsvisible_3veh_x10_recoveryaug_adamw_nodecay_node01_8gpu_bf16_bs512
INIT_WEIGHTS=${BASE_RUN}/latest.pth
INIT_ARGS=${BASE_RUN}/args.json
TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train_sft_exclude_unprotected_right_turn_after_entry_all_vehicles.json
EXTRA_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train_unprotected_right_turn_after_entry_all_vehicles.json
VALID_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_valid_sft_balanced.json
NORMALIZATION=${SOURCE_REPO}/diffusion_planner/normalization.json
PYTHON=${REPO}/.venv/bin/python
PORT=22641
MAX_ATTEMPTS=3
WANDB_RUN_ID=${HDP_WANDB_RUN_ID:-hdp-route-sft-${SLURM_JOB_ID}}

mkdir -p \
  "${SAVE_DIR}" \
  "${REPO}/tmp" \
  "${REPO}/cache/torchinductor-sft-${EXPECTED_COMMIT:0:12}" \
  "${REPO}/wandb_cache"
HEALTH_LOG=${SAVE_DIR}/health_monitor.log
TRAIN_LOG=${SAVE_DIR}/train_stdout.log

log_health() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${HEALTH_LOG}"
}

check_sha256() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    log_health "FATAL: SHA256 mismatch for ${path}: expected ${expected}, got ${actual}"
    exit 10
  fi
}

for required in \
  "${SOURCE_REPO}" \
  "${TRAIN_LIST}" \
  "${EXTRA_LIST}" \
  "${VALID_LIST}" \
  "${NORMALIZATION}" \
  "${INIT_WEIGHTS}" \
  "${INIT_ARGS}" \
  "${PYTHON}"; do
  if [[ ! -r "${required}" ]]; then
    log_health "FATAL: required input is not readable: ${required}"
    exit 10
  fi
done

ACTUAL_COMMIT=$(git -C "${SOURCE_REPO}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  log_health "FATAL: expected source commit ${EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}"
  exit 11
fi
if ! git -C "${SOURCE_REPO}" diff --quiet || ! git -C "${SOURCE_REPO}" diff --cached --quiet; then
  log_health "FATAL: immutable source worktree differs from ${EXPECTED_COMMIT}"
  exit 12
fi

check_sha256 39e4f71957bf73dc7b7ec0c28c8abed5938c38d27a0748e55da91fb0630172eb "${TRAIN_LIST}"
check_sha256 1e8393e49f9461018a359ab953d628e06e792153ba9b9863cd9d81e014413ec6 "${EXTRA_LIST}"
check_sha256 ef129d5052db364aa5bd8bda16cce0f03237bc1afded6969ce13e3259e4668bf "${VALID_LIST}"

"${PYTHON}" - "${INIT_ARGS}" "${INIT_WEIGHTS}" <<'PY' | tee -a "${HEALTH_LOG}"
import json
import pathlib
import sys

import torch

args_path = pathlib.Path(sys.argv[1])
weights_path = pathlib.Path(sys.argv[2])
args = json.loads(args_path.read_text(encoding="utf-8"))
expected = {
    "future_len": 80,
    "predicted_neighbor_num": 0,
    "decoder_depth": 6,
    "diffusion_sample_steps": 6,
    "use_velocity_representation": True,
    "planning_hybrid_loss": 0.01,
    "coeff_road_border_loss": 0.0,
    "coeff_neighbor_collision_loss": 0.0,
}
for key, value in expected.items():
    if args.get(key) != value:
        raise RuntimeError(f"incompatible Base args: {key}={args.get(key)!r}, expected {value!r}")
if weights_path.stat().st_size < 100_000_000:
    raise RuntimeError(f"Base checkpoint is unexpectedly small: {weights_path.stat().st_size} bytes")
checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
if checkpoint.get("epoch") != 20:
    raise RuntimeError(f"Base latest.pth is epoch {checkpoint.get('epoch')!r}, expected epoch 20")
if checkpoint.get("ema_state_dict") is None:
    raise RuntimeError("Base epoch-20 latest.pth has no EMA policy for SFT initialization")
print("verified Base epoch-20 latest.pth EMA policy")
PY

export HOME=/home/ubuntu
export TMPDIR=${REPO}/tmp
export XDG_CACHE_HOME=${REPO}/cache
export TORCHINDUCTOR_CACHE_DIR=${REPO}/cache/torchinductor-sft-${EXPECTED_COMMIT:0:12}
export TORCHINDUCTOR_COMPILE_THREADS=${HDP_COMPILE_THREADS:-$((SLURM_CPUS_PER_TASK / 8))}
export WANDB_DIR=${SAVE_DIR}
export WANDB_CACHE_DIR=${REPO}/wandb_cache
export WANDB_ENTITY=advanced-technology-department
export WANDB_PROJECT=Diffusion-Planner-Temporal
export WANDB_LOG_DATASET_ARTIFACT=0
export WANDB__SERVICE_WAIT=300
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export OMP_NUM_THREADS=1
export PYTHONPATH=${SOURCE_REPO}:${SOURCE_REPO}/diffusion_planner:${PYTHONPATH:-}

cd "${SOURCE_REPO}/diffusion_planner"

BASE_ARGS=(
  train_predictor.py
  --exp_name "${RUN_NAME}"
  --save_dir "${SAVE_DIR}"
  --train_set_list "${TRAIN_LIST}"
  --extra_train_set_list "${EXTRA_LIST}"
  --extra_train_set_repeat 10
  --extra_train_set_mask_traffic_lights False
  --valid_set_list "${VALID_LIST}"
  --train_subsample_step 1
  --align_legacy_neighbor_futures True
  --normalization_file_path "${NORMALIZATION}"
  --init_weights_path "${INIT_WEIGHTS}"
  --batch_size 512
  --learning_rate 1e-4
  --weight_decay 0.01
  --adamw_no_decay True
  --warm_up_epoch 5
  --train_epochs 20
  --save_utd 5
  --num_workers 16
  --enable_epdms_eval True
  --epdms_eval_use_agent_boxes True
  --epdms_eval_use_road_border True
  --multisample_eval_num_samples 0
  --ddp True
  --find_unused_parameters False
  --device cuda
  --tf32 True
  --amp_dtype bf16
  --fused_optimizer True
  --ddp_static_graph True
  --compile_model True
  --use_data_augment True
  --augment_prob 0.5
  --num_refine 20
  --ego_past_noise_std 0.1
  --use_smoothing_future_trajectory True
  --use_wandb True
  --wandb_project_name Diffusion-Planner-Temporal
  --wandb_step_log_interval 50
  --future_len 80
  --time_len 31
  --agent_num 320
  --predicted_neighbor_num 0
  --lane_num 140
  --lane_len 20
  --route_num 25
  --route_len 20
  --static_objects_num 5
  --hidden_dim 256
  --num_heads 8
  --encoder_mixer_depth 6
  --encoder_fusion_depth 6
  --decoder_depth 6
  --diffusion_model_type x_start
  --diffusion_supervision_type x_start
  --diffusion_time_sample_method uniform
  --diffusion_sample_steps 6
  --use_velocity_representation True
  --planning_hybrid_loss 0.01
  --hybrid_loss_window 10
  --turn_indicator_generated_loss_weight 1.0
  --turn_indicator_expert_loss_weight 1.0
  --coeff_road_border_loss 0.0
  --road_border_margin 0.25
  --road_border_n_interp 2
  --coeff_neighbor_collision_loss 0.0
  --use_ema True
  --export_onnx_on_save False
  --port "${PORT}"
)

monitor_attempt() {
  local train_pid=$1
  local attempt=$2
  local last_mtime=0
  local unchanged=0
  while kill -0 "${train_pid}" 2>/dev/null; do
    sleep 60
    local current_mtime=0
    if [[ -e "${TRAIN_LOG}" ]]; then
      current_mtime=$(stat -c %Y "${TRAIN_LOG}")
    fi
    if (( current_mtime == last_mtime )); then
      unchanged=$((unchanged + 1))
    else
      unchanged=0
      last_mtime=${current_mtime}
    fi
    local gpu_state
    gpu_state=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
      --format=csv,noheader,nounits 2>&1 | tr '\n' ';')
    local disk_state
    disk_state=$(df -h "${SAVE_DIR}" | tail -1 | awk '{print $4 " free"}')
    log_health "attempt=${attempt} pid=${train_pid} log_stale_min=${unchanged} disk=${disk_state} gpu=${gpu_state}"
    if (( unchanged == 30 )); then
      log_health "ALERT: training log has not changed for 30 minutes; inspect process and NCCL state"
    fi
  done
}

status=1
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  if [[ -r "${SAVE_DIR}/latest.pth" ]]; then
    START_ARGS=(--resume_model_path "${SAVE_DIR}/latest.pth")
    log_health "Attempt ${attempt}/${MAX_ATTEMPTS}: strict resume from latest.pth"
  else
    START_ARGS=(--wandb_run_id "${WANDB_RUN_ID}")
    log_health "Attempt ${attempt}/${MAX_ATTEMPTS}: initialize from Base epoch-20 latest EMA with W&B ${WANDB_RUN_ID}"
  fi

  set +e
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port="${PORT}" \
    "${BASE_ARGS[@]}" \
    "${START_ARGS[@]}" >>"${TRAIN_LOG}" 2>&1 &
  TRAIN_PID=$!
  monitor_attempt "${TRAIN_PID}" "${attempt}"
  wait "${TRAIN_PID}"
  status=$?
  set -e

  if (( status == 0 )); then
    log_health "Training process exited successfully"
    break
  fi
  log_health "ERROR: training attempt ${attempt} exited with status ${status}"
  attempt=$((attempt + 1))
  if (( attempt <= MAX_ATTEMPTS )); then
    sleep 60
  fi
done

if (( status != 0 )); then
  log_health "FATAL: all ${MAX_ATTEMPTS} training attempts failed"
  touch "${SAVE_DIR}/RUN_FAILED"
  exit "${status}"
fi

"${PYTHON}" -c \
  'import csv, pathlib, sys, torch
p = pathlib.Path(sys.argv[1])
ckpt = torch.load(p / "latest.pth", map_location="cpu", weights_only=True)
with (p / "train_log.tsv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
assert int(ckpt["epoch"]) == 20, ckpt["epoch"]
assert rows and int(rows[-1]["epoch"]) == 20, rows[-1].get("epoch") if rows else None
assert (p / "best_model" / "best_model.pth").is_file()
assert (p / "best_epdms_model" / "best_model.pth").is_file()
print(f"verified epoch={ckpt['"'"'epoch'"'"']} rows={len(rows)}")' \
  "${SAVE_DIR}" | tee -a "${HEALTH_LOG}"

touch "${SAVE_DIR}/RUN_COMPLETE"
log_health "RUN_COMPLETE: verified 20 epochs and both best checkpoints"
