#!/usr/bin/env bash
#SBATCH --job-name=hdp-route-base60
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node01,node02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1900G
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output=/mnt/storage_rdma/workspaces/wang/slurm/%x-%j.out
#SBATCH --error=/mnt/storage_rdma/workspaces/wang/slurm/%x-%j.err

set -euo pipefail

# Slurm runs these jobs as a dedicated account; every path below is provisioned for it.
# Running as anyone else silently produces unreadable credentials and unwritable outputs,
# so fail immediately and say so instead of dying deep inside training.
REQUIRED_USER=${HDP_REQUIRED_USER:-wang}
if [[ "$(id -un)" != "${REQUIRED_USER}" ]]; then
  echo "FATAL: this job must run as ${REQUIRED_USER}, got $(id -un)" >&2
  exit 20
fi

# Local-NVMe checkout on whichever of our two nodes the job landed on, node01 first.
# Every launcher here requires HDP_EXPECTED_COMMIT, so landing on a node whose checkout
# sits at another commit aborts rather than training stale code.
REPO=${HDP_REPO:-}
for candidate in \
  /mnt/nvme/wangbin/Diffusion-Planner-hyper-diffusion-planner \
  /mnt/nvme/Diffusion-Planner-hyper-diffusion-planner; do
  if [[ -z "${REPO}" && -r "${candidate}/diffusion_planner/train_predictor.py" ]]; then
    REPO=${candidate}
  fi
done
if [[ -z "${REPO}" ]]; then
  echo "FATAL: no checkout on $(hostname -s); pass HDP_REPO" >&2
  exit 10
fi
RUN_NAME=hdp_ego_only_route_adaln_base60_causal_red_light_v2_fullrt_x10_recoveryaug_adamw_nodecay_node01_8gpu_bf16_bs512
SAVE_DIR=${REPO}/outputs/${RUN_NAME}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with --export=ALL,HDP_EXPECTED_COMMIT=<commit>}
MANIFEST_DIR=${REPO}/artifacts/causal_red_light_full_right_turn_20260715_v3_nearest_controlled
TRAIN_LIST=${MANIFEST_DIR}/path_list_train_base_causal_red_light_filtered.json
VALID_LIST=${REPO}/artifacts/full_sequence_base_from_20260622_step3/path_list_valid_fullseq_from_20260622_step3.json
EXTRA_X2=${MANIFEST_DIR}/path_list_train_unprotected_right_turn_causal_red_light_filtered.json
EXTRA_XX1=${MANIFEST_DIR}/path_list_unprotected_right_turn_xx1_causal_red_light_filtered.json
EXTRA_PSIM=${MANIFEST_DIR}/path_list_unprotected_right_turn_xx1_psim_causal_red_light_filtered.json
NORMALIZATION=${REPO}/diffusion_planner/normalization.json
PYTHON=${REPO}/.venv/bin/python
PORT=22631
MAX_ATTEMPTS=3
FRESH_WANDB_RUN_ID=${HDP_WANDB_RUN_ID:-hdp-route-base-${SLURM_JOB_ID}}

mkdir -p \
  "${SAVE_DIR}" \
  "${REPO}/tmp" \
  "${REPO}/cache/torchinductor" \
  "${REPO}/wandb_cache"
HEALTH_LOG=${SAVE_DIR}/health_monitor.log
TRAIN_LOG=${SAVE_DIR}/train_stdout.log

log_health() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${HEALTH_LOG}"
}

for required in \
  "${TRAIN_LIST}" \
  "${MANIFEST_DIR}/metadata.json" \
  "${VALID_LIST}" \
  "${EXTRA_X2}" \
  "${EXTRA_XX1}" \
  "${EXTRA_PSIM}" \
  "${NORMALIZATION}" \
  "${PYTHON}"; do
  if [[ ! -r "${required}" ]]; then
    log_health "FATAL: required input is not readable: ${required}"
    exit 10
  fi
done

check_manifest_sha256() {
  local expected=$1
  local path=$2
  local actual
  if [[ -z "${expected}" || ! -r "${path}" ]]; then
    log_health "FATAL: missing manifest checksum or list: ${path}"
    exit 10
  fi
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    log_health "FATAL: manifest SHA256 mismatch for ${path}: expected ${expected}, got ${actual}"
    exit 10
  fi
}

check_manifest_sha256 "$(jq -r '.generated_sha256.base // empty' "${MANIFEST_DIR}/metadata.json")" "${TRAIN_LIST}"
check_manifest_sha256 "$(jq -r '.generated_sha256.right_turn_lists[0] // empty' "${MANIFEST_DIR}/metadata.json")" "${EXTRA_X2}"
check_manifest_sha256 "$(jq -r '.generated_sha256.right_turn_lists[1] // empty' "${MANIFEST_DIR}/metadata.json")" "${EXTRA_XX1}"
check_manifest_sha256 "$(jq -r '.generated_sha256.right_turn_lists[2] // empty' "${MANIFEST_DIR}/metadata.json")" "${EXTRA_PSIM}"

cd "${REPO}"
ACTUAL_COMMIT=$(git rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  log_health "FATAL: expected commit ${EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}"
  exit 11
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  log_health "FATAL: tracked working tree differs from ${EXPECTED_COMMIT}"
  exit 12
fi

log_health "Slurm job ${SLURM_JOB_ID} allocated node01 at commit ${ACTUAL_COMMIT}"
log_health "Waiting for all pre-existing GPU compute processes to clear"
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
  GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
  log_health "GPU compute processes still present: ${GPU_PIDS}"
  sleep 30
done
sleep 30

export HOME=${HDP_JOB_HOME:-${REPO}/home}
export NETRC=${HOME}/.netrc
mkdir -p "${HOME}"
if [[ ! -r "${NETRC}" ]]; then
  echo "FATAL: W&B credentials are not readable at ${NETRC}. Provision that file for" >&2
  echo "       $(id -un), or point HDP_JOB_HOME at a directory that has one." >&2
  exit 13
fi
export TMPDIR=${REPO}/tmp
export XDG_CACHE_HOME=${REPO}/cache
export TORCHINDUCTOR_CACHE_DIR=${REPO}/cache/torchinductor
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
export PYTHONPATH=${REPO}:${REPO}/diffusion_planner:${PYTHONPATH:-}

cd "${REPO}/diffusion_planner"

BASE_ARGS=(
  train_predictor.py
  --exp_name "${RUN_NAME}"
  --save_dir "${SAVE_DIR}"
  --train_set_list "${TRAIN_LIST}"
  --extra_train_set_list "${EXTRA_X2}"
  --extra_train_set_list "${EXTRA_XX1}"
  --extra_train_set_list "${EXTRA_PSIM}"
  --extra_train_set_repeat 10
  --valid_set_list "${VALID_LIST}"
  --filter_skipped True
  --skip_filter_workers "${HDP_BASE_SKIP_FILTER_WORKERS:-64}"
  --train_subsample_step 1
  --align_legacy_neighbor_futures True
  --normalization_file_path "${NORMALIZATION}"
  --batch_size 512
  --learning_rate 2e-4
  --weight_decay 0.01
  --adamw_no_decay True
  --warm_up_epoch 5
  --train_epochs 60
  --save_utd 10
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
  --supervised_training_stage policy
  --coeff_road_border_loss 0.0
  --road_border_margin 0.25
  --road_border_n_interp 2
  --coeff_neighbor_collision_loss 0.0
  --use_ema True
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

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  RESUME_ARGS=()
  if [[ -r "${SAVE_DIR}/latest.pth" ]]; then
    RESUME_ARGS=(--resume_model_path "${SAVE_DIR}/latest.pth")
    log_health "Attempt ${attempt}/${MAX_ATTEMPTS}: strict resume from latest.pth"
  else
    RESUME_ARGS=(--wandb_run_id "${FRESH_WANDB_RUN_ID}")
    log_health "Attempt ${attempt}/${MAX_ATTEMPTS}: start clean Base training with W&B ${FRESH_WANDB_RUN_ID}"
  fi

  set +e
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port="${PORT}" \
    "${BASE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" >>"${TRAIN_LOG}" 2>&1 &
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
    log_health "Cooling down for 60 seconds before retry"
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
assert int(ckpt["epoch"]) == 60, ckpt["epoch"]
assert rows and int(rows[-1]["epoch"]) == 60, rows[-1].get("epoch") if rows else None
assert (p / "best_model" / "best_model.pth").is_file()
assert (p / "best_epdms_model" / "best_model.pth").is_file()
print(f"verified epoch={ckpt['"'"'epoch'"'"']} rows={len(rows)}")' \
  "${SAVE_DIR}" | tee -a "${HEALTH_LOG}"

touch "${SAVE_DIR}/RUN_COMPLETE"
log_health "RUN_COMPLETE: verified 60 epochs and both best checkpoints"
