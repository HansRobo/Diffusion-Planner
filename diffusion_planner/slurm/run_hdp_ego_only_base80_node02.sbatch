#!/usr/bin/env bash
# Fresh ego-only HDP Base training on the complete vehicle-parameter corpus.
# This launcher intentionally has no SFT initialization.  The three persistent
# is_skipped-filtered right-turn manifests are each repeated ten times.
# The source lists are treated as immutable inputs; the job never rewrites them.
# shellcheck disable=SC1091
#SBATCH --job-name=hdp-base80
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1900G
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/mnt/nvme/Diffusion-Planner-hdp-node02-current/outputs/hdp_ego_only_route_adaln_base80_data_robust_v1_vehicle_params_mirror_rightturnx10_node02_8gpu_bf16_bs512_no_roadborder/slurm-%j.out
#SBATCH --error=/mnt/nvme/Diffusion-Planner-hdp-node02-current/outputs/hdp_ego_only_route_adaln_base80_data_robust_v1_vehicle_params_mirror_rightturnx10_node02_8gpu_bf16_bs512_no_roadborder/slurm-%j.err

set -euo pipefail

# Slurm runs these jobs as a dedicated account; every path below is provisioned for it.
# Running as anyone else silently produces unreadable credentials and unwritable outputs,
# so fail immediately and say so instead of dying deep inside training.
REQUIRED_USER=${HDP_REQUIRED_USER:-wang}
if [[ "$(id -un)" != "${REQUIRED_USER}" ]]; then
  echo "FATAL: this job must run as ${REQUIRED_USER}, got $(id -un)" >&2
  exit 20
fi

REPO=${HDP_REPO:-/mnt/nvme/Diffusion-Planner-hdp-node02-current}
RUN_NAME=${HDP_RUN_NAME:-hdp_ego_only_route_adaln_base80_data_robust_v1_vehicle_params_mirror_rightturnx10_node02_8gpu_bf16_bs512_no_roadborder}
SAVE_DIR=${REPO}/outputs/${RUN_NAME}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with --export=ALL,HDP_EXPECTED_COMMIT=<commit>}
# The staged node repository must be a clean checkout of the expected commit.
# Set HDP_ALLOW_DIRTY=1 only for an intentional dirty-worktree snapshot.
ALLOW_DIRTY=${HDP_ALLOW_DIRTY:-0}
TRAIN_LIST=${HDP_TRAIN_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}
VALID_LIST=${HDP_VALID_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}
NORMALIZATION=${HDP_NORMALIZATION:-${REPO}/diffusion_planner/normalization.json}
RIGHT_TURN_DIR=${HDP_RIGHT_TURN_DIR:-${REPO}/artifacts/right_turn_is_skipped_filtered_20260716}
EXTRA_X2=${HDP_EXTRA_X2:-${RIGHT_TURN_DIR}/path_list_train_unprotected_right_turn_is_skipped_filtered.json}
EXTRA_XX1=${HDP_EXTRA_XX1:-${RIGHT_TURN_DIR}/path_list_unprotected_right_turn_xx1_is_skipped_filtered.json}
EXTRA_PSIM=${HDP_EXTRA_PSIM:-${RIGHT_TURN_DIR}/path_list_unprotected_right_turn_xx1_psim_is_skipped_filtered.json}
# This venv is shared on the GPU nodes and contains the pinned CUDA/PyTorch stack.
PYTHON=${HDP_PYTHON:-/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner/.venv/bin/python}
PORT=${HDP_PORT:-22633}
MAX_ATTEMPTS=${HDP_MAX_ATTEMPTS:-3}
FRESH_WANDB_RUN_ID=${HDP_WANDB_RUN_ID:-hdp-base80-no-road-border-${SLURM_JOB_ID}}
# A SIGTERM that is not our own failure means the node is being taken away
# (slurmd restart, package upgrade, maintenance).  Requeue so the job comes back
# and strict-resumes from latest.pth.  Set to 0, or touch ${SAVE_DIR}/NO_REQUEUE,
# to make a deliberate scancel stick.
REQUEUE_ON_SIGTERM=${HDP_REQUEUE_ON_SIGTERM:-1}

TORCHINDUCTOR_RUN_CACHE=${SAVE_DIR}/torchinductor_cache
mkdir -p "${SAVE_DIR}" "${REPO}/tmp" "${TORCHINDUCTOR_RUN_CACHE}" \
  "${REPO}/wandb_cache" "${REPO}/wandb_config" "${REPO}/home"
HEALTH_LOG=${SAVE_DIR}/health_monitor.log
TRAIN_LOG=${SAVE_DIR}/train_stdout.log

log_health() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${HEALTH_LOG}"
}

# Interruptible sleep.  Bash defers a trap until the current foreground command
# finishes, so a plain `sleep 60` would swallow SIGTERM for up to a minute -- far
# longer than the few seconds Slurm gives us before SIGKILL.  Blocking in `wait`
# instead lets the handler run immediately.
snooze() {
  sleep "$1" &
  wait $! 2>/dev/null || true
}

on_term() {
  trap '' TERM
  log_health "SIGTERM received (slurmd restart, node maintenance, or scancel)"
  if [[ -e "${SAVE_DIR}/NO_REQUEUE" ]]; then
    log_health "NO_REQUEUE marker present; honouring the cancellation"
  elif [[ "${REQUEUE_ON_SIGTERM}" != "1" ]]; then
    log_health "HDP_REQUEUE_ON_SIGTERM=${REQUEUE_ON_SIGTERM}; honouring the cancellation"
  else
    log_health "requeueing job ${SLURM_JOB_ID}; it will strict-resume from latest.pth"
    scontrol requeue "${SLURM_JOB_ID}" ||
      log_health "WARNING: scontrol requeue failed; job will stay dead"
  fi
  # Reap the trainer explicitly.  When slurmd itself is going down it never gets
  # to clean up the step, and the eight ranks survive as orphans holding the GPUs
  # -- that is what drained node02 on 2026-07-25.
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    log_health "terminating trainer pid ${TRAIN_PID}"
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${TRAIN_PID}" 2>/dev/null || break
      snooze 1
    done
    kill -KILL "${TRAIN_PID}" 2>/dev/null || true
  fi
  exit 143
}
trap on_term TERM

for required in "${TRAIN_LIST}" "${VALID_LIST}" "${NORMALIZATION}" "${EXTRA_X2}" "${EXTRA_XX1}" "${EXTRA_PSIM}" "${PYTHON}"; do
  if [[ ! -r "${required}" ]]; then
    log_health "FATAL: required input is not readable: ${required}"
    exit 10
  fi
done

cd "${REPO}"
GIT_SAFE_ARGS=(-c "safe.directory=${REPO}")
ACTUAL_COMMIT=$(git "${GIT_SAFE_ARGS[@]}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" && "${ACTUAL_COMMIT}" != "$(git rev-parse --verify "${EXPECTED_COMMIT}" 2>/dev/null || true)" ]]; then
  log_health "FATAL: expected commit ${EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}"
  exit 11
fi
if [[ "${ALLOW_DIRTY}" != "1" ]] && {
  ! git "${GIT_SAFE_ARGS[@]}" diff --quiet ||
  ! git "${GIT_SAFE_ARGS[@]}" diff --cached --quiet;
}; then
  log_health "FATAL: tracked working tree differs from ${ACTUAL_COMMIT}; set HDP_ALLOW_DIRTY=1 only for an intentional snapshot"
  exit 12
fi

# Guard the model path against accidentally running the stale pre-cleanup staged
# copy.  The launcher is allowed to continue only when the source list and valid
# list are the exact files requested by this run.
log_health "commit=${ACTUAL_COMMIT} allow_dirty=${ALLOW_DIRTY}"
log_health "train_list=${TRAIN_LIST} valid_list=${VALID_LIST}"
log_health "right_turn_lists=${EXTRA_X2},${EXTRA_XX1},${EXTRA_PSIM} repeat=10"
log_health "train_bytes=$(stat -c %s "${TRAIN_LIST}") valid_bytes=$(stat -c %s "${VALID_LIST}")"

export HOME=${HDP_JOB_HOME:-${REPO}/home}
export NETRC=${HOME}/.netrc
mkdir -p "${HOME}"
# Fail loudly on unreadable credentials instead of depending on a world-readable
# .netrc: the file must be owned by the job account (mode 600), and a permission
# regression has to stop the launch rather than be papered over by "other" bits.
if [[ ! -r "${NETRC}" ]]; then
  echo "FATAL: W&B credentials are not readable at ${NETRC}. Provision that file for" >&2
  echo "       $(id -un), or point HDP_JOB_HOME at a directory that has one." >&2
  exit 13
fi
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:/usr/local/cuda-12.8/bin:${PATH}
export XDG_CONFIG_HOME=${REPO}/config
export TMPDIR=${REPO}/tmp
export XDG_CACHE_HOME=${REPO}/cache
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_RUN_CACHE}
export TORCHINDUCTOR_COMPILE_THREADS=${HDP_COMPILE_THREADS:-$((SLURM_CPUS_PER_TASK / 8))}
export WANDB_DIR=${SAVE_DIR}
export WANDB_CACHE_DIR=${REPO}/wandb_cache
export WANDB_CONFIG_DIR=${REPO}/wandb_config
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

log_health "python=${PYTHON}"
log_health "nvcc=$(command -v nvcc || echo MISSING)"
if [[ ! -x "${PYTHON}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  log_health "FATAL: Python or CUDA compiler is unavailable"
  # 13 is reserved for the credential guard above, so the exit code alone
  # identifies which precondition failed.
  exit 14
fi

cd "${REPO}/diffusion_planner"

# The supplied lists are immutable corpus manifests.  The three right-turn lists
# are already filtered and carry no traffic-light masking flag.  Runtime filtering
# is disabled to avoid a second multi-million sidecar scan.  Set
# HDP_FILTER_SKIPPED=True only for a deliberately raw alternate list; that mode
# writes a run-local cache and never modifies source files.
FILTER_SKIPPED=${HDP_FILTER_SKIPPED:-False}

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
  --filter_skipped "${FILTER_SKIPPED}"
  --train_subsample_step 1
  --align_legacy_neighbor_futures False
  --normalization_file_path "${NORMALIZATION}"
  --batch_size 512
  --learning_rate 2e-4
  --weight_decay 0.01
  --adamw_no_decay True
  --warm_up_epoch 5
  --train_epochs 80
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
  --ego_history_frames 21
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
  # Road-border is evaluated and logged, but deliberately not part of the SFT
  # objective.  Its scale is much larger than the HDP trajectory loss and can
  # trade route-following for excessive clearance; RL owns this safety tradeoff.
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
    snooze 60
    local current_mtime=0
    [[ -e "${TRAIN_LOG}" ]] && current_mtime=$(stat -c %Y "${TRAIN_LOG}")
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
  RESUME_ARGS=()
  if [[ -r "${SAVE_DIR}/latest.pth" ]]; then
    # No --wandb_run_id here on purpose: the checkpoint carries the original id and
    # train.py falls back to it, so a strict resume continues the same W&B run.
    RESUME_ARGS=(--resume_model_path "${SAVE_DIR}/latest.pth")
    log_health "Attempt ${attempt}/${MAX_ATTEMPTS}: strict resume from latest.pth"
  else
    RESUME_ARGS=(--wandb_run_id "${FRESH_WANDB_RUN_ID}")
    log_health "Attempt ${attempt}/${MAX_ATTEMPTS}: fresh Base run, no SFT initialization"
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
    snooze 60
  fi
done

if (( status != 0 )); then
  log_health "FATAL: all ${MAX_ATTEMPTS} training attempts failed"
  touch "${SAVE_DIR}/RUN_FAILED"
  exit "${status}"
fi

"${PYTHON}" - "${SAVE_DIR}" <<'PY' | tee -a "${HEALTH_LOG}"
import csv
import pathlib
import sys
import torch

save_dir = pathlib.Path(sys.argv[1])
checkpoint = torch.load(save_dir / "latest.pth", map_location="cpu")
assert int(checkpoint["epoch"]) == 80, checkpoint.get("epoch")
with (save_dir / "train_log.tsv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
assert rows and int(rows[-1]["epoch"]) == 80, rows[-1].get("epoch") if rows else None
assert (save_dir / "best_model" / "best_model.pth").is_file()
print(f"verified epoch={checkpoint['epoch']} rows={len(rows)}")
PY

touch "${SAVE_DIR}/RUN_COMPLETE"
log_health "RUN_COMPLETE: verified fresh 80-epoch Base training and best checkpoint"
