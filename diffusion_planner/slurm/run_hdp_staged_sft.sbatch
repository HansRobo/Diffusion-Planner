#!/usr/bin/env bash
#SBATCH --job-name=hdp-staged-sft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node01,node02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1900G
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --chdir=/tmp
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

SOURCE_REPO=${HDP_SOURCE_REPO:?set HDP_SOURCE_REPO to the immutable source worktree}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?set HDP_EXPECTED_COMMIT}
BASE_RUN=${HDP_BASE_RUN:?set HDP_BASE_RUN to the Base80 data-contract run}
BASE_EPOCH=${HDP_BASE_EPOCH:-72}
POLICY_INIT_RUN=${HDP_POLICY_INIT_RUN:-${BASE_RUN}}
POLICY_INIT_EPOCH=${HDP_POLICY_INIT_EPOCH:-${BASE_EPOCH}}
BASE_REPO=$(dirname "$(dirname "${BASE_RUN}")")
ROOT=${HDP_OUTPUT_ROOT:-${SOURCE_REPO}}
POLICY_EPOCHS=${HDP_POLICY_EPOCHS:-20}
POLICY_WARMUP=${HDP_POLICY_WARMUP:-2}
POLICY_LR=${HDP_POLICY_LR:-1e-4}
RUN_POLICY_STAGE=${HDP_RUN_POLICY_STAGE:-True}
HEAD_EXPERT_EPOCHS=${HDP_HEAD_EXPERT_EPOCHS:-1}
HEAD_EXPERT_WARMUP=${HDP_HEAD_EXPERT_WARMUP:-0}
HEAD_EXPERT_LR=${HDP_HEAD_EXPERT_LR:-3e-4}
POLICY_NAME=${HDP_POLICY_NAME:-hdp_policy_ft${POLICY_EPOCHS}_from_e${POLICY_INIT_EPOCH}}
HEAD_EXPERT_NAME=${HDP_HEAD_EXPERT_NAME:-hdp_head_expert${HEAD_EXPERT_EPOCHS}_from_${POLICY_NAME}}
POLICY_DIR=${ROOT}/outputs/${POLICY_NAME}
HEAD_EXPERT_DIR=${ROOT}/outputs/${HEAD_EXPERT_NAME}

TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
EXTRA_ROOT=${HDP_EXTRA_ROOT:-${BASE_REPO}/artifacts/right_turn_is_skipped_filtered_20260716}
EXTRA_1=${EXTRA_ROOT}/path_list_train_unprotected_right_turn_is_skipped_filtered.json
EXTRA_2=${EXTRA_ROOT}/path_list_unprotected_right_turn_xx1_is_skipped_filtered.json
EXTRA_3=${EXTRA_ROOT}/path_list_unprotected_right_turn_xx1_psim_is_skipped_filtered.json
NORMALIZATION=${SOURCE_REPO}/diffusion_planner/normalization.json
PYTHON=${SOURCE_REPO}/.venv/bin/python
BASE_WEIGHTS=${POLICY_INIT_RUN}/latest.pth
BASE_ARGS_JSON=${BASE_RUN}/args.json
PORT=${HDP_MASTER_PORT:-22657}

mkdir -p "${ROOT}" "${ROOT}/outputs" "${ROOT}/tmp" "${ROOT}/cache" "${ROOT}/wandb_cache"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

check_sha256() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    log "FATAL: SHA256 mismatch for ${path}: expected ${expected}, got ${actual}"
    exit 10
  fi
}

for required in \
  "${SOURCE_REPO}" \
  "${BASE_WEIGHTS}" \
  "${BASE_ARGS_JSON}" \
  "${TRAIN_LIST}" \
  "${VALID_LIST}" \
  "${EXTRA_1}" \
  "${EXTRA_2}" \
  "${EXTRA_3}" \
  "${NORMALIZATION}" \
  "${PYTHON}"; do
  if [[ ! -r "${required}" ]]; then
    log "FATAL: required input is not readable: ${required}"
    exit 10
  fi
done

# The staged source may be checked out by a different account than the job runs as;
# scope the exception to this path instead of trusting it globally.
GIT_SAFE_ARGS=(-c "safe.directory=${SOURCE_REPO}")
ACTUAL_COMMIT=$(git "${GIT_SAFE_ARGS[@]}" -C "${SOURCE_REPO}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  log "FATAL: expected source commit ${EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}"
  exit 11
fi
if ! git "${GIT_SAFE_ARGS[@]}" -C "${SOURCE_REPO}" diff --quiet ||
  ! git "${GIT_SAFE_ARGS[@]}" -C "${SOURCE_REPO}" diff --cached --quiet; then
  log "FATAL: immutable source worktree differs from ${EXPECTED_COMMIT}"
  exit 12
fi

check_sha256 ed61f00053ace1c4955dd2994ee78672b7c7a9ae9238577ceb1816c74f1ae395 "${TRAIN_LIST}"
check_sha256 854556f6dd2e4fd709a3fd23ca2b1086aacb9df20159be227c26091ff52de62c "${VALID_LIST}"
check_sha256 48c2eb476659241beba8cc4b9795eabf2e44ecb593a4af138fb8a4d7a38abf20 "${EXTRA_1}"
check_sha256 7bba1ea74a6f90bbb62a4daab9fc4ebd78ed7efa02c7504ef4deb811724b741b "${EXTRA_2}"
check_sha256 1a85f292129316662d80a8556e3265b22a441d163ebce8ff11ae3f852bfa98d5 "${EXTRA_3}"

"${PYTHON}" - "${BASE_ARGS_JSON}" "${BASE_WEIGHTS}" "${POLICY_INIT_EPOCH}" \
  "${TRAIN_LIST}" "${VALID_LIST}" "${EXTRA_1}" "${EXTRA_2}" "${EXTRA_3}" <<'PY'
import json
import pathlib
import sys

import torch

args_path, weights_path = map(pathlib.Path, sys.argv[1:3])
expected_epoch = int(sys.argv[3])
train_list, valid_list, *extra_lists = sys.argv[4:]
args = json.loads(args_path.read_text(encoding="utf-8"))
expected = {
    "train_set_list": train_list,
    "valid_set_list": valid_list,
    "extra_train_set_repeat": 10,
    "filter_skipped": False,
    "seed": 3407,
    "train_subsample_step": 1,
    "align_legacy_neighbor_futures": False,
    "use_data_augment": True,
    "augment_prob": 0.5,
    "num_refine": 20,
    "ego_past_noise_std": 0.1,
    "use_smoothing_future_trajectory": True,
    "future_len": 80,
    "time_len": 31,
    "ego_history_frames": 21,
    "predicted_neighbor_num": 0,
    "decoder_depth": 6,
    "diffusion_sample_steps": 6,
    "use_velocity_representation": True,
    "planning_hybrid_loss": 0.01,
    "coeff_road_border_loss": 0.0,
    "coeff_neighbor_collision_loss": 0.0,
    "supervised_training_stage": "policy",
}
for key, value in expected.items():
    if args.get(key) != value:
        raise RuntimeError(f"Base data/model contract mismatch: {key}={args.get(key)!r}, expected {value!r}")
base_extra_names = [pathlib.Path(path).name for path in args.get("extra_train_set_list", [])]
expected_extra_names = [pathlib.Path(path).name for path in extra_lists]
if base_extra_names != expected_extra_names:
    raise RuntimeError(
        f"Base extra-list contract mismatch: {base_extra_names!r}, expected {expected_extra_names!r}"
    )
checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
if checkpoint.get("epoch") != expected_epoch:
    raise RuntimeError(
        f"Base latest checkpoint is epoch {checkpoint.get('epoch')!r}, expected {expected_epoch}"
    )
if checkpoint.get("ema_state_dict") is None:
    raise RuntimeError("Base latest checkpoint has no EMA policy")
print(f"verified policy-init epoch {expected_epoch} EMA and exact Base80 data contract")
PY

export HOME=${HDP_JOB_HOME:-${ROOT}/home}
export NETRC=${HOME}/.netrc
mkdir -p "${HOME}"
if [[ ! -r "${NETRC}" ]]; then
  echo "FATAL: W&B credentials are not readable at ${NETRC}. Provision that file for" >&2
  echo "       $(id -un), or point HDP_JOB_HOME at a directory that has one." >&2
  exit 13
fi
export TMPDIR=${ROOT}/tmp
export XDG_CACHE_HOME=${ROOT}/cache
export WANDB_DIR=${ROOT}
export WANDB_CACHE_DIR=${ROOT}/wandb_cache
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

COMMON_ARGS=(
  --train_set_list "${TRAIN_LIST}"
  --extra_train_set_list "${EXTRA_1}"
  --extra_train_set_list "${EXTRA_2}"
  --extra_train_set_list "${EXTRA_3}"
  --extra_train_set_repeat 10
  --valid_set_list "${VALID_LIST}"
  --filter_skipped False
  --seed 3407
  --train_subsample_step 1
  --align_legacy_neighbor_futures False
  --normalization_file_path "${NORMALIZATION}"
  --batch_size 512
  --weight_decay 0.01
  --adamw_no_decay True
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
  --coeff_road_border_loss 0.0
  --road_border_margin 0.25
  --road_border_n_interp 2
  --coeff_neighbor_collision_loss 0.0
  --use_ema True
  --port "${PORT}"
)

checkpoint_epoch() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch

print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True).get("epoch", -1)))
PY
}

run_stage() {
  local phase=$1
  local stage=$2
  local run_name=$3
  local save_dir=$4
  local init_weights=$5
  local epochs=$6
  local warmup=$7
  local learning_rate=$8
  local save_utd=$9
  local max_attempts=3
  local attempt=1
  local status=1
  local wandb_id="hdp-${phase}-${SLURM_JOB_ID}"
  local train_log=${save_dir}/train_stdout.log

  mkdir -p "${save_dir}"
  export TORCHINDUCTOR_CACHE_DIR=${ROOT}/cache/torchinductor-${EXPECTED_COMMIT:0:12}-${phase}
  export TORCHINDUCTOR_COMPILE_THREADS=${HDP_COMPILE_THREADS:-$((SLURM_CPUS_PER_TASK / 8))}
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

  if [[ -r "${save_dir}/latest.pth" ]] && [[ "$(checkpoint_epoch "${save_dir}/latest.pth")" == "${epochs}" ]]; then
    log "${phase}: already complete at epoch ${epochs}; skipping"
    return 0
  fi

  while (( attempt <= max_attempts )); do
    local -a start_args
    if [[ -r "${save_dir}/latest.pth" ]]; then
      start_args=(--resume_model_path "${save_dir}/latest.pth")
      log "${phase}: attempt ${attempt}/${max_attempts}, strict resume"
    else
      start_args=(--init_weights_path "${init_weights}" --wandb_run_id "${wandb_id}")
      log "${phase}: attempt ${attempt}/${max_attempts}, EMA weights-only initialization"
    fi

    set +e
    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node=8 \
      --master_port="${PORT}" \
      "${SOURCE_REPO}/diffusion_planner/train_predictor.py" \
      --exp_name "${run_name}" \
      --save_dir "${save_dir}" \
      --supervised_training_stage "${stage}" \
      --learning_rate "${learning_rate}" \
      --warm_up_epoch "${warmup}" \
      --train_epochs "${epochs}" \
      --save_utd "${save_utd}" \
      "${COMMON_ARGS[@]}" \
      "${start_args[@]}" >>"${train_log}" 2>&1
    status=$?
    set -e
    if (( status == 0 )); then
      break
    fi
    log "${phase}: attempt ${attempt} exited with status ${status}"
    attempt=$((attempt + 1))
    sleep 60
  done

  if (( status != 0 )); then
    touch "${save_dir}/RUN_FAILED"
    return "${status}"
  fi
  if [[ "$(checkpoint_epoch "${save_dir}/latest.pth")" != "${epochs}" ]]; then
    log "FATAL: ${phase} latest checkpoint did not reach epoch ${epochs}"
    return 20
  fi
  touch "${save_dir}/RUN_COMPLETE"
  log "${phase}: verified epoch ${epochs} latest checkpoint"
}

cd "${SOURCE_REPO}/diffusion_planner"

# Stage 1: remove signal feedback and adapt only the trajectory planner. A dependent
# head-only job can skip this stage and consume a completed policy checkpoint directly.
case "${RUN_POLICY_STAGE}" in
  True|true|1)
    run_stage policy policy \
      "${POLICY_NAME}" "${POLICY_DIR}" "${BASE_WEIGHTS}" \
      "${POLICY_EPOCHS}" "${POLICY_WARMUP}" "${POLICY_LR}" 5
    ;;
  False|false|0)
    POLICY_DIR=${POLICY_INIT_RUN}
    log "policy: using completed epoch ${POLICY_INIT_EPOCH} checkpoint at ${POLICY_DIR}"
    ;;
  *)
    log "FATAL: HDP_RUN_POLICY_STAGE must be True or False, got ${RUN_POLICY_STAGE}"
    exit 21
    ;;
esac

# Stage 2, and the last one: train the head on expert trajectories. This evaluates the
# frozen encoder once per batch and entirely skips the diffusion policy. The expert head
# at ${HEAD_EXPERT_DIR} is the final artifact. Validation still scores the head applied to
# the generated trajectory, so exposure drift would show up in the metrics either way.
run_stage turn-expert turn_indicator \
  "${HEAD_EXPERT_NAME}" "${HEAD_EXPERT_DIR}" "${POLICY_DIR}/latest.pth" \
  "${HEAD_EXPERT_EPOCHS}" "${HEAD_EXPERT_WARMUP}" "${HEAD_EXPERT_LR}" 1

touch "${ROOT}/RUN_COMPLETE"
log "All staged SFT training completed"
