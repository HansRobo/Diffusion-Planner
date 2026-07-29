#!/usr/bin/env bash
#SBATCH --job-name=hdp-turn-head
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node01,node02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=450G
#SBATCH --gres=gpu:h100:4
#SBATCH --requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/mnt/storage_rdma/workspaces/wang/slurm/%x-%j.out
#SBATCH --error=/mnt/storage_rdma/workspaces/wang/slurm/%x-%j.err

# The `gpu` partition spans twelve nodes, ten of which belong to other teams, and no node
# carries a Slurm feature to select on. `--nodelist` is therefore the only way to say "one
# of ours": it is a candidate pool, node01 leading, so the job can never land on a foreign
# node. Slurm's own stdout/stderr must exist before the job starts and so live on shared
# storage; everything the training writes stays on the allocated node's local NVMe.

set -euo pipefail

# Turn-indicator head training, ONE round, on the Base80 policy that the node02 RL
# campaign is fine-tuning. Two facts make this safe to run concurrently with that
# campaign:
#
#   * RL trains only decoder.dit.* and decoder.global_route_encoder.* (see
#     configure_rl_trainable_parameters); decoder.turn_indicator_predictor.* is
#     excluded under BOTH rl_trainable_scope values, so the head is untouched there.
#   * TurnIndicatorHead detaches all four of its inputs, so head training sends no
#     gradient into the encoder or the policy. supervised_training_stage=turn_indicator
#     additionally freezes every non-head parameter, so the trainable count is the head
#     alone: 939,523 for the legacy probe, 2,323,203 at num_queries=4 num_layers=2.
#
# Deliberate deltas from run_hdp_staged_sft.sbatch:
#
#   * enable_epdms_eval=False. In a head-only stage the trajectory policy is frozen,
#     so the EPDMS proxy re-scores identical trajectories every epoch. Measured on the
#     node02 lr2e5 lineage it produced 0.8605571 then 0.8605341 across two stages -- a
#     2.3e-05 bf16 wobble -- while costing 17-21 min per validation pass out of a 65 min
#     stage. It is pure wall-clock with no information in it.
#   * a single turn_indicator round. The head trains on the expert future, so DiT is not
#     evaluated at all; validation still scores turn_indicator_logit, the head applied to
#     the GENERATED trajectory (validate_model.py).
#   * warm_up_epoch=0 makes LinearWarmupConstantLR a pure constant, so the per-epoch
#     validation curve carries no schedule confound and each epoch is comparable.

REQUIRED_USER=${HDP_REQUIRED_USER:-wang}
if [[ "$(id -un)" != "${REQUIRED_USER}" ]]; then
  echo "FATAL: this job must run as ${REQUIRED_USER}, got $(id -un)" >&2
  exit 20
fi

# Runs from a checkout on the allocated node's local NVMe. The candidate list is ordered,
# so node01's path is the default and node02's is the fallback; the checkouts there sit at
# different commits, which is exactly why HDP_EXPECTED_COMMIT is required rather than
# defaulted -- a floating job must not silently train whatever that node happens to hold.
SOURCE_REPO=${HDP_SOURCE_REPO:-}
if [[ -z "${SOURCE_REPO}" ]]; then
  for candidate in \
    /mnt/nvme/wangbin/Diffusion-Planner-hyper-diffusion-planner \
    /mnt/nvme/Diffusion-Planner-hyper-diffusion-planner; do
    if [[ -r "${candidate}/diffusion_planner/train_predictor.py" ]]; then
      SOURCE_REPO=${candidate}
      break
    fi
  done
fi
if [[ -z "${SOURCE_REPO}" ]]; then
  echo "FATAL: no checkout found on $(hostname -s); pass HDP_SOURCE_REPO" >&2
  exit 10
fi
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with HDP_EXPECTED_COMMIT=<commit>}
ROOT=${HDP_OUTPUT_ROOT:-${SOURCE_REPO}}

POLICY_CKPT_DIR=${HDP_POLICY_CKPT_DIR:-/mnt/storage_rdma/diffusion_planner/checkpoints/hdp_base80_data_robust_v1_no_roadborder_e80_latest}
POLICY_EPOCH=${HDP_POLICY_EPOCH:-80}
INIT_WEIGHTS=${POLICY_CKPT_DIR}/latest.pth

# Two epochs: this frozen-policy head turns over at epoch 2 on this configuration, and
# epochs 3+ only overfit at ~50 min of node01 each. This is a training LENGTH, not a
# checkpoint selection -- the deliverable is still `latest.pth` (../../CLAUDE.md), which
# after two epochs simply *is* the epoch-2 head. Because HEAD_WARMUP=0 makes
# LinearWarmupConstantLR a pure MultiplicativeLR(1.0) that ignores total_epochs, and
# args._train_epochs only ever reaches a wandb log field, shortening the run cannot change
# what those epochs contain. Raise it with HDP_HEAD_EPOCHS when the head arch or lr changes
# and the turnover must be re-measured -- the run name carries ${HEAD_EPOCHS}, so a longer
# run gets its own directory.
HEAD_EPOCHS=${HDP_HEAD_EPOCHS:-2}
HEAD_WARMUP=${HDP_HEAD_WARMUP:-0}
HEAD_LR=${HDP_HEAD_LR:-3e-4}
RUN_NAME=${HDP_RUN_NAME:-hdp_turn_indicator_head${HEAD_EPOCHS}ep_lr${HEAD_LR}_from_base80_e${POLICY_EPOCH}_bf16_bs512}
SAVE_DIR=${ROOT}/outputs/${RUN_NAME}

# Data contract copied verbatim from the RL campaign's args.json so the head is trained
# on exactly the corpus and augmentation the policy was trained on.
TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
EXTRA_ROOT=${SOURCE_REPO}/artifacts/right_turn_is_skipped_filtered_20260716
EXTRA_1=${EXTRA_ROOT}/path_list_train_unprotected_right_turn_is_skipped_filtered.json
EXTRA_2=${EXTRA_ROOT}/path_list_unprotected_right_turn_xx1_is_skipped_filtered.json
EXTRA_3=${EXTRA_ROOT}/path_list_unprotected_right_turn_xx1_psim_is_skipped_filtered.json
NORMALIZATION=${SOURCE_REPO}/diffusion_planner/normalization.json
PYTHON=${SOURCE_REPO}/.venv/bin/python
PORT=${HDP_MASTER_PORT:-22671}
MAX_ATTEMPTS=${HDP_MAX_ATTEMPTS:-3}

# Rank count follows the allocation so two 4-GPU arms can share one node. batch_size is
# GLOBAL: train.py:853 sets local_batch_size = batch_size // world_size, so two arms at
# the same batch_size carry no learning-rate confound whatever rank count they run at.
#
# The LOCAL batch is not a free variable, though. At 8 ranks (local 64) validation is
# finite; at 4 ranks (local 128) with an otherwise byte-identical args.json it died in
# pre-training validation with "prediction_finite=False" on several ranks at once, 6/6
# attempts. A frozen policy cannot become unstable from rank count alone, so the local
# shape is the suspect: under --amp_dtype bf16 a different local batch selects different
# matmul kernels, and bf16's 8-bit mantissa can round an intermediate denominator to
# exactly zero. Keep batch_size // NPROC at the value observed finite (64).
NPROC=${HDP_NPROC:-$(nvidia-smi --list-gpus | wc -l)}
BATCH_SIZE=${HDP_BATCH_SIZE:-512}
if [[ ! "${NPROC}" =~ ^[1-8]$ ]] || (( BATCH_SIZE % NPROC != 0 )); then
  echo "FATAL: NPROC=${NPROC} must be 1-8 and divide batch_size ${BATCH_SIZE}" >&2
  exit 21
fi
LOCAL_BATCH=$((BATCH_SIZE / NPROC))
if (( LOCAL_BATCH > 64 )); then
  echo "FATAL: local batch ${LOCAL_BATCH} (${BATCH_SIZE}/${NPROC}) exceeds the 64 that" >&2
  echo "       validated finite; local 128 reproduced non-finite predictions on 6/6" >&2
  echo "       attempts. Override with HDP_ALLOW_LOCAL_BATCH=1" >&2
  echo "       only together with a precision change that explains the fix." >&2
  [[ "${HDP_ALLOW_LOCAL_BATCH:-0}" == "1" ]] || exit 22
fi

# Two arms share node01 (224 CPUs, 2 TB) alongside other users' non-Slurm work, which
# Slurm does not account for, so the per-arm request is half the node rather than all of
# it: 96 CPUs and 450 GB each leaves headroom instead of overcommitting real memory.
mkdir -p "${ROOT}" "${ROOT}/outputs" "${ROOT}/tmp" "${ROOT}/cache" "${ROOT}/wandb_cache" "${SAVE_DIR}"

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
  "${INIT_WEIGHTS}" \
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
check_sha256 1678d994187bc2b93fadfc34e3997cc06417b00ff97392e7827a20e792e2c552 "${INIT_WEIGHTS}"

# The head is initialized from the Base80 policy the RL campaign started from, so the
# trained head transfers to the RL output: RL moves decoder.dit.* only, at lr 1e-7.
"${PYTHON}" - "${POLICY_CKPT_DIR}/args.json" "${INIT_WEIGHTS}" "${POLICY_EPOCH}" <<'PY'
import json
import pathlib
import sys

import torch

args_path, weights_path = map(pathlib.Path, sys.argv[1:3])
expected_epoch = int(sys.argv[3])
args = json.loads(args_path.read_text(encoding="utf-8"))
expected = {
    "supervised_training_stage": "policy",
    "predicted_neighbor_num": 0,
    "use_velocity_representation": True,
    "future_len": 80,
    "time_len": 31,
    "ego_history_frames": 21,
    "decoder_depth": 6,
    "diffusion_sample_steps": 6,
    "policy_uses_turn_indicator_history": False,
    "turn_indicator_output_dim": 3,
}
for key, value in expected.items():
    if args.get(key) != value:
        raise RuntimeError(f"Policy contract mismatch: {key}={args.get(key)!r}, expected {value!r}")
checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
if checkpoint.get("epoch") != expected_epoch:
    raise RuntimeError(
        f"Policy latest checkpoint is epoch {checkpoint.get('epoch')!r}, expected {expected_epoch}"
    )
if checkpoint.get("ema_state_dict") is None:
    raise RuntimeError("Policy latest checkpoint has no EMA policy")
head_tensors = [k for k in checkpoint["ema_state_dict"] if "turn_indicator_predictor" in k]
if len(head_tensors) != 16:
    raise RuntimeError(f"expected 16 turn-indicator head tensors, found {len(head_tensors)}")
print(f"verified policy-init epoch {expected_epoch} EMA with an untrained 16-tensor intent head")
PY

export HOME=${HDP_JOB_HOME:-/mnt/storage_rdma/workspaces/wang/home}
export NETRC=${HOME}/.netrc
if [[ ! -r "${NETRC}" ]]; then
  echo "FATAL: W&B credentials are not readable at ${NETRC}" >&2
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
# Per-run inductor cache: two arms compile concurrently on the same node and must not
# share a cache directory.
export TORCHINDUCTOR_CACHE_DIR=${ROOT}/cache/torchinductor-${EXPECTED_COMMIT:0:12}-${RUN_NAME}
export TORCHINDUCTOR_COMPILE_THREADS=${HDP_COMPILE_THREADS:-$((SLURM_CPUS_PER_TASK / 8))}
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

TRAIN_ARGS=(
  --exp_name "${RUN_NAME}"
  --save_dir "${SAVE_DIR}"
  --supervised_training_stage turn_indicator
  --learning_rate "${HEAD_LR}"
  --warm_up_epoch "${HEAD_WARMUP}"
  --train_epochs "${HEAD_EPOCHS}"
  --save_utd 1
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
  --batch_size "${BATCH_SIZE}"
  --weight_decay 0.01
  --adamw_no_decay True
  --num_workers 16
  --enable_epdms_eval False
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

# Head architecture and label smoothing are passed explicitly so the run records the
# values it trained rather than inheriting whatever the defaults happen to be. The
# defaults here are the same ones `TrainConfig` carries.
HEAD_NUM_QUERIES=${HDP_HEAD_NUM_QUERIES:-4}
HEAD_NUM_LAYERS=${HDP_HEAD_NUM_LAYERS:-2}
HEAD_DROPOUT=${HDP_HEAD_DROPOUT:-0.1}
LABEL_SMOOTHING=${HDP_LABEL_SMOOTHING:-0.05}
TRAIN_ARGS+=(
  --turn_indicator_head_num_queries "${HEAD_NUM_QUERIES}"
  --turn_indicator_head_num_layers "${HEAD_NUM_LAYERS}"
  --turn_indicator_head_dropout "${HEAD_DROPOUT}"
  --turn_indicator_label_smoothing "${LABEL_SMOOTHING}"
)
log "head: q=${HEAD_NUM_QUERIES} l=${HEAD_NUM_LAYERS} dropout=${HEAD_DROPOUT} ls=${LABEL_SMOOTHING}"

checkpoint_epoch() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch

print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True).get("epoch", -1)))
PY
}

TRAIN_LOG=${SAVE_DIR}/train_stdout.log

if [[ -r "${SAVE_DIR}/latest.pth" ]] && [[ "$(checkpoint_epoch "${SAVE_DIR}/latest.pth")" == "${HEAD_EPOCHS}" ]]; then
  log "already complete at epoch ${HEAD_EPOCHS}; nothing to do"
  touch "${SAVE_DIR}/RUN_COMPLETE"
  exit 0
fi

attempt=1
status=1
while ((attempt <= MAX_ATTEMPTS)); do
  declare -a start_args
  if [[ -r "${SAVE_DIR}/latest.pth" ]]; then
    start_args=(--resume_model_path "${SAVE_DIR}/latest.pth")
    log "attempt ${attempt}/${MAX_ATTEMPTS}, strict resume from latest.pth"
  else
    start_args=(--init_weights_path "${INIT_WEIGHTS}" --wandb_run_id "hdp-turn-head-${SLURM_JOB_ID}")
    log "attempt ${attempt}/${MAX_ATTEMPTS}, EMA weights-only initialization from Base80"
  fi

  set +e
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="${PORT}" \
    "${SOURCE_REPO}/diffusion_planner/train_predictor.py" \
    "${TRAIN_ARGS[@]}" \
    "${start_args[@]}" >>"${TRAIN_LOG}" 2>&1
  status=$?
  set -e
  if ((status == 0)); then
    break
  fi
  log "attempt ${attempt} exited with status ${status}"
  attempt=$((attempt + 1))
  sleep 60
done

if ((status != 0)); then
  touch "${SAVE_DIR}/RUN_FAILED"
  log "FATAL: training failed after ${MAX_ATTEMPTS} attempts"
  exit "${status}"
fi
if [[ "$(checkpoint_epoch "${SAVE_DIR}/latest.pth")" != "${HEAD_EPOCHS}" ]]; then
  log "FATAL: latest checkpoint did not reach epoch ${HEAD_EPOCHS}"
  exit 21
fi
touch "${SAVE_DIR}/RUN_COMPLETE"
log "verified epoch ${HEAD_EPOCHS} latest checkpoint at ${SAVE_DIR}/latest.pth"
