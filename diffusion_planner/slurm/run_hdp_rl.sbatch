#!/usr/bin/env bash
#SBATCH --job-name=hdp-rl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:h100:8

set -euo pipefail

REPO=${HDP_RL_REPO:-/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner}
NNODES=${SLURM_NNODES:-1}
NPROC_PER_NODE=${HDP_RL_NPROC_PER_NODE:-${HDP_RL_NPROC:-8}}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [[ -n "${HDP_RL_COMPILE_THREADS:-}" ]]; then
  COMPILE_THREADS=${HDP_RL_COMPILE_THREADS}
else
  COMPILE_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))
  ((COMPILE_THREADS <= 8)) || COMPILE_THREADS=8
fi
((COMPILE_THREADS >= 1)) || COMPILE_THREADS=1
NUM_WORKERS=${HDP_RL_NUM_WORKERS:-8}
TRAIN_SUBSAMPLE_STEP=${HDP_RL_TRAIN_SUBSAMPLE_STEP:-1}
DEFAULT_RL_TRAIN_LIST=${HDP_DEFAULT_RL_TRAIN_LIST:-${HDP_DEFAULT_SFT_TRAIN_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}}
TRAIN_LIST=${HDP_RL_TRAIN_LIST:-${DEFAULT_RL_TRAIN_LIST}}
VALID_LIST=${HDP_RL_VALID_LIST:?submit with HDP_RL_VALID_LIST=<manifest>}
if [[ -n "${HDP_RL_FILTER_SKIPPED+x}" ]]; then
  FILTER_SKIPPED=${HDP_RL_FILTER_SKIPPED}
elif [[ "${TRAIN_LIST}" == "${DEFAULT_RL_TRAIN_LIST}" ]]; then
  FILTER_SKIPPED=False
else
  FILTER_SKIPPED=True
fi
EXTRA_LISTS=${HDP_RL_EXTRA_LISTS:-}
EXTRA_REPEAT=${HDP_RL_EXTRA_REPEAT:-0}
INIT_CKPT=${HDP_RL_INIT_CKPT:-}
RESUME_CKPT=${HDP_RL_RESUME_CKPT:-}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with HDP_EXPECTED_COMMIT=<commit>}
NUM_GENERATIONS=${HDP_RL_NUM_GENERATIONS:-8}
BATCH_SIZE=${HDP_RL_BATCH_SIZE:-512}
VALID_BATCH_SIZE=${HDP_RL_VALID_BATCH_SIZE:-512}
TRAIN_EPOCHS=${HDP_RL_TRAIN_EPOCHS:-1}
VALIDATE_BEFORE_TRAINING=${HDP_RL_VALIDATE_BEFORE_TRAINING:-True}
MAX_VALID_LOSS_REGRESSION=${HDP_RL_MAX_VALID_LOSS_REGRESSION:-0.25}
MAX_VALID_SAFETY_REGRESSION=${HDP_RL_MAX_VALID_SAFETY_REGRESSION:-0.001}
MAX_VALID_EPDMS_REGRESSION=${HDP_RL_MAX_VALID_EPDMS_REGRESSION:-0.001}
BEST_SCORE_MIN_DELTA=${HDP_RL_BEST_SCORE_MIN_DELTA:-0.0001}
EARLY_STOP_PATIENCE=${HDP_RL_EARLY_STOP_PATIENCE:-0}
FULL_EVAL_UTD=${HDP_RL_FULL_EVAL_UTD:-5}
LEARNING_RATE=${HDP_RL_LEARNING_RATE:-1e-7}
WEIGHT_DECAY=${HDP_RL_WEIGHT_DECAY:-0.0}
REWARD_BETA=${HDP_RL_REWARD_BETA:-0.5}
REWARD_W_SAFETY=${HDP_RL_REWARD_W_SAFETY:-0.0}
REWARD_W_RISK=${HDP_RL_REWARD_W_RISK:-1.0}
REWARD_W_FOLLOW=${HDP_RL_REWARD_W_FOLLOW:-3.0}
REWARD_W_LANE=${HDP_RL_REWARD_W_LANE:-2.5}
REWARD_W_PROGRESS=${HDP_RL_REWARD_W_PROGRESS:-3.0}
REWARD_W_ROAD_BORDER=${HDP_RL_REWARD_W_ROAD_BORDER:-0.0}
BEHAVIOR_GATE=${HDP_RL_BEHAVIOR_GATE:-safety}
OCCUPANCY_USE_ROAD_BORDER=${HDP_RL_OCCUPANCY_USE_ROAD_BORDER:-True}
STATIONARY_PROGRESS_MODE=${HDP_RL_STATIONARY_PROGRESS_MODE:-distance}
STATIONARY_REFERENCE_THRESHOLD_M=${HDP_RL_STATIONARY_REFERENCE_THRESHOLD_M:-1.0}
STATIONARY_PROGRESS_TOLERANCE_M=${HDP_RL_STATIONARY_PROGRESS_TOLERANCE_M:-2.0}
STOPPED_NEIGHBOR_VEL_THRESH=${HDP_RL_STOPPED_NEIGHBOR_VEL_THRESH:-0.1}
STOPPED_NEIGHBOR_DISP_THRESH=${HDP_RL_STOPPED_NEIGHBOR_DISP_THRESH:-0.5}
RED_LIGHT_CONSTRAINT=${HDP_RL_RED_LIGHT_CONSTRAINT:-True}
RED_LIGHT_LANE_TOLERANCE_M=${HDP_RL_RED_LIGHT_LANE_TOLERANCE_M:-2.0}
BC_WEIGHT=${HDP_RL_BC_WEIGHT:-0.0}
EMA_UPDATE_RATE=${HDP_RL_EMA_UPDATE_RATE:-0.05}
NOISE_SCALE=${HDP_RL_NOISE_SCALE:-1.5}
EVAL_NOISE_SCALE=${HDP_RL_EVAL_NOISE_SCALE:-0.5}
EVAL_NUM_GENERATIONS=${HDP_RL_EVAL_NUM_GENERATIONS:-32}
EVAL_REWARD_W_SAFETY=${HDP_RL_EVAL_REWARD_W_SAFETY:-0.0}
EVAL_REWARD_W_RISK=${HDP_RL_EVAL_REWARD_W_RISK:-1.0}
EVAL_REWARD_W_FOLLOW=${HDP_RL_EVAL_REWARD_W_FOLLOW:-3.0}
EVAL_REWARD_W_LANE=${HDP_RL_EVAL_REWARD_W_LANE:-2.5}
EVAL_REWARD_W_PROGRESS=${HDP_RL_EVAL_REWARD_W_PROGRESS:-3.0}
EVAL_REWARD_W_ROAD_BORDER=${HDP_RL_EVAL_REWARD_W_ROAD_BORDER:-0.0}
EVAL_BEHAVIOR_GATE=${HDP_RL_EVAL_BEHAVIOR_GATE:-safety}
EVAL_OCCUPANCY_USE_ROAD_BORDER=${HDP_RL_EVAL_OCCUPANCY_USE_ROAD_BORDER:-True}
EVAL_STATIONARY_PROGRESS_MODE=${HDP_RL_EVAL_STATIONARY_PROGRESS_MODE:-distance}
EVAL_STATIONARY_REFERENCE_THRESHOLD_M=${HDP_RL_EVAL_STATIONARY_REFERENCE_THRESHOLD_M:-1.0}
EVAL_STATIONARY_PROGRESS_TOLERANCE_M=${HDP_RL_EVAL_STATIONARY_PROGRESS_TOLERANCE_M:-2.0}
EVAL_STOPPED_NEIGHBOR_VEL_THRESH=${HDP_RL_EVAL_STOPPED_NEIGHBOR_VEL_THRESH:-0.1}
EVAL_STOPPED_NEIGHBOR_DISP_THRESH=${HDP_RL_EVAL_STOPPED_NEIGHBOR_DISP_THRESH:-0.5}
EVAL_RED_LIGHT_CONSTRAINT=${HDP_RL_EVAL_RED_LIGHT_CONSTRAINT:-True}
EVAL_RED_LIGHT_LANE_TOLERANCE_M=${HDP_RL_EVAL_RED_LIGHT_LANE_TOLERANCE_M:-2.0}
ROAD_BORDER_CRITICAL=${HDP_RL_ROAD_BORDER_CRITICAL_M:-0.20}
ROAD_BORDER_SAFE=${HDP_RL_ROAD_BORDER_SAFE_M:-0.60}
ROLLOUT_STEPS=${HDP_RL_ROLLOUT_STEPS:-6}
UPDATES_PER_ROLLOUT=${HDP_RL_UPDATES_PER_ROLLOUT:-1}
UPDATE_MAX_CANDIDATES=${HDP_RL_UPDATE_MAX_CANDIDATES_PER_RANK:-2048}
COMPILE_MODEL=${HDP_RL_COMPILE_MODEL:-True}
TRAIN_SCOPE=${HDP_RL_TRAIN_SCOPE:-decoder}
ENCODER_DROP_PATH_RATE=${HDP_RL_ENCODER_DROP_PATH_RATE:-0.1}
DECODER_DROP_PATH_RATE=${HDP_RL_DECODER_DROP_PATH_RATE:-0.1}
USE_WANDB=${HDP_RL_USE_WANDB:-True}
WANDB_PROJECT=${HDP_RL_WANDB_PROJECT:-hdp-rl}
WANDB_RUN_ID=${HDP_RL_WANDB_RUN_ID:-}
STEP_LOG_INTERVAL=${HDP_RL_STEP_LOG_INTERVAL:-100}
RUN_NAME=${HDP_RL_RUN_NAME:-hdp_rl_${WORLD_SIZE}gpu_n${NUM_GENERATIONS}_${SLURM_JOB_ID}}
SAVE_ROOT=${HDP_RL_SAVE_ROOT:-${REPO}/outputs}
for gate_name in "${BEHAVIOR_GATE}" "${EVAL_BEHAVIOR_GATE}"; do
  case "${gate_name}" in
    none | safety | risk) ;;
    *)
      echo "behavior gate must be one of none, safety, or risk; got ${gate_name}" >&2
      exit 2
      ;;
  esac
done
for stationary_mode in "${STATIONARY_PROGRESS_MODE}" "${EVAL_STATIONARY_PROGRESS_MODE}"; do
  case "${stationary_mode}" in
    constant | distance) ;;
    *)
      echo "stationary progress mode must be constant or distance; got ${stationary_mode}" >&2
      exit 2
      ;;
  esac
done
if ((NNODES > 1)) && [[ "${SAVE_ROOT}" == /mnt/nvme/* ]]; then
  echo "multi-node RL requires a shared HDP_RL_SAVE_ROOT, not local NVMe: ${SAVE_ROOT}" >&2
  exit 2
fi
SAVE_DIR=${SAVE_ROOT}/${RUN_NAME}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
PORT=$((24000 + SLURM_JOB_ID % 10000))

mkdir -p "${SAVE_DIR}" "${REPO}/tmp" "${REPO}/cache/torchinductor"
if [[ -f "${SAVE_DIR}/RL_RUN_OK" && -z "${RESUME_CKPT}" ]]; then
  echo "RL run is already complete: ${SAVE_DIR}"
  exit 0
fi
if [[ -z "${RESUME_CKPT}" && -f "${SAVE_DIR}/latest.pth" ]]; then
  completed_epoch=$(
    "${REPO}/.venv/bin/python" - "${SAVE_DIR}/latest.pth" <<'PY'
import pathlib
import sys

import torch

checkpoint = torch.load(pathlib.Path(sys.argv[1]), map_location="cpu", weights_only=True)
print(int(checkpoint["epoch"]))
PY
  )
  if ((completed_epoch >= TRAIN_EPOCHS)); then
    echo "RL checkpoint reached epoch ${completed_epoch}/${TRAIN_EPOCHS} but RL_RUN_OK is missing" >&2
    exit 2
  fi
  RESUME_CKPT=${SAVE_DIR}/latest.pth
  echo "Auto-resuming interrupted RL run from epoch ${completed_epoch}: ${RESUME_CKPT}"
fi
rm -f "${SAVE_DIR}/RL_RUN_OK"
cd "${REPO}"
EXPECTED_COMMIT_FULL=$(git rev-parse --verify "${EXPECTED_COMMIT}^{commit}")
EXPECTED_TREE=$(git rev-parse "${EXPECTED_COMMIT_FULL}^{tree}")
[[ "$(git rev-parse HEAD^{tree})" == "${EXPECTED_TREE}" ]]
git diff --quiet
git diff --cached --quiet
if [[ -n "${RESUME_CKPT}" ]]; then
  SOURCE_CKPT=${RESUME_CKPT}
  CKPT_ARGS=(--resume_model_path "${RESUME_CKPT}")
  VERIFY_MODE=resume
else
  [[ -n "${INIT_CKPT}" ]] || {
    echo "submit a fresh RL run with HDP_RL_INIT_CKPT=<SFT checkpoint>" >&2
    exit 2
  }
  SOURCE_CKPT=${INIT_CKPT}
  CKPT_ARGS=(--init_weights_path "${INIT_CKPT}")
  VERIFY_MODE=init
fi
if [[ "${VERIFY_MODE}" == init ]]; then
  "${REPO}/.venv/bin/python" - "${SOURCE_CKPT}" <<'PY'
import pathlib
import sys

import torch

checkpoint_path = pathlib.Path(sys.argv[1])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if not isinstance(checkpoint, dict) or checkpoint.get("ema_state_dict") is None:
    raise RuntimeError(
        f"fresh HDP-RL requires an SFT checkpoint with EMA weights: {checkpoint_path}"
    )
PY
fi
EXTRA_ARGS=(--extra_train_set_repeat "${EXTRA_REPEAT}")
if [[ -n "${EXTRA_LISTS}" ]]; then
  IFS=: read -r -a extra_list_array <<<"${EXTRA_LISTS}"
  for extra_list in "${extra_list_array[@]}"; do
    [[ -r "${extra_list}" ]] || { echo "missing extra list: ${extra_list}" >&2; exit 2; }
    EXTRA_ARGS+=(--extra_train_set_list "${extra_list}")
  done
fi
for path in "${TRAIN_LIST}" "${VALID_LIST}" "${SOURCE_CKPT}" "${SOURCE_CKPT%/*}/args.json"; do
  [[ -r "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
input_paths=("${TRAIN_LIST}" "${VALID_LIST}" "${SOURCE_CKPT}" "${SOURCE_CKPT%/*}/args.json")
data_paths=("${TRAIN_LIST}" "${VALID_LIST}" "${REPO}/diffusion_planner/normalization.json")
if [[ -n "${EXTRA_LISTS}" ]]; then
  input_paths+=("${extra_list_array[@]}")
  data_paths+=("${extra_list_array[@]}")
fi
INPUT_FINGERPRINT=$(sha256sum "${input_paths[@]}" | sha256sum | cut -d ' ' -f 1)
DATA_FINGERPRINT=$(sha256sum "${data_paths[@]}" | sha256sum | cut -d ' ' -f 1)
SOURCE_CKPT_SHA256=$(sha256sum "${SOURCE_CKPT}" | cut -d ' ' -f 1)
PYTHON_ENV_FINGERPRINT=$(
  PYTHONPATH="${REPO}:${REPO}/diffusion_planner" "${REPO}/.venv/bin/python" -c \
    'import numpy, torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), numpy.__version__)'
)
PROVENANCE_PATH=${SAVE_DIR}/run_provenance.json
"${REPO}/.venv/bin/python" - \
  "${PROVENANCE_PATH}" "${VERIFY_MODE}" "${EXPECTED_COMMIT_FULL}" "${EXPECTED_TREE}" \
  "${DATA_FINGERPRINT}" "${PYTHON_ENV_FINGERPRINT}" "${SOURCE_CKPT}" \
  "${SOURCE_CKPT_SHA256}" "${data_paths[@]}" <<'PY'
import json
import os
import pathlib
import sys

(
    provenance_path,
    mode,
    commit,
    tree,
    data_fingerprint,
    python_environment,
    source_checkpoint,
    source_checkpoint_sha256,
    *data_paths,
) = sys.argv[1:]
path = pathlib.Path(provenance_path)
current = {
    "version": 1,
    "code_commit": commit,
    "code_tree": tree,
    "data_fingerprint": data_fingerprint,
    "python_environment": python_environment,
    "data_paths": data_paths,
}
if path.exists():
    with path.open(encoding="utf-8") as f:
        recorded = json.load(f)
    checked = ("code_tree", "data_fingerprint", "python_environment")
    mismatches = {
        key: {"recorded": recorded.get(key), "current": current[key]}
        for key in checked
        if recorded.get(key) != current[key]
    }
    if mismatches:
        raise RuntimeError(f"Strict RL resume provenance mismatch: {mismatches}")
    if mode == "init" and recorded.get("initial_checkpoint_sha256") != source_checkpoint_sha256:
        raise RuntimeError("Fresh-run retry changed the initialization checkpoint")
else:
    current.update(
        {
            "created_from_mode": mode,
            "legacy_resume_anchor": mode == "resume",
            "initial_checkpoint": source_checkpoint,
            "initial_checkpoint_sha256": source_checkpoint_sha256,
        }
    )
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    if mode == "resume":
        print("Anchored legacy RL resume provenance at the current code/data/environment state")
PY
VERIFY_SOURCE_CKPT=${SOURCE_CKPT}
if [[ "${VERIFY_MODE}" == resume ]]; then
  VERIFY_SOURCE_CKPT=${SAVE_DIR}/.rl_smoke_source_${SLURM_JOB_ID}.pth
  cp --reflink=auto "${SOURCE_CKPT}" "${VERIFY_SOURCE_CKPT}"
  trap 'rm -f "${VERIFY_SOURCE_CKPT}"' EXIT
fi

export HOME=/home/ubuntu
export TMPDIR=${REPO}/tmp
export XDG_CACHE_HOME=${REPO}/cache
export TORCHINDUCTOR_CACHE_DIR=${REPO}/cache/torchinductor
export TORCHINDUCTOR_COMPILE_THREADS=${COMPILE_THREADS}
export PYTHONPATH=${REPO}:${REPO}/diffusion_planner
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export REPO NNODES NPROC_PER_NODE MASTER_ADDR PORT SAVE_DIR
export EXPECTED_TREE INPUT_FINGERPRINT PYTHON_ENV_FINGERPRINT
export TRAIN_LIST VALID_LIST SOURCE_CKPT EXTRA_LISTS

# Validate and prepare every allocated host before starting torchrun. This catches an
# unsynchronized repository, checkpoint, manifest, or virtualenv without stranding the
# remaining nodes inside rendezvous.
srun --kill-on-bad-exit=1 --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    cd "${REPO}"
    [[ "$(git rev-parse HEAD^{tree})" == "${EXPECTED_TREE}" ]]
    git diff --quiet
    git diff --cached --quiet
    required=(
      "${TRAIN_LIST}"
      "${VALID_LIST}"
      "${SOURCE_CKPT}"
      "${SOURCE_CKPT%/*}/args.json"
    )
    if [[ -n "${EXTRA_LISTS}" ]]; then
      IFS=: read -r -a extra_paths <<<"${EXTRA_LISTS}"
      required+=("${extra_paths[@]}")
    fi
    for path in "${required[@]}"; do
      [[ -r "${path}" ]] || { echo "$(hostname -s): missing required input: ${path}" >&2; exit 2; }
    done
    [[ -x "${REPO}/.venv/bin/python" ]] || {
      echo "$(hostname -s): missing Python environment: ${REPO}/.venv/bin/python" >&2
      exit 2
    }
    actual_python_env=$(
      PYTHONPATH="${REPO}:${REPO}/diffusion_planner" "${REPO}/.venv/bin/python" -c \
        "import numpy, torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), numpy.__version__)"
    )
    [[ "${actual_python_env}" == "${PYTHON_ENV_FINGERPRINT}" ]] || {
      echo "$(hostname -s): Python/CUDA/NCCL environment mismatch" >&2
      exit 2
    }
    visible_gpus=$("${REPO}/.venv/bin/python" -c "import torch; print(torch.cuda.device_count())")
    ((visible_gpus >= NPROC_PER_NODE)) || {
      echo "$(hostname -s): requested ${NPROC_PER_NODE} GPUs, found ${visible_gpus}" >&2
      exit 2
    }
    actual_fingerprint=$(sha256sum "${required[@]}" | sha256sum | cut -d " " -f 1)
    [[ "${actual_fingerprint}" == "${INPUT_FINGERPRINT}" ]] || {
      echo "$(hostname -s): RL input fingerprint mismatch" >&2
      exit 2
    }
    mkdir -p "${SAVE_DIR}" "${REPO}/tmp" "${REPO}/cache/torchinductor"
  '

TRAIN_ARGS=(
  train_hdp_rl_predictor.py
  --exp_name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --train_set_list "${TRAIN_LIST}" \
  --valid_set_list "${VALID_LIST}" \
  --filter_skipped "${FILTER_SKIPPED}" \
  --skip_filter_workers "${HDP_RL_SKIP_FILTER_WORKERS:-32}" \
  "${EXTRA_ARGS[@]}" \
  --align_legacy_neighbor_futures False \
  --normalization_file_path "${REPO}/diffusion_planner/normalization.json" \
  --num_workers "${NUM_WORKERS}" \
  --train_subsample_step "${TRAIN_SUBSAMPLE_STEP}" \
  --batch_size "${BATCH_SIZE}" \
  --valid_batch_size "${VALID_BATCH_SIZE}" \
  --train_epochs "${TRAIN_EPOCHS}" \
  --rl_validate_before_training "${VALIDATE_BEFORE_TRAINING}" \
  --rl_max_valid_loss_regression "${MAX_VALID_LOSS_REGRESSION}" \
  --rl_max_valid_safety_regression "${MAX_VALID_SAFETY_REGRESSION}" \
  --rl_max_valid_epdms_regression "${MAX_VALID_EPDMS_REGRESSION}" \
  --rl_best_score_min_delta "${BEST_SCORE_MIN_DELTA}" \
  --rl_early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --save_utd 1 \
  --rl_full_eval_utd "${FULL_EVAL_UTD}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --adamw_no_decay True \
  --warm_up_epoch 1 \
  --num_generations "${NUM_GENERATIONS}" \
  --rl_noise_scale "${NOISE_SCALE}" \
  --rl_eval_noise_scale "${EVAL_NOISE_SCALE}" \
  --rl_eval_num_generations "${EVAL_NUM_GENERATIONS}" \
  --rl_eval_reward_w_safety "${EVAL_REWARD_W_SAFETY}" \
  --rl_eval_reward_w_risk "${EVAL_REWARD_W_RISK}" \
  --rl_eval_reward_w_follow "${EVAL_REWARD_W_FOLLOW}" \
  --rl_eval_reward_w_lane "${EVAL_REWARD_W_LANE}" \
  --rl_eval_reward_w_progress "${EVAL_REWARD_W_PROGRESS}" \
  --rl_eval_reward_w_road_border "${EVAL_REWARD_W_ROAD_BORDER}" \
  --rl_eval_behavior_gate "${EVAL_BEHAVIOR_GATE}" \
  --rl_eval_occupancy_use_road_border "${EVAL_OCCUPANCY_USE_ROAD_BORDER}" \
  --rl_eval_stationary_progress_mode "${EVAL_STATIONARY_PROGRESS_MODE}" \
  --rl_eval_stationary_reference_threshold_m "${EVAL_STATIONARY_REFERENCE_THRESHOLD_M}" \
  --rl_eval_stationary_progress_tolerance_m "${EVAL_STATIONARY_PROGRESS_TOLERANCE_M}" \
  --rl_eval_stopped_neighbor_vel_thresh "${EVAL_STOPPED_NEIGHBOR_VEL_THRESH}" \
  --rl_eval_stopped_neighbor_disp_thresh "${EVAL_STOPPED_NEIGHBOR_DISP_THRESH}" \
  --rl_eval_red_light_constraint "${EVAL_RED_LIGHT_CONSTRAINT}" \
  --rl_eval_red_light_lane_tolerance_m "${EVAL_RED_LIGHT_LANE_TOLERANCE_M}" \
  --rl_road_border_critical_m "${ROAD_BORDER_CRITICAL}" \
  --rl_road_border_safe_m "${ROAD_BORDER_SAFE}" \
  --rl_rollout_steps "${ROLLOUT_STEPS}" \
  --rl_updates_per_rollout "${UPDATES_PER_ROLLOUT}" \
  --rl_update_max_candidates_per_rank "${UPDATE_MAX_CANDIDATES}" \
  --rl_reward_normalize group \
  --rl_reward_beta "${REWARD_BETA}" \
  --rl_reward_w_safety "${REWARD_W_SAFETY}" \
  --rl_reward_w_risk "${REWARD_W_RISK}" \
  --rl_reward_w_follow "${REWARD_W_FOLLOW}" \
  --rl_reward_w_lane "${REWARD_W_LANE}" \
  --rl_reward_w_progress "${REWARD_W_PROGRESS}" \
  --rl_reward_w_road_border "${REWARD_W_ROAD_BORDER}" \
  --rl_behavior_gate "${BEHAVIOR_GATE}" \
  --rl_occupancy_use_road_border "${OCCUPANCY_USE_ROAD_BORDER}" \
  --rl_stationary_progress_mode "${STATIONARY_PROGRESS_MODE}" \
  --rl_stationary_reference_threshold_m "${STATIONARY_REFERENCE_THRESHOLD_M}" \
  --rl_stationary_progress_tolerance_m "${STATIONARY_PROGRESS_TOLERANCE_M}" \
  --rl_stopped_neighbor_vel_thresh "${STOPPED_NEIGHBOR_VEL_THRESH}" \
  --rl_stopped_neighbor_disp_thresh "${STOPPED_NEIGHBOR_DISP_THRESH}" \
  --rl_red_light_constraint "${RED_LIGHT_CONSTRAINT}" \
  --rl_red_light_lane_tolerance_m "${RED_LIGHT_LANE_TOLERANCE_M}" \
  --rl_bc_weight "${BC_WEIGHT}" \
  --rl_ema_update_rate "${EMA_UPDATE_RATE}" \
  --rl_init_use_ema True \
  --rl_train_scope "${TRAIN_SCOPE}" \
  "${CKPT_ARGS[@]}" \
  --use_data_augment False \
  --enable_epdms_eval True \
  --epdms_eval_use_agent_boxes True \
  --epdms_eval_use_road_border True \
  --multisample_eval_num_samples 0 \
  --predicted_neighbor_num 0 \
  --encoder_mixer_depth 6 \
  --encoder_fusion_depth 6 \
  --decoder_depth 6 \
  --encoder_drop_path_rate "${ENCODER_DROP_PATH_RATE}" \
  --decoder_drop_path_rate "${DECODER_DROP_PATH_RATE}" \
  --hidden_dim 256 \
  --num_heads 8 \
  --diffusion_sample_steps 6 \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --coeff_road_border_loss 0.0 \
  --coeff_neighbor_collision_loss 0.0 \
  --use_ema True \
  --ddp True \
  --tf32 True \
  --amp_dtype bf16 \
  --fused_optimizer True \
  --ddp_static_graph True \
  --compile_model "${COMPILE_MODEL}" \
  --use_wandb "${USE_WANDB}" \
  --wandb_project_name "${WANDB_PROJECT}" \
  --wandb_step_log_interval "${STEP_LOG_INTERVAL}" \
  --export_onnx_on_save False \
  --port "${PORT}"
)
if [[ -n "${WANDB_RUN_ID}" ]]; then
  TRAIN_ARGS+=(--wandb_run_id "${WANDB_RUN_ID}")
fi

# One Slurm task launches one torchrun agent per node. Rank 0 stays on the first
# allocated host, so checkpoints and W&B files remain in that host's SAVE_DIR while
# every node keeps its own TorchInductor cache on local NVMe.
srun --kill-on-bad-exit=1 --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    cd "${REPO}/diffusion_planner"
    exec "${REPO}/.venv/bin/python" -m torch.distributed.run \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${SLURM_PROCID}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${PORT}" \
      "$@"
  ' bash "${TRAIN_ARGS[@]}" >"${SAVE_DIR}/train_stdout.log" 2>&1

"${REPO}/.venv/bin/python" - \
  "${VERIFY_SOURCE_CKPT}" "${SAVE_DIR}" "${TRAIN_SCOPE}" "${VERIFY_MODE}" \
  "${TRAIN_EPOCHS}" "${BEHAVIOR_GATE}" "${EVAL_BEHAVIOR_GATE}" \
  "${OCCUPANCY_USE_ROAD_BORDER}" "${EVAL_OCCUPANCY_USE_ROAD_BORDER}" \
  "${RED_LIGHT_CONSTRAINT}" "${EVAL_RED_LIGHT_CONSTRAINT}" <<'PY'
import csv
import json
import math
import pathlib
import sys

import torch

initial_path = pathlib.Path(sys.argv[1])
save_dir = pathlib.Path(sys.argv[2])
scope = sys.argv[3]
mode = sys.argv[4]
expected_epoch = int(sys.argv[5])
expected_behavior_gate = sys.argv[6]
expected_eval_behavior_gate = sys.argv[7]
expected_occupancy_use_road_border = sys.argv[8].lower() == "true"
expected_eval_occupancy_use_road_border = sys.argv[9].lower() == "true"
expected_red_light_constraint = sys.argv[10].lower() == "true"
expected_eval_red_light_constraint = sys.argv[11].lower() == "true"
source = torch.load(initial_path, map_location="cpu", weights_only=True)
if mode == "resume":
    initial_live = source["model"]
    initial_shadow = source["ema_state_dict"]
else:
    initial_live = source["ema_state_dict"]
    initial_shadow = source["ema_state_dict"]
final_ckpt = torch.load(save_dir / "latest.pth", map_location="cpu", weights_only=True)
completed_epoch = int(final_ckpt["epoch"])
assert 1 <= completed_epoch <= expected_epoch
assert not final_ckpt["optimizer"]["state"], "policy boundary kept stale optimizer moments"
live = final_ckpt["model"]
shadow = final_ckpt["ema_state_dict"]

def bare_state(state):
    result = {key.removeprefix("module."): value for key, value in state.items()}
    assert len(result) == len(state), "state dict has colliding bare/DDP keys"
    return result

initial_live = bare_state(initial_live)
initial_shadow = bare_state(initial_shadow)
live = bare_state(live)
shadow = bare_state(shadow)
assert live.keys() == initial_live.keys() == initial_shadow.keys() == shadow.keys()

decoder_changed = []
encoder_changed = []
turn_changed = []
policy_sync_error = []
ema_changed = []
source_global_step = int(source.get("global_step", 0)) if mode == "resume" else 0
optimizer_steps = int(final_ckpt.get("global_step", 0)) - source_global_step
assert optimizer_steps >= 1
for key in live:
    if not torch.is_floating_point(live[key]):
        continue
    changed = not torch.equal(live[key], initial_live[key])
    if key.startswith("decoder.dit.") or key.startswith(
        "decoder.global_route_encoder."
    ):
        decoder_changed.append(changed)
        ema_changed.append(not torch.equal(shadow[key], initial_shadow[key]))
        policy_sync_error.append((shadow[key].float() - live[key].float()).abs().max().item())
    elif key.startswith("encoder."):
        encoder_changed.append(changed)
    elif key.startswith("decoder.turn_indicator_predictor."):
        turn_changed.append(changed)

assert any(decoder_changed), "RL optimizer did not update decoder"
assert any(ema_changed), "RL optimizer did not update EMA"
assert not any(turn_changed), "frozen turn head changed"
if scope == "decoder":
    assert not any(encoder_changed), "decoder-only RL changed encoder"
else:
    assert any(encoder_changed), "all-scope RL did not update encoder"
assert policy_sync_error and max(policy_sync_error) < 2e-5, max(policy_sync_error)

with (save_dir / "train_log.tsv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
assert len(rows) == completed_epoch
with (save_dir / "args.json").open(encoding="utf-8") as f:
    run_args = json.load(f)
assert run_args["rl_behavior_gate"] == expected_behavior_gate
assert run_args["rl_eval_behavior_gate"] == expected_eval_behavior_gate
assert run_args["rl_occupancy_use_road_border"] is expected_occupancy_use_road_border
assert (
    run_args["rl_eval_occupancy_use_road_border"]
    is expected_eval_occupancy_use_road_border
)
assert run_args["rl_red_light_constraint"] is expected_red_light_constraint
assert run_args["rl_eval_red_light_constraint"] is expected_eval_red_light_constraint
required = (
    "train_reward_mean",
    "train_reward_std",
    "train_reward_group_range",
    "train_reward_collision_safety_score",
    "train_reward_red_light_score",
    "train_reward_red_light_constraint_active_fraction",
    "train_reward_red_light_violation_fraction",
    "train_grad_norm",
    "train_grad_clip_fraction",
    "train_valid_group_fraction",
    "train_optimizer_step_fraction",
)
for key in required:
    value = float(rows[0][key])
    assert math.isfinite(value), (key, value)
assert float(rows[0]["train_optimizer_step_fraction"]) > 0.0
assert float(rows[0]["train_valid_group_fraction"]) > 0.0
full_eval_rows = [
    row
    for row in rows
    if str(row.get("valid_full_eval", "")).strip().lower() in {"1", "true", "yes"}
    and row.get("valid_reward_mean") not in {None, ""}
    and math.isfinite(float(row["valid_reward_mean"]))
]
assert full_eval_rows, "training produced no finite full-evaluation reward"
for key in (
    "valid_reward_collision_safety",
    "valid_reward_red_light",
    "valid_reward_red_light_violation_fraction",
):
    value = float(full_eval_rows[-1][key])
    assert math.isfinite(value), (key, value)
with (save_dir / "source_baseline_metrics.json").open(encoding="utf-8") as f:
    source_metrics = json.load(f)
if source_metrics.get("baseline_available", True):
    assert math.isfinite(float(source_metrics["valid_reward_mean"]))
else:
    assert run_args["rl_validate_before_training"] is False

print(
    "RL_RUN_OK",
    f"completed_epoch={completed_epoch}/{expected_epoch}",
    f"optimizer_steps={optimizer_steps}",
    f"decoder_tensors_changed={sum(decoder_changed)}",
    f"encoder_tensors_changed={sum(encoder_changed)}",
    f"max_policy_sync_error={max(policy_sync_error):.3e}",
)
PY

touch "${SAVE_DIR}/RL_RUN_OK"
