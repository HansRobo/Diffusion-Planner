#!/usr/bin/env bash
#SBATCH --job-name=hdp-lr-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1900G
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --output=/mnt/nvme/Diffusion-Planner-hdp-lr-sweep-20260721/slurm-%j.out
#SBATCH --error=/mnt/nvme/Diffusion-Planner-hdp-lr-sweep-20260721/slurm-%j.err

set -euo pipefail

SOURCE_REPO=${HDP_SOURCE_REPO:?set HDP_SOURCE_REPO}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?set HDP_EXPECTED_COMMIT}
BASE_RUN=${HDP_BASE_RUN:?set HDP_BASE_RUN}
VALID_PROBE=${HDP_VALID_PROBE:?set HDP_VALID_PROBE}
EXTRA_ROOT=${HDP_EXTRA_ROOT:?set HDP_EXTRA_ROOT}
ROOT=${HDP_OUTPUT_ROOT:-/mnt/nvme/Diffusion-Planner-hdp-lr-sweep-20260721}
PYTHON=/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner/.venv/bin/python
TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
EXTRA_1=${EXTRA_ROOT}/path_list_train_unprotected_right_turn_is_skipped_filtered.json
EXTRA_2=${EXTRA_ROOT}/path_list_unprotected_right_turn_xx1_is_skipped_filtered.json
EXTRA_3=${EXTRA_ROOT}/path_list_unprotected_right_turn_xx1_psim_is_skipped_filtered.json
NORMALIZATION=${SOURCE_REPO}/diffusion_planner/normalization.json
BASE_WEIGHTS=${BASE_RUN}/latest.pth
read -r -a LR_VALUES <<<"${HDP_LR_VALUES:-1e-5 3e-5 5e-5}"

mkdir -p "${ROOT}/outputs" "${ROOT}/tmp" "${ROOT}/cache" "${ROOT}/wandb_cache"

if [[ "$(git -C "${SOURCE_REPO}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "source commit mismatch" >&2
  exit 10
fi
if ! git -C "${SOURCE_REPO}" diff --quiet || ! git -C "${SOURCE_REPO}" diff --cached --quiet; then
  echo "source worktree is dirty" >&2
  exit 10
fi
for path in "${PYTHON}" "${BASE_WEIGHTS}" "${VALID_PROBE}" "${TRAIN_LIST}" \
  "${EXTRA_1}" "${EXTRA_2}" "${EXTRA_3}" "${NORMALIZATION}"; do
  [[ -r "${path}" ]] || { echo "unreadable input: ${path}" >&2; exit 10; }
done

"${PYTHON}" - "${BASE_WEIGHTS}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if checkpoint.get("epoch") != 72 or checkpoint.get("ema_state_dict") is None:
    raise RuntimeError(
        f"LR probes require Base epoch-72 EMA, got epoch={checkpoint.get('epoch')!r}"
    )
PY

export HOME=/home/ubuntu
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
export OMP_NUM_THREADS=1
export PYTHONPATH=${SOURCE_REPO}:${SOURCE_REPO}/diffusion_planner:${PYTHONPATH:-}
export TORCHINDUCTOR_CACHE_DIR=${ROOT}/cache/torchinductor-${EXPECTED_COMMIT:0:12}-policy
export TORCHINDUCTOR_COMPILE_THREADS=${HDP_COMPILE_THREADS:-28}

COMMON_ARGS=(
  --train_set_list "${TRAIN_LIST}"
  --extra_train_set_list "${EXTRA_1}"
  --extra_train_set_list "${EXTRA_2}"
  --extra_train_set_list "${EXTRA_3}"
  --extra_train_set_repeat 10
  --valid_set_list "${VALID_PROBE}"
  --filter_skipped False
  --seed 3407
  --train_subsample_step 6
  --align_legacy_neighbor_futures True
  --normalization_file_path "${NORMALIZATION}"
  --batch_size 512
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
  --augment_type quintic
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
  --turn_indicator_generated_loss_weight 1.0
  --turn_indicator_expert_loss_weight 1.0
  --coeff_road_border_loss 1.0
  --road_border_margin 0.25
  --road_border_n_interp 2
  --coeff_neighbor_collision_loss 0.0
  --supervised_training_stage policy
  --use_ema True
  --export_onnx_on_save False
  --warm_up_epoch 0
  --train_epochs 1
  --save_utd 1
)

cd "${SOURCE_REPO}/diffusion_planner"
for index in "${!LR_VALUES[@]}"; do
  lr=${LR_VALUES[$index]}
  tag=${lr//-/m}
  tag=${tag//./p}
  name=hdp_no_signal_policy_lr_probe_${tag}_base72_step6_node02_8gpu_bf16_bs512
  save_dir=${ROOT}/outputs/${name}
  mkdir -p "${save_dir}"
  echo "[$(date '+%F %T %Z')] starting ${name}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port="$((22710 + index))" \
    "${SOURCE_REPO}/diffusion_planner/train_predictor.py" \
    --exp_name "${name}" \
    --save_dir "${save_dir}" \
    --learning_rate "${lr}" \
    --init_weights_path "${BASE_WEIGHTS}" \
    --wandb_run_id "hdp-policy-lrprobe-${tag}-${SLURM_JOB_ID}" \
    "${COMMON_ARGS[@]}" >>"${save_dir}/train_stdout.log" 2>&1
  "${PYTHON}" - "${save_dir}" "${lr}" <<'PY'
import csv
import json
import math
import pathlib
import sys
import torch

save_dir = pathlib.Path(sys.argv[1])
expected_lr = float(sys.argv[2])
checkpoint = torch.load(save_dir / "latest.pth", map_location="cpu", weights_only=True)
rows = list(csv.DictReader((save_dir / "train_log.tsv").open(), delimiter="\t"))
args = json.loads((save_dir / "args.json").read_text())
if checkpoint.get("epoch") != 1 or len(rows) != 1:
    raise RuntimeError("LR probe did not complete exactly one epoch")
if not math.isclose(float(args["learning_rate"]), expected_lr, rel_tol=1e-8):
    raise RuntimeError(f"LR mismatch: {args['learning_rate']} versus {expected_lr}")
for key, value in rows[0].items():
    if key != "epoch" and value and not math.isfinite(float(value)):
        raise RuntimeError(f"non-finite probe metric: {key}={value}")
print(f"verified lr={expected_lr:g} train_loss={rows[0]['train_loss']}")
PY
done

touch "${ROOT}/RUN_COMPLETE"
