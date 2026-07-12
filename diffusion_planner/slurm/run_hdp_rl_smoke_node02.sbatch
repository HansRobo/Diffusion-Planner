#!/usr/bin/env bash
#SBATCH --job-name=hdp-rl-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00

set -euo pipefail

REPO=/mnt/nvme/Diffusion-Planner-hyper-diffusion-planner
NPROC=${HDP_RL_NPROC:-1}
NUM_WORKERS=${HDP_RL_NUM_WORKERS:-0}
TRAIN_LIST=${HDP_RL_TRAIN_LIST:-${REPO}/artifacts/rl_smoke_node02/train_1.json}
VALID_LIST=${HDP_RL_VALID_LIST:-${REPO}/artifacts/rl_smoke_node02/valid_16.json}
EXTRA_LISTS=${HDP_RL_EXTRA_LISTS:-}
EXTRA_REPEAT=${HDP_RL_EXTRA_REPEAT:-0}
EXTRA_MASK_TRAFFIC_LIGHTS=${HDP_RL_EXTRA_MASK_TRAFFIC_LIGHTS:-False}
INIT_CKPT=${HDP_RL_INIT_CKPT:-${REPO}/checkpoints/base_epoch8_best_epdms/best_model.pth}
RESUME_CKPT=${HDP_RL_RESUME_CKPT:-}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with HDP_EXPECTED_COMMIT=<commit>}
NUM_GENERATIONS=${HDP_RL_NUM_GENERATIONS:-32}
BATCH_SIZE=${HDP_RL_BATCH_SIZE:-${NPROC}}
TRAIN_EPOCHS=${HDP_RL_TRAIN_EPOCHS:-1}
COMPILE_MODEL=${HDP_RL_COMPILE_MODEL:-False}
TRAIN_SCOPE=${HDP_RL_TRAIN_SCOPE:-decoder}
USE_WANDB=${HDP_RL_USE_WANDB:-False}
WANDB_PROJECT=${HDP_RL_WANDB_PROJECT:-hdp-rl}
STEP_LOG_INTERVAL=${HDP_RL_STEP_LOG_INTERVAL:-0}
RUN_NAME=${HDP_RL_RUN_NAME:-rl_smoke_${NPROC}gpu_n${NUM_GENERATIONS}_${SLURM_JOB_ID}}
SAVE_DIR=${REPO}/outputs/${RUN_NAME}
PORT=$((24000 + SLURM_JOB_ID % 10000))

mkdir -p "${SAVE_DIR}" "${REPO}/tmp" "${REPO}/cache/torchinductor"
rm -f "${SAVE_DIR}/RL_SMOKE_OK"
cd "${REPO}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
git diff --quiet
git diff --cached --quiet
if [[ -n "${RESUME_CKPT}" ]]; then
  SOURCE_CKPT=${RESUME_CKPT}
  CKPT_ARGS=(--resume_model_path "${RESUME_CKPT}")
  VERIFY_MODE=resume
else
  SOURCE_CKPT=${INIT_CKPT}
  CKPT_ARGS=(--init_weights_path "${INIT_CKPT}")
  VERIFY_MODE=init
fi
EXTRA_ARGS=(--extra_train_set_repeat "${EXTRA_REPEAT}")
if [[ -n "${EXTRA_LISTS}" ]]; then
  IFS=: read -r -a extra_list_array <<<"${EXTRA_LISTS}"
  for extra_list in "${extra_list_array[@]}"; do
    [[ -r "${extra_list}" ]] || { echo "missing extra list: ${extra_list}" >&2; exit 2; }
    EXTRA_ARGS+=(--extra_train_set_list "${extra_list}")
  done
fi
EXTRA_ARGS+=(--extra_train_set_mask_traffic_lights "${EXTRA_MASK_TRAFFIC_LIGHTS}")
for path in "${TRAIN_LIST}" "${VALID_LIST}" "${SOURCE_CKPT}" "${SOURCE_CKPT%/*}/args.json"; do
  [[ -r "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
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
export PYTHONPATH=${REPO}:${REPO}/diffusion_planner
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=error

cd "${REPO}/diffusion_planner"
"${REPO}/.venv/bin/python" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${PORT}" \
  train_hdp_rl_predictor.py \
  --exp_name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --train_set_list "${TRAIN_LIST}" \
  --valid_set_list "${VALID_LIST}" \
  "${EXTRA_ARGS[@]}" \
  --normalization_file_path "${REPO}/diffusion_planner/normalization.json" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --train_epochs "${TRAIN_EPOCHS}" \
  --save_utd 1 \
  --rl_full_eval_utd 1 \
  --learning_rate 5e-6 \
  --weight_decay 0.01 \
  --warm_up_epoch 1 \
  --num_generations "${NUM_GENERATIONS}" \
  --rl_noise_scale 0.5 \
  --rl_rollout_steps 6 \
  --rl_reward_normalize group \
  --rl_reward_beta 1.0 \
  --rl_ema_update_rate 0.05 \
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
  --compile_reward_kernels False \
  --use_wandb "${USE_WANDB}" \
  --wandb_project_name "${WANDB_PROJECT}" \
  --wandb_step_log_interval "${STEP_LOG_INTERVAL}" \
  --export_onnx_on_save False \
  --port "${PORT}" \
  >"${SAVE_DIR}/train_stdout.log" 2>&1

"${REPO}/.venv/bin/python" - \
  "${VERIFY_SOURCE_CKPT}" "${SAVE_DIR}" "${TRAIN_SCOPE}" "${VERIFY_MODE}" \
  "${TRAIN_EPOCHS}" <<'PY'
import csv
import math
import pathlib
import sys

import torch

initial_path = pathlib.Path(sys.argv[1])
save_dir = pathlib.Path(sys.argv[2])
scope = sys.argv[3]
mode = sys.argv[4]
expected_epoch = int(sys.argv[5])
source = torch.load(initial_path, map_location="cpu", weights_only=True)
if mode == "resume":
    initial_live = source["model"]
    initial_shadow = source["ema_state_dict"]
else:
    initial_live = source["ema_state_dict"]
    initial_shadow = source["ema_state_dict"]
final_ckpt = torch.load(save_dir / "latest.pth", map_location="cpu", weights_only=True)
assert final_ckpt["epoch"] == expected_epoch
live = final_ckpt["model"]
shadow = final_ckpt["ema_state_dict"]
assert live.keys() == initial_live.keys() == initial_shadow.keys() == shadow.keys()

decoder_changed = []
encoder_changed = []
turn_changed = []
ema_relation_error = []
ema_changed = []
source_global_step = int(source.get("global_step", 0)) if mode == "resume" else 0
optimizer_steps = int(final_ckpt.get("global_step", 0)) - source_global_step
assert optimizer_steps >= 1
for key in live:
    if not torch.is_floating_point(live[key]):
        continue
    changed = not torch.equal(live[key], initial_live[key])
    bare_key = key.removeprefix("module.")
    if bare_key.startswith("decoder.dit.") or bare_key.startswith(
        "decoder.global_route_encoder."
    ):
        decoder_changed.append(changed)
        ema_changed.append(not torch.equal(shadow[key], initial_shadow[key]))
        if changed and optimizer_steps == 1:
            expected = initial_shadow[key].float().mul(0.95).add(live[key].float(), alpha=0.05)
            ema_relation_error.append((shadow[key].float() - expected).abs().max().item())
    elif bare_key.startswith("encoder."):
        encoder_changed.append(changed)
    elif bare_key.startswith("decoder.turn_indicator_predictor."):
        turn_changed.append(changed)

assert any(decoder_changed), "RL optimizer did not update decoder"
assert any(ema_changed), "RL optimizer did not update EMA"
assert not any(turn_changed), "frozen turn head changed"
if scope == "decoder":
    assert not any(encoder_changed), "decoder-only RL changed encoder"
else:
    assert any(encoder_changed), "all-scope RL did not update encoder"
if optimizer_steps == 1:
    assert ema_relation_error and max(ema_relation_error) < 2e-5, max(ema_relation_error)

with (save_dir / "train_log.tsv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
assert len(rows) == expected_epoch
required = (
    "train_reward_mean",
    "train_reward_std",
    "train_reward_group_range",
    "train_grad_norm",
    "train_valid_group_fraction",
    "train_optimizer_step_fraction",
)
for key in required:
    value = float(rows[0][key])
    assert math.isfinite(value), (key, value)
assert float(rows[0]["train_optimizer_step_fraction"]) > 0.0
assert float(rows[0]["train_valid_group_fraction"]) > 0.0

print(
    "RL_SMOKE_OK",
    f"optimizer_steps={optimizer_steps}",
    f"decoder_tensors_changed={sum(decoder_changed)}",
    f"encoder_tensors_changed={sum(encoder_changed)}",
    f"max_ema_relation_error="
    f"{max(ema_relation_error):.3e}" if ema_relation_error else "multi_step_checked",
)
PY

touch "${SAVE_DIR}/RL_SMOKE_OK"
