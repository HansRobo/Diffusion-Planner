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
TRAIN_LIST=${HDP_RL_TRAIN_LIST:-${REPO}/artifacts/rl_smoke_node02/train_1.json}
VALID_LIST=${HDP_RL_VALID_LIST:-${REPO}/artifacts/rl_smoke_node02/valid_16.json}
INIT_CKPT=${HDP_RL_INIT_CKPT:-${REPO}/checkpoints/base_epoch8_best_epdms/best_model.pth}
EXPECTED_COMMIT=${HDP_EXPECTED_COMMIT:?submit with HDP_EXPECTED_COMMIT=<commit>}
NUM_GENERATIONS=${HDP_RL_NUM_GENERATIONS:-32}
BATCH_SIZE=${HDP_RL_BATCH_SIZE:-${NPROC}}
COMPILE_MODEL=${HDP_RL_COMPILE_MODEL:-False}
TRAIN_SCOPE=${HDP_RL_TRAIN_SCOPE:-decoder}
RUN_NAME=${HDP_RL_RUN_NAME:-rl_smoke_${NPROC}gpu_n${NUM_GENERATIONS}_${SLURM_JOB_ID}}
SAVE_DIR=${REPO}/outputs/${RUN_NAME}
PORT=$((24000 + SLURM_JOB_ID % 10000))

mkdir -p "${SAVE_DIR}" "${REPO}/tmp" "${REPO}/cache/torchinductor"
cd "${REPO}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
git diff --quiet
git diff --cached --quiet
for path in "${TRAIN_LIST}" "${VALID_LIST}" "${INIT_CKPT}" "${INIT_CKPT%/*}/args.json"; do
  [[ -r "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done

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
  --normalization_file_path "${REPO}/diffusion_planner/normalization.json" \
  --num_workers 0 \
  --batch_size "${BATCH_SIZE}" \
  --train_epochs 1 \
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
  --init_weights_path "${INIT_CKPT}" \
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
  --use_wandb False \
  --wandb_step_log_interval 0 \
  --export_onnx_on_save False \
  --port "${PORT}" \
  >"${SAVE_DIR}/train_stdout.log" 2>&1

"${REPO}/.venv/bin/python" - "${INIT_CKPT}" "${SAVE_DIR}" "${TRAIN_SCOPE}" <<'PY'
import csv
import math
import pathlib
import sys

import torch

initial_path = pathlib.Path(sys.argv[1])
save_dir = pathlib.Path(sys.argv[2])
scope = sys.argv[3]
initial = torch.load(initial_path, map_location="cpu", weights_only=True)["ema_state_dict"]
final_ckpt = torch.load(save_dir / "latest.pth", map_location="cpu", weights_only=True)
assert final_ckpt["epoch"] == 1
live = final_ckpt["model"]
shadow = final_ckpt["ema_state_dict"]
assert live.keys() == initial.keys() == shadow.keys()

decoder_changed = []
encoder_changed = []
turn_changed = []
ema_relation_error = []
for key in live:
    if not torch.is_floating_point(live[key]):
        continue
    changed = not torch.equal(live[key], initial[key])
    bare_key = key.removeprefix("module.")
    if bare_key.startswith("decoder.dit.") or bare_key.startswith(
        "decoder.global_route_encoder."
    ):
        decoder_changed.append(changed)
        if changed:
            expected = initial[key].float().mul(0.95).add(live[key].float(), alpha=0.05)
            ema_relation_error.append((shadow[key].float() - expected).abs().max().item())
    elif bare_key.startswith("encoder."):
        encoder_changed.append(changed)
    elif bare_key.startswith("decoder.turn_indicator_predictor."):
        turn_changed.append(changed)

assert any(decoder_changed), "RL optimizer did not update decoder"
assert not any(turn_changed), "frozen turn head changed"
if scope == "decoder":
    assert not any(encoder_changed), "decoder-only RL changed encoder"
assert ema_relation_error and max(ema_relation_error) < 2e-5, max(ema_relation_error, default=None)

with (save_dir / "train_log.tsv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
assert len(rows) == 1
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
    f"decoder_tensors_changed={sum(decoder_changed)}",
    f"encoder_tensors_changed={sum(encoder_changed)}",
    f"max_ema_relation_error={max(ema_relation_error):.3e}",
)
PY

touch "${SAVE_DIR}/RL_SMOKE_OK"
