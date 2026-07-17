#!/bin/sh
set -eu

# Full-sequence AWR replacement model for the original four-channel DP.
# The 20260707 train and 20260702 balanced validation lists were audited in
# full for is_skipped before launch.  Keep the expensive sidecar scan out of
# every rank; the exact source paths remain in run provenance.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if test -n "${PYTHON:-}"; then
  PY=$PYTHON
elif test -x "$ROOT/.venv/bin/python"; then
  PY="$ROOT/.venv/bin/python"
else
  PY=python3
fi
TRAIN_LIST="${TRAIN_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}"
VALID_LIST="${VALID_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}"
MODEL="${MODEL_PATH:-/tmp/t4_v5_original_dp/model/best_model.pth}"
ARGS="${ARGS_PATH:-/tmp/t4_v5_original_dp/model/args.json}"
CONFIG="$ROOT/rlvr/configs/awr_original_dp_t4_hdp_faithful.json"
OUT="${OUTPUT_DIR:-$ROOT/outputs/awr_t4_full_sequence_filtered}"

if ! test -r "$TRAIN_LIST" || ! test -s "$TRAIN_LIST"; then
  echo "train list is missing or unreadable: $TRAIN_LIST" >&2
  exit 1
fi
if ! test -r "$VALID_LIST" || ! test -s "$VALID_LIST"; then
  echo "valid list is unreadable: $VALID_LIST" >&2
  echo "wang is an ubuntu-group member, but this shell may be stale; use a new login shell or: sg ubuntu -c '$0'" >&2
  exit 1
fi
test -s "$MODEL"
test -s "$ARGS"

export PYTHONPATH="$ROOT/diffusion_planner:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Do not block a multi-day run on missing interactive credentials.  If W&B is
# already authenticated, stream online; otherwise keep a complete offline run
# that can be synced after `python -m wandb login`.
if test -z "${WANDB_MODE+x}"; then
  if test -n "${WANDB_API_KEY:-}" || { test -r "$HOME/.netrc" && grep -q 'api.wandb.ai' "$HOME/.netrc"; }; then
    WANDB_MODE=online
  else
    WANDB_MODE=offline
  fi
fi
export WANDB_MODE
echo "W&B mode: $WANDB_MODE"

exec "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT:-29617}" \
  -m rlvr.train_awr \
  --model_path "$MODEL" \
  --args_path "$ARGS" \
  --train_npz_list "$TRAIN_LIST" \
  --valid_npz_list "$VALID_LIST" \
  --config "$CONFIG" \
  --output_dir "$OUT" \
  --exp_name "full_sequence_20260707_hdp_plain_mse_e${EPOCHS:-100}_single_checkpoint" \
  --device cuda \
  --seed 3407 \
  --max_train_scenes 0 \
  --max_valid_scenes 2048 \
  --epochs "${EPOCHS:-100}" \
  --K 10 \
  --eval_k 10 \
  --noise_scale_range 0.5 0.5 \
  --sample_steps 5 \
  --hdp_trajectory_augmentation \
  --hdp_trajectory_augmentation_std 0.5 \
  --trainable_scope dit \
  --learning_rate 1e-6 \
  --bc_weight 0 \
  --hdp_hybrid_loss_weight 0 \
  --awr_loss_type plain_mse \
  --reward_profile hdp_pdm \
  --reward_mode gate \
  --gradient_accumulation_scenes 1 \
  --scene_batch_size 192 \
  --rollout_scene_batch_size 640 \
  --rollout_scene_load_workers 1 \
  --save_rollouts 0 \
  --amp_dtype bf16 \
  --enable_compile \
  --compile_mode default \
  --no-compile_encoder \
  --compile_decoder \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-original-dp-awr}" \
  --wandb_run_name "${WANDB_RUN_NAME:-original-dp-awr-20260707-e${EPOCHS:-100}}" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_tags original-dp awr hdp-faithful full-sequence 20260707 \
  --no-skip_filtered_scenes
