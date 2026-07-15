#!/usr/bin/env bash
# Smoke test: empty train/valid lists, preload checkpoint, grouped closed-loop → W&B.
#
# Usage (from Diffusion-Planner repo root):
#   wandb login   # entity yuxuanliu must have access
#   bash diffusion_planner/run_closed_loop_wandb_smoke.sh
#
# Multi-GPU (shard bags across GPUs via DDP closed-loop eval):
#   NPROC_PER_NODE=4 bash diffusion_planner/run_closed_loop_wandb_smoke.sh
#
# Overrides via environment:
#   MODEL_PATH, NPZ_ROOT, CLASSIFICATION_JSON_ROOT, DATASET_NAME, SAVE_DIR,
#   WANDB_ENTITY, WANDB_PROJECT, NPROC_PER_NODE (default 1)
#
# Classification JSON layout (under CLASSIFICATION_JSON_ROOT):
#   {project}/{map}/{date}.json
#   {corpus}/{project}/{map}/valid/{date}/{bag}/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP_DIR="$REPO_ROOT/diffusion_planner"
META_REPO="${META_REPO:-$REPO_ROOT/../Diffusion-Planner-Meta-Repository}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/best_model.pth}"
if [[ -z "${ARGS_JSON:-}" ]]; then
  if [[ -f "$(dirname "$MODEL_PATH")/args.json" ]]; then
    ARGS_JSON="$(dirname "$MODEL_PATH")/args.json"
  elif [[ -f "$REPO_ROOT/args.json" ]]; then
    ARGS_JSON="$REPO_ROOT/args.json"
  fi
fi
# NPZ valid layout (required):
#   {corpus}/{project}/{map}/valid/{date}/{bag}/
# Run setup_local_dataset_layout.sh once if your SSD copy is flat (sakuraf_*_valid/valid/...).
LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-/media/ukenryu/external_ssd/diffusion_planner_local_dataset}"
DATASET_NAME="${DATASET_NAME:-x2_dev/2231_odaiba_shinagawa_copied_from_xx1}"
NPZ_ROOT="${NPZ_ROOT:-${LOCAL_DATASET_ROOT}/${DATASET_NAME}/valid/2026-01-15}"
CLASSIFICATION_JSON_ROOT="${CLASSIFICATION_JSON_ROOT:-$META_REPO/dataset/scenario_classification_json}"
WANDB_ENTITY="${WANDB_ENTITY:-yuxuanliu}"
WANDB_PROJECT="${WANDB_PROJECT:-test_diffusion_planner_close_loop}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/checkpoints/cl_wandb_smoke_$TIMESTAMP}"

export WANDB_ENTITY="$WANDB_ENTITY"
export SCENARIO_CLASSIFICATION_JSON_ROOT="$CLASSIFICATION_JSON_ROOT"

# Native system python only (no venv).
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
PYTHON="python3"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=== closed-loop training smoke ==="
echo "repo:              $REPO_ROOT"
echo "model:             $MODEL_PATH"
echo "args_json:         ${ARGS_JSON:-<none>}"
echo "npz_root:          $NPZ_ROOT"
echo "class_json_root:   $CLASSIFICATION_JSON_ROOT"
echo "dataset_name:      ${DATASET_NAME:-<infer from npz path>}"
echo "save_dir:          $SAVE_DIR"
echo "wandb:             $WANDB_ENTITY/$WANDB_PROJECT"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
echo "nproc_per_node:    $NPROC_PER_NODE"
echo

test -f "$MODEL_PATH" || { echo "missing checkpoint: $MODEL_PATH"; exit 1; }
test -n "${ARGS_JSON:-}" && test -f "$ARGS_JSON" || {
  echo "missing args.json (set ARGS_JSON or place args.json next to MODEL_PATH or at repo root)"
  exit 1
}
test -d "$NPZ_ROOT" || {
  echo "missing npz_root: $NPZ_ROOT"
  echo "Run: bash diffusion_planner/setup_local_dataset_layout.sh"
  exit 1
}
test -d "$CLASSIFICATION_JSON_ROOT" || {
  echo "missing classification json root: $CLASSIFICATION_JSON_ROOT"
  exit 1
}
test -f "$DP_DIR/normalization.json" || { echo "missing $DP_DIR/normalization.json"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg required for MP4 output"; exit 1; }

"$PYTHON" -c "
import sys
import torch
if not torch.cuda.is_available():
    print('CUDA not available.', file=sys.stderr)
    sys.exit(1)
"

"$PYTHON" -c "import planner_metrics" 2>/dev/null || {
  echo "planner_metrics not importable. From repo root run: uv sync"
  exit 1
}

mkdir -p "$SAVE_DIR"
echo '[]' > "$SAVE_DIR/empty_train.json"
echo '[]' > "$SAVE_DIR/empty_valid.json"

TRAIN_EPOCHS="$("$PYTHON" -c "
import torch
ckpt = torch.load('${MODEL_PATH}', map_location='cpu', weights_only=False)
init_epoch = int(ckpt.get('epoch', 0))
print(init_epoch + 1)
")"
echo "resume checkpoint epoch -> training until train_epochs=$TRAIN_EPOCHS (one step)"

DATASET_ARGS=()
if [[ -n "$DATASET_NAME" ]]; then
  DATASET_ARGS=(--closed_loop_scenario_dataset_name "$DATASET_NAME")
fi

PROFILE_ARGS=()
if [[ "${CLOSED_LOOP_PROFILE:-0}" == "1" ]]; then
  PROFILE_ARGS+=(--closed_loop_profile true)
  if [[ "${CLOSED_LOOP_PROFILE_SYNC_GPU:-0}" == "1" ]]; then
    PROFILE_ARGS+=(--closed_loop_profile_sync_gpu true)
  fi
fi

cd "$DP_DIR"

TRAIN_ARGS=(
  --args_json "$ARGS_JSON"
  --exp_name cl_wandb_smoke
  --save_dir "$SAVE_DIR"
  --train_set_list "$SAVE_DIR/empty_train.json"
  --valid_set_list "$SAVE_DIR/empty_valid.json"
  --resume_model_path "$MODEL_PATH"
  --train_epochs "$TRAIN_EPOCHS"
  --save_utd 1
  --batch_size 1
  --learning_rate 0
  --num_workers 0
  --use_data_augment false
  --device cuda
  --use_wandb true
  --wandb_project_name "$WANDB_PROJECT"
  --normalization_file_path normalization.json
  --closed_loop_npz_root "$NPZ_ROOT"
  --closed_loop_classification_json_root "$CLASSIFICATION_JSON_ROOT"
  "${DATASET_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
  --closed_loop_near_miss_thresh 0.3
  --closed_loop_draw_every 8
  --closed_loop_replan_interval 40
)

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  # torchrun without --ddp true spawns N independent rank-0 jobs (duplicate wandb, all on GPU 0).
  export NCCL_NVLS_ENABLE=0
  export NCCL_P2P_DISABLE=0
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=lo
  TRAIN_ARGS+=(--ddp true)
  torchrun --nproc_per_node="$NPROC_PER_NODE" train_predictor.py "${TRAIN_ARGS[@]}"
else
  TRAIN_ARGS+=(--ddp true)
  torchrun --nproc_per_node="$NPROC_PER_NODE" train_predictor.py "${TRAIN_ARGS[@]}"
fi

echo
echo "Done. Local grouped CL outputs (if checkpoint save ran):"
find "$SAVE_DIR" -type d -name grouped 2>/dev/null | head -5 || true
echo "W&B project: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
