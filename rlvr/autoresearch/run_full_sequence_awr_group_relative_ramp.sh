#!/bin/sh
set -eu

# Formal full-data AWR run for the original x-start Diffusion Planner.
# Launch from a shell with access to the ubuntu-group validation manifest, e.g.
#   sg ubuntu -c 'rlvr/autoresearch/run_full_sequence_awr_group_relative_ramp.sh'
#
# This intentionally starts from the untouched v5 best DP.  It does not resume
# any checkpoint or replay cache from the 131k-scene representation ablations.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PY="${PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_LIST="${TRAIN_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}"
VALID_LIST="${VALID_LIST:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}"
MODEL="${MODEL_PATH:-/tmp/t4_v5_original_dp/model/best_model.pth}"
ARGS="${ARGS_PATH:-/tmp/t4_v5_original_dp/model/args.json}"
CONFIG="$ROOT/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json"
OUT="${OUTPUT_DIR:-$ROOT/outputs/awr_t4_full_sequence_filtered}"
EXPECTED_MODEL_SHA256=4ffaeea21cd29904da73349eea642e1d28f8ddbf02be363b7386e3a9b8ebcc75
EXPECTED_TRAIN_MANIFEST_LINES=5446155
EXPECTED_VALID_MANIFEST_LINES=46263

for path in "$PY" "$TRAIN_LIST" "$VALID_LIST" "$MODEL" "$ARGS" "$CONFIG"; do
  if ! test -r "$path" || ! test -s "$path"; then
    echo "required input is missing, empty, or unreadable: $path" >&2
    exit 1
  fi
done

# Both audited manifests are pretty-printed JSON arrays with one scene path
# per line plus the opening bracket.  Fail closed if a path is later replaced
# by a different split while retaining the same filename.  The semantic scene
# counts are therefore one less than these line counts.
TRAIN_MANIFEST_LINES=$(wc -l < "$TRAIN_LIST")
VALID_MANIFEST_LINES=$(wc -l < "$VALID_LIST")
if test "$TRAIN_MANIFEST_LINES" -ne "$EXPECTED_TRAIN_MANIFEST_LINES"; then
  echo "unexpected train manifest size: $TRAIN_LIST" >&2
  echo "expected lines: $EXPECTED_TRAIN_MANIFEST_LINES" >&2
  echo "actual lines:   $TRAIN_MANIFEST_LINES" >&2
  exit 1
fi
if test "$VALID_MANIFEST_LINES" -ne "$EXPECTED_VALID_MANIFEST_LINES"; then
  echo "unexpected valid manifest size: $VALID_LIST" >&2
  echo "expected lines: $EXPECTED_VALID_MANIFEST_LINES" >&2
  echo "actual lines:   $VALID_MANIFEST_LINES" >&2
  exit 1
fi

ACTUAL_MODEL_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
if test "$ACTUAL_MODEL_SHA256" != "$EXPECTED_MODEL_SHA256"; then
  echo "refusing to start from a non-original checkpoint: $MODEL" >&2
  echo "expected $EXPECTED_MODEL_SHA256" >&2
  echo "actual   $ACTUAL_MODEL_SHA256" >&2
  exit 1
fi

# The audited 131,072-scene replay cache occupies 78.63 GB.  Linear scaling
# predicts about 3.22 TB for all 5.446M scenes.  Require roughly 2x that amount
# at launch so cache refresh/checkpoint overhead cannot silently fill NVMe.
MIN_FREE_CACHE_KIB="${MIN_FREE_CACHE_KIB:-6442450944}"
AVAILABLE_CACHE_KIB=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
if test "$AVAILABLE_CACHE_KIB" -lt "$MIN_FREE_CACHE_KIB"; then
  echo "insufficient free NVMe for formal replay cache" >&2
  echo "available KiB: $AVAILABLE_CACHE_KIB" >&2
  echo "required  KiB: $MIN_FREE_CACHE_KIB" >&2
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if test "$GPU_COUNT" -lt 8; then
  echo "formal AWR requires 8 visible GPUs; found $GPU_COUNT" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/diffusion_planner:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
# The current legacy NPZ generation is off by one frame.  Remove this override
# only after the dataset owner confirms that newly generated data is aligned.
export DP_NEIGHBOR_FUTURE_OFFSET=1

if test -z "${WANDB_MODE+x}"; then
  if test -n "${WANDB_API_KEY:-}" || { test -r "$HOME/.netrc" && grep -q 'api.wandb.ai' "$HOME/.netrc"; }; then
    WANDB_MODE=online
  else
    WANDB_MODE=offline
  fi
fi
export WANDB_MODE

EPOCHS="${EPOCHS:-100}"
RUN_NAME="${WANDB_RUN_NAME:-original-dp-awr-full-group-relative-ramp-e${EPOCHS}}"

if test "${PREFLIGHT_ONLY:-0}" = 1; then
  echo "formal AWR preflight passed"
  echo "model sha256: $ACTUAL_MODEL_SHA256"
  echo "available cache KiB: $AVAILABLE_CACHE_KIB"
  echo "visible GPUs: $GPU_COUNT"
  echo "train scenes: $((TRAIN_MANIFEST_LINES - 1))"
  echo "valid scenes: $((VALID_MANIFEST_LINES - 1))"
  echo "epochs: $EPOCHS"
  exit 0
fi

exec "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT:-29627}" \
  -m rlvr.train_awr \
  --model_path "$MODEL" \
  --args_path "$ARGS" \
  --train_npz_list "$TRAIN_LIST" \
  --valid_npz_list "$VALID_LIST" \
  --config "$CONFIG" \
  --output_dir "$OUT" \
  --exp_name "full_sequence_20260707_group_relative_ramp20_preserve_e${EPOCHS}" \
  --device cuda \
  --seed 3407 \
  --start_epoch 1 \
  --epochs "$EPOCHS" \
  --max_train_scenes 0 \
  --max_valid_scenes 0 \
  --train_selector_scenes 65536 \
  --hdp_trajectory_augmentation_schedule first_five_epochs \
  --save_rollouts 0 \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-original-dp-awr}" \
  --wandb_entity "${WANDB_ENTITY:-advanced-technology-department}" \
  --wandb_run_name "$RUN_NAME" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_tags original-dp awr hdp-group-relative smooth-ramp20 heading-preserve full-sequence 20260707 \
  --no-skip_filtered_scenes
