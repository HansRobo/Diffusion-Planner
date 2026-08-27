#!/bin/bash
# Closed-loop feature-importance ablation (collision/near-miss/clearance deltas)
# over short self-driven rollouts. See scripts/token_importance_closed_loop.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set to a checkpoint .pth (args.json alongside it)}"
NPZ_ROOT="${NPZ_ROOT:?NPZ_ROOT must be set to a dir tree of route NPZ frames}"
SIDECAR_ROOT="${SIDECAR_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUT_TSV="${OUT_TSV:?OUT_TSV must be set to an output .tsv path}"

mkdir -p "$(dirname "$OUT_TSV")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(dirname "$OUT_TSV")/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

ARGS=(
  --model_path "$MODEL_PATH"
  --npz_root "$NPZ_ROOT"
  --chunk_len "${CHUNK_LEN:-80}"
  --start_stride "${START_STRIDE:-80}"
  --min_chunk_len "${MIN_CHUNK_LEN:-20}"
  --num_shards "${NUM_SHARDS:-1}"
  --shard_index "${SHARD_INDEX:-0}"
  --sample_fraction "${SAMPLE_FRACTION:-1.0}"
  --max_windows "${MAX_WINDOWS:--1}"
  --device "$DEVICE"
  --near_miss_thresh "${NEAR_MISS_THRESH:-0.5}"
  --search_radius "${SEARCH_RADIUS:-1.5}"
  --warmup_steps "${WARMUP_STEPS:-0}"
  --unstick_after "${UNSTICK_AFTER:-300}"
  --out_tsv "$OUT_TSV"
)
[ -n "$SIDECAR_ROOT" ] && ARGS+=(--sidecar_root "$SIDECAR_ROOT")
[ -n "${CONFIGS:-}" ] && ARGS+=(--configs "$CONFIGS")

"$PYTHON_BIN" scripts/token_importance_closed_loop.py "${ARGS[@]}"

echo "done"
echo "  tsv: $OUT_TSV"
