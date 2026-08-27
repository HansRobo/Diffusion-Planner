#!/bin/bash
# Render attention over a SELF-DRIVEN closed-loop rollout (not open-loop log replay).
# See scripts/visualize_closed_loop_attention.py for the full explanation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set to a checkpoint .pth (args.json alongside it)}"
NPZ_ROOT="${NPZ_ROOT:?NPZ_ROOT must be set to a dir tree of route NPZ frames}"
SIDECAR_ROOT="${SIDECAR_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set to an output directory}"
OUTPUT_NAME="${OUTPUT_NAME:-closed_loop_attention}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

ARGS=(
  --model_path "$MODEL_PATH"
  --npz_root "$NPZ_ROOT"
  --attention_mode "${ATTENTION_MODE:-all_token}"
  --layer "${LAYER:-mean}"
  --top_k "${TOP_K:-20}"
  --view_range "${VIEW_RANGE:-80}"
  --colormap "${COLORMAP:-plasma}"
  --marker_size_min "${MARKER_SIZE_MIN:-25}"
  --marker_size_max "${MARKER_SIZE_MAX:-700}"
  --device "$DEVICE"
  --search_radius "${SEARCH_RADIUS:-1.5}"
  --near_miss_thresh "${NEAR_MISS_THRESH:-0.5}"
  --warmup_steps "${WARMUP_STEPS:-0}"
  --unstick_after "${UNSTICK_AFTER:-300}"
  --fps "${FPS:-10}"
  --video_width "${VIDEO_WIDTH:-1920}"
  --video_height "${VIDEO_HEIGHT:-1080}"
  --out_dir "$OUT_DIR"
  --output_name "$OUTPUT_NAME"
)
[ -n "$SIDECAR_ROOT" ] && ARGS+=(--sidecar_root "$SIDECAR_ROOT")
[ -n "${ROUTE:-}" ] && ARGS+=(--route "$ROUTE")
[ -n "${START:-}" ] && ARGS+=(--start "$START")
[ -n "${END:-}" ] && ARGS+=(--end "$END")
[ -n "${CHUNK_LEN:-}" ] && ARGS+=(--chunk_len "$CHUNK_LEN")
[ "${KEEP_FRAMES:-0}" = "1" ] && ARGS+=(--keep_frames)

"$PYTHON_BIN" scripts/visualize_closed_loop_attention.py "${ARGS[@]}"

echo "done"
echo "  video:   $OUT_DIR/${OUTPUT_NAME}.mp4"
echo "  data:    $OUT_DIR/${OUTPUT_NAME}.jsonl"
echo "  summary: $OUT_DIR/${OUTPUT_NAME}_summary.json"
