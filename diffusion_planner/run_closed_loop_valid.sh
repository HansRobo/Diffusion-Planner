#!/usr/bin/env bash
# Local closed-loop validation for Diffusion-Planner.
# Usage:
#   MODE=smoke ./diffusion_planner/run_closed_loop_valid.sh
#   MODE=full  NPZ_ROOT=/path/to/date_dir ./diffusion_planner/run_closed_loop_valid.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-$REPO/best_model.pth}"
NPZ_ROOT="${NPZ_ROOT:-/media/ukenryu/external_ssd/sakuraf_0623_full_sequence_x2_odaiba_valid/valid/2026-01-15}"
MODE="${MODE:-smoke}"   # smoke | single_seq | full

if [[ -f "$REPO/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO/.venv/bin/activate"
fi
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ARGS_JSON="$(dirname "$MODEL")/args.json"
if [[ ! -f "$ARGS_JSON" && -f "$REPO/models/args.json" ]]; then
  echo "Linking models/args.json -> $(dirname "$MODEL")/args.json"
  ln -sf "$REPO/models/args.json" "$ARGS_JSON"
fi

test -f "$MODEL" || { echo "missing checkpoint: $MODEL"; exit 1; }
test -f "$ARGS_JSON" || {
  echo "missing args.json next to checkpoint (required by load_model)"
  echo "  expected: $ARGS_JSON"
  exit 1
}
test -d "$NPZ_ROOT" || { echo "missing npz_root: $NPZ_ROOT"; exit 1; }
python3 -c "
import sys
import torch
if not torch.cuda.is_available():
    print('CUDA not available.', file=sys.stderr)
    print(f'  torch={torch.__version__}  cuda_built={torch.version.cuda}', file=sys.stderr)
    print('  If driver reports CUDA 12.x but torch is +cu130, pin a cu124 wheel in uv.', file=sys.stderr)
    sys.exit(1)
"
command -v ffmpeg >/dev/null || { echo "ffmpeg required for MP4 output"; exit 1; }

case "$MODE" in
  smoke)
    SEG_LEN=150
    DRAW_EVERY=16
    NPZ_ROOT="${NPZ_ROOT}/13-49-19"
    ;;
  single_seq)
    SEG_LEN=600
    DRAW_EVERY=8
    NPZ_ROOT="${NPZ_ROOT}/13-49-19"
    ;;
  full)
    SEG_LEN=600
    DRAW_EVERY=8
    ;;
  *)
    echo "MODE must be smoke, single_seq, or full"; exit 1
    ;;
esac

echo "model:    $MODEL"
echo "npz_root: $NPZ_ROOT"
echo "mode:     $MODE (seg_len=$SEG_LEN draw_every=$DRAW_EVERY)"
echo "output:   $(dirname "$MODEL")/closed_loop/<timestamp>/"

python3 diffusion_planner/valid_predictor_closed_loop.py \
  --model_path "$MODEL" \
  --npz_root "$NPZ_ROOT" \
  --seg_len "$SEG_LEN" \
  --draw_every "$DRAW_EVERY" \
  --replan_interval 40 \
  --near_miss_thresh 0.3 \
  --device cuda
