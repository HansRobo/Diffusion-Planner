#!/usr/bin/env bash
# Create canonical dataset directory layout for grouped closed-loop resolution.
#
# Maps a flat local valid copy onto the corpus structure expected by
# scenario_classification.discover_classification_eval_jobs:
#
#   {LOCAL_DATASET_ROOT}/{project}/{map}/valid/{date}/{bag}/
#
# Example (Odaiba x2_dev valid symlink):
#   LOCAL_DATASET_ROOT/x2_dev/2231_odaiba_shinagawa_copied_from_xx1/valid
#     -> /media/.../sakuraf_0623_full_sequence_x2_odaiba_valid/valid
#
# Usage:
#   bash diffusion_planner/setup_local_dataset_layout.sh
#
# Env overrides:
#   LOCAL_DATASET_ROOT  default: /media/ukenryu/external_ssd/diffusion_planner_local_dataset
#   SRC_VALID           default: /media/ukenryu/external_ssd/sakuraf_0623_full_sequence_x2_odaiba_valid/valid
#   DATASET_NAME        default: x2_dev/2231_odaiba_shinagawa_copied_from_xx1
set -euo pipefail

LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-/media/ukenryu/external_ssd/diffusion_planner_local_dataset}"
SRC_VALID="${SRC_VALID:-/media/ukenryu/external_ssd/sakuraf_0623_full_sequence_x2_odaiba_valid/valid}"
DATASET_NAME="${DATASET_NAME:-x2_dev/2231_odaiba_shinagawa_copied_from_xx1}"

TARGET_VALID="${LOCAL_DATASET_ROOT}/${DATASET_NAME}/valid"

if [[ ! -d "$SRC_VALID" ]]; then
  echo "Source valid tree not found: $SRC_VALID" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET_VALID")"
ln -sfn "$SRC_VALID" "$TARGET_VALID"

echo "Canonical local dataset layout ready:"
echo "  ${TARGET_VALID} -> ${SRC_VALID}"
echo
echo "Smoke / grouped CL NPZ_ROOT example:"
echo "  export NPZ_ROOT=${TARGET_VALID}/2026-01-15"
echo "  export DATASET_NAME=${DATASET_NAME}"
