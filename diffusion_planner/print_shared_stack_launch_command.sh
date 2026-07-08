#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-dfp_shared_stack_unified_ego_lam01_sft20_tier4main_node02_8gpu_tf32}
PORT=${PORT:-22391}
DFP_LAMBDA=${DFP_LAMBDA:-0.1}

cat <<EOF
ssh node02 'cd /mnt/nvme/Diffusion-Planner-dfp-shared-stack/diffusion_planner && RUN_NAME=${RUN_NAME} PORT=${PORT} DFP_LAMBDA=${DFP_LAMBDA} bash ./run_dfp_shared_stack_unified_ego_lam01_sft20.sh'
EOF
