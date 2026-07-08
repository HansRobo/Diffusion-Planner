#!/usr/bin/env bash
set -euo pipefail

# Next single-decoder DFP candidate, used only if the current v4d run underperforms
# after epoch3. This keeps the same data/training setup as v4d while enabling the
# code-level chunk mask fix and restoring ego-history tokens in the encoder.
export RUN_NAME=${RUN_NAME:-dfp_joint_temporal_agent_chunkmask_history_step1_base_pdms_tier4main_node01_8gpu_tf32}
export SAVE_DIR=${SAVE_DIR:-/mnt/nvme/Diffusion-Planner-dfp-shared-stack/outputs/${RUN_NAME}}
export TRAIN_LIST=${TRAIN_LIST:-/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_train_sft_fullseq_from_20260622_step3.json}
export VALID_LIST=${VALID_LIST:-/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_valid_sft_fullseq_from_20260622_step3.json}
export DFP_DECODER_MODE=${DFP_DECODER_MODE:-joint_temporal_agent}
export USE_EGO_HISTORY=${USE_EGO_HISTORY:-True}
export SKIP_FILTER=${SKIP_FILTER:-False}
export PORT=${PORT:-22431}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

exec bash /mnt/nvme/Diffusion-Planner-dfp-shared-stack/diffusion_planner/run_dfp_shared_stack_step1_base_pdms.sh
