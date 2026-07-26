#!/usr/bin/env bash
# Continue the score-aligned PlannerRFT/AWR recipe only after its full-data
# epoch-2 train selector improves.  First finish Cycle 1 at epochs 3--10 from
# the selected epoch-2 checkpoint, then hand the verified run to the standard
# refresh/replay supervisor for epochs 11 through a configurable target.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
TORCHRUN=${ROOT}/.venv/bin/torchrun
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
TARGET_EPOCH=${TARGET_EPOCH:-100}
[[ ${TARGET_EPOCH} =~ ^[1-9][0-9]*$ ]] \
  || { echo "TARGET_EPOCH must be a positive integer" >&2; exit 2; }
(( TARGET_EPOCH >= 20 && TARGET_EPOCH % 10 == 0 )) \
  || { echo "TARGET_EPOCH must be a multiple of 10 and at least 20" >&2; exit 2; }
TOTAL_CYCLES=$((TARGET_EPOCH / 10))
MINE_RUN=${MINE_RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine}
OVERLAY=${OVERLAY:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_positive_overlay/replay_buffer}
COMPRESSED_CONTEXT=${COMPRESSED_CONTEXT:-${MINE_RUN}/replay_context_zstd_v1}
ABLATION_PARENT=${ABLATION_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_scored_prefix_active_expert_ablation}
CYCLE1_PARENT=${CYCLE1_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycle01_replay_e03_e10}
CAMPAIGN_OUT=${CAMPAIGN_OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_prefix_full_cycles02_to${TOTAL_CYCLES}_e${TARGET_EPOCH}}
SUPERVISOR=${ROOT}/rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh
HEALTH_MONITOR=${ROOT}/rlvr/autoresearch/monitor_plannerrft_epoch100.sh
EXPECTED_SELECTOR_SHA256=49fd1863a3c1d54db59440da426e3475df67ccc4edd8da6ddde7e32945f3620c
EXPECTED_VALID_SHA256=91a1c8ef4004c7074f024495d13416456a4dbe2f0320f0b1a43282659d435dff
LOCK=${CYCLE1_PARENT}/continuation.lock

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

die() {
  echo "prefix campaign continuation: $*" >&2
  exit 1
}

latest_run_with() {
  local parent=$1 artifact=$2
  [[ -d ${parent} ]] || return 0
  find "${parent}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
    -exec test -s "{}/${artifact}" ';' -print 2>/dev/null | sort | tail -n 1
}

main() {
  mkdir -p "${CYCLE1_PARENT}"
  exec 9>"${LOCK}"
  flock -n 9 || exit 0

  local ablation audit source args selector_gain cycle1_run log
  ablation=$(latest_run_with "${ABLATION_PARENT}" final_summary.json)
  [[ -n ${ablation} ]] || die "no completed score-aligned epoch-2 run"
  audit=${ablation}/epoch_002_paired_audit.json
  [[ -s ${audit} ]] || die "epoch-2 paired audit is missing"
  selector_gain=$(jq -er '.fixed_train_selector.metrics.det_reward.improvement' "${audit}")
  "${ROOT}/.venv/bin/python" - "${selector_gain}" <<'PY'
import math
import sys

gain = float(sys.argv[1])
assert math.isfinite(gain) and gain > 0.0, gain
PY
  jq -e '
    .best_train_epoch == 2 and
    .best_train_is_starting_incumbent == false and
    .checkpoint_selection == "fixed_train_selector_ema_mean_det_reward"
  ' "${ablation}/final_summary.json" >/dev/null \
    || die "epoch-2 checkpoint was not selected by the fixed train selector"
  source=${ablation}/best_train.pth
  args=${ablation}/model_args.json
  [[ -s ${source} && -s ${args} ]] || die "selected epoch-2 model/args are missing"
  for rank in $(seq 0 7); do
    [[ -s ${COMPRESSED_CONTEXT}/rank_$(printf '%04d' "${rank}")/manifest.json ]] \
      || die "compressed context rank ${rank} is incomplete"
  done

  cycle1_run=$(latest_run_with "${CYCLE1_PARENT}" final_summary.json)
  if [[ -z ${cycle1_run} ]]; then
    [[ -z $(find "${CYCLE1_PARENT}" -mindepth 1 -maxdepth 1 -type d -name '20*' -print -quit) ]] \
      || die "an incomplete Cycle-1 continuation exists"
    log=${CYCLE1_PARENT}/launch.log
    "${TORCHRUN}" --standalone --nproc_per_node=8 -m rlvr.train_awr \
      --model_path "${source}" --args_path "${args}" \
      --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
      --config "${CONFIG}" --output_dir "${CYCLE1_PARENT}" \
      --exp_name plannerrft_prefix_full_cycle01_replay_e03_e10 \
      --device cuda --seed 3407 --start_epoch 3 --epochs 10 \
      --resume_replay_root "${OVERLAY}" --replay_sampling with_replacement \
      --compressed_replay_context_root "${COMPRESSED_CONTEXT}" \
      --awr_beta 2 --positive_advantage_only --positive_advantage_margin 0.01 \
      --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
      --expert_anchor_weight 0.4 --expert_anchor_active_groups_only \
      --awr_candidate_loss_horizon 40 --drop_all_zero_groups \
      --awr_loss_type plain_mse --diffusion_t_range 0.001 0.2 \
      --trainable_scope output --learning_rate 0.000001 \
      --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy \
      --max_train_scenes 0 --max_valid_scenes 0 \
      --eval_k 1 --eval_sample_steps 10 --train_selector_scenes 65536 \
      --expected_train_selector_sha256 "${EXPECTED_SELECTOR_SHA256}" \
      --expected_valid_sha256 "${EXPECTED_VALID_SHA256}" \
      --full_valid_interval 0 --hard_valid_scenes 0 \
      --scene_batch_size 192 --gradient_accumulation_scenes 192 \
      --scene_load_workers 4 --eval_scene_load_workers 1 \
      --eval_scene_batch_size 512 --save_rollouts 0 \
      --wandb --wandb_project original-dp-awr \
      --wandb_entity advanced-technology-department \
      --wandb_run_name original-dp-awr-plannerrft-prefix-cycle01-e03-e10 \
      --wandb_mode "${WANDB_MODE:-online}" \
      --wandb_tags original-dp awr plannerrft-guidance scored-prefix40 \
        active-group-expert full-sequence cycle01 global-e03-e10 \
      --no-skip_filtered_scenes > "${log}" 2>&1
    cycle1_run=$(sed -n 's/^Run directory: //p' "${log}" | tail -n 1)
  fi
  [[ -n ${cycle1_run} && -s ${cycle1_run}/final_summary.json ]] \
    || die "Cycle-1 epochs 3--10 did not complete"

  # The supervisor re-validates every epoch, selector, cache contract and
  # checkpoint hash before launching refresh epoch 11.
  mkdir -p "${CAMPAIGN_OUT}"
  setsid env CAMPAIGN="${CAMPAIGN_OUT}" INTERVAL=60 \
    TARGET_EPOCH="${TARGET_EPOCH}" \
    bash "${HEALTH_MONITOR}" \
    > "${CAMPAIGN_OUT}/health_monitor.launch.log" 2>&1 < /dev/null &
  exec env \
    CYCLE1_REPLAY_PARENT="${CYCLE1_PARENT}" \
    CYCLE1_FIRST=3 CYCLE1_LAST=10 \
    CYCLE1_SOURCE_CHECKPOINT="${source}" \
    AWR_CANDIDATE_LOSS_HORIZON=40 \
    EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=1 \
    CYCLE1_MINE_RUN="${MINE_RUN}" \
    COMPRESSED_CONTEXT_ROOT="${COMPRESSED_CONTEXT}" \
    TARGET_EPOCH="${TARGET_EPOCH}" \
    OUT="${CAMPAIGN_OUT}" \
    bash "${SUPERVISOR}"
}

main "$@"
