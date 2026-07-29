#!/usr/bin/env bash
set -euo pipefail

# Wait for the accepted-policy cycle-2 rollout pass, validate its strict disk
# replay contract, and start the replay-only epochs without leaving the GPUs
# idle overnight.  This script never salvages or mutates an incomplete cache.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
MINE_RUN=${OUT}/20260719-152238_cycle02_selected_e004a005_onpolicy_everyrefresh_ramp20_mine
MINE_PID_FILE=${OUT}/cycle02_mine_20260719.pid
MINE_LOG=${OUT}/cycle02_mine_20260719.log
CACHE=${MINE_RUN}/replay_buffer
SOURCE_RUN=${OUT}/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100
MODEL=${SOURCE_RUN}/best_model_awr_retained_e004_a0p05.pth
ARGS=${SOURCE_RUN}/model_args.json
REPLAY_LOG=${OUT}/cycle02_replay_lr2e8_20260719.log
REPLAY_PID_FILE=${OUT}/cycle02_replay_lr2e8_20260719.pid
EXPECTED_PER_RANK=680960
EXPECTED_TOTAL=5447680

cd "${ROOT}"

mine_pid=$(cat "${MINE_PID_FILE}")
while kill -0 "${mine_pid}" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "${MINE_RUN}/final_summary.json" ]]; then
  echo "mine process ended without final_summary.json; refusing replay" >&2
  tail -200 "${MINE_LOG}" >&2 || true
  exit 1
fi

total=0
for rank in $(seq 0 7); do
  printf -v rank_name 'rank_%04d' "${rank}"
  rank_dir=${CACHE}/${rank_name}
  manifest=${rank_dir}/manifest.json
  [[ -f "${manifest}" ]] || {
    echo "missing ${manifest}; refusing replay" >&2
    exit 1
  }
  count=$(jq -er '.scene_count' "${manifest}")
  [[ "${count}" -eq "${EXPECTED_PER_RANK}" ]] || {
    echo "rank ${rank} has ${count}, expected ${EXPECTED_PER_RANK}" >&2
    exit 1
  }
  lines=$(wc -l < "${rank_dir}/scene_paths.jsonl")
  [[ "${lines}" -eq "${count}" ]] || {
    echo "rank ${rank} path index has ${lines}, manifest has ${count}" >&2
    exit 1
  }
  total=$((total + count))
done
[[ "${total}" -eq "${EXPECTED_TOTAL}" ]] || {
  echo "cache total ${total}, expected ${EXPECTED_TOTAL}" >&2
  exit 1
}

# Audit the exact full reward/weight memmaps before the first replay update.
# This reads only the small policy-dependent arrays (not trajectories or the
# multi-terabyte encoder) and makes the learning-signal contract durable for
# later analysis/visualisation.
SIGNAL_DIR=${MINE_RUN}/full_replay_signal_audit
"${ROOT}/.venv/bin/python" \
  "${ROOT}/rlvr/autoresearch/tools/analyze_full_replay_signal.py" \
  --replay-root "${CACHE}" --output-dir "${SIGNAL_DIR}" \
  --title "T4 Cycle 2 full replay: exact AWR learning signal"
jq -e --argjson expected "${EXPECTED_TOTAL}" '
  .scene_groups == $expected and
  .group_size == 10 and
  .candidate_trajectories == ($expected * 10) and
  (.distributions.weighted_reward_gain.mean > 0) and
  (.distributions.ess.mean >= 1) and
  (.distributions.ess.mean <= 10) and
  (.fractions.all_zero_reward >= 0) and
  (.fractions.all_zero_reward < 1)
' "${SIGNAL_DIR}/full_replay_signal.json" >/dev/null

# The resumed run owns a separate output directory and only reads this cache.
# Read-only mode prevents a later accidental refresh from truncating it.
chmod -R a-w "${CACHE}"

if [[ -f "${REPLAY_PID_FILE}" ]]; then
  old_pid=$(cat "${REPLAY_PID_FILE}")
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "cycle-2 replay is already running as ${old_pid}"
    exit 0
  fi
fi

sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${MODEL}' --args_path '${ARGS}' --train_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json --valid_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json --config '${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json' --output_dir '${OUT}' --exp_name cycle02_selected_e004a005_everyrefresh_ramp20_replay_lr2e8_e12to20 --device cuda --seed 3407 --start_epoch 12 --epochs 20 --resume_replay_root '${CACHE}' --inherit_refresh_baseline_root '${MINE_RUN}' --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --learning_rate 2e-8 --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle02-replay-lr2e8-e12to20 --wandb_mode online --wandb_tags original-dp awr cycle02 global-epoch-11-20 replay inherited-refresh-baseline hdp-group-relative smooth-ramp20 heading-preserve every-refresh lr2e-8 full-sequence 20260707 --no-skip_filtered_scenes > '${REPLAY_LOG}' 2>&1 < /dev/null & echo \$!" > "${REPLAY_PID_FILE}"

replay_pid=$(cat "${REPLAY_PID_FILE}")
sleep 5
kill -0 "${replay_pid}"
echo "cycle-2 replay launched: pid=${replay_pid} log=${REPLAY_LOG}"
