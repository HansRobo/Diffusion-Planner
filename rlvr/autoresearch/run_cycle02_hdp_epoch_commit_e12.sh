#!/usr/bin/env bash
set -euo pipefail

# Matched one-epoch policy-commit audit on the immutable Cycle-2 replay cache.
# It changes only optimizer LR / EMA transaction; candidate trajectories,
# rewards, weights, scene order, selector and validation are inherited exactly.
# The 5% boundary interpolation is a local T4/Original-DP adaptation. HDP's
# paper table lists EMA=0.05, but the clean public NAVSIM code disables EMA and
# does not establish this transaction.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
SOURCE_RUN=${OUT}/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100
SOURCE_CHECKPOINT=${SOURCE_RUN}/best_model_awr_retained_e004_a0p05.pth
MINE_RUN=${OUT}/20260719-152238_cycle02_selected_e004a005_onpolicy_everyrefresh_ramp20_mine
CACHE=${MINE_RUN}/replay_buffer
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json
LOG=${OUT}/cycle02_hdp_epoch_commit_e12_20260719.log
PID_FILE=${OUT}/cycle02_hdp_epoch_commit_e12_20260719.pid

cd "${ROOT}"

for rank in $(seq 0 7); do
  printf -v rank_name 'rank_%04d' "${rank}"
  manifest=${CACHE}/${rank_name}/manifest.json
  [[ -s ${manifest} ]] || { echo "missing ${manifest}" >&2; exit 1; }
  [[ $(jq -er '.scene_count' "${manifest}") -eq 680960 ]] || {
    echo "incomplete replay rank ${rank}" >&2
    exit 1
  }
done

# Never contend with another 8-GPU trainer. Evaluation/wait supervisors do not
# match this command; only a live rlvr.train_awr process blocks launch.
if pgrep -af 'rlvr.train_awr' >/tmp/awr_active_trainers.$$; then
  echo "refusing matched pilot while another rlvr.train_awr is active:" >&2
  cat /tmp/awr_active_trainers.$$ >&2
  rm -f /tmp/awr_active_trainers.$$
  exit 1
fi
rm -f /tmp/awr_active_trainers.$$

sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${SOURCE_CHECKPOINT}' --args_path '${SOURCE_RUN}/model_args.json' --train_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json --valid_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json --config '${CONFIG}' --output_dir '${OUT}' --exp_name cycle02_conservative_commit_e12_matched_cache --device cuda --seed 3407 --start_epoch 12 --epochs 12 --resume_replay_root '${CACHE}' --inherit_refresh_baseline_root '${MINE_RUN}' --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --no-drop_all_zero_groups --learning_rate 1e-6 --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle02-conservative-commit-e12 --wandb_mode online --wandb_tags original-dp awr cycle02 matched-cache conservative-accepted-policy-interpolation new-policy-rate-0.05 local-t4-adaptation not-exact-public-hdp global-epoch-12 lr-1e-6 full-valid full-sequence 20260707 --no-skip_filtered_scenes > '${LOG}' 2>&1 < /dev/null & echo \$!" > "${PID_FILE}"

pid=$(cat "${PID_FILE}")
sleep 5
[[ -r /proc/${pid}/cmdline ]] || {
  echo "matched policy-commit pilot failed to stay alive" >&2
  tail -200 "${LOG}" >&2 || true
  exit 1
}
echo "matched conservative Original-DP commit pilot launched: pid=${pid} log=${LOG}"
