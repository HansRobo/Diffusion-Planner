#!/usr/bin/env bash
set -euo pipefail

# Close Cycle 2 across its split conservative-commit processes, select the global best on
# the fixed train selector, report that checkpoint with the deployment sampler,
# and launch the next on-policy refresh at global epoch 21.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json
TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
SOURCE_RUN=${OUT}/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100
SOURCE_CHECKPOINT=${SOURCE_RUN}/best_model_awr_retained_e004_a0p05.pth
CYCLE2_MINE_RUN=${OUT}/20260719-152238_cycle02_selected_e004a005_onpolicy_everyrefresh_ramp20_mine
E12_RUN=${OUT}/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache
E13_LOG=${OUT}/cycle02_hdp_epoch_commit_e13to20_20260719.log
E13_PID_FILE=${OUT}/cycle02_hdp_epoch_commit_e13to20_20260719.pid
E13_SUPERVISOR_PID_FILE=${OUT}/cycle02_hdp_epoch_commit_e13to20_supervisor.pid
CYCLE3_LOG=${OUT}/cycle03_conservative_mine_global_e21.log
CYCLE3_PID_FILE=${OUT}/cycle03_conservative_mine_global_e21.pid
SELF_PID_FILE=${OUT}/cycle03_conservative_after_cycle02_supervisor.pid
DEPLOY_SCENES=${SOURCE_RUN}/scene_selection_valid.json
DEPLOY_BASELINE=${SOURCE_RUN}/deployment_full10/e004_a0p05/scenes.json

cd "${ROOT}"

process_matches() {
  local pid=$1
  local needle=$2
  [[ -r /proc/${pid}/cmdline ]] || return 1
  local command
  command=$(tr '\0' ' ' < "/proc/${pid}/cmdline")
  [[ ${command} == *"${needle}"* ]]
}

if [[ -s ${SELF_PID_FILE} ]]; then
  prior=$(cat "${SELF_PID_FILE}")
  if [[ ${prior} != "$$" ]] && process_matches "${prior}" \
      "continue_cycle03_exact_after_cycle02_hdp.sh"; then
    echo "Cycle-3 conservative-arm supervisor already running as ${prior}"
    exit 0
  fi
fi
printf '%s\n' "$$" > "${SELF_PID_FILE}"

while [[ ! -s ${E13_PID_FILE} ]]; do
  if [[ -s ${E13_SUPERVISOR_PID_FILE} ]]; then
    upstream=$(cat "${E13_SUPERVISOR_PID_FILE}")
    if ! kill -0 "${upstream}" 2>/dev/null; then
      echo "epoch 13--20 supervisor ended before launch" >&2
      tail -200 "${OUT}/cycle02_hdp_epoch_commit_e13to20_supervisor.log" >&2 || true
      exit 1
    fi
  fi
  sleep 30
done

e13_pid=$(cat "${E13_PID_FILE}")
while process_matches "${e13_pid}" "rlvr.train_awr"; do
  sleep 30
done

cycle2_run=$(sed -n 's/^Run directory: //p' "${E13_LOG}" | tail -n 1)
[[ -n ${cycle2_run} ]] || {
  echo "epoch 13--20 log contains no run directory" >&2
  tail -200 "${E13_LOG}" >&2 || true
  exit 1
}
cycle2_run=$(realpath -e "${cycle2_run}")
case "$(basename "${cycle2_run}")" in
  *_cycle02_hdp_epoch_commit_e13to20) ;;
  *) echo "unexpected Cycle-2 continuation run: ${cycle2_run}" >&2; exit 1 ;;
esac

[[ -s ${cycle2_run}/final_summary.json ]] || {
  echo "Cycle-2 continuation ended without final_summary.json" >&2
  tail -200 "${E13_LOG}" >&2 || true
  exit 1
}

"${ROOT}/.venv/bin/python" - "${cycle2_run}" <<'PY'
import json
import math
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
final = json.loads((run / "final_summary.json").read_text())
assert int(final["start_epoch"]) == 13
assert int(final["epochs"]) == 20
assert final["checkpoint_selection"] == "fixed_train_selector_ema_mean_det_reward"
for epoch in range(13, 21):
    summary = json.loads(
        (run / f"train_epoch_{epoch:03d}" / "summary.json").read_text()
    )
    proposal = float(summary["policy_proposal_relative_l2"])
    accepted = float(summary["policy_accepted_relative_l2"])
    assert proposal > 0.0
    assert math.isclose(accepted / proposal, 0.05, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(float(summary["ema_policy_commit_rate"]), 0.05, abs_tol=1e-12)
    assert float(summary["optimizer_state_reset_after_policy_commit"]) == 1.0
print("verified conservative Original-DP commits for epochs 13--20")
PY

selection_dir=${cycle2_run}/global_train_selection_through_epoch020
global_checkpoint=${selection_dir}/best_train_global.pth
global_manifest=${selection_dir}/selection.json
candidate_args=(
  --candidate "incumbent_e11|11|${SOURCE_CHECKPOINT}|${CYCLE2_MINE_RUN}/train_selector_epoch_000/summary.json|${CYCLE2_MINE_RUN}/train_selector_epoch_000/scenes.json"
  --candidate "accepted_e12|12|${E12_RUN}/epoch_012.pth|${E12_RUN}/train_selector_epoch_012/summary.json|${E12_RUN}/train_selector_epoch_012/scenes.json"
)
for epoch in $(seq 13 20); do
  printf -v epoch_name '%03d' "${epoch}"
  candidate_args+=(
    --candidate "accepted_e${epoch}|${epoch}|${cycle2_run}/epoch_${epoch_name}.pth|${cycle2_run}/train_selector_epoch_${epoch_name}/summary.json|${cycle2_run}/train_selector_epoch_${epoch_name}/scenes.json"
  )
done
"${ROOT}/.venv/bin/python" \
  -m rlvr.autoresearch.tools.select_global_train_checkpoint \
  "${candidate_args[@]}" --expected-scene-count 65536 \
  --output-checkpoint "${global_checkpoint}" \
  --output-manifest "${global_manifest}"

winner_epoch=$(jq -er '.winner.selection_epoch' "${global_manifest}")
winner_reward=$(jq -er '.winner.mean_det_reward' "${global_manifest}")
winner_sha=$(jq -er '.materialized_checkpoint_sha256' "${global_manifest}")
echo "Cycle-2 global train winner: epoch=${winner_epoch} reward=${winner_reward} sha=${winner_sha}"

# Report the selected train winner under the deployment-time 10-step sampler.
# This is intentionally report-only and cannot change the selection above.
deploy_out=${selection_dir}/deployment_full10
deploy_report=${selection_dir}/deployment_full10_paired_vs_incumbent.json
if [[ ! -s ${deploy_out}/manifest.json ]]; then
  sg ubuntu -c "cd '${ROOT}' && env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 '${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py' --model_path '${global_checkpoint}' --args_path '${SOURCE_RUN}/model_args.json' --scenes '${DEPLOY_SCENES}' --config '${cycle2_run}/effective_config.json' --output_dir '${deploy_out}' --label cycle02_global_train_best_full_valid_dp10 --batch_size 512 --scene_load_workers 1 --sample_steps 10 --amp_dtype bf16 --compile_mode default"
  DP_NEIGHBOR_FUTURE_OFFSET=1 PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
    "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
    --aggregate --model_path "${global_checkpoint}" \
    --args_path "${SOURCE_RUN}/model_args.json" --scenes "${DEPLOY_SCENES}" \
    --config "${cycle2_run}/effective_config.json" --output_dir "${deploy_out}" \
    --label cycle02_global_train_best_full_valid_dp10
fi
"${ROOT}/.venv/bin/python" -m rlvr.autoresearch.tools.full_paired_eval_report \
  --baseline "${DEPLOY_BASELINE}" --candidate "${deploy_out}/scenes.json" \
  --epoch "${winner_epoch}" --bootstrap 10000 --seed 3407 \
  --output "${deploy_report}"

if [[ -s ${CYCLE3_PID_FILE} ]]; then
  existing=$(cat "${CYCLE3_PID_FILE}")
  if process_matches "${existing}" "rlvr.train_awr"; then
    echo "Cycle-3 epoch-21 refresh already running as ${existing}"
    exit 0
  fi
fi

free_bytes=$(df -B1 --output=avail /mnt/nvme | tail -n 1 | tr -d ' ')
min_free_bytes=$((6 * 1024 * 1024 * 1024 * 1024))
[[ ${free_bytes} -ge ${min_free_bytes} ]] || {
  echo "only ${free_bytes} bytes free; refusing a new replay generation" >&2
  exit 1
}

sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${global_checkpoint}' --args_path '${SOURCE_RUN}/model_args.json' --train_npz_list '${TRAIN_LIST}' --valid_npz_list '${VALID_LIST}' --config '${CONFIG}' --output_dir '${OUT}' --exp_name cycle03_conservative_besttrain_onpolicy_ramp20_mine_global_e21 --device cuda --seed 3407 --start_epoch 21 --epochs 21 --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --learning_rate 1e-6 --shared_replay_encoding_root '${CYCLE2_MINE_RUN}/replay_buffer' --shared_replay_order_epoch 1 --rollout_prefetch_batches 1 --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle03-conservative-mine-global-e21 --wandb_mode online --wandb_tags original-dp awr cycle03 conservative-accepted-policy-interpolation local-t4-adaptation global-epoch-21 onpolicy-refresh mine-only-no-policy-update hdp-group-relative drop-tied-groups smooth-ramp20 heading-preserve every-refresh shared-frozen-encoder-cache full-sequence 20260707 --no-skip_filtered_scenes > '${CYCLE3_LOG}' 2>&1 < /dev/null & echo \$!" > "${CYCLE3_PID_FILE}"

cycle3_pid=$(cat "${CYCLE3_PID_FILE}")
sleep 5
process_matches "${cycle3_pid}" "rlvr.train_awr" || {
  echo "Cycle-3 epoch-21 refresh failed to stay alive" >&2
  tail -200 "${CYCLE3_LOG}" >&2 || true
  exit 1
}
echo "Cycle-3 epoch-21 conservative-arm refresh launched: pid=${cycle3_pid} log=${CYCLE3_LOG}"
