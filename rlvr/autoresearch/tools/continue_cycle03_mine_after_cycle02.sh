#!/usr/bin/env bash
set -euo pipefail

# Wait for the global-epoch 12--20 replay run, require a completed train-set
# selection artifact, rotate the already-accepted cycle-1 bulk cache, and
# launch only the next on-policy refresh (global epoch 21).  Replay epochs
# 22--30 are intentionally not launched here: their LR is chosen after the
# cycle-2 trajectory is inspected.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
CYCLE1_RUN=${OUT}/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100
CYCLE2_MINE_RUN=${OUT}/20260719-152238_cycle02_selected_e004a005_onpolicy_everyrefresh_ramp20_mine
CYCLE2_REPLAY_PID_FILE=${OUT}/cycle02_replay_lr2e8_20260719.pid
CYCLE2_REPLAY_LOG=${OUT}/cycle02_replay_lr2e8_20260719.log
CYCLE2_MINE_PID_FILE=${OUT}/cycle02_mine_20260719.pid
CYCLE2_SUPERVISOR_PID_FILE=${OUT}/cycle02_supervisor_20260719.pid
CYCLE3_LOG=${OUT}/cycle03_mine_global_e21_20260719.log
CYCLE3_PID_FILE=${OUT}/cycle03_mine_global_e21_20260719.pid
DEPLOY_SCENES=${CYCLE1_RUN}/scene_selection_valid.json
DEPLOY_BASELINE=${CYCLE1_RUN}/deployment_full10/e004_a0p05/scenes.json

cd "${ROOT}"

process_matches() {
  local pid=$1
  local needle=$2
  [[ -r /proc/${pid}/cmdline ]] || return 1
  local command
  command=$(tr '\0' ' ' < "/proc/${pid}/cmdline")
  [[ ${command} == *"${needle}"* ]]
}

# The first supervisor creates the replay PID file only after a strict 8-rank
# cache close. If both upstream processes end before then, fail instead of
# waiting forever or launching from an incomplete cache.
while [[ ! -s ${CYCLE2_REPLAY_PID_FILE} ]]; do
  mine_alive=0
  supervisor_alive=0
  if [[ -s ${CYCLE2_MINE_PID_FILE} ]]; then
    mine_pid=$(cat "${CYCLE2_MINE_PID_FILE}")
    kill -0 "${mine_pid}" 2>/dev/null && mine_alive=1
  fi
  if [[ -s ${CYCLE2_SUPERVISOR_PID_FILE} ]]; then
    supervisor_pid=$(cat "${CYCLE2_SUPERVISOR_PID_FILE}")
    kill -0 "${supervisor_pid}" 2>/dev/null && supervisor_alive=1
  fi
  if [[ ${mine_alive} -eq 0 && ${supervisor_alive} -eq 0 ]]; then
    echo "cycle-2 mining/supervisor ended before replay launch" >&2
    exit 1
  fi
  sleep 60
done

replay_pid=$(cat "${CYCLE2_REPLAY_PID_FILE}")
while process_matches "${replay_pid}" "rlvr.train_awr"; do
  sleep 60
done

[[ -f ${CYCLE2_REPLAY_LOG} ]] || {
  echo "missing cycle-2 replay log" >&2
  exit 1
}
cycle2_run=$(sed -n 's/^Run directory: //p' "${CYCLE2_REPLAY_LOG}" | tail -n 1)
[[ -n ${cycle2_run} ]] || {
  echo "cycle-2 replay log contains no run directory" >&2
  tail -200 "${CYCLE2_REPLAY_LOG}" >&2 || true
  exit 1
}
cycle2_run=$(realpath -e "${cycle2_run}")
case "$(basename "${cycle2_run}")" in
  *_cycle02_selected_e004a005_everyrefresh_ramp20_replay_lr2e8_e12to20) ;;
  *) echo "unexpected cycle-2 run: ${cycle2_run}" >&2; exit 1 ;;
esac

[[ -f ${cycle2_run}/final_summary.json && -s ${cycle2_run}/best_train.pth ]] || {
  echo "cycle-2 replay ended without final_summary.json/best_train.pth" >&2
  tail -200 "${CYCLE2_REPLAY_LOG}" >&2 || true
  exit 1
}
best_epoch=$(jq -er '.best_train_epoch' "${cycle2_run}/final_summary.json")
best_reward=$(jq -er '.best_train_reward' "${cycle2_run}/final_summary.json")
starting_epoch=$(jq -er '.starting_train_selection_epoch' "${cycle2_run}/final_summary.json")
starting_reward=$(jq -er '.starting_train_selection_reward' "${cycle2_run}/final_summary.json")
reward_delta=$(jq -er '.best_train_reward_delta_from_start' "${cycle2_run}/final_summary.json")
selection_rule=$(jq -er '.checkpoint_selection' "${cycle2_run}/final_summary.json")
[[ ${selection_rule} == "fixed_train_selector_ema_mean_det_reward" ]] || {
  echo "unexpected checkpoint selector: ${selection_rule}" >&2
  exit 1
}
"${ROOT}/.venv/bin/python" - "${best_reward}" "${starting_reward}" \
  "${reward_delta}" <<'PY'
import math
import sys

best, start, reported_delta = map(float, sys.argv[1:])
assert all(math.isfinite(value) for value in (best, start, reported_delta))
assert best + 1e-12 >= start, (best, start)
assert math.isclose(best - start, reported_delta, rel_tol=0.0, abs_tol=1e-12), (
    best,
    start,
    reported_delta,
)
PY
best_is_incumbent=$(jq -r '.best_train_is_starting_incumbent' "${cycle2_run}/final_summary.json")
[[ ${best_is_incumbent} == "true" || ${best_is_incumbent} == "false" ]] || {
  echo "invalid best_train_is_starting_incumbent=${best_is_incumbent}" >&2
  exit 1
}
if [[ ${best_is_incumbent} == "true" ]]; then
  [[ ${best_epoch} -eq ${starting_epoch} ]] || {
    echo "incumbent flag disagrees with epochs: best=${best_epoch} start=${starting_epoch}" >&2
    exit 1
  }
else
  [[ ${best_epoch} -ge 12 && ${best_epoch} -le 20 ]] || {
    echo "new cycle-2 best epoch ${best_epoch} is outside replay epochs 12--20" >&2
    exit 1
  }
fi
[[ ${best_reward} != "null" ]] || {
  echo "cycle-2 best reward is null" >&2
  exit 1
}
checkpoint_sha=$(sha256sum "${cycle2_run}/best_train.pth" | awk '{print $1}')
summary_checkpoint_sha=$(jq -er '.best_train_checkpoint_sha256' "${cycle2_run}/final_summary.json")
[[ ${checkpoint_sha} == "${summary_checkpoint_sha}" ]] || {
  echo "cycle-2 best checkpoint hash disagrees with final summary" >&2
  exit 1
}
resume_root=$(jq -er '.resume_replay_root' "${cycle2_run}/provenance.json")
[[ $(realpath -e "${resume_root}") == $(realpath -e "${CYCLE2_MINE_RUN}/replay_buffer") ]] || {
  echo "cycle-2 replay provenance points at unexpected cache: ${resume_root}" >&2
  exit 1
}
echo "cycle-2 accepted train candidate: epoch=${best_epoch} reward=${best_reward} incumbent=${best_is_incumbent} sha=${checkpoint_sha}"

# The per-epoch selector deliberately uses HDP's cheaper 5-step sampler. Before
# propagating the selected EMA into another on-policy refresh, measure it once
# with the actual 10-step deployment sampler on all 46,262 fixed validation
# scenes. This is report-only evidence: train-set reward remains the checkpoint
# selector and this diagnostic cannot veto or replace it.
deploy_out=${cycle2_run}/deployment_full10/best_train
deploy_report=${cycle2_run}/deployment_full10/paired_incremental_vs_cycle1.json
mkdir -p "${deploy_out}"
sg ubuntu -c "cd '${ROOT}' && env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 '${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py' --model_path '${cycle2_run}/best_train.pth' --args_path '${cycle2_run}/model_args.json' --scenes '${DEPLOY_SCENES}' --config '${cycle2_run}/effective_config.json' --output_dir '${deploy_out}' --label cycle02_best_train_full_valid_dp10 --batch_size 512 --scene_load_workers 1 --sample_steps 10 --amp_dtype bf16 --compile_mode default"
DP_NEIGHBOR_FUTURE_OFFSET=1 PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
  "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
  --aggregate --model_path "${cycle2_run}/best_train.pth" \
  --args_path "${cycle2_run}/model_args.json" --scenes "${DEPLOY_SCENES}" \
  --config "${cycle2_run}/effective_config.json" --output_dir "${deploy_out}" \
  --label cycle02_best_train_full_valid_dp10
"${ROOT}/.venv/bin/python" -m rlvr.autoresearch.tools.full_paired_eval_report \
  --baseline "${DEPLOY_BASELINE}" --candidate "${deploy_out}/scenes.json" \
  --epoch "${best_epoch}" --bootstrap 10000 --seed 3407 --output "${deploy_report}"
"${ROOT}/.venv/bin/python" - "${deploy_out}/manifest.json" "${deploy_report}" \
  "${checkpoint_sha}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
report_path = pathlib.Path(sys.argv[2])
expected_sha = sys.argv[3]
manifest = json.loads(manifest_path.read_text())
assert manifest["scene_contract_verified"] is True
assert int(manifest["actual_scene_count"]) == 46262
assert int(manifest["expected_scene_count"]) == 46262
assert int(manifest["sample_steps"]) == 10
assert int(manifest["candidate_count"]) == 1
assert float(manifest["noise_scale"]) == 0.0
assert manifest["checkpoint_payload"] == "ema_state_dict"
model = pathlib.Path(manifest["model"])
digest = hashlib.sha256(model.read_bytes()).hexdigest()
assert digest == expected_sha
report = json.loads(report_path.read_text())
row = next(iter(report["epochs"].values()))
assert int(row["scene_count"]) == 46262
print(
    "cycle-2 deployment audit: "
    f"delta={row['metrics']['det_reward']['improvement']:+.8f} "
    f"ci95={row['metrics']['det_reward']['improvement_ci95']}"
)
PY

# Preserve the full common-random paired policy-distribution audit before the
# cycle-1 trajectory/encoding arrays are rotated away.
"${ROOT}/.venv/bin/python" \
  "${ROOT}/rlvr/autoresearch/tools/audit_paired_replay_cache.py" \
  --reference "${CYCLE1_RUN}/replay_buffer" \
  --candidate "${CYCLE2_MINE_RUN}/replay_buffer" \
  --output "${CYCLE2_MINE_RUN}/paired_full_cache_vs_cycle1.json" \
  --diversity_groups_per_rank 512

# Cycle 1 is already independently selected and full-validated. Retain its
# manifests/config/checkpoint evidence while reclaiming only bulk replay arrays.
"${ROOT}/rlvr/autoresearch/tools/archive_prune_replay_cache.sh" \
  "${CYCLE1_RUN}" "${CYCLE1_RUN}" 5447680 --execute

if [[ -s ${CYCLE3_PID_FILE} ]]; then
  existing=$(cat "${CYCLE3_PID_FILE}")
  if process_matches "${existing}" "rlvr.train_awr"; then
    echo "cycle-3 global-epoch-21 mining already running as ${existing}"
    exit 0
  fi
fi

free_bytes=$(df -B1 --output=avail /mnt/nvme | tail -n 1 | tr -d ' ')
min_free_bytes=$((6 * 1024 * 1024 * 1024 * 1024))
[[ ${free_bytes} -ge ${min_free_bytes} ]] || {
  echo "only ${free_bytes} bytes free after rotation; refusing 3-TiB cache" >&2
  exit 1
}

sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${cycle2_run}/best_train.pth' --args_path '${cycle2_run}/model_args.json' --train_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json --valid_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json --config '${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json' --output_dir '${OUT}' --exp_name cycle03_besttrain_onpolicy_everyrefresh_ramp20_mine_global_e21 --device cuda --seed 3407 --start_epoch 21 --epochs 21 --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --learning_rate 2e-8 --shared_replay_encoding_root '${CYCLE2_MINE_RUN}/replay_buffer' --shared_replay_order_epoch 1 --rollout_prefetch_batches 1 --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle03-onpolicy-mine-global-e21 --wandb_mode online --wandb_tags original-dp awr cycle03 global-epoch-21 onpolicy-refresh hdp-group-relative drop-tied-groups smooth-ramp20 heading-preserve every-refresh shared-frozen-encoder-cache cached-replay-context rollout-prefetch-1 full-sequence 20260707 --no-skip_filtered_scenes > '${CYCLE3_LOG}' 2>&1 < /dev/null & echo \$!" > "${CYCLE3_PID_FILE}"

cycle3_pid=$(cat "${CYCLE3_PID_FILE}")
sleep 5
process_matches "${cycle3_pid}" "rlvr.train_awr"
echo "cycle-3 global-epoch-21 mining launched: pid=${cycle3_pid} log=${CYCLE3_LOG}"
