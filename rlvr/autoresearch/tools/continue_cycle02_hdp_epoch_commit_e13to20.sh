#!/usr/bin/env bash
set -euo pipefail

# Continue the conservative Original-DP epoch-boundary interpolation arm after
# the isolated epoch-12 matched-cache audit.  This is a local T4 adaptation,
# not an exact public-HDP reproduction.  The trajectory starts from the
# accepted epoch-12 policy (not the train-selected incumbent); global
# checkpoint selection remains an independent comparison against the fixed
# train-set incumbent after epoch 20.

[[ ${ALLOW_CONSERVATIVE_COMMIT_CONTINUATION:-0} == 1 ]] || {
  echo "refusing automatic continuation: set ALLOW_CONSERVATIVE_COMMIT_CONTINUATION=1 after the epoch-12 paired train-selector audit" >&2
  exit 2
}

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json
TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
MINE_RUN=${OUT}/20260719-152238_cycle02_selected_e004a005_onpolicy_everyrefresh_ramp20_mine
OVERLAY_RUN=${OUT}/20260719-cycle02_tied_zero_drop_overlay
E12_RUN=${OUT}/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache
E12_PID_FILE=${OUT}/cycle02_hdp_epoch_commit_e12_20260719.pid
E12_LOG=${OUT}/cycle02_hdp_epoch_commit_e12_20260719.log
E13_LOG=${OUT}/cycle02_hdp_epoch_commit_e13to20_20260719.log
E13_PID_FILE=${OUT}/cycle02_hdp_epoch_commit_e13to20_20260719.pid
SUPERVISOR_PID_FILE=${OUT}/cycle02_hdp_epoch_commit_e13to20_supervisor.pid

cd "${ROOT}"

process_matches() {
  local pid=$1
  local needle=$2
  [[ -r /proc/${pid}/cmdline ]] || return 1
  local command
  command=$(tr '\0' ' ' < "/proc/${pid}/cmdline")
  [[ ${command} == *"${needle}"* ]]
}

if [[ -s ${SUPERVISOR_PID_FILE} ]]; then
  prior=$(cat "${SUPERVISOR_PID_FILE}")
  if [[ ${prior} != "$$" ]] && process_matches "${prior}" \
      "continue_cycle02_hdp_epoch_commit_e13to20.sh"; then
    echo "epoch 13--20 continuation supervisor already running as ${prior}"
    exit 0
  fi
fi
printf '%s\n' "$$" > "${SUPERVISOR_PID_FILE}"

[[ -s ${E12_PID_FILE} ]] || {
  echo "missing epoch-12 pid file: ${E12_PID_FILE}" >&2
  exit 1
}
e12_pid=$(cat "${E12_PID_FILE}")
while process_matches "${e12_pid}" "rlvr.train_awr"; do
  sleep 30
done

[[ -s ${E12_RUN}/final_summary.json && -s ${E12_RUN}/epoch_012.pth ]] || {
  echo "epoch 12 ended without final summary/checkpoint" >&2
  tail -200 "${E12_LOG}" >&2 || true
  exit 1
}
[[ -s ${E12_RUN}/train_epoch_012/summary.json ]] || {
  echo "epoch 12 ended without train summary" >&2
  exit 1
}

"${ROOT}/.venv/bin/python" - "${E12_RUN}" "${MINE_RUN}" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

run = pathlib.Path(sys.argv[1]).resolve()
mine = pathlib.Path(sys.argv[2]).resolve()
summary = json.loads((run / "train_epoch_012" / "summary.json").read_text())
assert math.isclose(float(summary["ema_policy_commit_rate"]), 0.05, abs_tol=1e-12)
assert float(summary["optimizer_state_reset_after_policy_commit"]) == 1.0
proposal = float(summary["policy_proposal_relative_l2"])
accepted = float(summary["policy_accepted_relative_l2"])
assert proposal > 0.0
assert math.isclose(accepted / proposal, 0.05, rel_tol=1e-6, abs_tol=1e-9)

effective = json.loads((run / "effective_config.json").read_text())
assert effective["training"]["ema_per_epoch"] is True
assert effective["training"]["ema_commit_live_policy"] is True
assert math.isclose(float(effective["training"]["ema_decay"]), 0.95)
assert math.isclose(float(effective["training"]["learning_rate"]), 1e-6)
assert effective["awr"]["drop_all_zero_groups"] is False

resume = json.loads((run / "replay_resume.json").read_text())
assert pathlib.Path(resume["source_replay_root"]).resolve() == (
    mine / "replay_buffer"
).resolve()
assert int(resume["total_scene_groups"]) == 5_447_680

checkpoint = run / "epoch_012.pth"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
final = json.loads((run / "final_summary.json").read_text())
assert int(final["start_epoch"]) == 12
assert int(final["epochs"]) == 12
print(
    "verified conservative epoch-12 policy commit: "
    f"proposal_l2={proposal:.9g} accepted_l2={accepted:.9g} sha256={digest}"
)
PY

if [[ -s ${E13_PID_FILE} ]]; then
  existing=$(cat "${E13_PID_FILE}")
  if process_matches "${existing}" "rlvr.train_awr"; then
    echo "epoch 13--20 continuation already running as ${existing}"
    exit 0
  fi
fi

cache=${OVERLAY_RUN}/replay_buffer
"${ROOT}/.venv/bin/python" - "${OVERLAY_RUN}" "${MINE_RUN}" <<'PY'
import json
import math
import pathlib
import sys

overlay = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[2]).resolve()
manifest = json.loads((overlay / "overlay_manifest.json").read_text())
assert pathlib.Path(manifest["source_replay"]).resolve() == (source / "replay_buffer").resolve()
assert int(manifest["scene_groups"]) == 5_447_680
assert int(manifest["dropped_tied_zero_groups"]) == 102_037
assert math.isclose(float(manifest["dropped_fraction"]), 0.01873035861137218)
assert manifest["non_dropped_weights_bit_exact"] is True
signal = json.loads(
    (overlay / "full_replay_signal_audit" / "full_replay_signal.json").read_text()
)
fractions = signal["fractions"]
assert fractions["reward_tied"] == fractions["all_zero_reward"]
assert fractions["zero_weight_group"] == fractions["reward_tied"]
assert fractions["active_preference_group"] == 1.0 - fractions["zero_weight_group"]
effective = json.loads((overlay / "effective_config.json").read_text())
assert effective["awr"]["drop_all_zero_groups"] is True
print("verified paper-described tied-zero drop overlay")
PY
for rank in $(seq 0 7); do
  printf -v rank_name 'rank_%04d' "${rank}"
  manifest=${cache}/${rank_name}/manifest.json
  [[ -s ${manifest} ]] || {
    echo "missing immutable Cycle-2 manifest: ${manifest}" >&2
    exit 1
  }
  [[ $(jq -er '.scene_count' "${manifest}") -eq 680960 ]] || {
    echo "unexpected scene count in ${manifest}" >&2
    exit 1
  }
done

sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${E12_RUN}/epoch_012.pth' --args_path '${E12_RUN}/model_args.json' --train_npz_list '${TRAIN_LIST}' --valid_npz_list '${VALID_LIST}' --config '${CONFIG}' --output_dir '${OUT}' --exp_name cycle02_conservative_commit_e13to20 --device cuda --seed 3407 --start_epoch 13 --epochs 20 --resume_replay_root '${cache}' --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --drop_all_zero_groups --learning_rate 1e-6 --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle02-conservative-commit-e13to20 --wandb_mode online --wandb_tags original-dp awr cycle02 conservative-accepted-policy-interpolation new-policy-rate-0.05 local-t4-adaptation global-epoch-13-20 lr-1e-6 matched-cache-overlay drop-tied-groups full-valid full-sequence 20260707 --no-skip_filtered_scenes > '${E13_LOG}' 2>&1 < /dev/null & echo \$!" > "${E13_PID_FILE}"

e13_pid=$(cat "${E13_PID_FILE}")
sleep 5
process_matches "${e13_pid}" "rlvr.train_awr" || {
  echo "epoch 13--20 continuation failed to stay alive" >&2
  tail -200 "${E13_LOG}" >&2 || true
  exit 1
}
echo "epoch 13--20 conservative Original-DP continuation launched: pid=${e13_pid} log=${E13_LOG}"
