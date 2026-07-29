#!/usr/bin/env bash
set -euo pipefail

# Continue the conservative accepted-policy Original-DP AWR chain from the
# supervised global-epoch-21 refresh through global epoch 100.  Every process
# boundary is fail-closed: no replay starts from an incomplete cache, and no
# new refresh starts without a hash-verified train-selected checkpoint and a
# complete 46,262-scene report for every replay epoch.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json
TRAIN_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID_LIST=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CYCLE1_RUN=${OUT}/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100
CYCLE2_MINE_RUN=${OUT}/20260719-152238_cycle02_selected_e004a005_onpolicy_everyrefresh_ramp20_mine
ENCODING_ROOT=${CYCLE2_MINE_RUN}/replay_buffer
CYCLE2_GLOBAL_SELECTION_GLOB=${OUT}/*_cycle02_hdp_epoch_commit_e13to20/global_train_selection_through_epoch020
CYCLE3_SUPERVISOR_PID_FILE=${OUT}/cycle03_conservative_after_cycle02_supervisor.pid
CYCLE3_MINE_PID_FILE=${OUT}/cycle03_conservative_mine_global_e21.pid
CYCLE3_MINE_LOG=${OUT}/cycle03_conservative_mine_global_e21.log
DEPLOY_SCENES=${CYCLE1_RUN}/scene_selection_valid.json
DEPLOY_ORIGINAL=${CYCLE1_RUN}/deployment_full10/source/scenes.json
DEPLOY_CYCLE1=${CYCLE1_RUN}/deployment_full10/e004_a0p05/scenes.json
EXPECTED_PER_RANK=680960
EXPECTED_TOTAL=5447680
REPLAY_LR=1e-6
STATE=${OUT}/cycles03_to10_conservative_commit_state.jsonl
SELF_PID_FILE=${OUT}/cycles03_to10_conservative_commit_supervisor.pid

cd "${ROOT}"
if [[ -s ${SELF_PID_FILE} ]]; then
  prior_supervisor=$(cat "${SELF_PID_FILE}")
  if [[ ${prior_supervisor} != "$$" && -r /proc/${prior_supervisor}/cmdline ]]; then
    prior_command=$(tr '\0' ' ' < "/proc/${prior_supervisor}/cmdline")
    if [[ ${prior_command} == *"continue_cycles03_to10.sh"* ]]; then
      echo "conservative-commit epoch-100 supervisor already running as ${prior_supervisor}"
      exit 0
    fi
  fi
fi
printf '%s\n' "$$" > "${SELF_PID_FILE}"

process_matches() {
  local pid=$1
  local needle=$2
  [[ -r /proc/${pid}/cmdline ]] || return 1
  local command
  command=$(tr '\0' ' ' < "/proc/${pid}/cmdline")
  [[ ${command} == *"${needle}"* ]]
}

run_from_log() {
  local log=$1
  local run
  run=$(sed -n 's/^Run directory: //p' "${log}" | tail -n 1)
  [[ -n ${run} ]] || {
    echo "no run directory in ${log}" >&2
    tail -200 "${log}" >&2 || true
    return 1
  }
  realpath -e "${run}"
}

wait_for_training() {
  local pid_file=$1
  local log=$2
  local label=$3
  local upstream_pid_file=${4:-}
  while [[ ! -s ${pid_file} ]]; do
    if [[ -n ${upstream_pid_file} && -s ${upstream_pid_file} ]]; then
      local upstream_pid
      upstream_pid=$(cat "${upstream_pid_file}")
      if ! kill -0 "${upstream_pid}" 2>/dev/null; then
        echo "${label}: upstream supervisor ended before launch" >&2
        tail -200 "${log}" >&2 || true
        return 1
      fi
    fi
    sleep 60
  done
  local pid
  pid=$(cat "${pid_file}")
  while process_matches "${pid}" "rlvr.train_awr"; do
    sleep 60
  done
  [[ -f ${log} ]] || {
    echo "${label}: missing log ${log}" >&2
    return 1
  }
}

validate_mine_cache() {
  local run=$1
  local cycle=$2
  local cache=${run}/replay_buffer
  [[ -f ${run}/final_summary.json ]] || {
    echo "cycle ${cycle}: mining ended without final_summary.json" >&2
    return 1
  }
  local total=0
  for rank in $(seq 0 7); do
    local rank_name
    printf -v rank_name 'rank_%04d' "${rank}"
    local rank_dir=${cache}/${rank_name}
    local manifest=${rank_dir}/manifest.json
    [[ -f ${manifest} ]] || {
      echo "cycle ${cycle}: missing ${manifest}" >&2
      return 1
    }
    local count lines expected_lines
    count=$(jq -er '.scene_count' "${manifest}")
    [[ ${count} -eq ${EXPECTED_PER_RANK} ]] || {
      echo "cycle ${cycle} rank ${rank}: ${count} != ${EXPECTED_PER_RANK}" >&2
      return 1
    }
    lines=$(wc -l < "${rank_dir}/scene_paths.jsonl")
    expected_lines=$(wc -l < "${rank_dir}/expected_scene_paths.jsonl")
    [[ ${lines} -eq ${count} && ${expected_lines} -eq ${count} ]] || {
      echo "cycle ${cycle} rank ${rank}: path contract is incomplete" >&2
      return 1
    }
    [[ -L ${rank_dir}/scene_encoding.npy ]] || {
      echo "cycle ${cycle} rank ${rank}: frozen encoding is not shared" >&2
      return 1
    }
    jq -e '
      .version == 3 and
      .decoder_context_cached == true and
      .decoder_context_schema == "canonical_loader_aligned_v1" and
      (.arrays.context_ego_current_state | type == "string") and
      (.arrays.context_neighbor_agents_past | type == "string") and
      (.arrays.context_neighbor_agents_future | type == "string")
    ' "${manifest}" >/dev/null
    for key in ego_current_state neighbor_agents_past neighbor_agents_future; do
      [[ -e ${rank_dir}/context_${key}.npy ]] || {
        echo "cycle ${cycle} rank ${rank}: missing context_${key}.npy" >&2
        return 1
      }
      if [[ ${cycle} -gt 3 && ! -L ${rank_dir}/context_${key}.npy ]]; then
        echo "cycle ${cycle} rank ${rank}: decoder context was copied, not shared" >&2
        return 1
      fi
    done
    total=$((total + count))
  done
  [[ ${total} -eq ${EXPECTED_TOTAL} ]] || {
    echo "cycle ${cycle}: cache total ${total} != ${EXPECTED_TOTAL}" >&2
    return 1
  }
  printf '%s\n' "${cache}"
}

make_cache_read_only() {
  local cache=$1
  # find does not follow the encoder/context symlinks, so source generations
  # remain independently protected and no downstream target is chmod'ed by
  # accident.
  find "${cache}" -type f -exec chmod a-w {} +
  find "${cache}" -type d -exec chmod a-w {} +
}

audit_replay_signal() {
  local run=$1
  local cycle=$2
  local cache=$3
  local output=${run}/full_replay_signal_audit
  if [[ ${cycle} -ge 3 ]]; then
    jq -e '.awr.drop_all_zero_groups == true' \
      "${run}/effective_config.json" >/dev/null
  fi
  if [[ ! -s ${output}/full_replay_signal.json ]]; then
    "${ROOT}/.venv/bin/python" \
      "${ROOT}/rlvr/autoresearch/tools/analyze_full_replay_signal.py" \
      --replay-root "${cache}" --output-dir "${output}" \
      --title "T4 Cycle ${cycle} full replay: audited AWR learning signal" \
      > "${run}/full_replay_signal_audit.log" 2>&1
  fi
  jq -e --argjson expected "${EXPECTED_TOTAL}" '
    .scene_groups == $expected and
    .group_size == 10 and
    .candidate_trajectories == ($expected * 10) and
    (.distributions.weighted_reward_gain.mean > 0) and
    (.distributions.ess.mean >= 1) and
    (.distributions.ess.mean <= 10) and
    (.fractions.all_zero_reward >= 0) and
    (.fractions.all_zero_reward < 1)
  ' "${output}/full_replay_signal.json" >/dev/null
  if [[ ${cycle} -ge 3 ]]; then
    # The full Cycle-2 corpus established that every tied group under this
    # continuous reward is an all-zero terminal group.  From Cycle 3 onward,
    # prove that precisely this no-preference stratum carries zero AWR weight.
    jq -e '
      (.fractions.reward_tied == .fractions.all_zero_reward) and
      (.fractions.zero_weight_group == .fractions.reward_tied) and
      (.fractions.active_preference_group ==
        (1 - .fractions.zero_weight_group))
    ' "${output}/full_replay_signal.json" >/dev/null
  fi
  echo "cycle ${cycle}: full replay signal audited at ${output}"
}

launch_replay() {
  local cycle=$1
  local mine_epoch=$2
  local replay_end=$3
  local mine_run=$4
  local cache=$5
  local source_checkpoint source_args log pid_file
  source_checkpoint=$(jq -er '.staged_model' "${mine_run}/provenance.json")
  source_args=$(jq -er '.staged_args' "${mine_run}/provenance.json")
  [[ -s ${source_checkpoint} && -s ${source_args} ]] || {
    echo "cycle ${cycle}: missing staged source checkpoint/args" >&2
    return 1
  }
  log=${OUT}/cycle$(printf '%02d' "${cycle}")_replay_e$((mine_epoch + 1))to${replay_end}.log
  pid_file=${OUT}/cycle$(printf '%02d' "${cycle}")_replay.pid
  if [[ -s ${pid_file} ]]; then
    local old_pid
    old_pid=$(cat "${pid_file}")
    if process_matches "${old_pid}" "rlvr.train_awr"; then
      echo "cycle ${cycle}: replay already running as ${old_pid}"
      return 0
    fi
  fi
  sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${source_checkpoint}' --args_path '${source_args}' --train_npz_list '${TRAIN_LIST}' --valid_npz_list '${VALID_LIST}' --config '${CONFIG}' --output_dir '${OUT}' --exp_name cycle$(printf '%02d' "${cycle}")_conservative_commit_replay_e$((mine_epoch + 1))to${replay_end} --device cuda --seed 3407 --start_epoch $((mine_epoch + 1)) --epochs '${replay_end}' --resume_replay_root '${cache}' --inherit_refresh_baseline_root '${mine_run}' --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --learning_rate '${REPLAY_LR}' --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle$(printf '%02d' "${cycle}")-conservative-commit-replay-e$((mine_epoch + 1))to${replay_end} --wandb_mode online --wandb_tags original-dp awr cycle$(printf '%02d' "${cycle}") global-epoch-${mine_epoch}-${replay_end} replay conservative-accepted-policy-interpolation new-policy-rate-0.05 local-t4-adaptation optimizer-reset inherited-refresh-baseline hdp-group-relative drop-tied-groups smooth-ramp20 heading-preserve every-refresh lr-${REPLAY_LR} full-valid-every-epoch full-sequence 20260707 --no-skip_filtered_scenes > '${log}' 2>&1 < /dev/null & echo \$!" > "${pid_file}"
  local pid
  pid=$(cat "${pid_file}")
  sleep 5
  process_matches "${pid}" "rlvr.train_awr" || {
    echo "cycle ${cycle}: replay failed to stay alive" >&2
    tail -200 "${log}" >&2 || true
    return 1
  }
  echo "cycle ${cycle}: replay launched pid=${pid} log=${log}"
}

validate_replay_run() {
  local run=$1
  local cycle=$2
  local first_epoch=$3
  local last_epoch=$4
  [[ -f ${run}/final_summary.json && -s ${run}/best_train.pth ]] || {
    echo "cycle ${cycle}: replay lacks final_summary/best_train" >&2
    return 1
  }
  local checkpoint_sha summary_sha best_epoch starting_epoch incumbent
  local best_reward starting_reward reward_delta selection_rule
  checkpoint_sha=$(sha256sum "${run}/best_train.pth" | awk '{print $1}')
  summary_sha=$(jq -er '.best_train_checkpoint_sha256' "${run}/final_summary.json")
  [[ ${checkpoint_sha} == "${summary_sha}" ]] || {
    echo "cycle ${cycle}: selected checkpoint hash mismatch" >&2
    return 1
  }
  best_epoch=$(jq -er '.best_train_epoch' "${run}/final_summary.json")
  starting_epoch=$(jq -er '.starting_train_selection_epoch' "${run}/final_summary.json")
  best_reward=$(jq -er '.best_train_reward' "${run}/final_summary.json")
  starting_reward=$(jq -er '.starting_train_selection_reward' "${run}/final_summary.json")
  reward_delta=$(jq -er '.best_train_reward_delta_from_start' "${run}/final_summary.json")
  selection_rule=$(jq -er '.checkpoint_selection' "${run}/final_summary.json")
  [[ ${selection_rule} == "fixed_train_selector_ema_mean_det_reward" ]] || {
    echo "cycle ${cycle}: unexpected checkpoint selector ${selection_rule}" >&2
    return 1
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
  incumbent=$(jq -er '.best_train_is_starting_incumbent' "${run}/final_summary.json")
  if [[ ${incumbent} == true ]]; then
    [[ ${best_epoch} -eq ${starting_epoch} ]] || return 1
  else
    [[ ${best_epoch} -ge ${first_epoch} && ${best_epoch} -le ${last_epoch} ]] || {
      echo "cycle ${cycle}: best epoch ${best_epoch} outside replay range" >&2
      return 1
    }
  fi
  for epoch in $(seq "${first_epoch}" "${last_epoch}"); do
    local padded
    printf -v padded '%03d' "${epoch}"
    [[ -s ${run}/eval_epoch_${padded}/summary.json ]] || {
      echo "cycle ${cycle}: missing validation epoch ${epoch}" >&2
      return 1
    }
    [[ -s ${run}/train_selector_epoch_${padded}/summary.json ]] || {
      echo "cycle ${cycle}: missing train selector epoch ${epoch}" >&2
      return 1
    }
    [[ $(jq -er '.scene_count' "${run}/eval_epoch_${padded}/summary.json" | cut -d. -f1) -eq 46262 ]] || {
      echo "cycle ${cycle}: validation epoch ${epoch} is not full" >&2
      return 1
    }
    [[ $(jq -er '.scene_count' "${run}/train_selector_epoch_${padded}/summary.json" | cut -d. -f1) -eq 65536 ]] || {
      echo "cycle ${cycle}: train selector epoch ${epoch} is incomplete" >&2
      return 1
    }
    "${ROOT}/.venv/bin/python" - \
      "${run}/train_epoch_${padded}/summary.json" <<'PY'
import json
import math
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
proposal = float(summary["policy_proposal_relative_l2"])
accepted = float(summary["policy_accepted_relative_l2"])
assert proposal > 0.0
assert math.isclose(accepted / proposal, 0.05, rel_tol=1e-6, abs_tol=1e-9)
assert math.isclose(float(summary["ema_policy_commit_rate"]), 0.05, abs_tol=1e-12)
assert float(summary["optimizer_state_reset_after_policy_commit"]) == 1.0
PY
  done
  PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" - "${run}/best_train.pth" <<'PY'
import pathlib
import sys
import torch

checkpoint = torch.load(pathlib.Path(sys.argv[1]), map_location="cpu", weights_only=False)
assert isinstance(checkpoint, dict)
assert "model" in checkpoint and "ema_state_dict" in checkpoint
assert "optimizer_state_dict" in checkpoint
assert isinstance(checkpoint["optimizer_state_dict"], dict)
assert checkpoint["optimizer_state_dict"].get("state", {}) == {}
PY
  echo "cycle ${cycle}: selected epoch=${best_epoch} incumbent=${incumbent} sha=${checkpoint_sha}"
}

deployment_audit() {
  local run=$1
  local cycle=$2
  local previous_scenes=$3
  local checkpoint=${run}/best_train.pth
  local args=${run}/model_args.json
  local output=${run}/deployment_full10/best_train
  local manifest=${output}/manifest.json
  if [[ ! -s ${manifest} ]]; then
    mkdir -p "${output}"
    sg ubuntu -c "cd '${ROOT}' && env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 '${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py' --model_path '${checkpoint}' --args_path '${args}' --scenes '${DEPLOY_SCENES}' --config '${run}/effective_config.json' --output_dir '${output}' --label cycle$(printf '%02d' "${cycle}")_best_train_full_valid_dp10 --batch_size 512 --scene_load_workers 1 --sample_steps 10 --amp_dtype bf16 --compile_mode default"
    DP_NEIGHBOR_FUTURE_OFFSET=1 PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" \
      "${ROOT}/rlvr/autoresearch/tools/eval_awr_full_distributed.py" \
      --aggregate --model_path "${checkpoint}" --args_path "${args}" \
      --scenes "${DEPLOY_SCENES}" --config "${run}/effective_config.json" \
      --output_dir "${output}" --label cycle$(printf '%02d' "${cycle}")_best_train_full_valid_dp10
  fi
  PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" - "${manifest}" "${checkpoint}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
checkpoint = pathlib.Path(sys.argv[2]).resolve()
assert manifest["scene_contract_verified"] is True
assert int(manifest["actual_scene_count"]) == 46262
assert int(manifest["expected_scene_count"]) == 46262
assert int(manifest["sample_steps"]) == 10
assert int(manifest["candidate_count"]) == 1
assert float(manifest["noise_scale"]) == 0.0
assert manifest["checkpoint_payload"] == "ema_state_dict"
assert pathlib.Path(manifest["model"]).resolve() == checkpoint
PY
  "${ROOT}/.venv/bin/python" -m rlvr.autoresearch.tools.full_paired_eval_report \
    --baseline "${DEPLOY_ORIGINAL}" --candidate "${output}/scenes.json" \
    --epoch "$(jq -er '.best_train_epoch' "${run}/final_summary.json")" \
    --bootstrap 10000 --seed 3407 \
    --output "${run}/deployment_full10/paired_vs_original_dp.json"
  if [[ -s ${previous_scenes} ]]; then
    "${ROOT}/.venv/bin/python" -m rlvr.autoresearch.tools.full_paired_eval_report \
      --baseline "${previous_scenes}" --candidate "${output}/scenes.json" \
      --epoch "$(jq -er '.best_train_epoch' "${run}/final_summary.json")" \
      --bootstrap 10000 --seed 3407 \
      --output "${run}/deployment_full10/paired_vs_previous_cycle.json"
  fi
}

launch_mine() {
  local cycle=$1
  local epoch=$2
  local checkpoint=$3
  local args=$4
  local context_root=$5
  local log=${OUT}/cycle$(printf '%02d' "${cycle}")_mine_global_e${epoch}.log
  local pid_file=${OUT}/cycle$(printf '%02d' "${cycle}")_mine.pid
  local free_bytes
  free_bytes=$(df -B1 --output=avail /mnt/nvme | tail -n 1 | tr -d ' ')
  local min_free_bytes=$((2 * 1024 * 1024 * 1024 * 1024))
  [[ ${free_bytes} -ge ${min_free_bytes} ]] || {
    echo "cycle ${cycle}: only ${free_bytes} free bytes; refusing refresh" >&2
    return 1
  }
  sg ubuntu -c "cd '${ROOT}' && setsid env PYTHONPATH='${ROOT}' DP_NEIGHBOR_FUTURE_OFFSET=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 '${ROOT}/.venv/bin/torchrun' --standalone --nproc_per_node=8 -m rlvr.train_awr --model_path '${checkpoint}' --args_path '${args}' --train_npz_list '${TRAIN_LIST}' --valid_npz_list '${VALID_LIST}' --config '${CONFIG}' --output_dir '${OUT}' --exp_name cycle$(printf '%02d' "${cycle}")_conservative_besttrain_onpolicy_ramp20_mine_global_e${epoch} --device cuda --seed 3407 --start_epoch '${epoch}' --epochs '${epoch}' --max_train_scenes 0 --max_valid_scenes 0 --train_selector_scenes 65536 --full_valid_interval 0 --hard_valid_scenes 0 --hdp_trajectory_augmentation_schedule every_refresh --learning_rate '${REPLAY_LR}' --shared_replay_encoding_root '${ENCODING_ROOT}' --shared_replay_context_root '${context_root}' --shared_replay_order_epoch 1 --rollout_prefetch_batches 1 --scene_load_workers 4 --eval_scene_load_workers 1 --eval_scene_batch_size 512 --save_rollouts 0 --wandb --wandb_project original-dp-awr --wandb_entity advanced-technology-department --wandb_run_name original-dp-awr-cycle$(printf '%02d' "${cycle}")-conservative-onpolicy-mine-global-e${epoch} --wandb_mode online --wandb_tags original-dp awr cycle$(printf '%02d' "${cycle}") conservative-accepted-policy-interpolation local-t4-adaptation global-epoch-${epoch} onpolicy-refresh mine-only-no-policy-update hdp-group-relative drop-tied-groups smooth-ramp20 heading-preserve every-refresh shared-frozen-encoder-cache shared-decoder-context cached-replay-context rollout-prefetch-1 full-sequence 20260707 --no-skip_filtered_scenes > '${log}' 2>&1 < /dev/null & echo \$!" > "${pid_file}"
  local pid
  pid=$(cat "${pid_file}")
  sleep 5
  process_matches "${pid}" "rlvr.train_awr" || {
    echo "cycle ${cycle}: mining failed to stay alive" >&2
    tail -200 "${log}" >&2 || true
    return 1
  }
  echo "cycle ${cycle}: mining launched pid=${pid} log=${log}"
}

record_cycle() {
  local cycle=$1
  local mine_run=$2
  local replay_run=$3
  local deploy_scenes=$4
  PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" - \
    "${STATE}" "${cycle}" "${mine_run}" "${replay_run}" "${deploy_scenes}" <<'PY'
import datetime
import json
import pathlib
import sys

state, cycle, mine_run, replay_run, deploy_scenes = sys.argv[1:]
summary = json.loads((pathlib.Path(replay_run) / "final_summary.json").read_text())
row = {
    "time": datetime.datetime.now().astimezone().isoformat(),
    "cycle": int(cycle),
    "mine_run": str(pathlib.Path(mine_run).resolve()),
    "replay_run": str(pathlib.Path(replay_run).resolve()),
    "best_train_epoch": int(summary["best_train_epoch"]),
    "best_train_reward": float(summary["best_train_reward"]),
    "best_train_is_starting_incumbent": bool(summary["best_train_is_starting_incumbent"]),
    "best_train_checkpoint_sha256": summary["best_train_checkpoint_sha256"],
    "deployment_scenes": str(pathlib.Path(deploy_scenes).resolve()),
}
with pathlib.Path(state).open("a") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

[[ -d ${ENCODING_ROOT} ]] || {
  echo "missing in-progress Cycle-2 encoding root" >&2
  exit 1
}
[[ -s ${DEPLOY_SCENES} && -s ${DEPLOY_ORIGINAL} ]] || {
  echo "missing fixed deployment evaluation contract" >&2
  exit 1
}

wait_for_training "${CYCLE3_MINE_PID_FILE}" "${CYCLE3_MINE_LOG}" \
  "cycle 3 mining" "${CYCLE3_SUPERVISOR_PID_FILE}"
[[ -s ${ENCODING_ROOT}/rank_0000/manifest.json ]] || {
  echo "Cycle-3 launched without a completed immutable Cycle-2 encoding root" >&2
  exit 1
}
mine_run=$(run_from_log "${CYCLE3_MINE_LOG}")
context_root=$(validate_mine_cache "${mine_run}" 3)
make_cache_read_only "${context_root}"
audit_replay_signal "${mine_run}" 3 "${context_root}"

previous_deploy_scenes=${DEPLOY_CYCLE1}
cycle2_selection_dirs=( ${CYCLE2_GLOBAL_SELECTION_GLOB} )
if [[ ${#cycle2_selection_dirs[@]} -eq 1 && \
      -s ${cycle2_selection_dirs[0]}/deployment_full10/scenes.json ]]; then
  previous_deploy_scenes=${cycle2_selection_dirs[0]}/deployment_full10/scenes.json
fi

for cycle in $(seq 3 10); do
  mine_epoch=$((10 * cycle - 9))
  replay_end=$((10 * cycle))
  if [[ ${cycle} -gt 3 ]]; then
    mine_log=${OUT}/cycle$(printf '%02d' "${cycle}")_mine_global_e${mine_epoch}.log
    mine_pid_file=${OUT}/cycle$(printf '%02d' "${cycle}")_mine.pid
    wait_for_training "${mine_pid_file}" "${mine_log}" "cycle ${cycle} mining"
    mine_run=$(run_from_log "${mine_log}")
    cache=$(validate_mine_cache "${mine_run}" "${cycle}")
    make_cache_read_only "${cache}"
    audit_replay_signal "${mine_run}" "${cycle}" "${cache}"
  else
    cache=${context_root}
  fi

  launch_replay "${cycle}" "${mine_epoch}" "${replay_end}" "${mine_run}" "${cache}"
  replay_log=${OUT}/cycle$(printf '%02d' "${cycle}")_replay_e$((mine_epoch + 1))to${replay_end}.log
  replay_pid_file=${OUT}/cycle$(printf '%02d' "${cycle}")_replay.pid
  wait_for_training "${replay_pid_file}" "${replay_log}" "cycle ${cycle} replay"
  replay_run=$(run_from_log "${replay_log}")
  validate_replay_run "${replay_run}" "${cycle}" $((mine_epoch + 1)) "${replay_end}"
  deployment_audit "${replay_run}" "${cycle}" "${previous_deploy_scenes}"
  current_deploy_scenes=${replay_run}/deployment_full10/best_train/scenes.json
  record_cycle "${cycle}" "${mine_run}" "${replay_run}" "${current_deploy_scenes}"
  previous_deploy_scenes=${current_deploy_scenes}

  if [[ ${cycle} -lt 10 ]]; then
    next_cycle=$((cycle + 1))
    next_epoch=$((replay_end + 1))
    launch_mine "${next_cycle}" "${next_epoch}" \
      "${replay_run}/best_train.pth" "${replay_run}/model_args.json" \
      "${context_root}"
  fi
done

PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" - "${STATE}" \
  "${OUT}/epoch100_conservative_commit_completion.json" <<'PY'
import json
import pathlib
import sys

rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line]
latest = {}
for row in rows:
    latest[int(row["cycle"])] = row
assert set(latest) >= set(range(3, 11))
final = latest[10]
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "status": "epoch100_conservative_commit_training_and_open_loop_audit_complete",
    "cycles": [latest[index] for index in range(3, 11)],
    "final_selected_checkpoint": str(pathlib.Path(final["replay_run"]) / "best_train.pth"),
    "final_deployment_scenes": final["deployment_scenes"],
}, indent=2, sort_keys=True) + "\n")
PY
echo "conservative-commit AWR chain reached global epoch 100; see ${OUT}/epoch100_conservative_commit_completion.json"
