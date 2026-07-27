#!/usr/bin/env bash
# Select the replay beta for post-Cycle-1 refreshes with a paired, bounded
# policy-update probe.  All arms consume the same immutable full cache, use
# the same replay indices/seed/update count and are evaluated on the exact
# fixed 65,536-scene train selector.  This script never edits the source cache
# or the formal Cycle-1 output.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
TORCHRUN=${ROOT}/.venv/bin/torchrun
MODEL=${MODEL:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
ARGS=${ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${CONFIG:-${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json}
MINE_RUN=${MINE_RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine}
SOURCE_CACHE=${MINE_RUN}/replay_buffer
COMPRESSED_REPLAY_CONTEXT_ROOT=${COMPRESSED_REPLAY_CONTEXT_ROOT:-}
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_beta_sensitivity}
RESULT=${OUT}/selection.json
# One complete formal replay pass. A short 256-update direction probe saves
# little wall time because every arm must still evaluate the full 65,536-scene
# selector, yet it would choose the temperature used by nine later cycles from
# only 7.2% of an epoch. Keep the matched arms at production update depth.
UPDATES=${UPDATES:-3546}
# Derived from the caller; the literal was the un-augmented corpus and broke
# the moment right-turn oversampling changed the scene count.
TIE_TOLERANCE=${TIE_TOLERANCE:-0.00001}
AWR_CANDIDATE_LOSS_HORIZON=${AWR_CANDIDATE_LOSS_HORIZON:-40}
EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=${EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY:-1}
EVAL_SCENE_LOAD_WORKERS=${EVAL_SCENE_LOAD_WORKERS:-1}
[[ ${EVAL_SCENE_LOAD_WORKERS} == 1 || ${EVAL_SCENE_LOAD_WORKERS} == 4 || ${EVAL_SCENE_LOAD_WORKERS} == 8 ]] \
  || { echo "beta sensitivity: invalid EVAL_SCENE_LOAD_WORKERS=${EVAL_SCENE_LOAD_WORKERS}" >&2; exit 2; }
REPLAY_SCENE_LOAD_WORKERS=${REPLAY_SCENE_LOAD_WORKERS:-16}
[[ ${REPLAY_SCENE_LOAD_WORKERS} == 4 || ${REPLAY_SCENE_LOAD_WORKERS} == 8 || ${REPLAY_SCENE_LOAD_WORKERS} == 16 ]] \
  || { echo "beta sensitivity: invalid REPLAY_SCENE_LOAD_WORKERS=${REPLAY_SCENE_LOAD_WORKERS}" >&2; exit 2; }
[[ -s ${MODEL} && -s ${ARGS} ]] || {
  echo "beta sensitivity: missing selected policy or model args" >&2
  exit 1
}
MODEL=$(realpath -e "${MODEL}")
ARGS=$(realpath -e "${ARGS}")
MODEL_SHA256=$(sha256sum "${MODEL}" | awk '{print $1}')

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

die() {
  echo "beta sensitivity: $*" >&2
  exit 1
}

latest_completed_run() {
  local parent=$1
  [[ -d ${parent} ]] || return 0
  find "${parent}" -mindepth 1 -maxdepth 1 -type d -name '20*' \
    -exec test -s '{}/final_summary.json' ';' -print 2>/dev/null \
    | sort | tail -n 1
}

archive_interrupted_probe_log() {
  local beta=$1 path=$2 archive stamp
  [[ -e ${path} ]] || return 0
  stamp=$(date '+%Y%m%d-%H%M%S')
  archive=${OUT}/interrupted_attempts
  mkdir -p "${archive}"
  mv "${path}" "${archive}/beta${beta}_probe_${stamp}.log"
  echo "beta sensitivity: archived interrupted beta=${beta} probe; restarting from immutable cache" >&2
}

validate_source() {
  local total=0 rank rank_dir count
  [[ -s ${MINE_RUN}/decoder_context_attachment.json ]] \
    || die "source cache has no committed decoder-context attachment"
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${SOURCE_CACHE}" "${rank}"
    [[ -s ${rank_dir}/manifest.json && -s ${rank_dir}/expert_anchor_manifest.json ]] \
      || die "rank ${rank} has no strict replay/expert manifest"
    count=$(jq -er '.scene_count' "${rank_dir}/manifest.json")
    total=$((total + count))
  done
}

build_overlay() {
  local beta=$1
  local parent=${OUT}/beta${beta}_overlay
  local replay=${parent}/replay_buffer
  local manifest=${parent}/overlay_manifest.json
  if [[ ! -s ${manifest} ]]; then
    [[ ! -e ${replay} ]] || die "beta ${beta} overlay is incomplete: ${replay}"
    mkdir -p "${parent}"
    "${PYTHON}" -m rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
      --source-replay "${SOURCE_CACHE}" --output-replay "${replay}" \
      --beta "${beta}" --margin 0.01 \
      --behavior-anchor-weight 0 --unsafe-behavior-anchor-weight 0 \
      > "${parent}/build.log" 2>&1
  fi
  printf '%s\n' "${replay}"
}

run_arm() {
  local beta=$1 overlay=$2
  local parent=${OUT}/beta${beta}_probe
  local log=${OUT}/beta${beta}_probe.log
  local run expert_scope_arg=--expert_anchor_active_groups_only
  if [[ ${EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY} -ne 1 ]]; then
    expert_scope_arg=--no-expert_anchor_active_groups_only
  fi
  run=$(latest_completed_run "${parent}")
  if [[ -z ${run} ]]; then
    archive_interrupted_probe_log "${beta}" "${log}"
    mkdir -p "${parent}"
    local -a compressed_args=()
    if [[ -n ${COMPRESSED_REPLAY_CONTEXT_ROOT} ]]; then
      [[ -d ${COMPRESSED_REPLAY_CONTEXT_ROOT} ]] \
        || die "compressed replay context is missing: ${COMPRESSED_REPLAY_CONTEXT_ROOT}"
      compressed_args=(
        --compressed_replay_context_root "${COMPRESSED_REPLAY_CONTEXT_ROOT}"
      )
    fi
    "${TORCHRUN}" --standalone --nproc_per_node=8 -m rlvr.train_awr \
      --model_path "${MODEL}" --args_path "${ARGS}" \
      --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
      --config "${CONFIG}" --output_dir "${parent}" \
      --exp_name "plannerrft_full_cycle01_beta${beta}_probe_u${UPDATES}" \
      --device cuda --seed 3407 --start_epoch 2 --epochs 2 \
      --resume_replay_root "${overlay}" --replay_sampling with_replacement \
      "${compressed_args[@]}" \
      --replay_updates_per_epoch "${UPDATES}" \
      --awr_beta "${beta}" --positive_advantage_only \
      --positive_advantage_margin 0.01 \
      --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
      --expert_anchor_weight 0.4 \
      --awr_candidate_loss_horizon "${AWR_CANDIDATE_LOSS_HORIZON}" \
      "${expert_scope_arg}" \
      --drop_all_zero_groups \
      --awr_loss_type plain_mse --diffusion_t_range 0.001 0.2 \
      --trainable_scope output --learning_rate 0.000001 \
      --ema_decay 0.95 --ema_per_epoch --ema_commit_live_policy \
      --max_train_scenes 0 --max_valid_scenes 8 \
      --eval_k 1 --eval_sample_steps 10 \
      --train_selector_scenes 65536 \
      --full_valid_interval 0 --hard_valid_scenes 0 \
      --scene_batch_size 192 --gradient_accumulation_scenes 192 \
      --scene_load_workers "${REPLAY_SCENE_LOAD_WORKERS}" \
      --eval_scene_load_workers "${EVAL_SCENE_LOAD_WORKERS}" \
      --eval_prefetch_batches 0 \
      --eval_scene_batch_size 512 --save_rollouts 0 --no-wandb \
      --no-skip_filtered_scenes > "${log}" 2>&1
    run=$(sed -n 's/^Run directory: //p' "${log}" | tail -n 1)
  fi
  [[ -n ${run} && -s ${run}/final_summary.json ]] \
    || die "beta ${beta} probe did not complete"
  if [[ -n ${COMPRESSED_REPLAY_CONTEXT_ROOT} ]]; then
    local compressed_root
    compressed_root=$(realpath -e "${COMPRESSED_REPLAY_CONTEXT_ROOT}")
    jq -e --arg root "${compressed_root}" '
      .continuation.compressed_replay_context_root == $root
    ' "${run}/effective_config.json" >/dev/null \
      || die "beta ${beta} effective config did not use the compressed replay context"
    jq -e --arg root "${compressed_root}" '
      .compressed_replay_context_root == $root
    ' "${run}/provenance.json" >/dev/null \
      || die "beta ${beta} provenance did not use the compressed replay context"
  fi
  [[ -s ${run}/train_selector_epoch_000/summary.json && \
     -s ${run}/train_selector_epoch_002/summary.json ]] \
    || die "beta ${beta} lacks paired selector summaries"
  printf '%s\n' "${run}"
}

main() {
  if [[ -s ${RESULT} ]]; then
    jq -e --arg model "${MODEL}" --arg sha "${MODEL_SHA256}" '
      (.selected_beta == 0.5 or .selected_beta == 1 or .selected_beta == 2) and
      .model_checkpoint == $model and .model_checkpoint_sha256 == $sha
    ' "${RESULT}" >/dev/null \
      || die "existing selection result is malformed"
    echo "${RESULT}"
    return 0
  fi
  validate_source
  mkdir -p "${OUT}"
  local overlay05 overlay1 overlay2 run05 run1 run2
  overlay05=$(build_overlay 0.5)
  overlay1=$(build_overlay 1)
  overlay2=$(build_overlay 2)
  # Sequential execution keeps all arms on the same eight GPUs and preserves
  # the production global batch/update semantics.
  run05=$(run_arm 0.5 "${overlay05}")
  run1=$(run_arm 1 "${overlay1}")
  run2=$(run_arm 2 "${overlay2}")
  "${PYTHON}" - "${RESULT}" "${run05}" "${run1}" "${run2}" \
    "${UPDATES}" "${TIE_TOLERANCE}" \
    "${MODEL}" "${MODEL_SHA256}" "${ARGS}" <<'PY'
import datetime
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
runs = {
    0.5: pathlib.Path(sys.argv[2]),
    1.0: pathlib.Path(sys.argv[3]),
    2.0: pathlib.Path(sys.argv[4]),
}
updates = int(sys.argv[5])
tie_tolerance = float(sys.argv[6])
model = pathlib.Path(sys.argv[7])
model_sha256 = sys.argv[8]
model_args = pathlib.Path(sys.argv[9])
arms = {}
for beta, run in runs.items():
    before = json.loads((run / "train_selector_epoch_000" / "summary.json").read_text())
    after = json.loads((run / "train_selector_epoch_002" / "summary.json").read_text())
    start = float(before["mean_det_reward"])
    end = float(after["mean_det_reward"])
    arms[f"{beta:g}"] = {
        "run": str(run.resolve()),
        "selector_count": int(after["scene_count"]),
        "reward_before": start,
        "reward_after": end,
        "reward_delta": end - start,
        "ade_delta_m": float(after["mean_ade"]) - float(before["mean_ade"]),
        "fde_delta_m": float(after["mean_fde"]) - float(before["mean_fde"]),
        "collision_delta": float(after["mean_det_collision"]) - float(before["mean_det_collision"]),
        "road_border_delta": float(after["mean_det_rb_crossing"]) - float(before["mean_det_rb_crossing"]),
        "kinematic_delta": float(after["mean_det_kinematic_violation"]) - float(before["mean_det_kinematic_violation"]),
    }

best_delta = max(arm["reward_delta"] for arm in arms.values())
# Reward is the selection objective. When several arms are statistically
# indistinguishable at the declared numerical tolerance, use the lower beta:
# it has higher ESS and lower cross-scene weight concentration on this exact
# cache, so it is the conservative tie-break rather than a separate gate.
eligible = [
    beta
    for beta in sorted(runs)
    if arms[f"{beta:g}"]["reward_delta"]
    >= best_delta - tie_tolerance
]
selected = min(eligible)
selected_key = f"{selected:g}"
strict_best_beta = max(
    sorted(runs),
    key=lambda beta: arms[f"{beta:g}"]["reward_delta"],
)
reason = (
    "selector_deltas_within_tolerance_prefer_lower_weight_concentration"
    if selected != strict_best_beta
    else "highest_fixed_train_selector_reward_delta"
)

payload = {
    "version": 2,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "contract": "paired_same_cache_same_seed_same_updates_fixed_train_selector",
    "model_checkpoint": str(model.resolve()),
    "model_checkpoint_sha256": model_sha256,
    "model_args": str(model_args.resolve()),
    "updates_per_arm": updates,
    "tie_tolerance": tie_tolerance,
    "selected_beta": selected,
    "selected_reward_delta": arms[selected_key]["reward_delta"],
    "selection_reason": reason,
    "arms": arms,
    "note": "This probe selects replay temperature for later refreshes; it does not replace full-epoch checkpoint evaluation.",
}
result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(result_path)
PY
}

main "$@"
