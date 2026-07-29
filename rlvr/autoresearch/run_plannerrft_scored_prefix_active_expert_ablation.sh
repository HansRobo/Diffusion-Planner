#!/usr/bin/env bash
# One-variable-at-a-time full-data ablation for the PlannerRFT candidate cache.
#
# Relative to the formal Cycle-1 replay, this run changes only:
#   1. sampled AWR candidates receive loss on the reward-scored 40-step prefix;
#   2. the safe expert anchor is injected only in groups with an active AWR target.
# Deterministic candidate-0 and the safe expert anchor retain their full 80-step
# horizons.  The immutable source/overlay cache, seed, replay order, optimizer,
# EMA commit, reward and evaluation sets stay matched to the formal run.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
PYTHON=${ROOT}/.venv/bin/python
TORCHRUN=${ROOT}/.venv/bin/torchrun
MODEL=${MODEL:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
ARGS=${ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
TRAIN=/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json
VALID=/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json
CONFIG=${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json
MINE_RUN=${MINE_RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine}
SOURCE_CACHE=${MINE_RUN}/replay_buffer
OVERLAY=${OVERLAY:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_positive_overlay/replay_buffer}
COMPRESSED_CONTEXT=${COMPRESSED_CONTEXT:-${MINE_RUN}/replay_context_zstd_v1}
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_scored_prefix_active_expert_ablation}
EXPECTED_GROUPS=5446656
EXPECTED_RANK_GROUPS=680832
EXPECTED_SELECTOR_SHA256=49fd1863a3c1d54db59440da426e3475df67ccc4edd8da6ddde7e32945f3620c
EXPECTED_VALID_SHA256=91a1c8ef4004c7074f024495d13416456a4dbe2f0320f0b1a43282659d435dff
LOSS_HORIZON=${LOSS_HORIZON:-40}

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

die() {
  echo "scored-prefix ablation: $*" >&2
  exit 1
}

validate_inputs() {
  [[ -s ${MODEL} && -s ${ARGS} ]] || die "missing incumbent or model args"
  [[ ${LOSS_HORIZON} -eq 40 ]] \
    || die "formal reward contract scores exactly 40 future steps"
  [[ -s ${OVERLAY%/replay_buffer}/overlay_manifest.json ]] \
    || die "missing strict positive-overlay manifest"
  jq -e --arg source "$(realpath -e "${SOURCE_CACHE}")" \
    --argjson groups "${EXPECTED_GROUPS}" '
      .source_replay == $source and .groups == $groups and
      .parameters.beta == 2 and .parameters.margin == 0.01 and
      .parameters.behavior_anchor_weight == 0 and
      .parameters.unsafe_behavior_anchor_weight == 0 and
      .parameters.drop_all_zero_groups == true and
      .expert_anchor_sidecar == true
    ' "${OVERLAY%/replay_buffer}/overlay_manifest.json" >/dev/null \
    || die "positive-overlay contract differs"

  local rank count total=0
  for rank in $(seq 0 7); do
    [[ -s ${SOURCE_CACHE}/rank_$(printf '%04d' "${rank}")/manifest.json ]] \
      || die "source rank ${rank} has no strict manifest"
    [[ -s ${OVERLAY}/rank_$(printf '%04d' "${rank}")/manifest.json ]] \
      || die "overlay rank ${rank} has no strict manifest"
    count=$(jq -er '.scene_count' \
      "${SOURCE_CACHE}/rank_$(printf '%04d' "${rank}")/manifest.json")
    [[ ${count} -eq ${EXPECTED_RANK_GROUPS} ]] \
      || die "source rank ${rank} count ${count} differs"
    total=$((total + count))
  done
  [[ ${total} -eq ${EXPECTED_GROUPS} ]] \
    || die "source cache total ${total} differs"
}

validate_compressed_context() {
  local rank
  for rank in $(seq 0 7); do
    [[ -s ${COMPRESSED_CONTEXT}/rank_$(printf '%04d' "${rank}")/manifest.json ]] \
      || return 1
  done
  "${PYTHON}" - "${SOURCE_CACHE}" "${COMPRESSED_CONTEXT}" <<'PY'
import json
import pathlib
import sys

from rlvr.replay_context_codec import CompressedReplayContextReader

source = pathlib.Path(sys.argv[1])
compressed = pathlib.Path(sys.argv[2])
for rank in range(8):
    source_manifest = json.loads(
        (source / f"rank_{rank:04d}" / "manifest.json").read_text()
    )
    reader = CompressedReplayContextReader(
        compressed,
        rank,
        expected_scene_count=int(source_manifest["scene_count"]),
        expected_paths_sha256=str(
            source_manifest["decoder_context_attachment"][
                "expected_paths_sha256"
            ]
        ),
    )
    reader.close()
PY
}

ensure_compressed_context() {
  if validate_compressed_context; then
    echo "using verified lossless compressed replay context: ${COMPRESSED_CONTEXT}"
    return 0
  fi
  if pgrep -f 'rlvr.train_awr' >/dev/null; then
    die "refusing to compete with active training while building compressed context"
  fi
  mkdir -p "${COMPRESSED_CONTEXT}"
  local -a pids=()
  local rank rank_dir log
  for rank in $(seq 0 7); do
    printf -v rank_dir '%s/rank_%04d' "${COMPRESSED_CONTEXT}" "${rank}"
    [[ -s ${rank_dir}/manifest.json ]] && continue
    printf -v log '%s/build_rank_%04d.log' "${COMPRESSED_CONTEXT}" "${rank}"
    nice -n 5 "${PYTHON}" \
      -m rlvr.autoresearch.tools.build_compressed_replay_context \
      --source-replay "${SOURCE_CACHE}" --output-root "${COMPRESSED_CONTEXT}" \
      --rank "${rank}" --compression-level 1 --progress-interval 10000 \
      > "${log}" 2>&1 &
    pids+=("$!")
  done
  local pid status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  [[ ${status} -eq 0 ]] || die "lossless compressed-context build failed"
  validate_compressed_context \
    || die "compressed-context publication validation failed"
}

main() {
  validate_inputs
  if pgrep -f 'rlvr.train_awr' >/dev/null; then
    die "another AWR process is active; wait for the epoch-boundary decision"
  fi
  ensure_compressed_context
  mkdir -p "${OUT}"
  local stamp log
  stamp=$(date +%Y%m%d-%H%M%S)
  log=${OUT}/${stamp}_launch.log
  echo "launch log: ${log}"
  "${TORCHRUN}" --standalone --nproc_per_node=8 -m rlvr.train_awr \
    --model_path "${MODEL}" --args_path "${ARGS}" \
    --train_npz_list "${TRAIN}" --valid_npz_list "${VALID}" \
    --config "${CONFIG}" --output_dir "${OUT}" \
    --exp_name plannerrft_scored_prefix40_active_expert_full_e02 \
    --device cuda --seed 3407 --start_epoch 2 --epochs 2 \
    --resume_replay_root "${OVERLAY}" --replay_sampling with_replacement \
    --compressed_replay_context_root "${COMPRESSED_CONTEXT}" \
    --awr_beta 2 --positive_advantage_only --positive_advantage_margin 0.01 \
    --behavior_anchor_weight 0 --unsafe_behavior_anchor_weight 0 \
    --expert_anchor_weight 0.4 --expert_anchor_active_groups_only \
    --awr_candidate_loss_horizon "${LOSS_HORIZON}" \
    --drop_all_zero_groups --awr_loss_type plain_mse \
    --diffusion_t_range 0.001 0.2 --trainable_scope output \
    --learning_rate 0.000001 --ema_decay 0.95 --ema_per_epoch \
    --ema_commit_live_policy --max_train_scenes 0 --max_valid_scenes 0 \
    --eval_k 1 --eval_sample_steps 10 --train_selector_scenes 65536 \
    --expected_train_selector_sha256 "${EXPECTED_SELECTOR_SHA256}" \
    --expected_valid_sha256 "${EXPECTED_VALID_SHA256}" \
    --full_valid_interval 0 --hard_valid_scenes 0 \
    --scene_batch_size 192 --gradient_accumulation_scenes 192 \
    --scene_load_workers 4 --eval_scene_load_workers 1 \
    --eval_scene_batch_size 512 --save_rollouts 0 \
    --wandb --wandb_project original-dp-awr \
    --wandb_entity advanced-technology-department \
    --wandb_run_name original-dp-awr-plannerrft-prefix40-active-expert-full-e02 \
    --wandb_mode online \
    --wandb_tags original-dp awr plannerrft-guidance scored-prefix40 \
      active-group-expert matched-full-ablation positive-margin001 beta2 \
      full-sequence --no-skip_filtered_scenes 2>&1 | tee "${log}"
}

main "$@"
