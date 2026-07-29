#!/usr/bin/env bash
set -euo pipefail

# Idempotent phase-2 continuation for the formal PlannerRFT full-corpus run.
# It is intentionally separate from the mining launcher so a fully committed
# cache remains usable if the original shell exits at the phase boundary.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main

# The valid manifest is group-readable by ubuntu.  A long-lived shell may
# predate wang's supplementary-group refresh, so repair the execution context
# before any manifest access instead of failing later inside one DDP rank.
if ! id -Gn | tr ' ' '\n' | grep -qx 'ubuntu'; then
  if getent group ubuntu | awk -F: -v user="${USER:-$(id -un)}" '
      { n = split($4, members, ","); for (i = 1; i <= n; ++i) if (members[i] == user) found = 1 }
      END { exit(found ? 0 : 1) }
    '; then
    script_path=$(readlink -f "${BASH_SOURCE[0]}")
    printf -v command_q '%q ' "${script_path}" "$@"
    exec sg ubuntu -c "exec ${command_q}"
  fi
  echo "wang is a member of ubuntu but the current shell lacks the group; sg ubuntu re-exec failed" >&2
  exit 1
fi

MODEL=${MODEL:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
ARGS=${ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
TRAIN=${TRAIN:-/mnt/storage_rdma/diffusion_planner/dataset/20260707_vehicle_params_with_mirror/path_list_train_concatenated.json}
VALID=${VALID:-/mnt/storage_rdma/diffusion_planner/dataset/20260702_basic_dataset/path_list_valid_sft_balanced.json}
CONFIG=${CONFIG:-${ROOT}/rlvr/configs/awr_original_dp_t4_plannerrft_anchored.json}
MINE_PARENT=${MINE_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine}
OVERLAY_PARENT=${OVERLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_positive_overlay}
REPLAY_PARENT=${REPLAY_PARENT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_replay_e02_e10}
SOURCE_SCENES=5446154
GLOBAL_ROLLOUT_BATCH=$((8 * 192))
EXPECTED_CACHED_GROUPS=$(((SOURCE_SCENES + GLOBAL_ROLLOUT_BATCH - 1) / GLOBAL_ROLLOUT_BATCH * GLOBAL_ROLLOUT_BATCH))
EXPECTED_SELECTOR_SHA256=49fd1863a3c1d54db59440da426e3475df67ccc4edd8da6ddde7e32945f3620c
EXPECTED_VALID_SHA256=91a1c8ef4004c7074f024495d13416456a4dbe2f0320f0b1a43282659d435dff

cd "${ROOT}"
MINE_RUN=$(find "${MINE_PARENT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
if [[ -z "${MINE_RUN}" ]]; then
  echo "formal PlannerRFT mine run is missing" >&2
  exit 1
fi
MINE_CACHE=${MINE_RUN}/replay_buffer

if pgrep -f 'rlvr.train_awr.*plannerrft_full_cycle01_replay_e02_e10' >/dev/null; then
  echo "formal PlannerRFT replay is already active"
  exit 0
fi
if [[ $(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l) -ne 8 ]]; then
  echo "mine cache is not committed on all eight ranks" >&2
  exit 1
fi
if [[ $(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name expert_anchor_manifest.json | wc -l) -ne 8 ]]; then
  echo "expert sidecar is not committed on all eight ranks" >&2
  exit 1
fi
CONTEXT_MANIFESTS=$(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name context_build_manifest.json | wc -l)
INLINE_CONTEXT_RANKS=$(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json -print0 \
  | xargs -0 jq -s '[.[] | select(.decoder_context_cached == true and .decoder_context_schema == "canonical_loader_aligned_v1")] | length')
if [[ ${CONTEXT_MANIFESTS} -ne 8 && ${INLINE_CONTEXT_RANKS} -ne 8 ]]; then
  echo "decoder context is not committed on all eight ranks (sidecars=${CONTEXT_MANIFESTS}, inline=${INLINE_CONTEXT_RANKS})" >&2
  exit 1
fi
# Metadata attachment is idempotent and validates every rank, canonical path
# order, +1 neighbour alignment, X2 width and context array shape before the
# first atomic manifest replacement.  Calling it here as well as in the
# primary watcher makes the crash-recovery path unable to silently fall back
# to reopening full NPZs during replay.
"${ROOT}/.venv/bin/python" -m \
  rlvr.autoresearch.tools.attach_replay_decoder_context \
  --replay-root "${MINE_CACHE}" --expected-world-size 8 \
  >> "${MINE_RUN}/decoder_context_transition.log" 2>&1
CACHED_GROUPS=$(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json -print0 \
  | xargs -0 jq -s 'map(.scene_count) | add')
if [[ ${CACHED_GROUPS} -ne ${EXPECTED_CACHED_GROUPS} ]]; then
  echo "cache count ${CACHED_GROUPS} does not match padded contract ${EXPECTED_CACHED_GROUPS}" >&2
  exit 1
fi
EXPERT_GROUPS=$(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name expert_anchor_manifest.json -print0 \
  | xargs -0 jq -s 'map(.scene_count) | add')
if [[ ${EXPERT_GROUPS} -ne ${EXPECTED_CACHED_GROUPS} ]]; then
  echo "expert sidecar count ${EXPERT_GROUPS} does not match padded contract ${EXPECTED_CACHED_GROUPS}" >&2
  exit 1
fi
if [[ ${CONTEXT_MANIFESTS} -eq 8 ]]; then
  CONTEXT_GROUPS=$(find "${MINE_CACHE}" -mindepth 2 -maxdepth 2 -name context_build_manifest.json -print0 \
    | xargs -0 jq -s 'map(.scene_count) | add')
else
  # The clean Original-DP miner caches decoder context inline in each replay
  # manifest; there is no independent context_build_manifest.json to sum.
  # attach_replay_decoder_context has already validated all eight inline
  # arrays and their path hashes above.
  CONTEXT_GROUPS=${EXPECTED_CACHED_GROUPS}
fi
if [[ ${CONTEXT_GROUPS} -ne ${EXPECTED_CACHED_GROUPS} ]]; then
  echo "decoder-context count ${CONTEXT_GROUPS} does not match padded contract ${EXPECTED_CACHED_GROUPS}" >&2
  exit 1
fi
jq -n \
  --argjson source_scene_count "${SOURCE_SCENES}" \
  --argjson cached_group_count "${CACHED_GROUPS}" \
  --argjson global_rollout_batch "${GLOBAL_ROLLOUT_BATCH}" \
  '{
    version: 1,
    contract: "all_source_scenes_once_then_deterministic_prefix_padding",
    source_scene_count: $source_scene_count,
    cached_group_count: $cached_group_count,
    ddp_tail_padding_count: ($cached_group_count - $source_scene_count),
    global_rollout_batch: $global_rollout_batch
  }' > "${MINE_CACHE}/ddp_tail_padding_contract.json"

OVERLAY_CACHE=${OVERLAY_PARENT}/replay_buffer
OVERLAY_MANIFESTS=0
if [[ -d "${OVERLAY_CACHE}" ]]; then
  OVERLAY_MANIFESTS=$(find "${OVERLAY_CACHE}" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l)
fi
if [[ ${OVERLAY_MANIFESTS} -eq 0 ]]; then
  if [[ -e "${OVERLAY_PARENT}" ]]; then
    echo "incomplete derived overlay exists at ${OVERLAY_PARENT}; refusing overwrite" >&2
    exit 1
  fi
  "${ROOT}/.venv/bin/python" -m \
    rlvr.autoresearch.tools.build_positive_anchor_replay_overlay \
    --source-replay "${MINE_CACHE}" \
    --output-replay "${OVERLAY_CACHE}" \
    --beta 2 \
    --margin 0.01 \
    --behavior-anchor-weight 0 \
    --unsafe-behavior-anchor-weight 0
elif [[ ${OVERLAY_MANIFESTS} -ne 8 ]]; then
  echo "derived overlay has ${OVERLAY_MANIFESTS}/8 manifests" >&2
  exit 1
fi

TARGET_GEOMETRY=${OVERLAY_PARENT}/target_geometry.json
if [[ ! -s ${TARGET_GEOMETRY} ]]; then
  "${ROOT}/.venv/bin/python" -m \
    rlvr.autoresearch.tools.analyze_replay_target_geometry \
    --replay-root "${OVERLAY_CACHE}" \
    --output "${TARGET_GEOMETRY}" \
    --expert-weight 0.4 \
    > "${OVERLAY_PARENT}/target_geometry.log" 2>&1
fi
jq -e --argjson groups "${EXPECTED_CACHED_GROUPS}" '
  .source_complete == true and
  .prefix_groups == $groups and
  .group_size == 10 and .future_len == 80 and
  .contract.weight_source == "stored replay weights" and
  .parameters.expert_weight == 0.4 and
  .active_group_fraction > 0 and
  .active_candidate_target_reward_gain > 0
' "${TARGET_GEOMETRY}" >/dev/null || {
  echo "target-geometry audit differs from the formal replay contract" >&2
  exit 1
}

if [[ -e "${REPLAY_PARENT}" ]]; then
  if find "${REPLAY_PARENT}" -mindepth 2 -maxdepth 2 -name final_summary.json -print -quit | grep -q .; then
    echo "replay output already contains a committed run: ${REPLAY_PARENT}" >&2
    exit 1
  fi
  # A prior launch may have written only provenance/base_model metadata before
  # DDP initialization.  Keep that forensic record; train_awr creates a new
  # timestamped child run under this parent and starts from the same source
  # policy, so no replay progress or cache is discarded.
  echo "partial replay parent exists; starting a fresh timestamped child run: ${REPLAY_PARENT}" >&2
fi

export PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-0}
export DP_DDP_TIMEOUT_MINUTES=${DP_DDP_TIMEOUT_MINUTES:-180}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-online}

exec "${ROOT}/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
  -m rlvr.train_awr \
  --model_path "${MODEL}" \
  --args_path "${ARGS}" \
  --train_npz_list "${TRAIN}" \
  --valid_npz_list "${VALID}" \
  --config "${CONFIG}" \
  --output_dir "${REPLAY_PARENT}" \
  --exp_name plannerrft_full_cycle01_replay_e02_e10 \
  --device cuda \
  --seed 3407 \
  --start_epoch 2 \
  --epochs 10 \
  --resume_replay_root "${OVERLAY_CACHE}" \
  --replay_sampling with_replacement \
  --awr_beta 2 \
  --positive_advantage_only \
  --positive_advantage_margin 0.01 \
  --behavior_anchor_weight 0 \
  --unsafe_behavior_anchor_weight 0 \
  --expert_anchor_weight 0.4 \
  --drop_all_zero_groups \
  --awr_loss_type plain_mse \
  --diffusion_t_range 0.001 0.2 \
  --trainable_scope output \
  --learning_rate 0.000001 \
  --ema_decay 0.95 \
  --ema_per_epoch \
  --ema_commit_live_policy \
  --max_train_scenes 0 \
  --max_valid_scenes 0 \
  --eval_k 1 \
  --eval_sample_steps 10 \
  --train_selector_scenes 65536 \
  --expected_train_selector_sha256 "${EXPECTED_SELECTOR_SHA256}" \
  --expected_valid_sha256 "${EXPECTED_VALID_SHA256}" \
  --full_valid_interval 0 \
  --hard_valid_scenes 0 \
  --scene_batch_size 192 \
  --gradient_accumulation_scenes 192 \
  --scene_load_workers 4 \
  --eval_scene_load_workers 1 \
  --eval_scene_batch_size 512 \
  --save_rollouts 0 \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-original-dp-awr}" \
  --wandb_entity "${WANDB_ENTITY:-advanced-technology-department}" \
  --wandb_run_name "${WANDB_RUN_NAME:-original-dp-awr-plannerrft-full-cycle01-e02-e10}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_tags original-dp awr plannerrft-guidance conditional-guidance \
    positive-margin001 beta2 expert04 epoch-boundary-alpha005 cumulative-replay \
    full-sequence cycle01 \
  --no-skip_filtered_scenes
