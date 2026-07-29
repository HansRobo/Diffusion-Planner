#!/usr/bin/env bash
set -euo pipefail

# Full 192-scene stratified response-grid audit at the calibrated conservative
# PlannerRFT-style setting.  Each GPU owns one 24-scene stratum.  This is an
# inference/visualization study, not an AWR training run.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/response_grid_lat1_s2_stratified192
POLICY=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth
ARGS=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json
REFERENCE=/tmp/t4_v5_original_dp/model/best_model.pth
REFERENCE_ARGS=/tmp/t4_v5_original_dp/model/args.json
SCENES=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.json
SCENE_MANIFEST=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.manifest.json
REWARD=${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_faithful.json

strata=(
  all_zero_or_tied
  collision
  road_border
  kinematic
  low_reward_nonterminal
  native_mode_collapse
  native_high_diversity
  ordinary_random
)

mkdir -p "${OUT}"
pids=()
for gpu in "${!strata[@]}"; do
  stratum=${strata[$gpu]}
  start=$((gpu * 24))
  end=$((start + 23))
  run_dir=${OUT}/${stratum}
  mkdir -p "${run_dir}"
  indices=()
  for index in $(seq "${start}" "${end}"); do
    indices+=("${index}")
  done
  env \
    PYTHONPATH="${ROOT}" \
    DP_NEIGHBOR_FUTURE_OFFSET=1 \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "${ROOT}/.venv/bin/python" -u \
      "${ROOT}/rlvr/autoresearch/tools/diag_plannerrft_guidance.py" \
      --model_path "${POLICY}" \
      --args_path "${ARGS}" \
      --reference_model_path "${REFERENCE}" \
      --reference_args_path "${REFERENCE_ARGS}" \
      --scenes "${SCENES}" \
      --indices "${indices[@]}" \
      --reward_config "${REWARD}" \
      --output_dir "${run_dir}" \
      --noise_scale 0.5 \
      --lambda_lat 1.0 \
      --lambda_lon 0.25 \
      --guidance_scale 2.0 \
      --ramp_steps 20 \
      --compile_mode default \
      >"${run_dir}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "one or more stratified-response workers failed" >&2
  exit "${status}"
fi

"${ROOT}/.venv/bin/python" - "${OUT}" "${SCENE_MANIFEST}" "${POLICY}" "${strata[@]}" <<'PY'
from collections import Counter, defaultdict
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
source_manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
policy = pathlib.Path(sys.argv[3])
strata = sys.argv[4:]
baseline_by_path = {row["scene_path"]: row for row in source_manifest["rows"]}


def is_hard_safe(row):
    return not (
        row["collision"]
        or row["rb_crossing"]
        or row["kinematic"]
        or float(row.get("red_light", 0.0)) < -0.5
    )


rows = []
for stratum in strata:
    manifest = json.loads((root / stratum / "manifest.json").read_text())
    for scene in manifest["scenes_results"]:
        baseline = baseline_by_path[scene["scene_path"]]
        candidates = scene["ranked_candidates"]
        by_command = {row["candidate_label"]: row for row in candidates}
        best = max(candidates, key=lambda row: row["total"])
        guided_only = [row for row in candidates if row["candidate_label"] not in {"U", "C"}]
        best_guided = max(guided_only, key=lambda row: row["total"])
        u = by_command["U"]
        hard_safe_count = sum(is_hard_safe(row) for row in candidates)
        rows.append(
            {
                "stratum": stratum,
                "scene_index": scene["scene_index"],
                "scene_path": scene["scene_path"],
                "baseline_best_reward": baseline["best_group_reward"],
                "baseline_mean_reward": baseline["mean_group_reward"],
                "baseline_all_zero": bool(baseline["all_zero_reward_group"]),
                "baseline_safe_candidate_count": baseline["safe_candidate_count"],
                "response_u_reward": u["total"],
                "response_best_reward": best["total"],
                "response_best_guided_reward": best_guided["total"],
                "response_mean_reward": statistics.fmean(
                    row["total"] for row in candidates
                ),
                "response_hard_safe_count": hard_safe_count,
                "response_all_zero": all(row["total"] == 0.0 for row in candidates),
                "best_command": best["candidate_label"],
                "best_guided_command": best_guided["candidate_label"],
                "gain_best_vs_u": best["total"] - u["total"],
                "gain_best_vs_baseline_k10": best["total"]
                - baseline["best_group_reward"],
                "gain_best_guided_vs_u": best_guided["total"] - u["total"],
                "recovered_baseline_all_zero": bool(
                    baseline["all_zero_reward_group"] and best["total"] > 0.0
                ),
                "lost_baseline_nonzero": bool(
                    not baseline["all_zero_reward_group"] and best["total"] == 0.0
                ),
                "endpoint_pairwise_mean_m": scene["diversity"][
                    "endpoint_pairwise_mean_m"
                ],
                "endpoint_pairwise_max_m": scene["diversity"][
                    "endpoint_pairwise_max_m"
                ],
                "max_first_waypoint_from_unguided_m": max(
                    row["first_waypoint_from_unguided_m"] for row in candidates
                ),
                "max_first_waypoint_position_norm_m": max(
                    row["first_waypoint_position_norm_m"] for row in candidates
                ),
                "figure": str(
                    (
                        root
                        / stratum
                        / f"scene_{len([r for r in rows if r['stratum'] == stratum]):03d}_{scene['scene_index']:06d}.png"
                    ).resolve()
                ),
            }
        )


def summarize(group):
    return {
        "scene_count": len(group),
        "baseline_all_zero_count": sum(row["baseline_all_zero"] for row in group),
        "response_all_zero_count": sum(row["response_all_zero"] for row in group),
        "recovered_baseline_all_zero_count": sum(
            row["recovered_baseline_all_zero"] for row in group
        ),
        "lost_baseline_nonzero_count": sum(row["lost_baseline_nonzero"] for row in group),
        "mean_baseline_best_reward": statistics.fmean(
            row["baseline_best_reward"] for row in group
        ),
        "mean_response_u_reward": statistics.fmean(
            row["response_u_reward"] for row in group
        ),
        "mean_response_best_reward": statistics.fmean(
            row["response_best_reward"] for row in group
        ),
        "mean_response_best_guided_reward": statistics.fmean(
            row["response_best_guided_reward"] for row in group
        ),
        "mean_gain_best_vs_u": statistics.fmean(
            row["gain_best_vs_u"] for row in group
        ),
        "mean_gain_best_vs_baseline_k10": statistics.fmean(
            row["gain_best_vs_baseline_k10"] for row in group
        ),
        "positive_gain_vs_u_count": sum(row["gain_best_vs_u"] > 0.0 for row in group),
        "positive_gain_vs_baseline_k10_count": sum(
            row["gain_best_vs_baseline_k10"] > 0.0 for row in group
        ),
        "mean_hard_safe_count_of_10": statistics.fmean(
            row["response_hard_safe_count"] for row in group
        ),
        "mean_endpoint_pairwise_m": statistics.fmean(
            row["endpoint_pairwise_mean_m"] for row in group
        ),
        "max_first_waypoint_from_unguided_m": max(
            row["max_first_waypoint_from_unguided_m"] for row in group
        ),
        "best_command_counts": dict(Counter(row["best_command"] for row in group)),
        "best_guided_command_counts": dict(
            Counter(row["best_guided_command"] for row in group)
        ),
    }


grouped = defaultdict(list)
for row in rows:
    grouped[row["stratum"]].append(row)
payload = {
    "protocol": {
        "role": "deterministic guidance response grid; not fixed-Beta sampling and not training",
        "policy": str(policy.resolve()),
        "frozen_reference": "/tmp/t4_v5_original_dp/model/best_model.pth",
        "lambda_lat_m": 1.0,
        "lambda_lon_fraction": 0.25,
        "guidance_scale": 2.0,
        "same_noise_within_scene": True,
        "neighbor_future_offset": 1,
    },
    "overall": summarize(rows),
    "by_stratum": {stratum: summarize(grouped[stratum]) for stratum in strata},
    "rows": rows,
}
(root / "pilot_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"overall": payload["overall"], "by_stratum": payload["by_stratum"]}, indent=2))
PY

echo "stratified response pilot complete: ${OUT}/pilot_summary.json"
