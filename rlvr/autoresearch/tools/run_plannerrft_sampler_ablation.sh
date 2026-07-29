#!/usr/bin/env bash
set -euo pipefail

# Paired 384-scene candidate-discovery ablation. Every arm reuses the same
# K=10 independent noise latents. Eight GPUs each own one 48-scene stratum.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
ETA_SCHEME=${ETA_SCHEME:-iid_beta}
REFERENCE_MODE=${REFERENCE_MODE:-zero_noise}
REFERENCE_NOISE_SCALE=${REFERENCE_NOISE_SCALE:-0.5}
LONGITUDINAL_MODE=${LONGITUDINAL_MODE:-candidate_stretch}
LAMBDA_LAT=${LAMBDA_LAT:-1.0}
SAMPLE_STEPS=${SAMPLE_STEPS:-10}
ARMS=${ARMS:-"all_guided mixed2 mixed5"}
[[ ${SAMPLE_STEPS} =~ ^[0-9]+$ ]] && (( SAMPLE_STEPS >= 2 )) || {
  echo "SAMPLE_STEPS must be an integer >= 2" >&2
  exit 2
}
read -r -a arm_args <<< "${ARMS}"
case "${REFERENCE_MODE}" in
  zero_noise|stochastic) ;;
  *)
    echo "REFERENCE_MODE must be zero_noise or stochastic" >&2
    exit 2
    ;;
esac
case "${LONGITUDINAL_MODE}" in
  candidate_stretch|reference_speed_push) ;;
  *)
    echo "LONGITUDINAL_MODE must be candidate_stretch or reference_speed_push" >&2
    exit 2
    ;;
esac
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/${ETA_SCHEME}_${REFERENCE_MODE}_reference_${LONGITUDINAL_MODE}_sampler_ablation_lat${LAMBDA_LAT}_s2_stratified384}
POLICY=${POLICY:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
ARGS=${POLICY_ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
REFERENCE=${REFERENCE:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/base_model/best_model.pth}
REFERENCE_ARGS=${REFERENCE_ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/base_model/args.json}
SCENES=${SCENES:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes_384.json}
REWARD=${REWARD:-${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_faithful.json}

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

if [[ -e ${OUT} ]]; then
  echo "ablation output already exists; refusing to overwrite ${OUT}" >&2
  exit 1
fi
mkdir -p "${OUT}"
pids=()
for gpu in "${!strata[@]}"; do
  stratum=${strata[$gpu]}
  start=$((gpu * 48))
  end=$((start + 47))
  run_dir=${OUT}/${stratum}
  mkdir -p "${run_dir}"
  indices=()
  for index in $(seq "${start}" "${end}"); do
    indices+=("${index}")
  done
  env \
    PYTHONPATH="${ROOT}" \
    DP_NEIGHBOR_FUTURE_OFFSET=${DP_NEIGHBOR_FUTURE_OFFSET:-1} \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "${ROOT}/.venv/bin/python" -u \
      "${ROOT}/rlvr/autoresearch/tools/diag_plannerrft_sampler_ablation.py" \
      --model_path "${POLICY}" \
      --args_path "${ARGS}" \
      --reference_model_path "${REFERENCE}" \
      --reference_args_path "${REFERENCE_ARGS}" \
      --scenes "${SCENES}" \
      --indices "${indices[@]}" \
      --reward_config "${REWARD}" \
      --output_dir "${run_dir}" \
      --group_size 10 \
      --noise_scale 0.5 \
      --sample_steps "${SAMPLE_STEPS}" \
      --reference_mode "${REFERENCE_MODE}" \
      --reference_noise_scale "${REFERENCE_NOISE_SCALE}" \
      --eta_scheme "${ETA_SCHEME}" \
      --longitudinal_mode "${LONGITUDINAL_MODE}" \
      --arms "${arm_args[@]}" \
      --lambda_lat "${LAMBDA_LAT}" \
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
  echo "one or more fixed-Beta sampler workers failed" >&2
  exit "${status}"
fi

"${ROOT}/.venv/bin/python" - "${OUT}" "${POLICY}" "${REFERENCE}" "${strata[@]}" <<'PY'
from collections import Counter, defaultdict
import json
import pathlib
import statistics
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
policy = pathlib.Path(sys.argv[2])
reference = pathlib.Path(sys.argv[3])
strata = sys.argv[4:]


def hard_safe(candidate):
    return not (
        bool(candidate.get("collision", False))
        or bool(candidate.get("rb_crossing", False))
        or bool(candidate.get("kinematic", False))
        or float(candidate.get("red_light", 0.0)) < -0.5
    )


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(q * 100):02d}": float(np.quantile(array, q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


rows = []
protocol = None
for stratum in strata:
    manifest = json.loads((root / stratum / "manifest.json").read_text())
    protocol = protocol or manifest["protocol"]
    for scene in manifest["scenes_results"]:
        arm_stats = {}
        for arm, payload in scene["arms"].items():
            candidates = payload["ranked_candidates"]
            rewards = [float(row["total"]) for row in candidates]
            best = max(candidates, key=lambda row: float(row["total"]))
            arm_stats[arm] = {
                "best_reward": max(rewards),
                "mean_reward": statistics.fmean(rewards),
                "reward_std": statistics.pstdev(rewards),
                "nonzero_count": sum(value > 0.0 for value in rewards),
                "all_zero": all(value == 0.0 for value in rewards),
                "hard_safe_count": sum(hard_safe(row) for row in candidates),
                "collision_count": sum(bool(row.get("collision", False)) for row in candidates),
                "road_border_count": sum(bool(row.get("rb_crossing", False)) for row in candidates),
                "kinematic_count": sum(bool(row.get("kinematic", False)) for row in candidates),
                "best_index": int(best["k"]),
                "best_label": best["candidate_label"],
                "best_is_guided": bool(best["is_guided"]),
                "endpoint_pairwise_mean_m": float(
                    payload["diversity"]["endpoint_pairwise_mean_m"]
                ),
                "endpoint_pairwise_max_m": float(
                    payload["diversity"]["endpoint_pairwise_max_m"]
                ),
                "max_first_waypoint_delta_from_paired_native_m": max(
                    float(row["first_waypoint_from_unguided_m"])
                    for row in candidates
                ),
            }
        native = arm_stats["native"]
        comparisons = {}
        for arm, stats in arm_stats.items():
            if arm == "native":
                continue
            comparisons[arm] = {
                "best_reward_gain": stats["best_reward"] - native["best_reward"],
                "mean_reward_gain": stats["mean_reward"] - native["mean_reward"],
                "hard_safe_count_gain": stats["hard_safe_count"] - native["hard_safe_count"],
                "recovered_native_all_zero": native["all_zero"] and not stats["all_zero"],
                "lost_native_nonzero": not native["all_zero"] and stats["all_zero"],
                "improved_best": stats["best_reward"] > native["best_reward"] + 1e-12,
                "degraded_best": stats["best_reward"] < native["best_reward"] - 1e-12,
                "preserved_or_improved_best": stats["best_reward"] >= native["best_reward"] - 1e-12,
            }
        rows.append(
            {
                "stratum": stratum,
                "scene_index": scene["scene_index"],
                "scene_path": scene["scene_path"],
                "arm_stats": arm_stats,
                "comparisons": comparisons,
                "arm_diagnostics": scene["arm_diagnostics"],
                "manifest": str((root / stratum / "manifest.json").resolve()),
            }
        )


arms = [arm for arm in rows[0]["arm_stats"] if arm != "native"]


def summarize(group):
    payload = {
        "scene_count": len(group),
        "native": {
            "all_zero_count": sum(row["arm_stats"]["native"]["all_zero"] for row in group),
            "mean_best_reward": statistics.fmean(
                row["arm_stats"]["native"]["best_reward"] for row in group
            ),
            "mean_candidate_reward": statistics.fmean(
                row["arm_stats"]["native"]["mean_reward"] for row in group
            ),
            "mean_hard_safe_count_of_10": statistics.fmean(
                row["arm_stats"]["native"]["hard_safe_count"] for row in group
            ),
            "mean_endpoint_pairwise_m": statistics.fmean(
                row["arm_stats"]["native"]["endpoint_pairwise_mean_m"] for row in group
            ),
        },
        "arms": {},
    }
    for arm in arms:
        gains = [row["comparisons"][arm]["best_reward_gain"] for row in group]
        payload["arms"][arm] = {
            "all_zero_count": sum(row["arm_stats"][arm]["all_zero"] for row in group),
            "recovered_native_all_zero_count": sum(
                row["comparisons"][arm]["recovered_native_all_zero"] for row in group
            ),
            "lost_native_nonzero_count": sum(
                row["comparisons"][arm]["lost_native_nonzero"] for row in group
            ),
            "improved_best_count": sum(
                row["comparisons"][arm]["improved_best"] for row in group
            ),
            "degraded_best_count": sum(
                row["comparisons"][arm]["degraded_best"] for row in group
            ),
            "preserved_or_improved_best_count": sum(
                row["comparisons"][arm]["preserved_or_improved_best"] for row in group
            ),
            "mean_best_reward": statistics.fmean(
                row["arm_stats"][arm]["best_reward"] for row in group
            ),
            "mean_best_reward_gain": statistics.fmean(gains),
            "best_reward_gain_quantiles": quantiles(gains),
            "mean_candidate_reward": statistics.fmean(
                row["arm_stats"][arm]["mean_reward"] for row in group
            ),
            "mean_candidate_reward_gain": statistics.fmean(
                row["comparisons"][arm]["mean_reward_gain"] for row in group
            ),
            "mean_hard_safe_count_of_10": statistics.fmean(
                row["arm_stats"][arm]["hard_safe_count"] for row in group
            ),
            "mean_hard_safe_count_gain": statistics.fmean(
                row["comparisons"][arm]["hard_safe_count_gain"] for row in group
            ),
            "mean_endpoint_pairwise_m": statistics.fmean(
                row["arm_stats"][arm]["endpoint_pairwise_mean_m"] for row in group
            ),
            "max_first_waypoint_delta_from_paired_native_m": max(
                row["arm_stats"][arm]["max_first_waypoint_delta_from_paired_native_m"]
                for row in group
            ),
            "best_guided_count": sum(
                row["arm_stats"][arm]["best_is_guided"] for row in group
            ),
        }
    return payload


grouped = defaultdict(list)
for row in rows:
    grouped[row["stratum"]].append(row)

max_equivalence_error = max(
    float(diagnostics.get("zero_strength_native_max_xy_m", 0.0))
    for row in rows
    for diagnostics in row["arm_diagnostics"].values()
)

summary = {
    "protocol": {
        **protocol,
        "policy": str(policy.resolve()),
        "frozen_reference": str(reference.resolve()),
        "aggregation_note": (
            "native and every guided arm share scene, K latent tensors, fixed noise "
            "temperature and fixed-Beta eta draws"
        ),
    },
    "numerical_equivalence": {
        "max_zero_strength_native_xy_error_m": max_equivalence_error,
        "passed": max_equivalence_error <= float(protocol["equivalence_atol_m"]),
    },
    "overall": summarize(rows),
    "by_stratum": {key: summarize(value) for key, value in grouped.items()},
    "top_improvements": {
        arm: sorted(
            (
                {
                    "stratum": row["stratum"],
                    "scene_index": row["scene_index"],
                    "scene_path": row["scene_path"],
                    **row["comparisons"][arm],
                    "native_best_reward": row["arm_stats"]["native"]["best_reward"],
                    "arm_best_reward": row["arm_stats"][arm]["best_reward"],
                    "arm_best_label": row["arm_stats"][arm]["best_label"],
                    "manifest": row["manifest"],
                }
                for row in rows
            ),
            key=lambda item: item["best_reward_gain"],
            reverse=True,
        )[:20]
        for arm in arms
    },
    "worst_regressions": {
        arm: sorted(
            (
                {
                    "stratum": row["stratum"],
                    "scene_index": row["scene_index"],
                    "scene_path": row["scene_path"],
                    **row["comparisons"][arm],
                    "native_best_reward": row["arm_stats"]["native"]["best_reward"],
                    "arm_best_reward": row["arm_stats"][arm]["best_reward"],
                    "arm_best_label": row["arm_stats"][arm]["best_label"],
                    "manifest": row["manifest"],
                }
                for row in rows
            ),
            key=lambda item: item["best_reward_gain"],
        )[:20]
        for arm in arms
    },
    "rows": rows,
}
(root / "sampler_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps({
    "numerical_equivalence": summary["numerical_equivalence"],
    "overall": summary["overall"],
}, indent=2))
PY
