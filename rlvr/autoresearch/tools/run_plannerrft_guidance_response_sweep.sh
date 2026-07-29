#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU, same-scene response sweep for the PlannerRFT-style Original-DP
# guidance probe.  This is diagnostic only: it writes figures/metrics and does
# not mutate a checkpoint or replay cache.

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/response_sweep_scene000002
POLICY=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth
ARGS=${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json
REFERENCE=/tmp/t4_v5_original_dp/model/best_model.pth
REFERENCE_ARGS=/tmp/t4_v5_original_dp/model/args.json
SCENES=${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes.json
REWARD=${ROOT}/rlvr/configs/awr_original_dp_t4_hdp_faithful.json

mkdir -p "${OUT}"
labels=(lat1_s0p5 lat1_s1 lat1_s2 lat1_s4 lat2p5_s0p5 lat2p5_s1 lat2p5_s2 lat2p5_s4)
lateral=(1.0 1.0 1.0 1.0 2.5 2.5 2.5 2.5)
scale=(0.5 1.0 2.0 4.0 0.5 1.0 2.0 4.0)
pids=()

for gpu in "${!labels[@]}"; do
  label=${labels[$gpu]}
  run_dir=${OUT}/${label}
  mkdir -p "${run_dir}"
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
      --indices 2 \
      --reward_config "${REWARD}" \
      --output_dir "${run_dir}" \
      --noise_scale 0.5 \
      --lambda_lat "${lateral[$gpu]}" \
      --lambda_lon 0.25 \
      --guidance_scale "${scale[$gpu]}" \
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
  echo "one or more response-sweep workers failed" >&2
  exit "${status}"
fi

"${ROOT}/.venv/bin/python" - "${OUT}" "${labels[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for label in sys.argv[2:]:
    manifest = json.loads((root / label / "manifest.json").read_text())
    scene = manifest["scenes_results"][0]
    candidates = scene["ranked_candidates"]
    by_command = {row["candidate_label"]: row for row in candidates}
    rows.append(
        {
            "label": label,
            "lambda_lat_m": manifest["lambda_lat"],
            "guidance_scale": manifest["guidance_scale"],
            "best_reward": max(row["total"] for row in candidates),
            "hard_safe_count": sum(
                not (row["collision"] or row["rb_crossing"] or row["kinematic"])
                for row in candidates
            ),
            "max_signed_obb_clearance_m": max(
                row["min_obb_clearance_m"] for row in candidates
            ),
            "endpoint_pairwise_mean_m": scene["diversity"][
                "endpoint_pairwise_mean_m"
            ],
            "endpoint_pairwise_max_m": scene["diversity"][
                "endpoint_pairwise_max_m"
            ],
            "max_first_waypoint_from_reference_m": max(
                row["first_waypoint_from_reference_m"] for row in candidates
            ),
            "max_first_waypoint_from_unguided_m": max(
                row["first_waypoint_from_unguided_m"] for row in candidates
            ),
            "realized_endpoint_lateral_m": {
                command: by_command[command]["endpoint_lateral_from_reference_m"]
                for command in ("L+", "L-", "L+V+", "L-V+")
            },
            "realized_endpoint_longitudinal_m": {
                command: by_command[command]["endpoint_longitudinal_from_reference_m"]
                for command in ("V+", "V-", "L+V+", "L+V-")
            },
            "manifest": str((root / label / "manifest.json").resolve()),
            "figure": str(next((root / label).glob("scene_*.png")).resolve()),
        }
    )

payload = {
    "scene_index": 2,
    "same_policy_reference_noise_and_reward": True,
    "selection_role": "response calibration only; no checkpoint selection",
    "rows": rows,
}
(root / "sweep_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "response sweep complete: ${OUT}/sweep_summary.json"
