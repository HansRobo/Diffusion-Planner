#!/usr/bin/env bash
# Factorial real-scene candidate-generation audit for later PlannerRFT refreshes.
# The twelve arms share the same 384 stratified scenes, policy latents, Beta
# commands and reward.  No checkpoint or replay cache is modified.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUNNER=${ROOT}/rlvr/autoresearch/tools/run_plannerrft_sampler_ablation.sh
PYTHON=${ROOT}/.venv/bin/python
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_candidate_sensitivity}
RESULT=${OUT}/selection.json
MIN_GAIN=${MIN_GAIN:-0.002}
MIN_DIVERSITY_RATIO=${MIN_DIVERSITY_RATIO:-0.80}
MAX_HARD_SAFE_COUNT_DROP=${MAX_HARD_SAFE_COUNT_DROP:-0.25}
SAMPLE_STEPS=${SAMPLE_STEPS:-5}
[[ ${SAMPLE_STEPS} == 5 || ${SAMPLE_STEPS} == 10 ]] || {
  echo "candidate sensitivity: SAMPLE_STEPS must be 5 or 10" >&2
  exit 2
}
POLICY=${POLICY:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
POLICY_ARGS=${POLICY_ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
[[ -s ${POLICY} && -s ${POLICY_ARGS} ]] || {
  echo "candidate sensitivity: missing selected policy or model args" >&2
  exit 1
}
POLICY=$(realpath -e "${POLICY}")
POLICY_ARGS=$(realpath -e "${POLICY_ARGS}")
POLICY_SHA256=$(sha256sum "${POLICY}" | awk '{print $1}')

die() {
  echo "candidate sensitivity: $*" >&2
  exit 1
}

run_arm() {
  local reference_mode=$1 longitudinal_mode=$2 lambda_lat=$3
  local lambda_label=${lambda_lat/./p}
  local name=${reference_mode}_${longitudinal_mode}_lat${lambda_label}
  local arm_out=${OUT}/${name}
  if [[ ! -s ${arm_out}/sampler_ablation_summary.json ]]; then
    if [[ -e ${arm_out} ]]; then
      local archive=${OUT}/interrupted_attempts stamp
      stamp=$(date '+%Y%m%d-%H%M%S')
      mkdir -p "${archive}"
      mv "${arm_out}" "${archive}/${name}_${stamp}"
      [[ ! -e ${OUT}/${name}.log ]] \
        || mv "${OUT}/${name}.log" "${archive}/${name}_${stamp}.log"
      echo "candidate sensitivity: archived interrupted ${name} arm; restarting" >&2
    fi
    ETA_SCHEME=stratified_beta \
      POLICY="${POLICY}" \
      POLICY_ARGS="${POLICY_ARGS}" \
      REFERENCE_MODE="${reference_mode}" \
      REFERENCE_NOISE_SCALE=0.5 \
      LONGITUDINAL_MODE="${longitudinal_mode}" \
      LAMBDA_LAT="${lambda_lat}" \
      SAMPLE_STEPS="${SAMPLE_STEPS}" \
      OUT="${arm_out}" \
      bash "${RUNNER}" > "${OUT}/${name}.log" 2>&1
  fi
  jq -e --arg ref "${reference_mode}" --arg lon "${longitudinal_mode}" \
    --argjson lambda_lat "${lambda_lat}" --argjson sample_steps "${SAMPLE_STEPS}" \
    --arg policy "${POLICY}" '
      .numerical_equivalence.passed == true and
      .protocol.reference_mode == $ref and
      .protocol.longitudinal_mode == $lon and
      .protocol.lambda_lat_m == $lambda_lat and
      .protocol.sample_steps == $sample_steps and
      .protocol.policy == $policy and
      .protocol.eta_sampling_scheme == "stratified_beta" and
      .overall.scene_count == 384
    ' "${arm_out}/sampler_ablation_summary.json" >/dev/null \
    || die "arm contract differs: ${name}"
}

main() {
  if [[ -s ${RESULT} ]]; then
    jq -e --arg policy "${POLICY}" --arg sha "${POLICY_SHA256}" \
      --argjson diversity "${MIN_DIVERSITY_RATIO}" \
      --argjson hard_drop "${MAX_HARD_SAFE_COUNT_DROP}" \
      --argjson sample_steps "${SAMPLE_STEPS}" '
      (.selected_reference_mode == "zero_noise" or
       .selected_reference_mode == "stochastic") and
      (.selected_longitudinal_mode == "candidate_stretch" or
       .selected_longitudinal_mode == "reference_speed_push") and
      (.selected_lambda_lat_m == 1 or
       .selected_lambda_lat_m == 1.5 or
       .selected_lambda_lat_m == 2.5) and
      .policy_checkpoint == $policy and .policy_checkpoint_sha256 == $sha and
      .version == 3 and .selected_sample_steps == $sample_steps and
      .minimum_diversity_ratio_vs_baseline == $diversity and
      .maximum_hard_safe_count_drop_vs_baseline == $hard_drop
    ' "${RESULT}" >/dev/null || die "existing result is malformed"
    echo "${RESULT}"
    return 0
  fi
  [[ -s ${RUNNER} ]] || die "missing sampler runner ${RUNNER}"
  mkdir -p "${OUT}"
  local reference_mode longitudinal_mode lambda_lat
  for reference_mode in zero_noise stochastic; do
    for longitudinal_mode in candidate_stretch reference_speed_push; do
      for lambda_lat in 1.0 1.5 2.5; do
        run_arm "${reference_mode}" "${longitudinal_mode}" "${lambda_lat}"
      done
    done
  done

  "${PYTHON}" - "${OUT}" "${RESULT}" "${MIN_GAIN}" \
    "${MIN_DIVERSITY_RATIO}" "${MAX_HARD_SAFE_COUNT_DROP}" \
    "${POLICY}" "${POLICY_SHA256}" "${POLICY_ARGS}" \
    "${SAMPLE_STEPS}" <<'PY'
import datetime
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
result_path = pathlib.Path(sys.argv[2])
minimum_gain = float(sys.argv[3])
minimum_diversity_ratio = float(sys.argv[4])
maximum_hard_safe_count_drop = float(sys.argv[5])
policy = pathlib.Path(sys.argv[6])
policy_sha256 = sys.argv[7]
policy_args = pathlib.Path(sys.argv[8])
sample_steps = int(sys.argv[9])
if not 0.0 <= minimum_diversity_ratio <= 1.0:
    raise ValueError("minimum diversity ratio must be in [0, 1]")
if maximum_hard_safe_count_drop < 0.0:
    raise ValueError("maximum hard-safe count drop must be non-negative")
names = tuple(
    f"{reference_mode}_{longitudinal_mode}_lat{str(lambda_lat).replace('.', 'p')}"
    for reference_mode in ("zero_noise", "stochastic")
    for longitudinal_mode in ("candidate_stretch", "reference_speed_push")
    for lambda_lat in (1.0, 1.5, 2.5)
)
arms = {}
for name in names:
    summary_path = root / name / "sampler_ablation_summary.json"
    payload = json.loads(summary_path.read_text())
    rows = payload["rows"]
    gains = [
        max(0.0, float(row["comparisons"]["all_guided"]["best_reward_gain"]))
        for row in rows
    ]
    all_guided = payload["overall"]["arms"]["all_guided"]
    protocol = payload["protocol"]
    if int(protocol["sample_steps"]) != sample_steps:
        raise RuntimeError(
            f"{name}: sample steps {protocol['sample_steps']} != {sample_steps}"
        )
    arms[name] = {
        "summary": str(summary_path.resolve()),
        "reference_mode": protocol["reference_mode"],
        "reference_noise_scale": float(protocol["reference_noise_scale"]),
        "longitudinal_mode": protocol["longitudinal_mode"],
        "lambda_lat_m": float(protocol["lambda_lat_m"]),
        # This is the formal top-union discovery quantity: a worse guided bank
        # cannot lower it because the native K remains available.
        "mean_native_guided_union_best_gain": statistics.fmean(gains),
        "improved_union_scene_count": sum(value > 0.0 for value in gains),
        "recovered_native_all_zero_count": int(
            all_guided["recovered_native_all_zero_count"]
        ),
        "guided_mean_hard_safe_count_of_10": float(
            all_guided["mean_hard_safe_count_of_10"]
        ),
        "guided_endpoint_pairwise_mean_m": float(
            all_guided["mean_endpoint_pairwise_m"]
        ),
        "max_first_waypoint_delta_m": float(
            all_guided["max_first_waypoint_delta_from_paired_native_m"]
        ),
        "numerical_equivalence_error_m": float(
            payload["numerical_equivalence"][
                "max_zero_strength_native_xy_error_m"
            ]
        ),
    }

baseline_name = "zero_noise_candidate_stretch_lat1p0"
baseline = arms[baseline_name]
eligible = []
for name, arm in arms.items():
    gain_over_baseline = (
        arm["mean_native_guided_union_best_gain"]
        - baseline["mean_native_guided_union_best_gain"]
    )
    arm["union_gain_delta_vs_cycle1"] = gain_over_baseline
    # Preserve the Original-DP current-state boundary and do not trade away
    # more than one of the finite all-zero recoveries for a tiny mean gain.
    # Reward is primary, but an exploration-policy change is not allowed to
    # discard most of the multi-modal support it is intended to provide or to
    # materially reduce the number of hard-safe candidates. PlannerRFT's own
    # uniform-policy ablation shows why diversity is a guard, not the maximand.
    if (
        arm["max_first_waypoint_delta_m"] <= 0.30
        and arm["recovered_native_all_zero_count"]
        >= baseline["recovered_native_all_zero_count"] - 1
        and arm["guided_endpoint_pairwise_mean_m"]
        >= minimum_diversity_ratio
        * baseline["guided_endpoint_pairwise_mean_m"]
        and arm["guided_mean_hard_safe_count_of_10"]
        >= baseline["guided_mean_hard_safe_count_of_10"]
        - maximum_hard_safe_count_drop
    ):
        eligible.append((arm["mean_native_guided_union_best_gain"], name))

if not eligible:
    raise RuntimeError("baseline candidate arm failed its own adoption guards")
best_gain, best_name = max(eligible)
if best_name != baseline_name and best_gain >= (
    baseline["mean_native_guided_union_best_gain"] + minimum_gain
):
    selected_name = best_name
    reason = "paired_union_gain_improves_beyond_minimum_and_clears_all_guards"
else:
    selected_name = baseline_name
    reason = "no_alternative_clears_conservative_paired_adoption_threshold"
selected = arms[selected_name]

result = {
    "version": 3,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "contract": "paired_384_scene_candidate_discovery_only",
    "policy_checkpoint": str(policy.resolve()),
    "policy_checkpoint_sha256": policy_sha256,
    "policy_args": str(policy_args.resolve()),
    "selected_sample_steps": sample_steps,
    "baseline": baseline_name,
    "minimum_adoption_gain": minimum_gain,
    "minimum_diversity_ratio_vs_baseline": minimum_diversity_ratio,
    "maximum_hard_safe_count_drop_vs_baseline": maximum_hard_safe_count_drop,
    "selected_arm": selected_name,
    "selected_reference_mode": selected["reference_mode"],
    "selected_reference_noise_scale": selected["reference_noise_scale"],
    "selected_longitudinal_mode": selected["longitudinal_mode"],
    "selected_lambda_lat_m": selected["lambda_lat_m"],
    "selection_reason": reason,
    "arms": arms,
    "note": (
        "Selection controls only future guided candidate generation; native K, "
        "reward ranking, deployment inference and the Cycle-1 cache are unchanged."
    ),
}
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(result_path)
PY
}

main "$@"
