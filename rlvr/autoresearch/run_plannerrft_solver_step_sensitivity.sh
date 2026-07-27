#!/usr/bin/env bash
# Paired 5-vs-10-step PlannerRFT candidate-discovery audit and conservative
# selection for future refreshes. Both arms reuse the same 384 scene IDs,
# policy/reference checkpoints, latent
# seeds, Beta commands, reward and K=10 budget.  This is read-only inference.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
RUNNER=${ROOT}/rlvr/autoresearch/tools/run_plannerrft_sampler_ablation.sh
PYTHON=${ROOT}/.venv/bin/python
OUT=${OUT:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_solver_step_sensitivity}
RESULT=${OUT}/paired_5_vs_10.json
POLICY=${POLICY:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/best_model_awr_retained_e004_a0p05.pth}
POLICY_ARGS=${POLICY_ARGS:-${ROOT}/outputs/awr_t4_full_sequence_filtered/20260719-223552_cycle02_hdp_epoch_commit_e12_matched_cache/model_args.json}
MINE_RUN=${MINE_RUN:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine}
SCENES=${SCENES:-${ROOT}/outputs/awr_t4_full_sequence_filtered/plannerrft_guidance_pilot/valid_stratified_scenes_384.json}
MIN_UNION_GAIN=${MIN_UNION_GAIN:-0.002}
MIN_DIVERSITY_RATIO=${MIN_DIVERSITY_RATIO:-0.80}
MAX_HARD_SAFE_COUNT_DROP=${MAX_HARD_SAFE_COUNT_DROP:-0.25}
POLICY=$(realpath -e "${POLICY}")
POLICY_ARGS=$(realpath -e "${POLICY_ARGS}")
POLICY_SHA256=$(sha256sum "${POLICY}" | awk '{print $1}')

die() {
  echo "solver-step sensitivity: $*" >&2
  exit 1
}

run_arm() {
  local steps=$1
  local arm=${OUT}/steps${steps}
  if [[ ! -s ${arm}/sampler_ablation_summary.json ]]; then
    if [[ -e ${arm} ]]; then
      local archive=${OUT}/interrupted_attempts stamp
      stamp=$(date '+%Y%m%d-%H%M%S')
      mkdir -p "${archive}"
      mv "${arm}" "${archive}/steps${steps}_${stamp}"
      [[ ! -e ${OUT}/steps${steps}.log ]] \
        || mv "${OUT}/steps${steps}.log" "${archive}/steps${steps}_${stamp}.log"
      echo "solver-step sensitivity: archived interrupted steps=${steps} arm; restarting" >&2
    fi
    SAMPLE_STEPS=${steps} ARMS=all_guided ETA_SCHEME=stratified_beta \
      REFERENCE_MODE=zero_noise LONGITUDINAL_MODE=candidate_stretch \
      LAMBDA_LAT=1.0 POLICY="${POLICY}" POLICY_ARGS="${POLICY_ARGS}" \
      SCENES="${SCENES}" OUT="${arm}" bash "${RUNNER}" \
      > "${OUT}/steps${steps}.log" 2>&1
  fi
}

main() {
  [[ -s ${RUNNER} && -s ${POLICY} && -s ${POLICY_ARGS} ]] \
    || die "missing runner, policy or model args"
  if [[ -s ${RESULT} ]]; then
    jq -e --arg policy "${POLICY}" --arg sha "${POLICY_SHA256}" \
      --argjson minimum_gain "${MIN_UNION_GAIN}" \
      --argjson diversity_ratio "${MIN_DIVERSITY_RATIO}" \
      --argjson hard_safe_drop "${MAX_HARD_SAFE_COUNT_DROP}" '
      .version == 2 and
      .protocol == "paired_same_scene_latent_eta_reward" and
      .policy == $policy and .policy_sha256 == $sha and
      .minimum_union_gain_for_step10 == $minimum_gain and
      .minimum_diversity_ratio_vs_step5 == $diversity_ratio and
      .maximum_hard_safe_count_drop_vs_step5 == $hard_safe_drop and
      (.selected_sample_steps == 5 or .selected_sample_steps == 10) and
      .arms["5"].sample_steps == 5 and .arms["10"].sample_steps == 10 and
      .arms["5"].scene_count == 384 and .arms["10"].scene_count == 384' \
      "${RESULT}" >/dev/null || die "existing result is malformed"
    echo "${RESULT}"
    return 0
  fi
  if pgrep -f 'rlvr.train_awr|eval_awr_full_distributed|diag_plannerrft_sampler_ablation' >/dev/null; then
    die "refusing to compete with active GPU training/evaluation"
  fi
  mkdir -p "${OUT}"
  [[ -s ${SCENES} ]] || die "missing fixed 384-scene sensitivity list ${SCENES}"
  run_arm 5
  run_arm 10
  "${PYTHON}" - "${OUT}" "${RESULT}" "${POLICY}" "${POLICY_ARGS}" \
    "${POLICY_SHA256}" "${MIN_UNION_GAIN}" "${MIN_DIVERSITY_RATIO}" \
    "${MAX_HARD_SAFE_COUNT_DROP}" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
result = pathlib.Path(sys.argv[2])
policy = pathlib.Path(sys.argv[3])
args = pathlib.Path(sys.argv[4])
policy_sha256 = sys.argv[5]
minimum_union_gain = float(sys.argv[6])
minimum_diversity_ratio = float(sys.argv[7])
maximum_hard_safe_count_drop = float(sys.argv[8])
if minimum_union_gain < 0.0:
    raise ValueError("minimum union gain must be non-negative")
if not 0.0 <= minimum_diversity_ratio <= 1.0:
    raise ValueError("minimum diversity ratio must be in [0, 1]")
if maximum_hard_safe_count_drop < 0.0:
    raise ValueError("maximum hard-safe count drop must be non-negative")

payloads = {
    steps: json.loads(
        (root / f"steps{steps}" / "sampler_ablation_summary.json").read_text()
    )
    for steps in (5, 10)
}


def summarize(payload):
    rows = payload["rows"]
    union_gains = [
        max(0.0, float(row["comparisons"]["all_guided"]["best_reward_gain"]))
        for row in rows
    ]
    arm = payload["overall"]["arms"]["all_guided"]
    return {
        "sample_steps": int(payload["protocol"]["sample_steps"]),
        "scene_count": len(rows),
        "mean_native_guided_union_best_gain": statistics.fmean(union_gains),
        "improved_union_scene_count": sum(value > 0.0 for value in union_gains),
        "recovered_native_all_zero_count": int(
            arm["recovered_native_all_zero_count"]
        ),
        "guided_mean_hard_safe_count_of_10": float(
            arm["mean_hard_safe_count_of_10"]
        ),
        "guided_endpoint_pairwise_mean_m": float(
            arm["mean_endpoint_pairwise_m"]
        ),
        "source_summary": str(
            (root / f"steps{payload['protocol']['sample_steps']}" /
             "sampler_ablation_summary.json").resolve()
        ),
    }


rows_by_steps = {
    steps: {row["scene_path"]: row for row in payload["rows"]}
    for steps, payload in payloads.items()
}
assert set(rows_by_steps[5]) == set(rows_by_steps[10])
paired = []
for path in rows_by_steps[5]:
    record = {"scene_path": path}
    for steps in (5, 10):
        row = rows_by_steps[steps][path]
        native = float(row["arm_stats"]["native"]["best_reward"])
        guided = float(row["arm_stats"]["all_guided"]["best_reward"])
        record[str(steps)] = {
            "native_best_reward": native,
            "guided_best_reward": guided,
            "union_gain": max(0.0, guided - native),
        }
    paired.append(record)

arms = {str(steps): summarize(payloads[steps]) for steps in (5, 10)}
step5 = arms["5"]
step10 = arms["10"]
step10_clears_guards = (
    step10["mean_native_guided_union_best_gain"]
    >= step5["mean_native_guided_union_best_gain"] + minimum_union_gain
    and step10["recovered_native_all_zero_count"]
    >= step5["recovered_native_all_zero_count"] - 1
    and step10["guided_mean_hard_safe_count_of_10"]
    >= step5["guided_mean_hard_safe_count_of_10"]
    - maximum_hard_safe_count_drop
    and step10["guided_endpoint_pairwise_mean_m"]
    >= minimum_diversity_ratio
    * step5["guided_endpoint_pairwise_mean_m"]
)
selected_steps = 10 if step10_clears_guards else 5
selection_reason = (
    "step10_improves_union_reward_and_clears_candidate_guards"
    if step10_clears_guards
    else "step10_does_not_justify_its_extra_denoising_cost"
)

result.write_text(
    json.dumps(
        {
            "version": 2,
            "protocol": "paired_same_scene_latent_eta_reward",
            "policy": str(policy.resolve()),
            "policy_sha256": policy_sha256,
            "model_args": str(args.resolve()),
            "scene_list": str(pathlib.Path(payloads[5]["protocol"]["scenes"]).resolve())
            if payloads[5]["protocol"].get("scenes")
            else None,
            "minimum_union_gain_for_step10": minimum_union_gain,
            "minimum_diversity_ratio_vs_step5": minimum_diversity_ratio,
            "maximum_hard_safe_count_drop_vs_step5": maximum_hard_safe_count_drop,
            "selected_sample_steps": selected_steps,
            "selection_reason": selection_reason,
            "arms": arms,
            "paired_scene_rows": paired,
            "decision_role": (
                "selects only future training-time candidate generation; checkpoint "
                "selection and deployment remain ten-step deterministic train reward"
            ),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(result)
PY
  echo "${RESULT}"
}

main "$@"
