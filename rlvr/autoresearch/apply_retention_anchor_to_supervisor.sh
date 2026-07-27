#!/usr/bin/env bash
# Apply the behaviour-retention-anchor change to the cycles-2..10 supervisor.
#
# Why this is a script rather than an inline edit: bash reads a script
# incrementally as it executes, so editing run_plannerrft_full_to_epoch100.sh
# while the supervisor is mid-run can make it execute garbage.  The supervisor
# runs for days.  This script therefore refuses to touch the file unless the
# supervisor is stopped, applies the edit, verifies it, and leaves relaunching to
# launch_jitterfix_campaign.sh.
#
# What it changes and why
# ----------------------
# The overlay was built with --behavior-anchor-weight 0.  Combined with
# EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=1 that meant only the 18.3% of scenes holding
# a positive-advantage candidate contributed anything to the loss; the policy
# drifted unconstrained on the other 81.7%, which is where essentially all of the
# 0.15% collision events live.  Cycle-1 replay showed the consequence: every
# safety class degraded (collision +10.3%, kinematic +17.8%, rb +0.4%,
# lane +0.5%) while the aggregate reward rose +0.0006.
#
# Measured on the cycle-1 cache: any positive anchor weight lifts
# loss-constrained scenes from 19.2% to 98.3% and leaves the 33,880 improvement
# targets untouched.  0.25 puts the anchor's total weight at 27% of the
# improvement targets'.

set -Eeuo pipefail

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
SUPERVISOR=${ROOT}/rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh
PYTHON=${ROOT}/.venv/bin/python

if pgrep -f "run_plannerrft_full_to_epoch10[0]" > /dev/null; then
  echo "refusing to edit a running supervisor; stop it first (stop window only)" >&2
  exit 2
fi
[[ -f ${SUPERVISOR} ]] || { echo "missing ${SUPERVISOR}" >&2; exit 2; }

ANCHOR=$(cd "${ROOT}" && PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}" "${PYTHON}" -c \
  'from rlvr.campaign_contract import BEHAVIOR_ANCHOR_WEIGHT as w; print(f"{w:g}")')

cp "${SUPERVISOR}" "${SUPERVISOR}.bak.$(date '+%Y%m%d-%H%M%S')"

cd "${ROOT}" && PYTHONPATH="${ROOT}/diffusion_planner:${ROOT}" "${PYTHON}" - "${SUPERVISOR}" "${ANCHOR}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
anchor = sys.argv[2]
text = path.read_text()

edits = [
    # overlay build invocation
    (
        "      --behavior-anchor-weight 0 --unsafe-behavior-anchor-weight 0 \\",
        f"      --behavior-anchor-weight {anchor} "
        f"--unsafe-behavior-anchor-weight {anchor} \\",
    ),
    # overlay manifest contract
    (
        "    .parameters.behavior_anchor_weight == 0 and\n"
        "    .parameters.unsafe_behavior_anchor_weight == 0 and",
        f"    .parameters.behavior_anchor_weight == {anchor} and\n"
        f"    .parameters.unsafe_behavior_anchor_weight == {anchor} and",
    ),
    # replay runtime contract
    (
        "    .awr.behavior_anchor_weight == 0 and\n"
        "    .awr.unsafe_behavior_anchor_weight == 0 and",
        f"    .awr.behavior_anchor_weight == {anchor} and\n"
        f"    .awr.unsafe_behavior_anchor_weight == {anchor} and",
    ),
]
for old, new in edits:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one occurrence of:\n{old}\ngot {text.count(old)}")
    text = text.replace(old, new)
path.write_text(text)
print(f"applied retention anchor {anchor} at {len(edits)} sites")
PY

bash -n "${SUPERVISOR}" || { echo "syntax check failed" >&2; exit 1; }
echo "supervisor updated; relaunch with rlvr/autoresearch/launch_jitterfix_campaign.sh"
