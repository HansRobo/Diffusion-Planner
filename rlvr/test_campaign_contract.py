"""The contract module must agree with what the pipeline actually runs.

These tests are the guard against config drift — the failure mode that caused
three campaign restarts on 2026-07-26 (beta declared in three places with two
different values, an alpha ladder missing its only audited-good step, corpus
constants duplicated as literals).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rlvr import campaign_contract as C

ROOT = Path(__file__).resolve().parents[1]
AWR_CONFIG = ROOT / "rlvr/configs/awr_original_dp_t4_plannerrft_clean_sft.json"
LINE_SEARCH = ROOT / "rlvr/autoresearch/run_conditional_train_selector_line_search.sh"
SUPERVISOR = ROOT / "rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh"


def test_awr_config_matches_the_contract():
    awr = json.loads(AWR_CONFIG.read_text())["awr"]
    assert awr["beta"] == C.REPLAY_BETA
    assert (
        awr["original_dp_first_waypoint_gate_max_implied_steer_rad"]
        == C.FIRST_WAYPOINT_MAX_IMPLIED_STEER_RAD
    )
    assert (
        awr["original_dp_first_waypoint_gate_tangent_min_step_m"]
        == C.FIRST_WAYPOINT_TANGENT_MIN_STEP_M
    )


def test_alpha_ladder_contains_the_only_audited_good_step():
    """0.05 is the step the 2026-07-18 audit showed gains on; never drop it."""
    assert 0.05 in C.COMMIT_ALPHA_LADDER
    assert C.COMMIT_ALPHA_LADDER == tuple(sorted(C.COMMIT_ALPHA_LADDER))
    assert max(C.COMMIT_ALPHA_LADDER) < 1.0, "the unscaled policy is not a rung"


def test_line_search_script_uses_the_contract_ladder():
    text = LINE_SEARCH.read_text()
    for alpha in C.COMMIT_ALPHA_LADDER:
        label = f"a{alpha:g}".replace(".", "p")
        assert label in text, f"{label} missing from the line-search ladder"
    assert C.COMMIT_STEP_PREFERENCE in text


def test_unscaled_commit_is_opt_in_only():
    assert C.ALLOW_UNSCALED_COMMIT is False
    assert "ALLOW_UNSCALED_COMMIT" in LINE_SEARCH.read_text()


def test_supervisor_replay_beta_matches_the_contract():
    """Compare the resolved value, not the literal: 1.0 and 1 are the same beta
    but never the same string, and a text match fails on formatting alone."""
    line = next(
        ln for ln in SUPERVISOR.read_text().splitlines()
        if ln.startswith("REPLAY_BETA=")
    )
    resolved = subprocess.run(
        ["bash", "-c", f'{line}; printf %s "$REPLAY_BETA"'],
        capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin"},
    ).stdout
    assert float(resolved) == C.REPLAY_BETA, (
        f"supervisor default beta {resolved} disagrees with contract {C.REPLAY_BETA}"
    )


@pytest.mark.parametrize("script", ["run_plannerrft_full_to_epoch100.sh",
                                    "run_plannerrft_jitterfix_to_epoch100.sh"])
def test_right_turn_emphasis_matches_the_contract(script):
    """This contract said REPEAT=10 while both entrypoints defaulted to 1.

    Nothing read the value, so the drift was invisible.  What has to hold is the
    product: REPEAT * WEIGHT is the total gradient mass on the right-turn lists,
    and the user's standing requirement is 10x.
    """

    text = (SUPERVISOR.parent / script).read_text()
    found = {}
    for name in ("EXTRA_TRAIN_REPEAT", "EXTRA_TRAIN_WEIGHT"):
        line = next(
            ln for ln in text.splitlines() if ln.startswith(f"{name}=${{{name}:-")
        )
        found[name] = int(line.split(":-", 1)[1].split("}", 1)[0])
    assert found["EXTRA_TRAIN_REPEAT"] == C.EXTRA_TRAIN_REPEAT
    assert found["EXTRA_TRAIN_WEIGHT"] == C.EXTRA_TRAIN_WEIGHT
    assert found["EXTRA_TRAIN_REPEAT"] * found["EXTRA_TRAIN_WEIGHT"] == 10, (
        "the three right-turn lists must keep their 10x emphasis"
    )
    assert C.EXTRA_TRAIN_REPEAT * C.EXTRA_TRAIN_WEIGHT == 10


def test_supervisor_expert_improves_gate_resolves_without_unbound_variables():
    """The supervisor cannot be dry-run, and it runs under ``set -u``.

    A first attempt at this default referenced POSITIVE_ADVANTAGE_MARGIN, which
    this script never defines because it does not eval the contract module.
    ``bash -n`` accepts that happily and the campaign would have died on an
    unbound variable at the cycle-1 handover, hours in.  Resolve the real line
    under ``set -u`` instead, and pin it to the contract that documents it.
    """

    line = next(
        ln for ln in SUPERVISOR.read_text().splitlines()
        if ln.startswith("EXPERT_IMPROVES_MARGIN=")
    )
    resolved = subprocess.run(
        ["bash", "-uc", f'{line}; printf %s "$EXPERT_IMPROVES_MARGIN"'],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    assert resolved.returncode == 0, f"unbound variable in {line!r}: {resolved.stderr}"
    if C.EXPERT_IMPROVES_MARGIN is None:
        assert resolved.stdout == "", "contract disables the gate, supervisor enables it"
    else:
        assert float(resolved.stdout) == C.EXPERT_IMPROVES_MARGIN, (
            f"supervisor margin {resolved.stdout!r} disagrees with contract "
            f"{C.EXPERT_IMPROVES_MARGIN}"
        )


def test_expert_improves_gate_implies_an_unrestricted_expert_anchor():
    """The gate moves the decision into the overlay, so the trainer must not
    also restrict the anchor to AWR-active groups -- that would re-kill exactly
    the dead groups the gate exists to reach."""

    text = SUPERVISOR.read_text()
    assert "--no-expert_anchor_active_groups_only" in text
    gate = text.index("EXPERT_IMPROVES_MARGIN=${EXPERT_IMPROVES_MARGIN")
    override = text.index("EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=0")
    inherited = text.index("EXPERT_ANCHOR_ACTIVE_GROUPS_ONLY=${")
    assert inherited < gate < override, (
        "the gate must override the inherited anchor scope, not be overridden by it"
    )


@pytest.mark.parametrize(
    ("scenes", "expected_groups", "expected_per_rank"),
    [
        (5_446_154, 5_446_656, 680_832),          # un-augmented corpus
        (6_130_664, 6_131_712, 766_464),          # with right-turn oversampling
        (C.GLOBAL_ROLLOUT_BATCH, C.GLOBAL_ROLLOUT_BATCH, C.ROLLOUT_SCENE_BATCH_SIZE),
    ],
)
def test_corpus_padding_matches_the_observed_cache_shapes(
    scenes, expected_groups, expected_per_rank
):
    assert C.padded_group_count(scenes) == expected_groups
    assert C.per_rank_group_count(scenes) == expected_per_rank


def test_padding_never_truncates():
    for scenes in (1, 1535, 1536, 1537, 5_557_744):
        assert C.padded_group_count(scenes) >= scenes


def test_rejects_nonpositive_corpus():
    with pytest.raises(ValueError):
        C.padded_group_count(0)


def test_shell_output_is_evaluable_and_round_trips():
    emitted = C.shell_assignments()
    probe = (
        f"{emitted}\n"
        'printf "%s|%s|%s|%s\\n" '
        '"$REPLAY_BETA" "$EXTRA_TRAIN_REPEAT" '
        '"${COMMIT_ALPHAS[1]}" "${COMMIT_ALPHA_LABELS[1]}"'
    )
    out = subprocess.run(
        ["bash", "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    beta, repeat, alpha1, label1 = out.split("|")
    assert float(beta) == C.REPLAY_BETA
    assert int(repeat) == C.EXTRA_TRAIN_REPEAT
    assert float(alpha1) == C.COMMIT_ALPHA_LADDER[1]
    assert label1 == "a0p05"


def test_module_cli_runs():
    out = subprocess.run(
        [sys.executable, "-m", "rlvr.campaign_contract", "--shell"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout
    assert "REPLAY_BETA=" in out
    assert "XX1_LEGACY_EGO_WIDTH_M=1.7" in out


def test_retention_anchor_defaults_to_the_validated_runs_value():
    """The only run that produced a cycle-level gain used anchor=0.  A positive
    value is an experiment against that baseline, never part of it -- the
    "anchor=0 causes safety degradation" claim was refuted by that run's own
    cycle-5 epochs, which degrade identically at anchor=0."""
    assert C.BEHAVIOR_ANCHOR_WEIGHT == 0.0
    # Raising it is allowed, but 1.0 puts the anchor's total weight at 1.08x the
    # improvement targets' and drowns the signal.
    assert C.BEHAVIOR_ANCHOR_WEIGHT <= C.BEHAVIOR_ANCHOR_WEIGHT_MAX <= 0.5


def test_entrypoint_uses_the_contract_retention_anchor():
    """The anchor must come from the contract, never a literal."""
    text = (ROOT / "rlvr/autoresearch/run_plannerrft_jitterfix_to_epoch100.sh").read_text()
    assert "--behavior-anchor-weight \"${BEHAVIOR_ANCHOR_WEIGHT}\"" in text
    assert "--behavior-anchor-weight 0" not in text


def test_live_chain_never_hardcodes_neighbor_future_offset_zero():
    """Offset 0 says "future[0] is already t+1".  Measured on the 20260707 corpus
    (200 scenes, 15,997 agents): median ||present - future[0]|| == 0.0000 while
    ||present - future[1]|| == 0.1368 ~= one 0.1 s step.  future[0] IS the present
    state, so offset must be 1 or every neighbour is one frame stale -- in the
    model target *and* in the reward's OBB collision replay that AWR ranks
    candidates with.  0 is only correct once a converter writes pre-aligned files.
    """
    offenders = []
    for name in ("run_plannerrft_jitterfix_to_epoch100.sh",
                 "run_plannerrft_full_to_epoch100.sh"):
        for line in (ROOT / "rlvr/autoresearch" / name).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "DP_NEIGHBOR_FUTURE_OFFSET=0" in stripped.replace('"', "").replace("'", ""):
                offenders.append(f"{name}: {stripped[:70]}")
            if "DP_NEIGHBOR_FUTURE_OFFSET:-0}" in stripped:
                offenders.append(f"{name}: {stripped[:70]}")
    assert not offenders, "neighbour-future offset 0 reintroduced:\n" + "\n".join(offenders)


def test_no_corpus_size_constants_remain_in_the_live_chain():
    """Constants compared against already-produced artifacts caused >100 hard
    failures: a corpus change retroactively condemned valid caches."""
    chain = [
        "run_plannerrft_jitterfix_to_epoch100.sh",
        "run_plannerrft_full_to_epoch100.sh",
        "run_conditional_train_selector_line_search.sh",
        "run_plannerrft_replay_beta_sensitivity.sh",
        "run_plannerrft_solver_step_sensitivity.sh",
        "run_plannerrft_candidate_sensitivity.sh",
    ]
    banned = ("5446154", "5446656", "680832", "766464", "6131712", "46262")
    offenders = []
    for name in chain:
        text = (ROOT / "rlvr/autoresearch" / name).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for token in banned:
                if token in stripped:
                    offenders.append(f"{name}: {stripped[:70]}")
    assert not offenders, "corpus-size constants reintroduced:\n" + "\n".join(offenders)


def test_no_expected_hash_or_scene_count_pins_remain():
    chain = ROOT / "rlvr/autoresearch"
    banned = ("EXPECTED_SELECTOR_SHA256", "EXPECTED_VALID_SHA256",
              "EXPECTED_MODEL_SHA256", "EXPECTED_VALID_SCENES",
              "EXPECTED_SELECTOR_SCENES", "EXPECTED_PADDED_GROUPS",
              "EXPECTED_RANK_GROUPS")
    offenders = []
    for name in ("run_plannerrft_full_to_epoch100.sh",
                 "run_conditional_train_selector_line_search.sh",
                 "run_plannerrft_replay_beta_sensitivity.sh"):
        text = (chain / name).read_text()
        for token in banned:
            if token in text:
                offenders.append(f"{name}: {token}")
    assert not offenders, "pinned hash/count reintroduced: " + ", ".join(offenders)


def test_supervisor_reuses_a_mine_cache_when_the_policy_is_unchanged():
    """A vetoed cycle keeps its incumbent, so the next cycle's policy is
    byte-identical to the one that already produced a cache. Re-rolling it costs
    ~10 GPU-hours for a statistically identical sample."""
    text = (ROOT / "rlvr/autoresearch/run_plannerrft_full_to_epoch100.sh").read_text()
    assert "reusable_mine_run_for_checkpoint" in text
    # Matching must be on content hash, not path, so a copied checkpoint matches
    # and a different policy never does.
    assert "staged_model_sha256" in text
    reuse = text.index("reusable_mine_run_for_checkpoint()")
    loop = text.index("mine_run=$(reusable_mine_run_for_checkpoint")
    assert reuse < loop, "helper must be defined before the cycle loop uses it"
    assert "mine_cache_shape_complete" in text[reuse:reuse + 1200], (
        "a reused cache must still pass the completeness check"
    )
