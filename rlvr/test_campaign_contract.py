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
    text = SUPERVISOR.read_text()
    assert f"REPLAY_BETA=${{AWR_EXPECTED_REPLAY_BETA:-{C.REPLAY_BETA:g}}}" in text or (
        f"REPLAY_BETA={C.REPLAY_BETA:g}" in text
    ), "supervisor default beta disagrees with the contract"


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
