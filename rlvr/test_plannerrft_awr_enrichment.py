import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import betainc

from rlvr.awr import (
    _sample_fixed_beta_eta_batch,
    _select_topk_native_guided_union,
)
from rlvr.train_awr import (
    _iter_prefetched_eval_batches,
    _load_scene_batch,
    _move_prefetched_eval_data,
)


def test_formal_replay_keeps_training_continuous_while_selecting_best_separately():
    root = Path(__file__).resolve().parent
    config = json.loads(
        (root / "configs/awr_original_dp_t4_plannerrft_anchored.json").read_text()
    )
    assert config["training"]["rollback_rejected_epoch"] is False

    scripts = (
        root / "autoresearch/continue_plannerrft_full_cycle01_after_mine.sh",
        root / "autoresearch/run_plannerrft_full_to_epoch100.sh",
    )
    for script in scripts:
        source = script.read_text()
        assert "--rollback_rejected_epoch" not in source
        assert "cumulative-replay" in source


def test_formal_candidate_modes_are_explicit_and_future_supervisor_is_evidence_selected():
    root = Path(__file__).resolve().parent
    config = json.loads(
        (root / "configs/awr_original_dp_t4_plannerrft_anchored.json").read_text()
    )
    awr = config["awr"]
    assert awr["plannerrft_reference_mode"] == "zero_noise"
    assert awr["plannerrft_reference_noise_scale"] == 0.5
    assert awr["plannerrft_longitudinal_mode"] == "candidate_stretch"
    assert awr["plannerrft_lambda_lat"] == 1.0

    supervisor = (
        root / "autoresearch/run_plannerrft_full_to_epoch100.sh"
    ).read_text()
    beta_sensitivity = (
        root / "autoresearch/run_plannerrft_replay_beta_sensitivity.sh"
    ).read_text()
    candidate_sensitivity = (
        root / "autoresearch/run_plannerrft_candidate_sensitivity.sh"
    ).read_text()
    solver_sensitivity = (
        root / "autoresearch/run_plannerrft_solver_step_sensitivity.sh"
    ).read_text()
    assert "run_plannerrft_replay_beta_sensitivity.sh" in supervisor
    assert "run_plannerrft_candidate_sensitivity.sh" in supervisor
    assert "run_plannerrft_solver_step_sensitivity.sh" in supervisor
    assert '--plannerrft_reference_mode "${PLANNERRFT_REFERENCE_MODE}"' in supervisor
    assert (
        '--plannerrft_longitudinal_mode "${PLANNERRFT_LONGITUDINAL_MODE}"'
        in supervisor
    )
    assert '--plannerrft_lambda_lat "${PLANNERRFT_LAMBDA_LAT}"' in supervisor
    assert "selected_lambda_lat_m" in supervisor
    assert "for lambda_lat in 1.0 1.5 2.5" in candidate_sensitivity
    assert '"version": 3' in candidate_sensitivity
    assert 'SAMPLE_STEPS="${ROLLOUT_SAMPLE_STEPS}"' in supervisor
    assert '--sample_steps "${ROLLOUT_SAMPLE_STEPS}"' in supervisor
    assert 'SAMPLE_STEPS="${SAMPLE_STEPS}"' in candidate_sensitivity
    assert '"selected_sample_steps": selected_steps' in solver_sensitivity
    assert "REPLAY_SCENE_LOAD_WORKERS=${REPLAY_SCENE_LOAD_WORKERS:-16}" in supervisor
    assert '--scene_load_workers "${REPLAY_SCENE_LOAD_WORKERS}"' in supervisor
    assert "overlay05=$(build_overlay 0.5)" in beta_sensitivity
    assert "run05=$(run_arm 0.5" in beta_sensitivity
    assert ".selected_beta == 0.5" in beta_sensitivity
    assert "REPLAY_BETA} == 0.5" in supervisor


def test_conditional_selector_accepts_false_incumbent_flag_under_fail_fast_shell():
    """A valid false JSON boolean must not be interpreted as a shell failure."""

    root = Path(__file__).resolve().parent
    source = (
        root / "autoresearch/run_conditional_train_selector_line_search.sh"
    ).read_text()
    assert (
        "incumbent_retained=$(jq -r '.best_train_is_starting_incumbent'"
        in source
    )
    assert (
        "incumbent_retained=$(jq -er '.best_train_is_starting_incumbent'"
        not in source
    )
    assert (
        "[[ ${incumbent_retained} == true || ${incumbent_retained} == false ]]"
        in source
    )


def test_campaign_target_and_process_recovery_are_automatic_and_bounded():
    root = Path(__file__).resolve().parent
    supervisor = (root / "autoresearch/run_plannerrft_full_to_epoch100.sh").read_text()
    prefix = (root / "autoresearch/run_plannerrft_prefix_to_epoch100.sh").read_text()
    monitor = (root / "autoresearch/monitor_plannerrft_epoch100.sh").read_text()
    autonomous = (
        root / "autoresearch/run_plannerrft_autonomous_campaign.sh"
    ).read_text()

    for source in (supervisor, prefix, monitor, autonomous):
        assert "TARGET_EPOCH=${TARGET_EPOCH:-100}" in source
    assert 'TOTAL_CYCLES=$((TARGET_EPOCH / 10))' in supervisor
    assert 'plannerrft_full_cycles02_to${TOTAL_CYCLES}_e${TARGET_EPOCH}' in supervisor
    assert 'for cycle in $(seq 2 "${TOTAL_CYCLES}")' in supervisor
    assert 'epoch${TARGET_EPOCH}_completion.json' in supervisor
    assert 'range(1, total_cycles + 1)' in supervisor
    assert 'TARGET_EPOCH="${TARGET_EPOCH}"' in prefix
    assert 'COMPLETION=${CAMPAIGN}/epoch${TARGET_EPOCH}_completion.json' in monitor
    assert 'MAX_STAGNANT_RESTARTS=${MAX_STAGNANT_RESTARTS:-3}' in autonomous
    assert 'committed_signature()' in autonomous
    assert 'supervisor_alive()' in autonomous
    assert 'repeated supervisor exits produced no new committed artifact' in autonomous


def test_eval_batch_benchmark_has_repeat_noise_and_speed_guards():
    root = Path(__file__).resolve().parent
    script = (
        root / "autoresearch/benchmark_eval_scene_batch_size.sh"
    ).read_text()
    assert "BATCHES=(${BATCHES:-1024 2048 4096})" in script
    assert 'REFERENCE_DIR=$(realpath -e "$5")' in script
    assert "3.0 * abs" in script
    assert "minimum_speedup_for_adoption" in script
    assert "equivalent_to_repeat_envelope" in script


def test_eval_prefetch_preserves_cpu_batch_values_and_padding(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"scene_{index}.npz"
        np.savez(
            path,
            ego_current_state=np.full((8,), index, dtype=np.float32),
            ego_shape=np.asarray([2.75, 4.34, 1.70], dtype=np.float32),
            arbitrary=np.arange(6, dtype=np.float32).reshape(2, 3) + index,
        )
        paths.append(str(path))

    prepared = list(
        _iter_prefetched_eval_batches(
            paths,
            batch_size=2,
            scene_load_workers=1,
            prefetch_batches=2,
        )
    )
    assert len(prepared) == 2
    assert prepared[0][1] == paths[:2]
    assert prepared[1][1] == paths[2:]
    assert prepared[1][2] == [paths[2], paths[2]]
    for _, _, model_paths, prefetched_data in prepared:
        direct = _load_scene_batch(
            model_paths, torch.device("cpu"), workers=1
        )
        moved = _move_prefetched_eval_data(
            prefetched_data, torch.device("cpu")
        )
        assert set(direct) == set(moved)
        for key in direct:
            if isinstance(direct[key], torch.Tensor):
                torch.testing.assert_close(
                    direct[key], moved[key], rtol=0.0, atol=0.0
                )


def test_stratified_fixed_beta_covers_every_cdf_bin_and_is_reproducible():
    concentration = 1.0 + np.log(2.0)
    np.random.seed(17)
    first_lat, first_lon = _sample_fixed_beta_eta_batch(
        3,
        10,
        concentration=concentration,
        scheme="stratified_beta",
        device=torch.device("cpu"),
    )
    np.random.seed(17)
    second_lat, second_lon = _sample_fixed_beta_eta_batch(
        3,
        10,
        concentration=concentration,
        scheme="stratified_beta",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first_lat, second_lat, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_lon, second_lon, rtol=0.0, atol=0.0)
    assert bool(((first_lat > -1.0) & (first_lat < 1.0)).all())
    assert bool(((first_lon > -1.0) & (first_lon < 1.0)).all())

    # Map eta back through the target Beta CDF. Every row must contain exactly
    # one sample in each of the ten equal-probability strata, independent of
    # the random lateral/longitudinal permutations.
    for tensor in (first_lat, first_lon):
        cdf = betainc(concentration, concentration, (tensor.numpy() + 1.0) / 2.0)
        bins = np.floor(np.minimum(cdf, np.nextafter(1.0, 0.0)) * 10).astype(int)
        for row in bins:
            assert sorted(row.tolist()) == list(range(10))


def test_topk_union_preserves_native_best_and_prefers_native_on_ties():
    native_trajectories = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    guided_trajectories = 100.0 + native_trajectories
    native_noise = torch.tensor([0.5, 0.5, 0.5])
    guided_noise = torch.tensor([0.6, 0.6, 0.6])
    native_rewards = [
        SimpleNamespace(total=0.1),
        SimpleNamespace(total=0.9),
        SimpleNamespace(total=0.8),
    ]
    guided_rewards = [
        SimpleNamespace(total=0.8),  # tie: native 0.8 must rank first
        SimpleNamespace(total=0.95),
        SimpleNamespace(total=0.2),
    ]

    trajectories, noise, rewards, diagnostics = _select_topk_native_guided_union(
        native_trajectories,
        native_noise,
        native_rewards,
        guided_trajectories,
        guided_noise,
        guided_rewards,
        keep=3,
    )

    assert [reward.total for reward in rewards] == [0.95, 0.9, 0.8]
    torch.testing.assert_close(trajectories[0], guided_trajectories[1])
    torch.testing.assert_close(trajectories[1], native_trajectories[1])
    torch.testing.assert_close(trajectories[2], native_trajectories[2])
    torch.testing.assert_close(noise, torch.tensor([0.6, 0.5, 0.5]))
    assert diagnostics["native_candidate_count_selected"] == 2.0
    assert diagnostics["guided_candidate_count_selected"] == 1.0
    assert diagnostics["enriched_best_reward_gain"] == pytest.approx(0.05)


def test_topk_union_can_preserve_zero_noise_behavior_anchor():
    native_trajectories = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(
        3, 2, 4
    )
    guided_trajectories = 100.0 + native_trajectories
    native_noise = torch.tensor([0.0, 0.5, 0.5])
    guided_noise = torch.tensor([0.5, 0.5, 0.5])
    native_rewards = [
        SimpleNamespace(total=0.2),
        SimpleNamespace(total=0.9),
        SimpleNamespace(total=0.8),
    ]
    guided_rewards = [
        SimpleNamespace(total=0.95),
        SimpleNamespace(total=0.85),
        SimpleNamespace(total=0.1),
    ]

    trajectories, noise, rewards, diagnostics = _select_topk_native_guided_union(
        native_trajectories,
        native_noise,
        native_rewards,
        guided_trajectories,
        guided_noise,
        guided_rewards,
        keep=3,
        preserve_native_index0=True,
    )

    assert [reward.total for reward in rewards] == [0.2, 0.95, 0.9]
    torch.testing.assert_close(trajectories[0], native_trajectories[0])
    torch.testing.assert_close(trajectories[1], guided_trajectories[0])
    torch.testing.assert_close(trajectories[2], native_trajectories[1])
    torch.testing.assert_close(noise, torch.tensor([0.0, 0.5, 0.5]))
    assert diagnostics["behavior_anchor_preserved"] == 1.0
    assert diagnostics["native_candidate_count_selected"] == 2.0
    assert diagnostics["guided_candidate_count_selected"] == 1.0
    assert diagnostics["enriched_best_reward_gain"] == pytest.approx(0.05)


def test_behavior_anchor_union_evicts_native_best_only_when_guided_slots_are_better():
    """Candidate 0 is the identity anchor; native-best is a reward floor, not a slot."""

    native_trajectories = torch.arange(
        3 * 2 * 4, dtype=torch.float32
    ).reshape(3, 2, 4)
    guided_trajectories = 100.0 + native_trajectories
    native_noise = torch.tensor([0.0, 0.5, 0.5])
    guided_noise = torch.tensor([0.5, 0.5, 0.5])
    native_rewards = [
        SimpleNamespace(total=0.2),
        SimpleNamespace(total=0.9),
        SimpleNamespace(total=0.8),
    ]
    guided_rewards = [
        SimpleNamespace(total=0.93),
        SimpleNamespace(total=0.92),
        SimpleNamespace(total=0.91),
    ]

    trajectories, _, rewards, diagnostics = _select_topk_native_guided_union(
        native_trajectories,
        native_noise,
        native_rewards,
        guided_trajectories,
        guided_noise,
        guided_rewards,
        keep=3,
        preserve_native_index0=True,
    )

    # Two non-anchor slots are available, so two guided trajectories that are
    # both better than native-best correctly replace every stochastic native
    # member. The deterministic deployment identity remains first.
    assert [reward.total for reward in rewards] == [0.2, 0.93, 0.92]
    torch.testing.assert_close(trajectories[0], native_trajectories[0])
    torch.testing.assert_close(trajectories[1], guided_trajectories[0])
    torch.testing.assert_close(trajectories[2], guided_trajectories[1])
    assert diagnostics["behavior_anchor_preserved"] == 1.0
    assert diagnostics["native_candidate_count_selected"] == 1.0
    assert diagnostics["enriched_best_reward_gain"] == pytest.approx(0.03)
