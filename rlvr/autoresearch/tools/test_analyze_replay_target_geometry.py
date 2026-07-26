import json

import numpy as np

from rlvr.autoresearch.tools.analyze_replay_target_geometry import analyze


def test_target_geometry_exact_positive_and_expert_mix(tmp_path):
    run = tmp_path / "run"
    replay = run / "replay_buffer"
    rank = replay / "rank_0000"
    rank.mkdir(parents=True)
    (run / "effective_config.json").write_text(
        json.dumps({"awr": {"deterministic_first": True}})
    )
    (rank / "scene_paths.jsonl").write_text('"a.npz"\n"b.npz"\n')

    trajectories = np.zeros((2, 3, 2, 4), dtype=np.float32)
    trajectories[..., 2] = 1.0
    trajectories[:, 0, :, 0] = np.asarray([1.0, 2.0])
    trajectories[0, 1, :, 0] = np.asarray([2.0, 4.0])
    trajectories[0, 2, :, 0] = np.asarray([0.5, 1.0])
    trajectories[1, 1:, :, 0] = trajectories[1, :1, :, 0]
    rewards = np.asarray([[0.5, 0.7, 0.4], [0.8, 0.8, 0.7]], dtype=np.float32)
    expert = np.zeros((2, 2, 4), dtype=np.float32)
    expert[..., 2] = 1.0
    expert[0, :, 0] = np.asarray([1.5, 3.0])
    expert[1, :, 0] = np.asarray([2.0, 4.0])
    np.save(rank / "trajectories.npy", trajectories)
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "weights.npy", np.ones((2, 3), dtype=np.float32))
    np.save(rank / "expert_trajectories.npy", expert)
    np.save(rank / "expert_safe.npy", np.ones(2, dtype=np.float32))

    summary = analyze(
        replay,
        allow_partial_prefix=True,
        recompute_positive_weights=True,
        beta=2.0,
        margin=0.01,
        behavior_anchor_weight=0.0,
        unsafe_behavior_anchor_weight=0.0,
        expert_weight=0.4,
        chunk_size=1,
    )

    assert summary["source_complete"] is False
    assert summary["prefix_groups"] == 2
    assert summary["active_group_fraction"] == 0.5
    assert summary["mean_active_candidate_count"] == 1.0
    assert np.isclose(summary["global_expert_weight_share"], 0.8 / 1.8)
    assert np.isclose(summary["active_candidate_target_reward_gain"], 0.2)
    assert np.isclose(summary["active_candidate_path_length_delta_m"], 2.0)
    assert np.isclose(summary["active_candidate_endpoint_x_delta_m"], 2.0)
    modes = summary["active_target_mode_distribution"]
    assert modes["faster_only"]["groups"] == 1
    assert modes["faster_only"]["fraction_of_active_groups"] == 1.0
    assert np.isclose(modes["faster_only"]["mean_route_s_delta_m"], 2.0)
    assert np.isclose(modes["faster_only"]["mean_reward_gain"], 0.2)
    assert sum(row["groups"] for row in modes.values()) == 1
    assert summary["ordinary_expert_only_group_fraction"] == 0.5
    assert np.isclose(summary["ordinary_expert_vs_det_path_length_delta_m"], 2.0)
    assert summary["all_zero_group_fraction"] == 0.0
    assert summary["no_current_candidate_or_safe_expert_fraction"] == 0.0
    assert summary["deterministic_hard_recovery_fraction"] == 0.0
    assert summary["hard_recovery_tiny_5cm_fraction"] == 0.0


def test_quantifies_millimetre_hard_gate_recovery(tmp_path):
    run = tmp_path / "run"
    replay = run / "replay_buffer"
    rank = replay / "rank_0000"
    rank.mkdir(parents=True)
    (run / "effective_config.json").write_text(
        json.dumps({"awr": {"deterministic_first": True}})
    )
    (rank / "scene_paths.jsonl").write_text('"boundary.npz"\n')
    trajectories = np.zeros((1, 3, 4, 4), dtype=np.float32)
    trajectories[0, 1, :, 1] = 0.005
    trajectories[0, 2, :, 0] = 1.0
    np.save(rank / "trajectories.npy", trajectories)
    np.save(rank / "rewards.npy", np.asarray([[0.0, 0.99, 0.0]], dtype=np.float32))
    np.save(rank / "weights.npy", np.ones((1, 3), dtype=np.float32))
    np.save(rank / "expert_trajectories.npy", np.zeros((1, 4, 4), dtype=np.float32))
    np.save(rank / "expert_safe.npy", np.zeros(1, dtype=np.float32))

    summary = analyze(
        replay,
        allow_partial_prefix=True,
        recompute_positive_weights=True,
        beta=2.0,
        margin=0.01,
        behavior_anchor_weight=0.0,
        unsafe_behavior_anchor_weight=0.0,
        expert_weight=0.4,
        chunk_size=1,
    )

    assert summary["deterministic_hard_recovery_fraction"] == 1.0
    assert summary["hard_recovery_tiny_1cm_fraction"] == 1.0
    assert summary["hard_recovery_tiny_5cm_fraction"] == 1.0
    assert np.isclose(summary["hard_recovery_mean_path_displacement_m"], 0.005)
    assert np.isclose(summary["hard_recovery_mean_max_displacement_m"], 0.005)
    assert np.isclose(summary["hard_recovery_mean_endpoint_displacement_m"], 0.005)


def test_reports_multimodality_only_for_positive_awr_targets(tmp_path):
    run = tmp_path / "run"
    replay = run / "replay_buffer"
    rank = replay / "rank_0000"
    rank.mkdir(parents=True)
    (run / "effective_config.json").write_text(
        json.dumps({"awr": {"deterministic_first": True}})
    )
    (rank / "scene_paths.jsonl").write_text('"multi.npz"\n')
    trajectories = np.zeros((1, 3, 2, 4), dtype=np.float32)
    trajectories[0, 1, -1, :2] = [3.0, 0.0]
    trajectories[0, 2, -1, :2] = [1.0, 2.0]
    np.save(rank / "trajectories.npy", trajectories)
    np.save(rank / "rewards.npy", np.asarray([[0.5, 0.9, 0.8]], dtype=np.float32))
    np.save(rank / "weights.npy", np.ones((1, 3), dtype=np.float32))
    np.save(rank / "expert_trajectories.npy", np.zeros((1, 2, 4), dtype=np.float32))
    np.save(rank / "expert_safe.npy", np.zeros(1, dtype=np.float32))

    summary = analyze(
        replay,
        allow_partial_prefix=True,
        recompute_positive_weights=True,
        beta=2.0,
        margin=0.01,
        behavior_anchor_weight=0.0,
        unsafe_behavior_anchor_weight=0.0,
        expert_weight=0.4,
        chunk_size=1,
    )

    assert summary["positive_multitarget_fraction_active"] == 1.0
    assert summary["positive_endpoint_separation_2m_fraction_active"] == 1.0
    assert summary["positive_lateral_separation_1m_fraction_active"] == 1.0
    assert summary["positive_longitudinal_separation_2m_fraction_active"] == 1.0
    assert np.isclose(summary["positive_multitarget_mean_route_s_range_m"], 2.0)
    assert np.isclose(summary["positive_multitarget_mean_route_d_range_m"], 2.0)


def test_splits_positive_target_displacement_at_reward_horizon(tmp_path):
    run = tmp_path / "run"
    replay = run / "replay_buffer"
    rank = replay / "rank_0000"
    rank.mkdir(parents=True)
    (run / "effective_config.json").write_text(
        json.dumps(
            {
                "awr": {"deterministic_first": True},
                "reward": {"reward_horizon_steps": 2},
            }
        )
    )
    (rank / "scene_paths.jsonl").write_text('"tail.npz"\n')
    trajectories = np.zeros((1, 2, 4, 4), dtype=np.float32)
    trajectories[0, 1, :, 1] = [1.0, 1.0, 3.0, 5.0]
    np.save(rank / "trajectories.npy", trajectories)
    np.save(rank / "rewards.npy", np.asarray([[0.5, 0.9]], dtype=np.float32))
    np.save(rank / "weights.npy", np.ones((1, 2), dtype=np.float32))
    np.save(rank / "expert_trajectories.npy", np.zeros((1, 4, 4), dtype=np.float32))
    np.save(rank / "expert_safe.npy", np.zeros(1, dtype=np.float32))

    summary = analyze(
        replay,
        allow_partial_prefix=True,
        recompute_positive_weights=True,
        beta=2.0,
        margin=0.01,
        behavior_anchor_weight=0.0,
        unsafe_behavior_anchor_weight=0.0,
        expert_weight=0.4,
        chunk_size=1,
    )

    assert summary["reward_horizon_steps"] == 2
    assert summary["unscored_tail_steps"] == 2
    assert np.isclose(summary["active_candidate_mean_xy_displacement_scored_m"], 1.0)
    assert np.isclose(
        summary["active_candidate_mean_xy_displacement_unscored_tail_m"], 4.0
    )
    assert np.isclose(
        summary["active_candidate_unscored_to_scored_displacement_ratio"], 4.0
    )
    assert np.isclose(
        summary["active_candidate_reward_boundary_displacement_m"], 1.0
    )
    assert np.isclose(
        summary["active_candidate_first_unscored_displacement_m"], 3.0
    )
    assert np.isclose(summary["active_candidate_endpoint_displacement_m"], 5.0)
