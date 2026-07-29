import json

import numpy as np

from rlvr.autoresearch.tools.audit_replay_weight_concentration import analyze


def test_beta_exposes_weight_concentration_on_the_same_active_groups(tmp_path):
    replay = tmp_path / "run" / "replay_buffer"
    rank = replay / "rank_0000"
    rank.mkdir(parents=True)
    (rank / "scene_paths.jsonl").write_text('"a"\n"b"\n"c"\n')
    rewards = np.asarray(
        [
            [0.2, 0.3, 0.6, 0.9],
            [0.8, 0.81, 0.82, 0.99],
            [0.5, 0.4, 0.3, 0.2],
        ],
        dtype=np.float32,
    )
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "expert_safe.npy", np.asarray([1.0, 1.0, 0.0], dtype=np.float32))

    report = analyze(
        replay,
        betas=(0.5, 2.0),
        margin=0.01,
        behavior_anchor_weight=0.0,
        unsafe_behavior_anchor_weight=0.0,
        expert_weight=0.4,
        allow_partial_prefix=True,
        chunk_size=2,
    )

    low = report["beta_results"]["0.5"]
    high = report["beta_results"]["2"]
    assert report["prefix_groups"] == 3
    assert low["active_groups"] == high["active_groups"] == 2
    assert low["active_group_fraction"] == high["active_group_fraction"] == 2 / 3
    assert low["effective_sample_size_fraction"]["mean"] > high[
        "effective_sample_size_fraction"
    ]["mean"]
    assert low["top1_weight_share"]["mean"] < high["top1_weight_share"]["mean"]
    assert low["global_expert_weight_share"] > high["global_expert_weight_share"]


def test_closed_manifest_uses_declared_reward_array(tmp_path):
    replay = tmp_path / "run" / "replay_buffer"
    rank = replay / "rank_0000"
    rank.mkdir(parents=True)
    np.save(rank / "formal_rewards.npy", np.asarray([[0.0, 0.8]], dtype=np.float32))
    np.save(rank / "expert_safe.npy", np.asarray([1.0], dtype=np.float32))
    (rank / "manifest.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "scene_count": 1,
                "group_size": 2,
                "arrays": {"rewards": "formal_rewards.npy"},
            }
        )
    )

    report = analyze(
        replay,
        betas=(2.0,),
        margin=0.01,
        behavior_anchor_weight=0.0,
        unsafe_behavior_anchor_weight=0.0,
        expert_weight=0.4,
        allow_partial_prefix=False,
    )

    assert report["source_complete"] is True
    assert report["beta_results"]["2"]["active_groups"] == 1
    assert report["beta_results"]["2"]["top1_weight_share"]["mean"] == 1.0
