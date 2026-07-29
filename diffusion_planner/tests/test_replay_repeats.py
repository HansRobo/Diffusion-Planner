# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for diffusion_planner/sampling/replay_repeats.py.

The load-bearing tests are the equivalence tests: they build real
``ClusterWeightedDistributedSampler`` instances (one per rank, independent of the tool's
own iteration strategy), iterate them, count repeats off ``repeat_flags``, and assert the
tool reports exactly those numbers. Also covers the drop_last-adjusted forced-row count
against ``ForcedAugmentationSelector.forced_count``, data-list reconstruction, and the
refusal to guess a world size.

Usage:
    pytest diffusion_planner/tests/test_replay_repeats.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from diffusion_planner.utils.forced_augmentation import ForcedAugmentationSelector
from diffusion_planner.utils.weighted_sampler import ClusterWeightedDistributedSampler

SAMPLING_DIR = Path(__file__).resolve().parent.parent / "sampling"
sys.path.insert(0, str(SAMPLING_DIR))

import replay_repeats  # noqa: E402

SEED = 3407
ALPHA = 1.0


def make_inputs(tmp_path: Path, n: int = 240) -> tuple[str, str, list[str]]:
    """Write a train_set_list and a skewed 3-cluster JSON. Returns (list, clusters, paths)."""
    paths = [f"/data/loc/train/2026-01-01/12-00-00/frame_{i:06d}.npz" for i in range(n)]
    list_path = tmp_path / "train_list.json"
    cluster_path = tmp_path / "clusters.json"
    rare = max(2, n // 40)
    medium = max(4, n // 8)
    clusters = {
        "cluster_id0": paths[:rare],
        "cluster_id1": paths[rare : rare + medium],
        "cluster_id2": paths[rare + medium :],
    }
    list_path.write_text(json.dumps(paths))
    cluster_path.write_text(json.dumps(clusters))
    return str(list_path), str(cluster_path), paths


def write_run_dir(
    tmp_path: Path,
    list_path: str,
    cluster_path: str,
    *,
    train_epochs: int = 3,
    batch_size: int | None = None,
    subsample_step: int = 1,
    pool_spec: str = "",
    augment_type: str = "quintic",
    force: bool = True,
    data_list_size: int | None = None,
    extra: dict | None = None,
) -> Path:
    """Write an args.json (+ cluster_sampling.json) shaped like train.py's."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    args = {
        "exp_name": "unit_test",
        "save_dir": str(run_dir),
        "train_set_list": list_path,
        "valid_set_list": "/unused/valid.json",
        "train_subsample_step": subsample_step,
        "cluster_json": cluster_path,
        "cluster_weight_alpha": ALPHA,
        "force_aug_on_repeat": force,
        "repeat_aug_pool": pool_spec,
        "augment_type": augment_type,
        "use_data_augment": True,
        "use_flip_augment": True,
        "use_neighbor_dropout": True,
        "use_neighbor_noise": False,
        "use_turn_indicator_dropout": False,
        "seed": SEED,
        "train_epochs": train_epochs,
        "ddp": True,
        "major_version": 5,
    }
    if batch_size is not None:
        args["batch_size"] = batch_size
    args.update(extra or {})
    (run_dir / "args.json").write_text(json.dumps(args, indent=4))
    if data_list_size is not None:
        (run_dir / "cluster_sampling.json").write_text(
            json.dumps(
                {
                    "cluster_json": cluster_path,
                    "cluster_weight_alpha": ALPHA,
                    "train_subsample_step": subsample_step,
                    "data_list_size": data_list_size,
                    "matched_count": data_list_size,
                    "forced_augmentation": {"enabled": force, "pool": []},
                },
                indent=4,
            )
        )
    return run_dir


def ground_truth(
    data_list: list[str],
    cluster_path: str,
    world_size: int,
    epoch: int,
) -> dict:
    """Iterate a fresh sampler per rank and aggregate GLOBAL counts the obvious way.

    Deliberately independent of the tool: separate sampler objects per rank, no rank
    mutation, flags summed rather than reassembled.
    """
    total = 0
    repeats = 0
    drawn: set[int] = set()
    per_rank_flags: list[list[bool]] = []
    for rank in range(world_size):
        sampler = ClusterWeightedDistributedSampler(
            data_list,
            cluster_path,
            num_replicas=world_size,
            rank=rank,
            seed=SEED,
            alpha=ALPHA,
            track_repeats=True,
        )
        sampler.set_epoch(epoch)
        indices = list(sampler)
        flags = sampler.repeat_flags
        total += len(indices)
        repeats += sum(flags)
        drawn.update(indices)
        per_rank_flags.append(list(flags))
    return {
        "total_draws": total,
        "distinct_samples": len(drawn),
        "repeat_draws": repeats,
        "per_rank_flags": per_rank_flags,
    }


class TestEquivalenceWithTheRealSampler:
    @pytest.mark.parametrize("world_size", [1, 2, 4, 8])
    def test_replay_matches_direct_sampler_iteration(self, tmp_path, world_size):
        list_path, cluster_path, paths = make_inputs(tmp_path, n=240)
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--seed",
                str(SEED),
                "--cluster_weight_alpha",
                str(ALPHA),
                "--world_size",
                str(world_size),
                "--epochs",
                "1",
                "--repeat_aug_pool",
                "flip,neighbor_dropout",
            ]
        )
        truth = ground_truth(paths, cluster_path, world_size, epoch=0)
        row = result["per_epoch"][0]
        assert row["total_draws"] == truth["total_draws"]
        assert row["distinct_samples"] == truth["distinct_samples"]
        assert row["repeat_draws"] == truth["repeat_draws"]
        assert row["repeat_fraction"] == pytest.approx(truth["repeat_draws"] / truth["total_draws"])
        assert row["expected_forced_per_pool_member"] == pytest.approx(truth["repeat_draws"] / 2)
        # A non-trivial fixture: repeats must actually happen, or the test proves nothing.
        assert truth["repeat_draws"] > 0

    def test_replay_matches_direct_sampler_every_epoch(self, tmp_path):
        """Each epoch reseeds with seed + epoch, so every epoch must match individually."""
        list_path, cluster_path, paths = make_inputs(tmp_path, n=200)
        world_size = 4
        epochs = 5
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--seed",
                str(SEED),
                "--world_size",
                str(world_size),
                "--epochs",
                str(epochs),
                "--repeat_aug_pool",
                "flip,neighbor_noise,state_perturbation",
            ]
        )
        assert len(result["per_epoch"]) == epochs
        for epoch in range(epochs):
            truth = ground_truth(paths, cluster_path, world_size, epoch=epoch)
            row = result["per_epoch"][epoch]
            assert row["epoch"] == epoch
            assert row["repeat_draws"] == truth["repeat_draws"], f"epoch {epoch}"
            assert row["distinct_samples"] == truth["distinct_samples"], f"epoch {epoch}"
        # Epochs must not be identical, otherwise the per-epoch match is vacuous.
        assert len({row["repeat_draws"] for row in result["per_epoch"]}) > 1
        assert result["total"]["repeat_draws"] == sum(
            row["repeat_draws"] for row in result["per_epoch"]
        )
        assert result["total"]["total_draws"] == sum(
            row["total_draws"] for row in result["per_epoch"]
        )

    def test_start_epoch_shifts_the_replayed_epochs(self, tmp_path):
        list_path, cluster_path, paths = make_inputs(tmp_path, n=160)
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--seed",
                str(SEED),
                "--world_size",
                "2",
                "--epochs",
                "2",
                "--start_epoch",
                "40",
            ]
        )
        assert [row["epoch"] for row in result["per_epoch"]] == [40, 41]
        for row in result["per_epoch"]:
            truth = ground_truth(paths, cluster_path, 2, epoch=row["epoch"])
            assert row["repeat_draws"] == truth["repeat_draws"]

    def test_padded_length_accounts_for_world_size(self, tmp_path):
        """N=250 over 8 ranks pads to 256 draws; a world_size guess of 1 would say 250."""
        list_path, cluster_path, _ = make_inputs(tmp_path, n=250)
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--world_size",
                "8",
                "--epochs",
                "1",
            ]
        )
        assert result["sampler"]["num_samples_per_rank"] == 32
        assert result["sampler"]["padded_length"] == 256
        assert result["per_epoch"][0]["total_draws"] == 256

    def test_inconsistent_flags_are_refused_not_reported(self, tmp_path, monkeypatch):
        """The cross-check on the reassembled global draw list must not be dead code."""

        class CorruptingSampler(ClusterWeightedDistributedSampler):
            def __iter__(self):
                indices = super().__iter__()
                if self.rank == 1 and self.repeat_flags:
                    flipped = list(self.repeat_flags)
                    flipped[0] = not flipped[0]
                    self.repeat_flags = flipped
                return indices

        monkeypatch.setattr(replay_repeats, "ClusterWeightedDistributedSampler", CorruptingSampler)
        list_path, cluster_path, _ = make_inputs(tmp_path, n=120)
        with pytest.raises(replay_repeats.ReplayError, match="do not match the flags recomputed"):
            replay_repeats.run(
                [
                    "--train_set_list",
                    list_path,
                    "--cluster_json",
                    cluster_path,
                    "--world_size",
                    "4",
                    "--epochs",
                    "1",
                ]
            )

    def test_distinct_plus_repeats_equals_total(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=300)
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--world_size",
                "4",
                "--epochs",
                "3",
            ]
        )
        for row in result["per_epoch"]:
            assert row["distinct_samples"] + row["repeat_draws"] == row["total_draws"]


class TestForcedRowsMatchTheSelector:
    def test_forced_rows_equals_selector_forced_count(self, tmp_path):
        """The delivered-row count must equal what ForcedAugmentationSelector would tally.

        The training DataLoader is ``drop_last=True`` with ``batch_size // world_size`` per
        rank, so each rank's partial tail batch is never delivered and its repeat flags are
        never consumed. This replays that consumption per rank with the real selector.
        """
        list_path, cluster_path, paths = make_inputs(tmp_path, n=250)
        world_size = 4
        batch_size = 32
        per_rank_batch = batch_size // world_size
        run_dir = write_run_dir(
            tmp_path,
            list_path,
            cluster_path,
            train_epochs=2,
            batch_size=batch_size,
        )
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", str(world_size)])
        assert result["config"]["per_rank_batch"] == per_rank_batch
        pool = result["config"]["repeat_aug_pool"]

        for row in result["per_epoch"]:
            truth = ground_truth(paths, cluster_path, world_size, epoch=row["epoch"])
            selector_total = 0
            for flags in truth["per_rank_flags"]:
                selector = ForcedAugmentationSelector(pool, seed=SEED)
                selector.start_epoch(row["epoch"], flags, row["epoch"])
                offset = 0
                while offset + per_rank_batch <= len(flags):  # drop_last=True
                    selector.masks_for_batch(per_rank_batch, "cpu")
                    offset += per_rank_batch
                selector_total += selector.forced_count
            assert row["forced_rows"] == selector_total, f"epoch {row['epoch']}"
            # Non-trivial: the tail really is dropped for this shape.
            assert row["dropped_rows"] > 0
            assert row["forced_rows"] < row["repeat_draws"]

    def test_forced_rows_equals_repeats_when_shard_divides_evenly(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=256)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path, train_epochs=1, batch_size=32)
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "4"])
        row = result["per_epoch"][0]
        assert row["dropped_rows"] == 0
        assert row["forced_rows"] == row["repeat_draws"]

    def test_no_batch_size_omits_delivered_rows(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=200)
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--world_size",
                "2",
                "--epochs",
                "1",
            ]
        )
        assert "forced_rows" not in result["per_epoch"][0]
        assert "forced_rows" not in result["total"]
        assert "drop_last is not modelled" in replay_repeats.format_report(result)


class TestDataListReconstruction:
    @pytest.mark.parametrize("step", [1, 2, 3, 7])
    def test_mirrors_train_subsample_step(self, tmp_path, step):
        list_path, cluster_path, paths = make_inputs(tmp_path, n=210)
        expected = paths[::step]
        data_list = replay_repeats.build_data_list(list_path, step)
        assert data_list == expected

    def test_subsample_step_changes_every_count(self, tmp_path):
        list_path, cluster_path, paths = make_inputs(tmp_path, n=210)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path, train_epochs=1, subsample_step=3)
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])
        assert result["sampler"]["data_list_size"] == len(paths[::3])
        truth = ground_truth(paths[::3], cluster_path, 2, epoch=0)
        assert result["per_epoch"][0]["repeat_draws"] == truth["repeat_draws"]

    def test_rejects_non_positive_step(self, tmp_path):
        list_path, _, _ = make_inputs(tmp_path, n=20)
        with pytest.raises(replay_repeats.ReplayError, match="train_subsample_step"):
            replay_repeats.build_data_list(list_path, 0)

    def test_rejects_non_list_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(replay_repeats.ReplayError, match="JSON list"):
            replay_repeats.build_data_list(str(bad), 1)


class TestWorldSize:
    def test_missing_world_size_exits_with_instructions(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path)
        with pytest.raises(SystemExit) as excinfo:
            replay_repeats.main(["--run_dir", str(run_dir)])
        message = str(excinfo.value)
        assert "--world_size is required" in message
        assert "train_log.txt" in message

    def test_world_size_read_from_args_json_when_present(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, extra={"world_size": 4}
        )
        result = replay_repeats.run(["--run_dir", str(run_dir)])
        assert result["config"]["world_size"] == 4
        assert result["config"]["sources"]["world_size"] == "args.json[world_size]"

    def test_flag_overrides_args_json(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, extra={"world_size": 4}
        )
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])
        assert result["config"]["world_size"] == 2
        assert result["config"]["sources"]["world_size"] == "flag"

    def test_rejects_zero_world_size(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        with pytest.raises(replay_repeats.ReplayError, match="world_size must be >= 1"):
            replay_repeats.run(
                [
                    "--train_set_list",
                    list_path,
                    "--cluster_json",
                    cluster_path,
                    "--world_size",
                    "0",
                    "--epochs",
                    "1",
                ]
            )


class TestPoolResolution:
    def test_empty_pool_spec_resolves_to_enabled_augmentations(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path, train_epochs=1)
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])
        assert result["config"]["repeat_aug_pool"] == [
            "flip",
            "neighbor_dropout",
            "state_perturbation",
        ]

    def test_bridge_excludes_state_perturbation_from_default_pool(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, augment_type="bridge"
        )
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])
        assert result["config"]["repeat_aug_pool"] == ["flip", "neighbor_dropout"]

    def test_pool_naming_a_disabled_augmentation_is_rejected(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, pool_spec="neighbor_noise"
        )
        with pytest.raises(replay_repeats.ReplayError, match="not enabled"):
            replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])

    def test_forcing_off_reports_empty_pool_and_warns(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path, train_epochs=1, force=False)
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])
        assert result["config"]["repeat_aug_pool"] == []
        assert result["per_epoch"][0]["expected_forced_per_pool_member"] is None
        assert any("empty pool" in w for w in result["warnings"])
        assert "forcing DISABLED" in replay_repeats.format_report(result)

    def test_force_flag_enables_hypothetical_replay(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path, train_epochs=1, force=False)
        result = replay_repeats.run(
            ["--run_dir", str(run_dir), "--world_size", "2", "--force_aug_on_repeat"]
        )
        assert result["config"]["repeat_aug_pool"] == [
            "flip",
            "neighbor_dropout",
            "state_perturbation",
        ]

    def test_unknown_pool_name_rejected_without_run_dir(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        with pytest.raises(replay_repeats.ReplayError, match="unknown augmentation"):
            replay_repeats.run(
                [
                    "--train_set_list",
                    list_path,
                    "--cluster_json",
                    cluster_path,
                    "--world_size",
                    "1",
                    "--epochs",
                    "1",
                    "--repeat_aug_pool",
                    "flip,not_an_aug",
                ]
            )


class TestCrossChecksAndErrors:
    def test_data_list_size_mismatch_raises(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, data_list_size=999
        )
        with pytest.raises(replay_repeats.ReplayError, match="Every count would be wrong"):
            replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])

    def test_data_list_size_mismatch_can_be_overridden(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, data_list_size=999
        )
        result = replay_repeats.run(
            [
                "--run_dir",
                str(run_dir),
                "--world_size",
                "2",
                "--allow_data_list_mismatch",
            ]
        )
        assert any("cross-check FAILED" in w for w in result["warnings"])

    def test_matching_cluster_sampling_json_passes_quietly(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(
            tmp_path, list_path, cluster_path, train_epochs=1, data_list_size=100
        )
        result = replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])
        assert not any("cross-check" in w for w in result["warnings"])

    def test_missing_args_json_is_a_clear_error(self, tmp_path):
        empty = tmp_path / "not_a_run"
        empty.mkdir()
        with pytest.raises(replay_repeats.ReplayError, match="args.json not found"):
            replay_repeats.run(["--run_dir", str(empty), "--world_size", "1"])

    def test_run_without_cluster_json_is_rejected(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        run_dir = write_run_dir(tmp_path, list_path, cluster_path, train_epochs=1)
        args = json.loads((run_dir / "args.json").read_text())
        args["cluster_json"] = ""
        (run_dir / "args.json").write_text(json.dumps(args))
        with pytest.raises(replay_repeats.ReplayError, match="WITHOUT replacement"):
            replay_repeats.run(["--run_dir", str(run_dir), "--world_size", "2"])

    def test_epochs_required_without_run_dir(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=100)
        with pytest.raises(replay_repeats.ReplayError, match="--epochs is required"):
            replay_repeats.run(
                [
                    "--train_set_list",
                    list_path,
                    "--cluster_json",
                    cluster_path,
                    "--world_size",
                    "1",
                ]
            )

    def test_train_set_list_required_without_run_dir(self, tmp_path):
        _, cluster_path, _ = make_inputs(tmp_path, n=100)
        with pytest.raises(replay_repeats.ReplayError, match="--train_set_list is required"):
            replay_repeats.run(
                ["--cluster_json", cluster_path, "--world_size", "1", "--epochs", "1"]
            )


class TestCliOutput:
    def test_main_prints_table_and_writes_json(self, tmp_path, capsys):
        list_path, cluster_path, paths = make_inputs(tmp_path, n=250)
        run_dir = write_run_dir(
            tmp_path,
            list_path,
            cluster_path,
            train_epochs=2,
            batch_size=32,
            data_list_size=250,
        )
        out_json = tmp_path / "nested" / "repeats.json"
        code = replay_repeats.main(
            [
                "--run_dir",
                str(run_dir),
                "--world_size",
                "4",
                "--json",
                str(out_json),
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "Offline repeat-draw replay" in printed
        assert "TOTAL" in printed
        assert "forced_rows" in printed

        payload = json.loads(out_json.read_text())
        assert len(payload["per_epoch"]) == 2
        truth = ground_truth(paths, cluster_path, 4, epoch=0)
        assert payload["per_epoch"][0]["repeat_draws"] == truth["repeat_draws"]
        assert payload["total"]["repeat_draws"] == sum(
            row["repeat_draws"] for row in payload["per_epoch"]
        )
        assert payload["config"]["world_size"] == 4
        assert payload["sampler"]["data_list_size"] == 250

    def test_total_distinct_is_a_union_not_a_sum(self, tmp_path):
        list_path, cluster_path, _ = make_inputs(tmp_path, n=200)
        result = replay_repeats.run(
            [
                "--train_set_list",
                list_path,
                "--cluster_json",
                cluster_path,
                "--world_size",
                "2",
                "--epochs",
                "4",
            ]
        )
        per_epoch_sum = sum(row["distinct_samples"] for row in result["per_epoch"])
        union = result["total"]["distinct_samples_union"]
        assert union <= 200
        assert union < per_epoch_sum
        assert union >= max(row["distinct_samples"] for row in result["per_epoch"])
