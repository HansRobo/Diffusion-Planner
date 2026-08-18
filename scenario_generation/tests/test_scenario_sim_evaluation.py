"""``ScenarioSimClosedLoopEvaluation``: job discovery, streaming, sharding and the DDP merge.

The rollout itself is stubbed. What is under test is the orchestration the scenario_sim path now
inherits from ``ClosedLoopEvaluation`` -- that rows reach ``segments.jsonl`` with a suite-unique
identity, that their digests reach the sidecar under the SAME identity (a mismatch reattaches
nothing and silently degrades the pooled percentile), that a failing scenario cannot take down
the rest of a shard, and that a partial set of ranks is refused rather than merged.
"""

import json

import numpy as np
import pytest

from scenario_generation.closed_loop_eval import attach_tdigest_sidecars
from scenario_generation.closed_loop_evaluation import ClosedLoopEvalConfig, RolloutParams
from scenario_generation.scenario_sim_evaluation import ScenarioSimClosedLoopEvaluation
from scenario_generation.scenario_sim_metrics import build_segment_row

NEAR_MISS = 1.0
N_STEPS = 5


def _suite(tmp_path, n_cases: int = 3):
    """A suite shaped like the real one: identical stems under per-case directories."""
    root = tmp_path / "suite"
    for i in range(n_cases):
        d = root / f"case{i}" / "v1"
        d.mkdir(parents=True)
        (d / "scenario_0.xosc").write_text("<OpenSCENARIO/>")
    return root


def _row(progress_m: float = 12.0):
    return build_segment_row(
        n_steps_run=N_STEPS,
        terminated="max_steps",
        result_kind="Pass",
        clearances=[3.0, 2.0, 1.5, 2.5, 4.0],
        collisions=[False] * N_STEPS,
        rb_dists=np.array([2.0, 2.1, 1.9, 2.2, 2.3]),
        speeds=np.array([5.0, 5.1, 5.2, 5.0, 4.9]),
        dt=0.1,
        near_miss_thresh=NEAR_MISS,
        strong_brake_mps2=-2.5,
        progress_m=progress_m,
    )


def _evaluator(tmp_path, root, monkeypatch, *, rank=0, world=1, fail_on=None):
    """Build an evaluator whose rollout is stubbed; ``fail_on`` raises for that job's scenario."""

    def fake_rollout(model, model_args, osc_path, output_dir, **kw):
        if fail_on is not None and fail_on in str(osc_path):
            raise RuntimeError("stub rollout blew up")
        return _row()

    monkeypatch.setattr(
        "scenario_generation.scenario_sim_evaluation.run_scenario_sim_rollout", fake_rollout
    )
    # The map is never parsed: a stubbed rollout has nothing to do with it.
    monkeypatch.setattr(
        ScenarioSimClosedLoopEvaluation,
        "_builder_for",
        lambda self, osc, override: ("map.osm", None),
    )
    config = ClosedLoopEvalConfig(
        out_dir=tmp_path / "out",
        params=RolloutParams(device="cpu", near_miss_thresh=NEAR_MISS),
        verbose=False,
    )
    return ScenarioSimClosedLoopEvaluation(
        object(), object(), config, root, ddp_rank=rank, ddp_world_size=world
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_rows_and_digests_share_one_identity(tmp_path, monkeypatch):
    root = _suite(tmp_path, 3)
    ev = _evaluator(tmp_path, root, monkeypatch)
    summary = ev.run()

    rows = _read_jsonl(ev.out_dir / "segments.jsonl")
    digests = _read_jsonl(ev.out_dir / "tdigests.jsonl")
    assert len(rows) == 3
    # The suite's stems all collide, so these keys also pin that a row's identity carries the
    # path that disambiguates them.
    assert {r["route"] for r in rows} == {f"case{i}__v1__scenario_0" for i in range(3)}
    # Same identity on both sides, so the sidecar actually reattaches.
    assert {d["route"] for d in digests} == {r["route"] for r in rows}
    attach_tdigest_sidecars(rows, digests)
    assert all("_tdigest" in r["object"] for r in rows)
    # segments.jsonl itself stays free of the raw blobs.
    assert all("_tdigest" not in line for line in (ev.out_dir / "segments.jsonl").read_text())

    assert summary["n_segments"] == 3
    assert summary["n_scenarios"] == 3
    assert summary["mode"] == "scenario_sim"
    assert summary["terminated_counts"] == {"max_steps": 3}
    assert json.loads((ev.out_dir / "summary.json").read_text())["n_segments"] == 3


def test_a_failing_scenario_is_counted_not_fatal(tmp_path, monkeypatch):
    root = _suite(tmp_path, 3)
    ev = _evaluator(tmp_path, root, monkeypatch, fail_on="case1")
    summary = ev.run()

    rows = _read_jsonl(ev.out_dir / "segments.jsonl")
    assert len(rows) == 3
    assert summary["terminated_counts"] == {"max_steps": 2, "worker_failed": 1}
    failed = next(r for r in rows if r["terminated"] == "worker_failed")
    assert failed["route"] == "case1__v1__scenario_0"
    assert "stub rollout blew up" in (ev.out_dir / failed["route"] / "error.txt").read_text()


def test_ddp_shards_partition_and_merge_to_the_single_rank_result(tmp_path, monkeypatch):
    root = _suite(tmp_path, 4)

    single = _evaluator(tmp_path / "solo", root, monkeypatch).run()

    shared = tmp_path / "ddp"
    parts = [_evaluator(shared, root, monkeypatch, rank=r, world=2).run() for r in range(2)]
    assert all(p["ddp_shard"] for p in parts)
    assert sum(p["n_rows"] for p in parts) == 4
    for r in range(2):
        assert (shared / "out" / f"segments_{r}.jsonl").is_file()

    rank0 = _evaluator(shared, root, monkeypatch, rank=0, world=2)
    merged = rank0.merge_ddp_shards(2)
    assert merged["n_segments"] == single["n_segments"] == 4
    assert merged["terminated_counts"] == single["terminated_counts"]
    assert merged["object"]["collision_count"] == single["object"]["collision_count"]
    # The unsharded name is what the report and downstream readers look for.
    assert len(_read_jsonl(shared / "out" / "segments.jsonl")) == 4


def test_merge_refuses_a_missing_rank(tmp_path, monkeypatch):
    root = _suite(tmp_path, 4)
    shared = tmp_path / "ddp"
    _evaluator(shared, root, monkeypatch, rank=0, world=2).run()

    rank0 = _evaluator(shared, root, monkeypatch, rank=0, world=2)
    with pytest.raises(RuntimeError, match="missing segments_<rank>.jsonl"):
        rank0.merge_ddp_shards(2)


def test_interpreter_is_forced_out_of_process(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, _suite(tmp_path, 1), monkeypatch)
    assert ev.rollout_config.sim_in_subprocess is True


def test_claimed_work_is_partitioned_without_being_pre_assigned(tmp_path, monkeypatch):
    """Every rank walks the whole list; each job is run exactly once, by whoever wins it."""
    root = _suite(tmp_path, 5)
    shared = tmp_path / "claimed"
    claim_dir = shared / "claims"

    parts = []
    for r in range(2):
        ev = _evaluator(shared, root, monkeypatch, rank=r, world=2)
        ev.config.claim_dir = claim_dir
        # Claiming replaces the static split, so a rank is offered every job.
        assert len(ev.shard_jobs(ev.discover_jobs())) == 5
        parts.append(ev.run())

    routes = []
    for r in range(2):
        routes += [row["route"] for row in _read_jsonl(shared / "out" / f"segments_{r}.jsonl")]
    assert sorted(routes) == sorted(f"case{i}__v1__scenario_0" for i in range(5))
    assert sum(p["n_rows"] for p in parts) == 5
