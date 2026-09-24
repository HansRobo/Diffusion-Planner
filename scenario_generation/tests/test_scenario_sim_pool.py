"""The pooled worker's bookkeeping, with the rollout and the model stubbed out.

What is worth pinning here is what the pool does around a scenario, not the rollout itself: which
cases it takes, what it writes where, and when it stops taking more.
"""

import json
import sys
import types

import pytest

from scenario_generation import scenario_sim_pool as pool
from scenario_generation.scenario_sim_rollout import ScenarioRejected


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Build a work list of ``n`` cases and run the pool over it with a stubbed rollout."""

    def _run(outcomes, argv_extra=(), work_n=None, model_path="/ckpt.pth"):
        work_n = work_n if work_n is not None else len(outcomes)
        out_root = tmp_path / "run"
        work = [[str(out_root / f"case{i}"), f"/scenarios/case{i}.xosc"] for i in range(work_n)]
        work_list = tmp_path / "work.json"
        work_list.write_text(json.dumps(work))

        # load_model is imported inside main(), so the stub has to live on the module it comes from.
        fake = types.ModuleType("scenario_generation.simulate")
        loaded = []
        fake.load_model = lambda path, device: (loaded.append(("pth", path)) or "model", "args")
        fake.load_onnx_model = lambda path, device: (loaded.append(("onnx", path)) or "model", "args")
        monkeypatch.setitem(sys.modules, "scenario_generation.simulate", fake)
        _run.loaded = loaded

        calls = []

        def fake_rollout(model, model_args, osc_path, out, **kw):
            index = int(str(osc_path).split("case")[-1].split(".")[0])
            calls.append(index)
            outcome = outcomes[index] if index < len(outcomes) else None
            if isinstance(outcome, Exception):
                raise outcome
            return {"n_steps_run": 10, "terminated": "sim_terminated"}

        monkeypatch.setattr(pool, "run_scenario_sim_rollout", fake_rollout)
        monkeypatch.setattr(pool, "tdigest_sidecar_row", lambda row: None)

        rc = pool.main([
            "--work_list", str(work_list),
            "--claim_dir", str(tmp_path / "claims"),
            "--run_dir", str(out_root),
            "--model_path", model_path,
            "--device", "cpu",
            "--slot", "3",
            "--gpu", "1",
            *argv_extra,
        ])
        return rc, calls, out_root

    return _run


def _lines(path):
    return [l for l in path.read_text().splitlines() if l] if path.exists() else []


def test_a_case_already_claimed_is_left_alone(run, tmp_path):
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "000001").write_text("")  # someone else took case 1
    rc, calls, out = run([None, None, None])
    assert rc == 0
    assert calls == [0, 2]


def test_model_load_is_reported_on_the_case_that_paid_it(run):
    _, _, out = run([None, None])
    first = json.loads((out / "case0" / "row.json").read_text())["timing"]
    second = json.loads((out / "case1" / "row.json").read_text())["timing"]
    assert "model_load" in first
    assert "model_load" not in second
    # Both cases still report their own wall, which is what a per-case report reads.
    assert "worker_process" in first and "worker_process" in second


def test_each_case_appends_its_own_wall_with_slot_and_gpu(run):
    _, _, out = run([None, None])
    rows = [l.split("\t") for l in _lines(out / "slot_log.tsv")]
    assert [r[0] for r in rows] == ["case0", "case1"]
    assert all(r[2:5] == ["3", "1", "0"] for r in rows)


def test_a_refused_scenario_is_counted_apart_and_does_not_retire_the_worker(run):
    refused = [ScenarioRejected("configure() did not reach 'inactive'")] * 4
    rc, calls, out = run([*refused, None])
    assert rc == 0
    # All four refusals were taken and the fifth case still ran: a refusal says nothing about
    # this process, so the three-strikes count must not see it.
    assert calls == [0, 1, 2, 3, 4]
    assert _lines(out / "rejected.txt") == ["case0", "case1", "case2", "case3"]
    assert not _lines(out / "failures.txt")
    assert [l.split("\t")[4] for l in _lines(out / "slot_log.tsv")] == ["3", "3", "3", "3", "0"]


def test_three_consecutive_failures_stop_the_worker_taking_more(run):
    rc, calls, out = run([RuntimeError("boom")] * 3, work_n=5)
    assert rc == 0
    assert calls == [0, 1, 2]  # cases 3 and 4 are left for a healthy worker
    assert len(_lines(out / "failures.txt")) == 3


def test_a_failure_between_successes_does_not_stop_the_worker(run):
    rc, calls, _ = run([None, RuntimeError("boom"), None, RuntimeError("boom"), None])
    assert rc == 0
    assert calls == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("path, kind", [("/e/diffusion_planner.onnx", "onnx"), ("/e/best_model.pth", "pth")])
def test_the_suffix_picks_the_loader_and_it_runs_once(run, path, kind):
    run([None, None, None], model_path=path)
    assert run.loaded == [(kind, path)]
