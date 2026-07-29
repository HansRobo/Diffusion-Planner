from pathlib import Path

from rlvr.posttrain import runner
from rlvr.posttrain.runner import completed_epoch, prepare, wait_for_path
from rlvr.posttrain.spec import ROOT, Spec


def arm(tmp_path, epochs=3) -> Spec:
    return Spec(
        name="dual",
        train={
            "model_path": "/base.pth",
            "output_dir": str(tmp_path / "dual"),
            "epochs": epochs,
        },
    )


def writes_an_epoch(tmp_path, calls, code=0):
    """A fake attempt that finishes exactly one epoch, like a preempted run."""

    def fake_run(spec, **kwargs):
        calls.append(spec)
        run_dir = tmp_path / "dual" / "20260101-000000_dual"
        run_dir.mkdir(parents=True, exist_ok=True)
        nxt = completed_epoch(tmp_path / "dual") + 1
        (run_dir / f"epoch_{nxt:03d}.pth").write_bytes(b"x")
        return code

    return fake_run


def spec(**step) -> Spec:
    return Spec(name="t", train={"model_path": "/x"}, prepare=[step])


def test_a_built_artifact_is_not_rebuilt(tmp_path):
    (tmp_path / "overlay").mkdir()
    step = {"creates": str(tmp_path / "overlay"), "argv": ["false"]}
    assert prepare(spec(**step), tmp_path, {}) == 0


def test_a_failed_pre_step_stops_the_arm(tmp_path):
    assert prepare(spec(argv=["false"]), tmp_path, {}) != 0
    assert (tmp_path / "prepare_0.log").exists()


def test_a_pre_step_that_does_not_produce_its_artifact_fails(tmp_path):
    step = {"creates": str(tmp_path / "missing"), "argv": ["true"]}
    assert prepare(spec(**step), tmp_path, {}) != 0


def test_a_pre_step_runs_from_the_repo_root(tmp_path):
    out = tmp_path / "cwd.txt"
    step = {"creates": str(out), "argv": ["bash", "-c", f"pwd > {out}"]}
    assert prepare(spec(**step), tmp_path, {}) == 0
    assert Path(out.read_text().strip()) == ROOT


def test_the_supervisor_keeps_going_until_the_target_epoch(tmp_path, monkeypatch):
    calls: list[Spec] = []
    monkeypatch.setattr(runner, "run", writes_an_epoch(tmp_path, calls))
    assert runner.run_until_complete(arm(tmp_path, epochs=3)) == 0
    assert len(calls) == 3


def test_only_the_first_attempt_starts_from_the_origin(tmp_path, monkeypatch):
    """Attempt one starts from the mine; every attempt after continues itself."""

    calls: list[Spec] = []
    monkeypatch.setattr(runner, "run", writes_an_epoch(tmp_path, calls))
    runner.run_until_complete(arm(tmp_path, epochs=2))
    assert calls[0].train["model_path"] == "/base.pth"
    assert "use_policy_state" not in calls[0].train
    output_dir = str(tmp_path / "dual")
    assert calls[1].train["model_path"] == {"latest_epoch": output_dir}
    assert calls[1].train["start_epoch"] == {"resume_after_latest_epoch": output_dir}
    # A mid-run checkpoint carries both, and continuing without them silently
    # restarts from the deployable EMA with fresh optimizer moments.
    assert calls[1].train["use_policy_state"] is True
    assert calls[1].train["resume_optimizer_state"] is True


def test_an_arm_that_is_already_finished_is_not_relaunched(tmp_path, monkeypatch):
    run_dir = tmp_path / "dual" / "20260101-000000_dual"
    run_dir.mkdir(parents=True)
    (run_dir / "epoch_003.pth").write_bytes(b"x")
    calls: list[Spec] = []
    monkeypatch.setattr(runner, "run", writes_an_epoch(tmp_path, calls))
    assert runner.run_until_complete(arm(tmp_path, epochs=3)) == 0
    assert calls == []


def test_a_run_that_finishes_no_epoch_is_not_retried_forever(tmp_path, monkeypatch):
    """A crash loop is a real failure repeating, not something to resume."""

    calls: list[Spec] = []

    def fake_run(spec, **kwargs):
        calls.append(spec)
        return 1

    monkeypatch.setattr(runner, "run", fake_run)
    assert runner.run_until_complete(arm(tmp_path, epochs=3), max_stalls=2) == 1
    assert len(calls) == 2


def test_progress_clears_the_stall_count(tmp_path, monkeypatch):
    calls: list[Spec] = []
    writer = writes_an_epoch(tmp_path, calls)
    attempts = {"n": 0}

    def flaky(spec, **kwargs):
        attempts["n"] += 1
        if attempts["n"] % 2:  # every other attempt dies without an epoch
            calls.append(spec)
            return 1
        return writer(spec, **kwargs)

    monkeypatch.setattr(runner, "run", flaky)
    assert runner.run_until_complete(arm(tmp_path, epochs=3), max_stalls=2) == 0
    assert completed_epoch(tmp_path / "dual") == 3


def test_waiting_for_an_existing_path_does_not_block(tmp_path):
    wait_for_path(tmp_path)


def test_waiting_accepts_a_glob_for_a_directory_nobody_can_name_yet(tmp_path):
    """The mine's completion marker sits under a stamp it chooses itself."""

    (tmp_path / "20260101-000000_mine").mkdir()
    (tmp_path / "20260101-000000_mine" / "final_summary.json").write_text("{}")
    wait_for_path(f"{tmp_path}/*/final_summary.json")


def test_an_unresolvable_spec_is_a_preflight_exit_not_a_crash(tmp_path):
    """The supervisor counts this as a stall; a traceback would end the run."""

    spec = Spec(
        name="dual",
        train={"model_path": {"latest_epoch": str(tmp_path)}, "output_dir": str(tmp_path)},
    )
    assert runner.run(spec, log_root=tmp_path / "logs") == 78
