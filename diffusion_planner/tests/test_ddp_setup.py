from types import SimpleNamespace

from diffusion_planner.utils import ddp


def test_direct_launch_disables_uninitialized_ddp(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(ddp=True, device="cuda", port="22323")

    rank, gpu, world_size = ddp.ddp_setup_universal(args=args)

    assert (rank, gpu, world_size) == (0, 0, 1)
    assert args.ddp is False


def test_ddp_rejects_cpu_before_process_group_initialization():
    args = SimpleNamespace(ddp=True, device="cpu", port="22323")

    try:
        ddp.ddp_setup_universal(args=args)
    except ValueError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("CPU DDP must be rejected explicitly")
