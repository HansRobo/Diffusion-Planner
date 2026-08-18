import os
import subprocess
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import init_process_group

from diffusion_planner.utils.dist_init import dist_init_file_path


def ddp_setup_universal(verbose=False, args=None):
    if args.ddp == False:
        print(f"do not use ddp, train on GPU 0")
        return 0, 0, 1

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        os.environ["MASTER_PORT"] = str(getattr(args, "port", "29529"))
        os.environ["MASTER_ADDR"] = "localhost"
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_PORT"] = str(args.port)
        os.environ["MASTER_ADDR"] = addr
    else:
        print("Not using DDP mode")
        return 0, 0, 1

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(gpu)
    os.environ["RANK"] = str(rank)

    torch.cuda.set_device(gpu)
    dist_backend = "nccl"
    # I don't know why but this is needed for DDP to work instead of 'env://'
    dist_url = "file://"
    file_path = str(dist_init_file_path())
    print("| distributed init (rank {}): {}, gpu {}".format(rank, dist_url, gpu), flush=True)
    init_process_group(
        init_method=f"{dist_url}{file_path}",
        backend=dist_backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=10000),
    )
    torch.distributed.barrier()
    if verbose:
        setup_for_distributed(rank == 0)
    return rank, gpu, world_size


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_model(model, use_ddp):
    if use_ddp:
        return model.module
    else:
        return model


def print_from_every_rank(message):
    """Print from *every* rank, not just rank 0.

    ``setup_for_distributed`` replaces ``builtins.print`` with a wrapper that drops output
    on non-master ranks unless ``force=True`` is passed -- so a plain ``print`` of a
    per-rank diagnostic is silently lost on exactly the ranks that had something to say.
    Passing ``force=True`` is not an option either: the wrapper is only installed when DDP
    actually initializes, so on the single-process path it would hit the real ``print`` and
    raise TypeError. Writing to stdout directly is correct in both modes.
    """
    sys.stdout.write(str(message) + "\n")
    sys.stdout.flush()


def gather_objects(obj):
    """Gather a picklable object from every rank into a list, on every rank.

    Collective: must be called by every rank. Returns ``[obj]`` when DDP is not
    initialized, so callers work unchanged in single-process runs.
    """
    if not is_dist_avail_and_initialized():
        return [obj]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def all_reduce_sum(value, device):
    """Sum a python scalar across all DDP ranks.

    Collective: must be called by every rank. Returns ``value`` unchanged when DDP
    is not initialized (single-process run), so callers work in both modes.
    """
    if not is_dist_avail_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def all_reduce_min(value, device):
    """Minimum of a python scalar across all DDP ranks.

    Collective: must be called by every rank. Returns ``value`` unchanged when DDP
    is not initialized (single-process run).
    """
    if not is_dist_avail_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t.item()


def reduce_and_average_losses(loss_dict, device):
    """Combine per-rank epoch metrics into one global dict.

    Most keys are averaged, but two kinds must not be:

    * ``*_max`` (epoch maximum of a gradient norm) is reduced with MAX -- averaging maxima
      across ranks would hide the one rank that saw the spike, which defeats the purpose.
    * ``*_steps`` is a step count, so it is summed.

    Ranks can also end an epoch with *different* key sets (a rank that skipped every step
    on which the gradient stats were due contributes no ``grad/*`` entry). all_reduce is
    collective, so iterating each rank's own keys would deadlock on the mismatch; the union
    is agreed on first, in a deterministic order, and averages divide by the number of
    ranks that actually reported the key rather than by world_size.
    """
    torch.distributed.barrier()
    world_size = dist.get_world_size()

    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, sorted(loss_dict.keys()))
    keys = sorted({key for part in gathered for key in part})

    for key in keys:
        present = key in loss_dict
        raw = loss_dict[key] if present else 0.0
        value = float(raw.item() if torch.is_tensor(raw) else raw)

        if key.endswith("_max"):
            t = torch.tensor(
                [value if present else float("-inf")], dtype=torch.float64, device=device
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            loss_dict[key] = t.item()
        elif key.endswith("_steps"):
            t = torch.tensor([value], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_dict[key] = t.item()
        else:
            # [sum, number of ranks reporting] in one collective.
            t = torch.tensor([value, 1.0 if present else 0.0], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_dict[key] = t[0].item() / max(t[1].item(), 1.0)

    return loss_dict
