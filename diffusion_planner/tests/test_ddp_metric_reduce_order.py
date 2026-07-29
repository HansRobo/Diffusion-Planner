"""Cross-rank pairing for the epoch metric reducers.

Both reducers pack values positionally, so they are only correct if every rank
agrees on the key order. The rest of the suite monkeypatches them, which is how a
caller that built its dict by iterating a set literal reached production: under
torchrun ``PYTHONHASHSEED`` is randomized per process, so eight ranks produced
eight insertion orders and every replay epoch died in the key-digest check. These
tests run the real reducers on real gloo ranks with deliberately divergent order.
"""

import ast
import inspect
import textwrap

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusion_planner.utils.ddp import reduce_and_average_losses, reduce_scalar_metrics

from diffusion_planner import hdp_rl_epoch

KEYS = ("loss", "rl_loss", "reward_weighted_loss", "bc_loss")
TRUTH = {"loss": 1.0, "rl_loss": 2.0, "reward_weighted_loss": 3.0, "bc_loss": 4.0}
WORLD_SIZE = 4


def _rank_body(rank: int, world_size: int, init_file: str, errors) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # A rank-specific rotation stands in for per-process set iteration order.
        order = KEYS[rank % len(KEYS) :] + KEYS[: rank % len(KEYS)]
        built = {key: torch.tensor(TRUTH[key]) for key in order}
        if list(built) != list(order):
            errors.append(f"rank {rank}: dict did not honour insertion order")
            return

        averaged = reduce_and_average_losses(dict(built), torch.device("cpu"))
        summed = reduce_scalar_metrics(dict(built), torch.device("cpu"), dist.ReduceOp.SUM)

        # Averaging identical per-rank values must return them unchanged, and summing
        # must scale by the world size. Any mispairing shows up as a swapped value.
        for key, value in averaged.items():
            if abs(float(value) - TRUTH[key]) > 1e-9:
                errors.append(f"rank {rank}: {key} averaged to {value}, expected {TRUTH[key]}")
        for key, value in summed.items():
            if abs(float(value) - TRUTH[key] * world_size) > 1e-9:
                errors.append(f"rank {rank}: {key} summed to {value}")
    finally:
        dist.destroy_process_group()


def test_reducers_pair_values_with_their_own_keys_across_ranks(tmp_path):
    init_file = tmp_path / "gloo_rendezvous"
    with mp.Manager() as manager:
        errors = manager.list()
        mp.spawn(
            _rank_body,
            args=(WORLD_SIZE, str(init_file), errors),
            nprocs=WORLD_SIZE,
            join=True,
        )
        assert not list(errors), list(errors)


def test_reduced_metric_keys_are_never_ordered_by_a_set():
    """Guard the root cause, not just the reducer that reported it.

    Sorting inside the reducers makes divergent insertion order survivable, but a
    caller that iterates a set is still building a dict whose order is a function of
    ``PYTHONHASHSEED``. Keep the sources of these key lists ordered types so the two
    halves of the fix cannot drift apart.
    """
    source = inspect.getsource(hdp_rl_epoch._backward_reward_weighted_update)
    tree = ast.parse(textwrap.dedent(source))

    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.endswith("_keys"):
                assigned[target.id] = node.value

    assert set(assigned) >= {"objective_keys", "mean_keys"}, sorted(assigned)
    for name, value in assigned.items():
        assert not isinstance(value, (ast.Set, ast.SetComp)), (
            f"{name} is a set literal; the metric dict it fills would be ordered by "
            "PYTHONHASHSEED, which torchrun randomizes per rank"
        )
