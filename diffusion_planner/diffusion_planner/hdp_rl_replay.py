"""Disk-backed mine/replay cycle cache for the 100-epoch HDP-RL relay.

Reproduces the source repository's epoch state machine: every
``(epoch - 1) % rl_rollout_interval == 0`` epoch is a rollout-only mining pass
that freezes this cache (no optimizer steps), and the following
``interval - 1`` epochs train exclusively from it. Advantage weights and gate
validity are frozen at mine time and never recomputed during replay — the
upstream overlay-resurrection bug class is structurally impossible here.

Epochs here are the source repository's one-based numbering, the same one the
trainer prints as ``Epoch {epoch + 1}/{train_epochs}``. The training loop counts
from zero, so call sites convert with :func:`relay_epoch` rather than passing
their loop variable straight in — epoch 0 would otherwise land in cycle -1 and
look like a replay epoch whose cache was never mined.

Storage is one ``.pt`` shard per mined batch per rank on local NVMe, plus a
fail-closed ``meta.json`` contract (reward fingerprint, group size, commit
marker). A cycle directory missing its ``MINE_COMPLETE`` marker is discarded
and re-mined; a fingerprint mismatch aborts instead of silently training on a
cache mined under a different objective. Encodings are stored in float32
(bfloat16 cache storage was audited as lossy upstream and rejected).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

_MARKER = "MINE_COMPLETE"
_META = "meta.json"

_FINGERPRINT_FIELDS = (
    "rl_reward_aggregation",
    "rl_reward_horizon_steps",
    "rl_reward_beta",
    "rl_reward_w_safety",
    "rl_reward_w_risk",
    "rl_reward_w_follow",
    "rl_reward_w_lane",
    "rl_reward_w_progress",
    "rl_reward_w_road_border",
    "rl_behavior_gate",
    "rl_candidate_aug_std",
    "num_generations",
    "rl_noise_scale",
    "rl_rollout_steps",
)


# Fields that no longer exist but are frozen into caches mined before their removal.
# Dropping them outright would change every historical fingerprint and abort the
# resume of an otherwise-valid multi-terabyte cache, so each is pinned to the value it
# held in every run ever mined: `rl_first_waypoint_gate` False (32/32 args.json),
# `rl_reward_normalize` "group" (the only value ever launched, now the sole code path
# per ap:implementation), the two candidate-augmentation knobs that were superseded by
# `rl_candidate_aug_epochs` at their off defaults (0.0 in every args.json), and the two
# knobs of the ported hdp_pdm objective, which no run ever selected. The comparison is
# dict equality, so these need only reproduce the recorded values, not their position.
_FINGERPRINT_RETIRED = {
    "rl_first_waypoint_gate": repr(False),
    "rl_reward_normalize": repr("group"),
    "rl_candidate_aug_prob": repr(0.0),
    "rl_candidate_aug_stretch": repr(0.0),
    "rl_reward_source": repr("native"),
    "rl_pdm_red_light_gate": repr(True),
}

# Fields added after caches already existed. Recording one unconditionally would
# invalidate every historical fingerprint, so it is recorded only when it deviates from
# the value history implies -- which is exactly when the cache genuinely differs.
_FINGERPRINT_ADDED_DEFAULTS = {"rl_candidate_aug_epochs": 0}


def reward_fingerprint(args) -> dict:
    """The frozen-cache contract: everything that shapes mined groups/weights."""
    current = {name: repr(getattr(args, name, None)) for name in _FINGERPRINT_FIELDS}
    added = {
        name: repr(getattr(args, name, default))
        for name, default in _FINGERPRINT_ADDED_DEFAULTS.items()
        if getattr(args, name, default) != default
    }
    return {**current, **_FINGERPRINT_RETIRED, **added}


def relay_epoch(trainer_epoch: int) -> int:
    """Map the trainer's zero-based epoch onto this module's one-based numbering."""
    return int(trainer_epoch) + 1


def cycle_index(epoch: int, interval: int) -> int:
    return (epoch - 1) // interval


def is_mine_epoch(epoch: int, interval: int) -> bool:
    return (epoch - 1) % interval == 0


class CycleReplayWriter:
    """Per-rank shard writer for one mining epoch."""

    def __init__(self, root: str | Path, cycle: int, rank: int, args):
        self.dir = Path(root) / f"cycle_{cycle:03d}"
        self.rank = rank
        self.dir.mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._scenes = 0
        if rank == 0:
            marker = self.dir / _MARKER
            if marker.exists():
                raise RuntimeError(f"refusing to overwrite a completed replay cycle: {self.dir}")
            (self.dir / _META).write_text(
                json.dumps(
                    {
                        "cycle": cycle,
                        "group_size": int(args.num_generations),
                        "fingerprint": reward_fingerprint(args),
                    },
                    indent=2,
                )
            )

    def append(self, batch: dict[str, torch.Tensor]) -> None:
        required = {
            "ego_world",
            "reward",
            "reward_weights",
            "valid_sample",
            "ego_current_state",
            "route_lanes",
            "encoding",
        }
        missing = required - set(batch)
        if missing:
            raise ValueError(f"replay shard is missing tensors: {sorted(missing)}")
        payload = {
            key: value.detach().to("cpu", torch.float32 if value.is_floating_point() else None)
            for key, value in batch.items()
        }
        path = self.dir / f"rank{self.rank:02d}_shard{self._count:05d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.rename(path)
        self._count += 1
        self._scenes += int(batch["ego_current_state"].shape[0])

    def finalize(self, use_ddp: bool) -> None:
        if use_ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.rank == 0:
            (self.dir / _MARKER).write_text(f"scenes_rank0={self._scenes}\n")
        if use_ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


class CycleReplayReader:
    """Shuffled per-rank shard reader for the replay epochs of one cycle."""

    def __init__(self, root: str | Path, cycle: int, rank: int, args):
        self.dir = Path(root) / f"cycle_{cycle:03d}"
        marker = self.dir / _MARKER
        if not marker.exists():
            raise RuntimeError(f"replay cycle is incomplete (no {_MARKER}): {self.dir}")
        meta = json.loads((self.dir / _META).read_text())
        expected = reward_fingerprint(args)
        if meta["fingerprint"] != expected:
            drift = {
                key: (meta["fingerprint"].get(key), expected[key])
                for key in expected
                if meta["fingerprint"].get(key) != expected[key]
            }
            raise RuntimeError(
                f"replay cache contract mismatch in {self.dir}: {drift}. "
                "The cache was mined under a different objective; re-mine instead of "
                "silently training on it."
            )
        if int(meta["group_size"]) != int(args.num_generations):
            raise RuntimeError(
                f"replay cache group size {meta['group_size']} != {args.num_generations}"
            )
        self.shards = sorted(self.dir.glob(f"rank{rank:02d}_shard*.pt"))
        if not self.shards:
            raise RuntimeError(f"no replay shards for rank {rank} in {self.dir}")

    def epoch_shards(self, seed: int) -> list[Path]:
        order = torch.randperm(
            len(self.shards), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        return [self.shards[i] for i in order]

    @staticmethod
    def load(path: Path, device) -> dict[str, torch.Tensor]:
        batch = torch.load(path, map_location="cpu", weights_only=True)
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cycle_is_replayable(root: str | Path, cycle: int, rank: int, args) -> bool:
    """True when this cycle was already mined to completion under the same objective.

    Mining is the expensive half of the relay and it takes no optimizer steps, so a
    run that restarts onto a mine epoch should train from the finalized cache rather
    than pay for it twice — :class:`CycleReplayWriter` refuses to overwrite one in
    any case. Every condition :class:`CycleReplayReader` would raise on is checked
    here, including shards for this rank, so a cache mined by a different world size
    is reported as not replayable instead of failing mid-epoch.
    """
    directory = Path(root) / f"cycle_{cycle:03d}"
    if not (directory / _MARKER).exists():
        return False
    try:
        meta = json.loads((directory / _META).read_text())
    except (OSError, ValueError, KeyError):
        return False
    if meta.get("fingerprint") != reward_fingerprint(args):
        return False
    if int(meta.get("group_size", -1)) != int(args.num_generations):
        return False
    return any(directory.glob(f"rank{rank:02d}_shard*.pt"))


def cleanup_previous_cycle(root: str | Path, cycle: int, rank: int) -> None:
    """Bound disk use to one cycle: drop the finished cycle before mining the next."""
    if rank != 0 or cycle <= 0:
        return
    previous = Path(root) / f"cycle_{cycle - 1:03d}"
    if (previous / _MARKER).exists():
        shutil.rmtree(previous, ignore_errors=True)
