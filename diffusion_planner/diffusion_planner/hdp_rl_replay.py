"""Disk-backed mine/replay cycle cache for the 100-epoch HDP-RL relay.

Reproduces the source repository's epoch state machine: every
``(epoch - 1) % rl_rollout_interval == 0`` epoch is a rollout-only mining pass
that freezes this cache (no optimizer steps), and the following
``interval - 1`` epochs train exclusively from it. Advantage weights and gate
validity are frozen at mine time and never recomputed during replay — the
upstream overlay-resurrection bug class is structurally impossible here.

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
    "rl_reward_source",
    "rl_reward_aggregation",
    "rl_reward_horizon_steps",
    "rl_reward_normalize",
    "rl_reward_beta",
    "rl_reward_w_safety",
    "rl_reward_w_risk",
    "rl_reward_w_follow",
    "rl_reward_w_lane",
    "rl_reward_w_progress",
    "rl_reward_w_road_border",
    "rl_behavior_gate",
    "rl_pdm_red_light_gate",
    "rl_first_waypoint_gate",
    "rl_candidate_aug_prob",
    "rl_candidate_aug_std",
    "rl_candidate_aug_stretch",
    "num_generations",
    "rl_noise_scale",
    "rl_rollout_steps",
)


def reward_fingerprint(args) -> dict:
    """The frozen-cache contract: everything that shapes mined groups/weights."""
    return {name: repr(getattr(args, name, None)) for name in _FINGERPRINT_FIELDS}


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


def cleanup_previous_cycle(root: str | Path, cycle: int, rank: int) -> None:
    """Bound disk use to one cycle: drop the finished cycle before mining the next."""
    if rank != 0 or cycle <= 0:
        return
    previous = Path(root) / f"cycle_{cycle - 1:03d}"
    if (previous / _MARKER).exists():
        shutil.rmtree(previous, ignore_errors=True)
