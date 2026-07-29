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

"""Replay a run's ClusterWeightedDistributedSampler offline to recover repeat-draw counts.

The sampler draws WITH replacement, so samples repeat within an epoch, and
``--force_aug_on_repeat`` forces one pool augmentation onto every repeat draw. Training
does not log how many rows that was. It is exactly reconstructible: the draws depend only
on the cluster JSON, the data list, ``seed``, ``cluster_weight_alpha``, the DDP world size
and the epoch index. ``_compute_weights`` reads the cluster JSON and canonicalizes path
strings -- it opens no NPZ files -- so this replay is CPU-only, needs no GPU, and is safe
to run next to a live training job.

Numbers are GLOBAL (all DDP ranks), because the sampler assigns repeat ordinals on the
global draw list before sharding. The replay iterates the real sampler once per rank and
reassembles the global draw list from the shards, then cross-checks the reassembled list
against the per-rank flags -- a mismatch raises instead of printing plausible numbers.

Usage:
    # Recover the numbers for a finished or in-flight run
    python replay_repeats.py \\
        --run_dir /mnt/nvme/training_result/20260722-101500_my_exp \\
        --world_size 8 \\
        [--epochs 80] [--json /tmp/repeats.json]

    # Replay a hypothetical config (no run dir); any flag also overrides args.json
    python replay_repeats.py \\
        --cluster_json /path/to/clusters.json \\
        --train_set_list /path/to/train_list.json \\
        --train_subsample_step 1 \\
        --seed 3407 --cluster_weight_alpha 1.0 \\
        --world_size 8 --epochs 80 \\
        --repeat_aug_pool flip,neighbor_noise

``--world_size`` is required whenever the run dir does not record it: ``padded_length =
ceil(N / world_size) * world_size``, so guessing 1 would silently report wrong totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diffusion_planner.utils.forced_augmentation import AUG_ORDER, resolve_repeat_aug_pool
from diffusion_planner.utils.train_utils import openjson
from diffusion_planner.utils.weighted_sampler import ClusterWeightedDistributedSampler

# train.py's build_aug_pipeline() call site, mirrored: TrainConfig flag -> registry name.
# Only these five are wired into the training pipeline; "traffic_light" exists in
# AUG_ORDER but train.py never passes it, so it can never be in a run's pool.
USE_FLAG_TO_AUG: dict[str, str] = {
    "use_flip_augment": "flip",
    "use_neighbor_dropout": "neighbor_dropout",
    "use_neighbor_noise": "neighbor_noise",
    "use_turn_indicator_dropout": "turn_indicator",
    "use_data_augment": "state_perturbation",
}

# TrainConfig defaults, used only in no---run_dir mode. Every resolved value is echoed in
# the report header with its source, so a default is never silently load-bearing.
DEFAULTS: dict[str, object] = {
    "seed": 3407,
    "cluster_weight_alpha": 1.0,
    "train_subsample_step": 1,
    "augment_type": "quintic",
}

# Keys args.json may carry for the DDP world size. TrainConfig has no such field today
# (train_run.py passes --nproc-per-node=gpu_count() to torchrun and it is never
# persisted), so this normally resolves to nothing and --world_size is required. Listed
# anyway so the tool picks the value up for free if a field is ever added.
WORLD_SIZE_KEYS: tuple[str, ...] = (
    "world_size",
    "num_replicas",
    "nproc_per_node",
    "ddp_world_size",
)


class ReplayError(Exception):
    """A configuration problem that must stop the replay rather than skew the counts."""


def load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_dir(run_dir: str) -> tuple[dict, dict | None]:
    """Read ``args.json`` (required) and ``cluster_sampling.json`` (optional) from a run.

    ``args.json`` is ``vars(TrainConfig)`` dumped by train.py at startup on global rank 0.
    ``cluster_sampling.json`` is written next to it when ``--cluster_json`` is set and
    records the LIVE, post-subsample data list size plus the resolved forced-aug pool --
    the cross-check that catches a wrong ``train_set_list`` before it skews every count.
    """
    root = Path(run_dir)
    args_path = root / "args.json"
    if not args_path.is_file():
        raise ReplayError(
            f"{args_path} not found. --run_dir must point at a training run directory "
            f"(train.py writes args.json there at startup on global rank 0)."
        )
    run_args = load_json_file(args_path)
    cluster_sampling_path = root / "cluster_sampling.json"
    cluster_sampling = (
        load_json_file(cluster_sampling_path) if cluster_sampling_path.is_file() else None
    )
    return run_args, cluster_sampling


def build_data_list(train_set_list: str, train_subsample_step: int) -> list[str]:
    """Reproduce train.py's live data list exactly.

    train.py does ``DiffusionPlannerData(args.train_set_list)`` -- which is
    ``openjson(train_set_list)`` -- then ``data_list = data_list[:: train_subsample_step]``.
    Both steps matter: the sampler weights, the draw count and therefore every number in
    this report are computed on the sliced list.
    """
    if train_subsample_step < 1:
        raise ReplayError(f"--train_subsample_step must be >= 1, got {train_subsample_step}")
    data_list = openjson(train_set_list)
    if not isinstance(data_list, list):
        raise ReplayError(
            f"{train_set_list} must contain a JSON list of NPZ paths, got {type(data_list).__name__}"
        )
    return data_list[::train_subsample_step]


def enabled_aug_names(run_args: dict) -> list[str]:
    """Registry names train.py would have put in the pipeline, in AUG_ORDER."""
    enabled = {
        aug_name for flag, aug_name in USE_FLAG_TO_AUG.items() if bool(run_args.get(flag, False))
    }
    return [name for name in AUG_ORDER if name in enabled]


def resolve_pool(
    run_args: dict | None,
    pool_override: str | None,
    force_override: bool | None,
) -> tuple[list[str], list[str]]:
    """Resolve the forced-augmentation pool the run used (or a hypothetical one).

    With a run dir this defers to ``resolve_repeat_aug_pool``, the same function train.py
    validates with, so pool defaulting and the ``--augment_type bridge`` exclusion match
    the run. Without one there is no pipeline to resolve against, so an explicit
    ``--repeat_aug_pool`` is taken literally (names still validated).

    Returns ``(pool, warnings)``; an empty pool means forcing was off for this config.
    """
    warnings: list[str] = []
    requested = None if pool_override is None else [p.strip() for p in pool_override.split(",")]
    requested = None if requested is None else [p for p in requested if p]

    if run_args is None:
        if not requested:
            return [], []
        unknown = [name for name in requested if name not in AUG_ORDER]
        if unknown:
            raise ReplayError(
                f"--repeat_aug_pool names unknown augmentation(s) {unknown}; "
                f"valid names are {list(AUG_ORDER)}"
            )
        return [name for name in AUG_ORDER if name in set(requested)], warnings

    forced_on = (
        bool(run_args.get("force_aug_on_repeat", False))
        if force_override is None
        else bool(force_override)
    )
    if not forced_on:
        return [], warnings

    pipeline_names = enabled_aug_names(run_args)
    if bool(run_args.get("use_traffic_light_dropout", False)):
        warnings.append(
            "args.json sets use_traffic_light_dropout, but train.py does not wire "
            "traffic_light into the augmentation pipeline, so it cannot be in the pool."
        )
    # resolve_repeat_aug_pool only reads the names off the pipeline pairs, never the
    # augmentation objects, so placeholders reproduce its decisions without importing
    # torch modules that would want a device.
    pipeline = [(name, object()) for name in pipeline_names]
    pool_spec = run_args.get("repeat_aug_pool", "") if pool_override is None else pool_override
    augment_type = str(run_args.get("augment_type", DEFAULTS["augment_type"]))
    try:
        pool, pool_warnings = resolve_repeat_aug_pool(pool_spec or "", pipeline, augment_type)
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    return pool, warnings + pool_warnings


def replay_epochs(
    data_list: list[str],
    cluster_json: str,
    seed: int,
    alpha: float,
    world_size: int,
    start_epoch: int,
    epochs: int,
    pool_size: int,
    per_rank_batch: int | None = None,
) -> tuple[list[dict], dict, dict]:
    """Replay the sampler and return ``(per_epoch, total, sampler_info)``.

    One sampler is built (the cluster JSON is parsed once) and iterated once per rank with
    ``rank`` retargeted between passes. Every rank seeds an identical generator with
    ``seed + epoch`` and holds identical weights, so each pass regenerates the same global
    draw list and differs only in which stride it slices -- the shards therefore partition
    the global list and reassemble into it exactly.
    """
    sampler = ClusterWeightedDistributedSampler(
        data_list,
        cluster_json,
        num_replicas=world_size,
        rank=0,
        seed=seed,
        alpha=alpha,
        track_repeats=True,
    )
    padded_length = sampler.num_samples * world_size
    sampler_info = {
        "data_list_size": len(data_list),
        "num_samples_per_rank": sampler.num_samples,
        "padded_length": padded_length,
        "matched_count": sampler.matched_count,
        "num_clusters": len(sampler.cluster_counts),
    }

    per_epoch: list[dict] = []
    union: set[int] = set()
    for epoch in range(start_epoch, start_epoch + epochs):
        sampler.set_epoch(epoch)
        global_draws: list[int] = [-1] * padded_length
        global_flags: list[bool] = [False] * padded_length
        repeats = 0
        forced_rows = 0
        delivered_rows = 0
        for rank in range(world_size):
            sampler.rank = rank
            indices = list(sampler)
            flags = sampler.repeat_flags
            if flags is None or sampler.repeat_flags_epoch != epoch:
                raise ReplayError(
                    f"sampler produced no repeat_flags for epoch {epoch} "
                    f"(stamp={sampler.repeat_flags_epoch}); this build of "
                    f"weighted_sampler.py does not track repeats as expected."
                )
            if len(indices) != sampler.num_samples or len(flags) != len(indices):
                raise ReplayError(
                    f"rank {rank} shard is {len(indices)} indices / {len(flags)} flags, "
                    f"expected {sampler.num_samples} of each."
                )
            global_draws[rank:padded_length:world_size] = indices
            global_flags[rank:padded_length:world_size] = flags
            repeats += sum(flags)
            if per_rank_batch:
                # The training DataLoader is drop_last=True, so the tail of each rank's
                # shard that cannot fill a batch is never delivered and its repeat flags
                # are never consumed by ForcedAugmentationSelector.
                delivered = (len(indices) // per_rank_batch) * per_rank_batch
                delivered_rows += delivered
                forced_rows += sum(flags[:delivered])

        # Independent recomputation on the reassembled global list. If the shards did not
        # come from one shared draw list, or the interleaving is wrong, these disagree.
        seen: set[int] = set()
        expected_flags = []
        for idx in global_draws:
            expected_flags.append(idx in seen)
            seen.add(idx)
        if expected_flags != global_flags:
            raise ReplayError(
                f"epoch {epoch}: per-rank repeat flags do not match the flags recomputed "
                f"from the reassembled global draw list. The replay does not reproduce "
                f"this run's sampler; refusing to report counts."
            )
        distinct = len(seen)
        if distinct + repeats != padded_length:
            raise ReplayError(
                f"epoch {epoch}: distinct({distinct}) + repeats({repeats}) != "
                f"total draws({padded_length}); aggregation is inconsistent."
            )

        union |= seen
        row = {
            "epoch": epoch,
            "total_draws": padded_length,
            "distinct_samples": distinct,
            "repeat_draws": repeats,
            "repeat_fraction": repeats / padded_length if padded_length else 0.0,
            "expected_forced_per_pool_member": (repeats / pool_size) if pool_size else None,
        }
        if per_rank_batch:
            row["delivered_rows"] = delivered_rows
            row["dropped_rows"] = padded_length - delivered_rows
            row["forced_rows"] = forced_rows
            row["forced_per_pool_member"] = (forced_rows / pool_size) if pool_size else None
        per_epoch.append(row)

    total_draws = sum(r["total_draws"] for r in per_epoch)
    total_repeats = sum(r["repeat_draws"] for r in per_epoch)
    total: dict = {
        "epochs": len(per_epoch),
        "total_draws": total_draws,
        "distinct_samples_union": len(union),
        "repeat_draws": total_repeats,
        "repeat_fraction": (total_repeats / total_draws) if total_draws else 0.0,
        "expected_forced_per_pool_member": (total_repeats / pool_size) if pool_size else None,
    }
    if per_rank_batch:
        total_forced = sum(r["forced_rows"] for r in per_epoch)
        total["delivered_rows"] = sum(r["delivered_rows"] for r in per_epoch)
        total["dropped_rows"] = sum(r["dropped_rows"] for r in per_epoch)
        total["forced_rows"] = total_forced
        total["forced_per_pool_member"] = (total_forced / pool_size) if pool_size else None
    return per_epoch, total, sampler_info


def format_report(result: dict) -> str:
    """Render the stdout report. ``result`` is the same dict written by --json."""
    config = result["config"]
    info = result["sampler"]
    pool = config["repeat_aug_pool"]
    has_rows = "forced_rows" in result["total"]

    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("Offline repeat-draw replay (ClusterWeightedDistributedSampler)")
    lines.append("=" * 96)
    if config.get("run_dir"):
        lines.append(f"  run_dir              : {config['run_dir']}")
    for key in (
        "train_set_list",
        "cluster_json",
        "train_subsample_step",
        "seed",
        "cluster_weight_alpha",
        "world_size",
        "batch_size",
        "start_epoch",
        "epochs",
    ):
        if config.get(key) is None:
            continue
        lines.append(f"  {key:<21}: {config[key]}  [{config['sources'].get(key, 'flag')}]")
    lines.append(
        f"  data_list_size       : {info['data_list_size']:,} "
        f"(after [::{config['train_subsample_step']}]);  "
        f"cluster-matched {info['matched_count']:,} in {info['num_clusters']} clusters"
    )
    lines.append(
        f"  draws per epoch      : {info['padded_length']:,} "
        f"= ceil({info['data_list_size']:,}/{config['world_size']}) "
        f"x {config['world_size']} = {info['num_samples_per_rank']:,} per rank"
    )
    lines.append(
        f"  repeat_aug_pool      : {', '.join(pool) if pool else '(forcing DISABLED)'}"
        f"{f'  ({len(pool)} members)' if pool else ''}"
    )
    if has_rows:
        lines.append(
            f"  delivered rows/epoch : {result['per_epoch'][0]['delivered_rows']:,} "
            f"of {info['padded_length']:,} "
            f"({result['per_epoch'][0]['dropped_rows']:,} dropped by DataLoader "
            f"drop_last=True, per-rank batch {config['per_rank_batch']})"
        )
    for message in result.get("warnings", []):
        lines.append(f"  WARNING: {message}")
    lines.append("")

    head = f"{'epoch':>6} {'total_draws':>13} {'distinct':>12} {'repeats':>12} {'repeat_frac':>12}"
    if has_rows:
        head += f" {'forced_rows':>13} {'per_pool_member':>16}"
    else:
        head += f" {'per_pool_member':>16}"
    lines.append(head)
    lines.append("-" * len(head))
    for row in result["per_epoch"]:
        line = (
            f"{row['epoch']:>6} {row['total_draws']:>13,} {row['distinct_samples']:>12,} "
            f"{row['repeat_draws']:>12,} {row['repeat_fraction']:>12.4f}"
        )
        if has_rows:
            member = row["forced_per_pool_member"]
            line += f" {row['forced_rows']:>13,}"
            line += f" {member:>16,.1f}" if member is not None else f" {'-':>16}"
        else:
            member = row["expected_forced_per_pool_member"]
            line += f" {member:>16,.1f}" if member is not None else f" {'-':>16}"
        lines.append(line)
    lines.append("-" * len(head))

    total = result["total"]
    line = (
        f"{'TOTAL':>6} {total['total_draws']:>13,} {total['distinct_samples_union']:>12,} "
        f"{total['repeat_draws']:>12,} {total['repeat_fraction']:>12.4f}"
    )
    if has_rows:
        member = total["forced_per_pool_member"]
        line += f" {total['forced_rows']:>13,}"
        line += f" {member:>16,.1f}" if member is not None else f" {'-':>16}"
    else:
        member = total["expected_forced_per_pool_member"]
        line += f" {member:>16,.1f}" if member is not None else f" {'-':>16}"
    lines.append(line)
    lines.append("")
    lines.append(
        f"  distinct column is per-epoch; TOTAL distinct is the UNION over "
        f"{total['epochs']} epoch(s), not a sum."
    )
    if has_rows:
        lines.append(
            "  forced_rows is the count that actually got force-augmented: repeat draws "
            "among the\n  rows the DataLoader delivered (drop_last=True discards each "
            "rank's partial tail batch)."
        )
        lines.append(
            f"  repeats not forced (dropped tail): {total['repeat_draws'] - total['forced_rows']:,}"
        )
    else:
        lines.append(
            "  No batch_size resolved, so drop_last is not modelled: repeats is an UPPER "
            "bound on\n  forced rows. Pass --batch_size to get the delivered-row count."
        )
    if pool:
        lines.append(
            f"  per_pool_member = rows / {len(pool)}: the selector picks one pool member "
            f"uniformly per\n  forced row, so each member's expected share is an equal split."
        )
    lines.append("=" * 96)
    return "\n".join(lines)


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a training run's cluster-weighted sampler offline and report "
        "per-epoch repeat-draw / forced-augmentation counts (CPU only, opens no NPZ files)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Training run directory containing args.json (and cluster_sampling.json). "
        "Any flag below overrides the value read from it.",
    )
    parser.add_argument("--cluster_json", type=str, default=None, help="Cluster assignment JSON")
    parser.add_argument("--train_set_list", type=str, default=None, help="JSON list of NPZ paths")
    parser.add_argument(
        "--train_subsample_step",
        type=int,
        default=None,
        help="Stride applied as data_list[::step], exactly as train.py does",
    )
    parser.add_argument("--seed", type=int, default=None, help="Run seed (sampler uses seed+epoch)")
    parser.add_argument(
        "--cluster_weight_alpha",
        type=float,
        default=None,
        help="Inverse-frequency exponent in [0, 1]",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="DDP world size of the original run. REQUIRED unless args.json records it: "
        "padded draws = ceil(N / world_size) * world_size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="GLOBAL batch size, to model the training DataLoader's drop_last=True and "
        "report the rows that actually got forced. Read from args.json when available.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to replay (default: train_epochs from args.json)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="First epoch index, matching train.py's range(init_epoch, train_epochs). "
        "Set this for a resumed run (default: 0).",
    )
    parser.add_argument(
        "--repeat_aug_pool",
        type=str,
        default=None,
        help="Comma-separated pool override. With --run_dir it is resolved exactly as "
        "train.py resolves it; without one it is taken literally.",
    )
    parser.add_argument(
        "--force_aug_on_repeat",
        action="store_true",
        default=None,
        help="Treat forcing as enabled even if args.json says it was off (hypothetical replay)",
    )
    parser.add_argument(
        "--allow_data_list_mismatch",
        action="store_true",
        help="Downgrade the cluster_sampling.json data-list-size cross-check to a warning",
    )
    parser.add_argument("--json", type=str, default=None, help="Write results to this JSON path")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> tuple[dict, dict | None, list[str]]:
    """Merge args.json with explicit flags. Returns ``(config, cluster_sampling, warnings)``."""
    run_args: dict | None = None
    cluster_sampling: dict | None = None
    if args.run_dir:
        run_args, cluster_sampling = load_run_dir(args.run_dir)

    sources: dict[str, str] = {}
    warnings: list[str] = []

    def pick(name: str, json_key: str | None = None, required: bool = True):
        json_key = json_key or name
        value = getattr(args, name)
        if value is not None:
            sources[name] = "flag"
            return value
        if run_args is not None and run_args.get(json_key) is not None:
            sources[name] = "args.json"
            return run_args[json_key]
        if name in DEFAULTS:
            sources[name] = "default"
            return DEFAULTS[name]
        if required:
            raise ReplayError(
                f"--{name} is required (not given, and not recoverable from "
                f"{'args.json' if run_args is not None else 'any run dir'})."
            )
        sources[name] = "unset"
        return None

    config: dict = {
        "run_dir": args.run_dir,
        "cluster_json": pick("cluster_json"),
        "train_set_list": pick("train_set_list"),
        "train_subsample_step": int(pick("train_subsample_step")),
        "seed": int(pick("seed")),
        "cluster_weight_alpha": float(pick("cluster_weight_alpha")),
        "epochs": pick("epochs", json_key="train_epochs"),
        "start_epoch": args.start_epoch,
        "batch_size": pick("batch_size", required=False),
    }
    if not config["cluster_json"]:
        raise ReplayError(
            "cluster_json is empty. Without --cluster_json the run used a plain "
            "DistributedSampler, which draws WITHOUT replacement: no sample repeats "
            "within an epoch, so there is nothing to replay."
        )
    config["epochs"] = int(config["epochs"])
    if config["epochs"] < 1:
        raise ReplayError(f"--epochs must be >= 1, got {config['epochs']}")

    world_size = args.world_size
    if world_size is not None:
        sources["world_size"] = "flag"
    elif run_args is not None:
        for key in WORLD_SIZE_KEYS:
            if run_args.get(key) is not None:
                world_size, sources["world_size"] = int(run_args[key]), f"args.json[{key}]"
                break
    if world_size is None:
        raise ReplayError(
            "--world_size is required: the run directory does not record the DDP world "
            "size (TrainConfig has no such field, so args.json cannot carry it), and "
            "guessing 1 would silently report wrong totals -- padded draws per epoch are "
            "ceil(N / world_size) * world_size.\n"
            "Recover it from the run's log, e.g.:\n"
            "  grep -o 'global_rank=[0-9]*' <run_dir>/train_log.txt | sort -u | wc -l\n"
            "(train.py prints 'global_rank=..., rank=...' once per rank at startup.)"
        )
    if world_size < 1:
        raise ReplayError(f"--world_size must be >= 1, got {world_size}")
    config["world_size"] = world_size

    per_rank_batch = None
    if config["batch_size"] is not None:
        config["batch_size"] = int(config["batch_size"])
        # train.py: DataLoader(batch_size=batch_size // world_size, drop_last=True)
        per_rank_batch = config["batch_size"] // world_size
        if per_rank_batch < 1:
            warnings.append(
                f"batch_size {config['batch_size']} // world_size {world_size} == 0, so "
                f"drop_last cannot be modelled; reporting draw-level counts only."
            )
            per_rank_batch = None
    config["per_rank_batch"] = per_rank_batch
    config["sources"] = sources

    pool, pool_warnings = resolve_pool(run_args, args.repeat_aug_pool, args.force_aug_on_repeat)
    config["repeat_aug_pool"] = pool
    warnings.extend(pool_warnings)
    if not pool:
        warnings.append(
            "Forced augmentation resolves to an empty pool (force_aug_on_repeat off, or "
            "no pool given without --run_dir): repeat counts below are reported, but no "
            "rows were force-augmented."
        )
    return config, cluster_sampling, warnings


def cross_check(config: dict, cluster_sampling: dict | None, data_list_size: int) -> list[str]:
    """Compare the reconstruction against cluster_sampling.json. Returns warnings."""
    warnings: list[str] = []
    if cluster_sampling is None:
        if config.get("run_dir"):
            warnings.append(
                "cluster_sampling.json not found in the run dir, so the reconstructed "
                "data list size cannot be cross-checked against what training used."
            )
        return warnings
    recorded = cluster_sampling.get("data_list_size")
    if recorded is not None and int(recorded) != data_list_size:
        raise ReplayError(
            f"reconstructed data list has {data_list_size:,} entries but the run's "
            f"cluster_sampling.json records {int(recorded):,}. Every count would be wrong. "
            f"Check the inputs: train_set_list={config['train_set_list']!r}, "
            f"train_subsample_step={config['train_subsample_step']} "
            f"(the run recorded train_subsample_step="
            f"{cluster_sampling.get('train_subsample_step')}). The path list itself may "
            f"also have been regenerated since the run started. Re-run with "
            f"--allow_data_list_mismatch to report anyway."
        )
    for key, recorded_key in (
        ("cluster_json", "cluster_json"),
        ("cluster_weight_alpha", "cluster_weight_alpha"),
        ("train_subsample_step", "train_subsample_step"),
    ):
        recorded_value = cluster_sampling.get(recorded_key)
        if recorded_value is not None and recorded_value != config[key]:
            warnings.append(
                f"{key}={config[key]!r} differs from cluster_sampling.json's "
                f"{recorded_value!r} (replaying a hypothetical config)."
            )
    recorded_pool = (cluster_sampling.get("forced_augmentation") or {}).get("pool")
    if recorded_pool is not None and list(recorded_pool) != list(config["repeat_aug_pool"]):
        warnings.append(
            f"repeat_aug_pool {config['repeat_aug_pool']} differs from "
            f"cluster_sampling.json's {list(recorded_pool)}; per-member numbers reflect "
            f"the pool used here."
        )
    return warnings


def run_from_args(args: argparse.Namespace) -> dict:
    """Full replay. Returns the result dict; raises ReplayError on bad configuration."""
    config, cluster_sampling, warnings = resolve_config(args)

    data_list = build_data_list(config["train_set_list"], config["train_subsample_step"])
    if not data_list:
        raise ReplayError(
            f"data list is empty after [::{config['train_subsample_step']}] "
            f"({config['train_set_list']})."
        )
    try:
        warnings += cross_check(config, cluster_sampling, len(data_list))
    except ReplayError as exc:
        if not args.allow_data_list_mismatch:
            raise
        warnings.append(f"data-list cross-check FAILED (ignored): {exc}")

    per_epoch, total, sampler_info = replay_epochs(
        data_list=data_list,
        cluster_json=config["cluster_json"],
        seed=config["seed"],
        alpha=config["cluster_weight_alpha"],
        world_size=config["world_size"],
        start_epoch=config["start_epoch"],
        epochs=config["epochs"],
        pool_size=len(config["repeat_aug_pool"]),
        per_rank_batch=config["per_rank_batch"],
    )
    return {
        "config": config,
        "sampler": sampler_info,
        "warnings": warnings,
        "per_epoch": per_epoch,
        "total": total,
    }


def run(argv: list[str] | None = None) -> dict:
    """Parse ``argv`` and replay. Convenience entry point for tests and importers."""
    return run_from_args(get_args(argv))


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    try:
        result = run_from_args(args)
    except ReplayError as exc:
        # SystemExit rather than a traceback: every ReplayError is an operator-fixable
        # configuration problem, and a stack trace buries the instruction.
        raise SystemExit(f"replay_repeats: {exc}") from exc

    print(format_report(result))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a truncating open leaves a half-written JSON behind if this
        # dies mid-dump, and the file is meant to be machine-read.
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        tmp.replace(out)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
