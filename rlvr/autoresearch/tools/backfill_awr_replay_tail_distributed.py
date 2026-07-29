#!/usr/bin/env python3
"""Mine only the verified missing tails and append them to an HDP replay cache.

Run as ``torchrun --nproc_per_node=8 -m
rlvr.autoresearch.tools.backfill_awr_replay_tail_distributed`` with the repo
root and ``diffusion_planner/`` on ``PYTHONPATH``, after the reader/training
process has exited.  Each worker owns one rank-local preallocated cache, resumes at the
published path prefix, recursively isolates a failing packed batch, and only
publishes a full manifest when every expected slot has been written.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlvr.train_awr import (
    _REPLAY_CONTEXT_KEYS,
    _REPLAY_DECODER_CONTEXT_SCHEMA,
    _REPLAY_EXPERT_SCHEMA,
    _compile_planner_modules,
    _configure_cuda_runtime,
    _DiskReplayWriter,
    _file_sha256,
    _load_config,
    _load_scene_batch,
    _train_scene,
    _validate_replay_source_contract,
    load_original_dp_checkpoint,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--args_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay_root", type=Path, required=True)
    parser.add_argument("--plan_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--scene_load_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    # See eval_awr_full_distributed.py: the default is the vehicle's precision, since
    # this path MINES the replay tail and bf16 collapses the candidate ordering.
    parser.add_argument("--amp_dtype", choices=["off", "bf16", "fp16"], default="off")
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually append in place; without this flag only validate the prefix contract",
    )
    return parser.parse_args()


def _active_training_uses(root: Path) -> list[tuple[int, str]]:
    needle = str(root.resolve())
    active: list[tuple[int, str]] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "rlvr.train_awr" in command and needle in command:
            active.append((int(entry.name), command))
    return active


def _paths(path: Path) -> list[str]:
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"invalid path list: {path}")
    return values


def _read_jsonl(path: Path) -> list[str]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _open_resume_writer(
    replay_root: Path,
    rank: int,
    expected_count: int,
    expected_prefix: int,
    writable: bool,
) -> _DiskReplayWriter:
    rank_dir = replay_root / f"rank_{rank:04d}"
    manifest_path = rank_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    writer_contract_path = rank_dir / "writer_contract.json"
    writer_contract = (
        json.loads(writer_contract_path.read_text())
        if writer_contract_path.is_file()
        else {}
    )
    paths_path = rank_dir / str(manifest["paths"])
    existing_paths = _read_jsonl(paths_path)
    if len(existing_paths) != expected_prefix:
        raise RuntimeError(
            f"rank {rank}: path prefix changed since plan: {len(existing_paths)} != {expected_prefix}"
        )
    expected_paths_name = manifest.get("expected_paths")
    if not expected_paths_name:
        raise RuntimeError(
            f"rank {rank}: cache has no canonical expected-path index"
        )
    expected_paths_path = rank_dir / str(expected_paths_name)
    expected_paths = _read_jsonl(expected_paths_path)
    if len(expected_paths) != expected_count:
        raise RuntimeError(
            f"rank {rank}: expected-path count {len(expected_paths)} != {expected_count}"
        )
    if existing_paths != expected_paths[:expected_prefix]:
        raise RuntimeError(
            f"rank {rank}: published paths are not the canonical prefix"
        )

    arrays: dict[str, np.memmap] = {}
    expert_arrays: dict[str, np.memmap] = {}
    shared_scene_encoding_path = manifest.get("shared_scene_encoding_path")
    if shared_scene_encoding_path is not None:
        shared_scene_encoding_path = Path(shared_scene_encoding_path).resolve()
        if not shared_scene_encoding_path.is_file():
            raise FileNotFoundError(shared_scene_encoding_path)
    shared_decoder_context_paths = {
        str(key): Path(path).resolve()
        for key, path in manifest.get("shared_decoder_context_paths", {}).items()
    }
    if shared_decoder_context_paths and not bool(
        manifest.get("decoder_context_cached", False)
    ):
        raise RuntimeError(
            f"rank {rank}: shared decoder paths declared without cached context"
        )
    if shared_decoder_context_paths and set(shared_decoder_context_paths) != set(
        _REPLAY_CONTEXT_KEYS
    ):
        raise RuntimeError(
            f"rank {rank}: shared decoder context is incomplete: "
            f"{sorted(shared_decoder_context_paths)}"
        )
    for key, source_path in shared_decoder_context_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    array_names = [
        "trajectories", "noise_scales", "weights", "rewards", "scene_encoding"
    ]
    context_names = [f"context_{key}" for key in _REPLAY_CONTEXT_KEYS]
    if bool(manifest.get("decoder_context_cached", False)):
        if manifest.get("decoder_context_schema") != _REPLAY_DECODER_CONTEXT_SCHEMA:
            raise RuntimeError(
                f"rank {rank}: unsafe/unknown decoder-context alignment schema"
            )
        missing_context = [
            name for name in context_names if name not in manifest.get("arrays", {})
        ]
        if missing_context:
            raise RuntimeError(
                f"rank {rank}: manifest decoder context is incomplete: {missing_context}"
            )
        array_names.extend(context_names)
    for name in array_names:
        filename = manifest.get("arrays", {}).get(name, f"{name}.npy")
        path = rank_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        context_key = (
            name.removeprefix("context_")
            if name.startswith("context_")
            else None
        )
        if context_key in shared_decoder_context_paths and (
            path.resolve() != shared_decoder_context_paths[context_key]
        ):
            raise RuntimeError(
                f"rank {rank}: shared decoder context link {name} points to "
                f"{path.resolve()}, expected "
                f"{shared_decoder_context_paths[context_key]}"
            )
        mmap_mode = (
            "r"
            if (
                name == "scene_encoding"
                and shared_scene_encoding_path is not None
            )
            or context_key in shared_decoder_context_paths
            else ("r+" if writable else "r")
        )
        arrays[name] = np.load(path, mmap_mode=mmap_mode)
    expert_manifest_path = rank_dir / "expert_anchor_manifest.json"
    expert_manifest = (
        json.loads(expert_manifest_path.read_text())
        if expert_manifest_path.is_file()
        else None
    )
    expert_expected = bool(
        expert_manifest is not None
        or writer_contract.get("expert_anchor_cached", False)
    )
    if expert_expected:
        expert_schema = (
            expert_manifest.get("schema")
            if expert_manifest is not None
            else writer_contract.get("expert_anchor_schema")
        )
        if expert_schema != _REPLAY_EXPERT_SCHEMA:
            raise RuntimeError(
                f"rank {rank}: unsafe/unknown expert sidecar schema"
            )
        if expert_manifest is not None and int(
            expert_manifest.get("future_len", -1)
        ) != int(arrays["trajectories"].shape[2]):
            raise RuntimeError(
                f"rank {rank}: expert sidecar future length mismatch"
            )
        expected_expert_names = {
            "expert_trajectories",
            "expert_rewards",
            "expert_safe",
        }
        expert_names = (
            expert_manifest.get("arrays", {})
            if expert_manifest is not None
            else {name: f"{name}.npy" for name in expected_expert_names}
        )
        if set(expert_names) != expected_expert_names:
            raise RuntimeError(
                f"rank {rank}: expert sidecar is incomplete: "
                f"{sorted(expert_names)}"
            )
        for name in sorted(expected_expert_names):
            path = rank_dir / str(expert_names[name])
            if not path.is_file():
                raise FileNotFoundError(path)
            expert_arrays[name] = np.load(
                path, mmap_mode=("r+" if writable else "r")
            )
        expected_expert_shapes = {
            "expert_trajectories": (
                expected_count,
                int(arrays["trajectories"].shape[2]),
                4,
            ),
            "expert_rewards": (expected_count,),
            "expert_safe": (expected_count,),
        }
        for name, expected_shape in expected_expert_shapes.items():
            array = expert_arrays[name]
            if array.dtype != np.float32 or tuple(array.shape) != expected_shape:
                raise RuntimeError(
                    f"rank {rank}: invalid expert sidecar {name}: "
                    f"dtype={array.dtype}, shape={tuple(array.shape)}, "
                    f"expected float32 {expected_shape}"
                )
            if expected_prefix and not np.isfinite(
                np.asarray(array[[0, expected_prefix - 1]])
            ).all():
                raise RuntimeError(
                    f"rank {rank}: non-finite {name} in published prefix"
                )
    owned_capacities = {
        int(array.shape[0])
        for name, array in arrays.items()
        if not (
            name == "scene_encoding"
            and shared_scene_encoding_path is not None
        )
    }
    shared_encoding_capacity = int(arrays["scene_encoding"].shape[0])
    if owned_capacities != {expected_count} or shared_encoding_capacity < expected_count:
        raise RuntimeError(
            f"rank {rank}: array capacity mismatch owned={owned_capacities}, "
            f"shared_encoding={shared_encoding_capacity}, "
            f"expected={expected_count}"
        )
    group_size = int(arrays["trajectories"].shape[1])
    future_len = int(arrays["trajectories"].shape[2])
    if tuple(arrays["trajectories"].shape[3:]) != (4,):
        raise RuntimeError(f"rank {rank}: invalid trajectory shape {arrays['trajectories'].shape}")
    if tuple(arrays["scene_encoding"].shape[1:]) != tuple(manifest["encoding_shape"]):
        raise RuntimeError("scene encoding shape disagrees with manifest")
    if expected_prefix:
        probe = np.asarray([0, expected_prefix - 1], dtype=np.int64)
        for name, array in arrays.items():
            values = np.asarray(array[probe])
            if not np.isfinite(values).all():
                raise RuntimeError(f"rank {rank}: non-finite {name} in published prefix")

    # Reuse the production writer's append/close implementation, but attach
    # it to existing r+ memmaps rather than calling __init__ (which truncates).
    writer = _DiskReplayWriter.__new__(_DiskReplayWriter)
    writer.root = replay_root
    writer.rank = rank
    writer.rank_dir = rank_dir
    writer.expected_count = expected_count
    writer.group_size = group_size
    writer.future_len = future_len
    writer.cache_decoder_context = bool(
        manifest.get("decoder_context_cached", False)
    )
    writer.cache_expert_anchor = bool(expert_arrays)
    writer.shared_scene_encoding_path = shared_scene_encoding_path
    writer.shared_scene_encoding_source_count = (
        shared_encoding_capacity
        if shared_scene_encoding_path is not None
        else None
    )
    writer.shared_decoder_context_paths = shared_decoder_context_paths
    writer.shared_decoder_context_source_counts = {
        key: int(arrays[f"context_{key}"].shape[0])
        for key in shared_decoder_context_paths
    }
    writer.expected_scene_paths = expected_paths
    writer._expected_paths_path = expected_paths_path
    writer.position = expected_prefix
    writer._encoding_shape = tuple(int(x) for x in arrays["scene_encoding"].shape[1:])
    writer._context_shapes = {
        key: tuple(int(x) for x in arrays[f"context_{key}"].shape[1:])
        for key in _REPLAY_CONTEXT_KEYS
        if f"context_{key}" in arrays
    }
    writer._arrays = arrays
    writer._expert_arrays = expert_arrays
    writer._paths_path = paths_path
    return writer


def _mine_and_append(
    writer: _DiskReplayWriter,
    paths: list[str],
    model: torch.nn.Module,
    model_args: Any,
    rollout_config: Any,
    reward_config: Any,
    train_config: dict[str, Any],
    device: torch.device,
    amp_dtype: str,
    scene_load_workers: int,
    failures_file: Path,
    exploration_reference_model: torch.nn.Module | None = None,
    exploration_reference_model_args: Any | None = None,
) -> None:
    """Append in order, recursively bisecting only a failing packed batch."""
    try:
        data_override = None
        if writer.shared_scene_encoding_path is not None:
            data_override = _load_scene_batch(
                paths,
                torch.device("cpu"),
                workers=max(1, int(scene_load_workers)),
            )
            start = int(writer.position)
            end = start + len(paths)
            encoding = np.array(
                writer._arrays["scene_encoding"][start:end],
                dtype=np.float32,
                copy=True,
            )
            if int(encoding.shape[0]) != len(paths):
                raise RuntimeError(
                    "shared tail encoding is shorter than the canonical path batch"
                )
            data_override["_cached_encoding"] = torch.from_numpy(encoding)
        row, rollout = _train_scene(
            model,
            model,
            model_args,
            None,
            paths,
            rollout_config,
            reward_config,
            train_config,
            device,
            amp_dtype,
            collect_only=True,
            scene_load_workers=scene_load_workers,
            data_override=data_override,
            exploration_reference_model=exploration_reference_model,
            exploration_reference_model_args=exploration_reference_model_args,
        )
        if rollout is None:
            raise RuntimeError("collect-only mining returned no rollout")
        if writer.shared_scene_encoding_path is not None and float(
            row.get("shared_encoder_cache_hit", 0.0)
        ) < 0.5:
            raise RuntimeError("shared tail mining did not use its cached encoding")
        writer.append(paths, rollout, group_size=writer.group_size)
        return
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as error:
        if len(paths) == 1:
            record = {
                "scene_path": paths[0],
                "writer_position": writer.position,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures_file.parent.mkdir(parents=True, exist_ok=True)
            with failures_file.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            raise RuntimeError(f"unrecoverable single-scene tail failure: {record}") from error
        middle = len(paths) // 2
        _mine_and_append(
            writer, paths[:middle], model, model_args, rollout_config, reward_config,
            train_config, device, amp_dtype, scene_load_workers, failures_file,
            exploration_reference_model, exploration_reference_model_args,
        )
        _mine_and_append(
            writer, paths[middle:], model, model_args, rollout_config, reward_config,
            train_config, device, amp_dtype, scene_load_workers, failures_file,
            exploration_reference_model, exploration_reference_model_args,
        )


def main() -> None:
    args = _args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    plan = json.loads((args.plan_dir / "manifest.json").read_text())
    if not plan.get("prefix_contract_verified"):
        raise RuntimeError("tail plan has no verified prefix contract")
    if world_size != int(plan["world_size"]):
        raise RuntimeError(f"world_size {world_size} != plan {plan['world_size']}")
    rank_plan = next(item for item in plan["ranks"] if int(item["rank"]) == rank)
    missing = _paths(args.plan_dir / str(rank_plan["missing_all_file"]))
    expected_count = int(rank_plan["expected_padded_count"])
    prefix_count = int(rank_plan["cached_prefix_count"])
    if len(missing) != int(rank_plan["missing_total_count"]):
        raise RuntimeError(f"rank {rank}: missing list count changed")

    # Six or seven ranks are commonly already strict-complete when one rank
    # encounters a malformed tail scene.  Leave those committed generations
    # byte-for-byte untouched and avoid loading/compiling two planners on GPUs
    # that have no work.  The campaign continuation performs the full
    # cross-rank array/sidecar validation after the missing ranks commit.
    if not missing:
        rank_dir = args.replay_root.resolve() / f"rank_{rank:04d}"
        manifest = json.loads((rank_dir / "manifest.json").read_text())
        expert_manifest = json.loads(
            (rank_dir / "expert_anchor_manifest.json").read_text()
        )
        if (
            int(manifest.get("scene_count", -1)) != expected_count
            or int(expert_manifest.get("scene_count", -1)) != expected_count
        ):
            raise RuntimeError(
                f"rank {rank}: zero-tail plan points at a non-complete manifest"
            )
        print(
            f"rank {rank}: strict-complete prefix={prefix_count:,}; no-op",
            flush=True,
        )
        return

    source_run = args.replay_root.resolve().parent
    provenance_path = source_run / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    source_provenance = json.loads(provenance_path.read_text())
    if bool(source_provenance.get("use_policy_state", False)):
        raise RuntimeError(
            "tail backfill currently supports only EMA-mined replay caches"
        )
    expected_model_sha = str(source_provenance.get("staged_model_sha256", ""))
    actual_model_sha = _file_sha256(args.model_path.resolve())
    if not expected_model_sha or actual_model_sha != expected_model_sha:
        raise RuntimeError(
            "tail model checkpoint differs from replay provenance: "
            f"actual={actual_model_sha}, expected={expected_model_sha}"
        )
    source_args = Path(source_provenance["staged_args"]).resolve()
    if _file_sha256(args.args_path.resolve()) != _file_sha256(source_args):
        raise RuntimeError("tail model args differ from replay provenance")

    if args.execute:
        active = _active_training_uses(args.replay_root)
        if active:
            raise RuntimeError(
                "refusing in-place backfill while a train_awr process uses this replay root: "
                + "; ".join(f"pid={pid} {command[:240]}" for pid, command in active)
            )

    writer = _open_resume_writer(
        args.replay_root.resolve(), rank, expected_count, prefix_count, args.execute,
    )
    print(
        f"rank {rank}: verified prefix={prefix_count:,} missing={len(missing):,} "
        f"capacity={expected_count:,} execute={args.execute}",
        flush=True,
    )
    if not args.execute:
        return

    backup = writer.rank_dir / "manifest.pre_tail_backfill.json"
    if not backup.exists():
        shutil.copy2(writer.rank_dir / "manifest.json", backup)
    failures_file = args.plan_dir / f"rank_{rank:04d}_failures.jsonl"
    failures_file.unlink(missing_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    _configure_cuda_runtime(device)
    random.seed(args.seed + 1000003 + rank)
    np.random.seed(args.seed + 1000003 + rank)
    torch.manual_seed(args.seed + 1000003 + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 1000003 + rank)

    raw_config, rollout_config, reward_config = _load_config(args.config)
    rollout_config.inference_amp_dtype = args.amp_dtype
    _validate_replay_source_contract(
        args.replay_root.resolve(),
        rollout_config,
        reward_config,
        allow_missing_defaults={
            # These candidate-generation choices were constants in the
            # Cycle-1 implementation and became explicit persisted config
            # fields afterward.  Tail recovery may accept only those exact
            # historical defaults; every other replay/reward field remains
            # fail-closed.
            "awr": {
                "plannerrft_reference_mode": "zero_noise",
                "plannerrft_reference_noise_scale": 0.5,
                "plannerrft_longitudinal_mode": "candidate_stretch",
            }
        },
    )
    train_config = dict(raw_config.get("training", raw_config))
    # The existing writer contract, not an incidental training-weight
    # default, decides whether every recovered row needs the immutable logged
    # expert trajectory/reward/safety sidecar.  _train_scene keeps this
    # persistence-only flag separate from target injection.
    train_config["cache_replay_expert_anchor"] = bool(
        writer.cache_expert_anchor
    )
    # Likewise, resume the decoder-context representation declared by the
    # writer (last observed neighbour state plus the already +1-aligned
    # future), rather than inheriting a possibly absent serialization toggle
    # from an older effective config.  Candidate generation still sees the
    # full scene; compaction happens only to rollout.reward_data immediately
    # before the disk append.
    train_config["cache_replay_decoder_context"] = bool(
        writer.cache_decoder_context
    )
    model, model_args = load_original_dp_checkpoint(
        args.model_path, device, args.args_path, use_ema=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    exploration_reference_model = None
    exploration_reference_model_args = None
    if rollout_config.plannerrft_guided_exploration:
        plannerrft_config = dict(raw_config.get("plannerrft", {}))
        reference_model_path = Path(
            plannerrft_config.get("reference_model_path", "")
        ).expanduser().resolve()
        reference_args_path = Path(
            plannerrft_config.get("reference_args_path", "")
        ).expanduser().resolve()
        if not reference_model_path.is_file():
            raise FileNotFoundError(reference_model_path)
        if not reference_args_path.is_file():
            raise FileNotFoundError(reference_args_path)
        (
            exploration_reference_model,
            exploration_reference_model_args,
        ) = load_original_dp_checkpoint(
            reference_model_path,
            device,
            reference_args_path,
            use_ema=True,
        )
        if (
            int(exploration_reference_model_args.future_len)
            != int(model_args.future_len)
            or int(exploration_reference_model_args.predicted_neighbor_num)
            != int(model_args.predicted_neighbor_num)
        ):
            raise RuntimeError(
                "PlannerRFT tail reference architecture differs from policy"
            )
        for parameter in exploration_reference_model.parameters():
            parameter.requires_grad_(False)
        exploration_reference_model.eval()
    if not args.disable_compile and device.type == "cuda":
        _compile_planner_modules(
            model,
            {
                "enabled": True,
                "backend": "inductor",
                "mode": args.compile_mode,
                "fullgraph": False,
                "dynamic": False,
                "encoder": False,
                "decoder": True,
            },
            f"tail_backfill_rank{rank}",
        )
        if exploration_reference_model is not None:
            _compile_planner_modules(
                exploration_reference_model,
                {
                    "enabled": True,
                    "backend": "inductor",
                    "mode": args.compile_mode,
                    "fullgraph": False,
                    "dynamic": False,
                    "encoder": False,
                    "decoder": True,
                },
                f"tail_backfill_reference_rank{rank}",
            )

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        _mine_and_append(
            writer, chunk, model, model_args, rollout_config, reward_config,
            train_config, device, args.amp_dtype,
            max(1, int(args.scene_load_workers)), failures_file,
            exploration_reference_model, exploration_reference_model_args,
        )
        if (start // batch_size) % 8 == 0:
            print(
                f"rank {rank}: tail {min(start + len(chunk), len(missing)):,}/{len(missing):,} "
                f"absolute={writer.position:,}/{writer.expected_count:,}",
                flush=True,
            )

    manifest_path = writer.close()
    provenance = {
        "rank": rank,
        "source_model": str(args.model_path.resolve()),
        "config": str(args.config.resolve()),
        "plan": str((args.plan_dir / "manifest.json").resolve()),
        "prefix_count": prefix_count,
        "backfilled_count": len(missing),
        "final_count": writer.position,
        "group_size": writer.group_size,
        "rng_seed": args.seed + 1000003 + rank,
        "batch_failure_policy": "recursive isolation; single-scene failure aborts; strict close retained",
        "manifest": str(manifest_path),
    }
    (writer.rank_dir / "tail_backfill_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
