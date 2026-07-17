#!/usr/bin/env python3
"""Train and evaluate AWR on the original four-channel Diffusion Planner.

The runner is intentionally independent of the HDP and GRPO training paths.
It consumes the v5.0 original-DP checkpoint, samples unguided DP diffusion
plans, scores them with the repository's OBB-aware T4 reward, and performs a
positive advantage-weighted diffusion regression update.

Examples:

    # The v5.0 release zip contains a .param.json instead of args.json.
    python -m rlvr.train_awr \
      --weights_zip ~/v5.0-20260715T090010Z-1-001.zip \
      --train_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train_sft.json \
      --valid_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_valid_sft_balanced.json \
      --config rlvr/configs/awr_original_dp_t4.json \
      --device cuda --exp_name original_dp_awr_t4

For a fast CPU plumbing smoke test, override ``max_train_scenes`` and
``max_valid_scenes`` on the command line.  The checkpoint is still validated
against the original DP architecture before any training starts.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.neighbor_future_alignment import (
    align_neighbor_future_numpy,
    get_neighbor_future_offset,
)
from planner_metrics.config import RewardConfig
from rlvr.awr import (
    AWRRollout,
    AWRRolloutConfig,
    breakdown_metrics,
    load_original_dp_checkpoint,
    load_scene,
    rollout_and_score_scene,
    rollout_and_score_scene_batch,
    rollout_to_json,
    reward_compatible_data,
    save_checkpoint_pair,
    update_ema,
)
from rlvr.grpo_loss import compute_batched_trajectory_losses
from rlvr.reward import compute_reward_batch
from diffusion_planner.utils.scene_skip import filter_scene_list


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.item())
        return _json_safe(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: Any) -> None:
    """Publish a manifest only after its complete contents reach one file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True) + "\n")


_EVAL_METRIC_SECTIONS = {
    # Reward and ranking signal.
    "det_reward": "01_reward",
    "best_group_reward": "01_reward",
    "mean_group_reward": "01_reward",
    "best_safe_reward": "01_reward",
    "best_vs_det_reward_gain": "01_reward",
    "candidate_reward_std": "01_reward",
    "reward_unique_count": "01_reward",
    "det_zero_reward": "01_reward",
    "zero_reward_candidate_fraction": "01_reward",
    "all_zero_reward_group": "01_reward",
    "all_equal_reward_group": "01_reward",
    # Hard-gate and candidate safety outcomes.
    "det_collision": "02_safety_gates",
    "det_rb_crossing": "02_safety_gates",
    "det_kinematic_violation": "02_safety_gates",
    "det_zero_collision": "02_safety_gates",
    "det_zero_rb_crossing": "02_safety_gates",
    "det_zero_kinematic": "02_safety_gates",
    "det_zero_without_collision_rb_kinematic": "02_safety_gates",
    "safe_candidate_count": "02_safety_gates",
    "safe_candidate_fraction": "02_safety_gates",
    "candidate_collision_count": "02_safety_gates",
    "candidate_collision_fraction": "02_safety_gates",
    "candidate_rb_crossing_count": "02_safety_gates",
    "candidate_rb_crossing_fraction": "02_safety_gates",
    "candidate_kinematic_violation_count": "02_safety_gates",
    "candidate_kinematic_violation_fraction": "02_safety_gates",
    # Lane/road diagnostics are deliberately separate from hard gates because
    # HDP-PDM uses lane as a soft 2/14 reward term in this experiment.
    "det_lane_crossing": "03_lane_road",
    "candidate_lane_crossing_fraction": "03_lane_road",
    "det_off_road_fraction": "03_lane_road",
    "det_centerline": "03_lane_road",
    # Driving behavior and comfort.
    "det_progress": "04_driving_quality",
    "det_safety": "04_driving_quality",
    "det_smoothness": "04_driving_quality",
    "path_len": "04_driving_quality",
    # Open-loop imitation diagnostics.
    "ade": "05_imitation",
    "fde": "05_imitation",
    # Diffusion sampling coverage/multimodality.
    "candidate_count": "06_multimodality",
    "candidate_endpoint_spread_mean": "06_multimodality",
    "candidate_endpoint_spread_max": "06_multimodality",
    "candidate_pairwise_ade": "06_multimodality",
    "candidate_temporal_spread": "06_multimodality",
}

# Percentiles of binary rates and counts add dozens of charts while conveying
# almost no information (for example collision P50 is normally exactly zero).
# Keep full mean/P10/P50/P90 summaries in the local JSON audit trail, but show
# P50/P90 in W&B only where the distribution shape is operationally useful.
_EVAL_WANDB_PERCENTILE_BASES = {
    "det_reward",
    "best_group_reward",
    "best_vs_det_reward_gain",
    "det_progress",
    "det_smoothness",
    "ade",
    "fde",
    "candidate_endpoint_spread_mean",
    "candidate_pairwise_ade",
    "candidate_temporal_spread",
}


def _split_summary_stat(key: str) -> tuple[str | None, str]:
    for statistic in ("mean", "p10", "p50", "p90"):
        prefix = f"{statistic}_"
        if key.startswith(prefix):
            return statistic, key[len(prefix) :]
    return None, key


def _wandb_metric_path(prefix: str, key: str) -> str | None:
    """Map flat audit metrics into small, stable W&B dashboard sections."""

    if prefix in {"eval", "eval_delta", "baseline"}:
        statistic, base = _split_summary_stat(key)
        if statistic == "p10":
            return None
        if statistic in {"p50", "p90"} and base not in _EVAL_WANDB_PERCENTILE_BASES:
            return None
        if base == "scene_count":
            section = "00_status"
        else:
            section = _EVAL_METRIC_SECTIONS.get(base, "07_diagnostics")
        return f"{prefix}_{section}/{key}"

    if prefix == "train":
        _, base = _split_summary_stat(key)
        if base in {
            "scene_count",
            "epoch",
            "cycle",
            "rollout_refresh",
            "replay_training",
        }:
            section = "00_status"
        elif base in {
            "loss",
            "grad_norm",
            "learning_rate",
            "optimizer_steps",
            "optimizer_step",
            "skipped",
            "fallback_nonfinite_update",
        }:
            section = "01_optimization"
        elif any(
            token in base
            for token in (
                "reward",
                "weight",
                "effective_sample_size",
                "top1_weight_share",
                "valid_group",
                "expert_anchor",
            )
        ):
            section = "02_awr_signal"
        elif any(token in base for token in ("collision", "rb_crossing", "kinematic")):
            section = "03_safety_gates"
        elif any(token in base for token in ("spread", "pairwise", "diversity", "candidate_count")):
            section = "04_multimodality"
        elif base in {
            "elapsed_sec",
            "scenes_per_sec",
            "gpu_peak_allocated_gib_rank0",
            "gpu_peak_reserved_gib_rank0",
        }:
            section = "05_system"
        else:
            section = "06_diagnostics"
        return f"train_{section}/{key}"

    return f"{prefix}/{key}"


def _init_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    effective_config: dict[str, Any],
):
    """Start one rank-zero W&B run only when explicitly requested."""

    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("--wandb was requested but the wandb package is unavailable") from error
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_dir.name,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        mode=args.wandb_mode,
        dir=str(run_dir),
        tags=args.wandb_tags or None,
        config=_json_safe(effective_config),
    )
    run.define_metric("epoch")
    namespaces = {
        "system/*",
        "best/*",
        *{
            f"{prefix}_{section}/*"
            for prefix in ("eval", "eval_delta", "baseline")
            for section in {
                "00_status",
                *_EVAL_METRIC_SECTIONS.values(),
                "07_diagnostics",
            }
        },
        *{
            f"train_{section}/*"
            for section in (
                "00_status",
                "01_optimization",
                "02_awr_signal",
                "03_safety_gates",
                "04_multimodality",
                "05_system",
                "06_diagnostics",
            )
        },
    }
    for namespace in sorted(namespaces):
        run.define_metric(namespace, step_metric="epoch")
    return run


def _wandb_log_summary(
    run: Any,
    prefix: str,
    summary: dict[str, Any],
    epoch: int,
    baseline: dict[str, Any] | None = None,
) -> None:
    if run is None:
        return
    payload: dict[str, float | int] = {"epoch": int(epoch)}
    for key, value in summary.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metric_path = _wandb_metric_path(prefix, key)
            if metric_path is not None:
                payload[metric_path] = float(value)
            if baseline is not None and key in baseline:
                base = baseline[key]
                if isinstance(base, (int, float)) and math.isfinite(float(base)):
                    delta_path = _wandb_metric_path("eval_delta", key)
                    if delta_path is not None:
                        payload[delta_path] = float(value) - float(base)
    run.log(payload)


def _copy_json_config(config_path: Path, run_dir: Path, raw_config: dict[str, Any]) -> None:
    _write_json(run_dir / "awr_config.json", raw_config)
    shutil.copy2(config_path, run_dir / "awr_config_source.json")


def _prepare_checkpoint(
    model_path: Path | None,
    weights_zip: Path | None,
    run_dir: Path,
    args_path: Path | None = None,
) -> tuple[Path, Path]:
    """Stage a checkpoint and its JSON args without modifying the source zip."""

    stage = run_dir / "base_model"
    stage.mkdir(parents=True, exist_ok=True)
    if weights_zip is not None:
        if not weights_zip.exists():
            raise FileNotFoundError(weights_zip)
        with zipfile.ZipFile(weights_zip) as archive:
            pth_members = [m for m in archive.namelist() if m.endswith("diffusion_planner.pth")]
            param_members = [m for m in archive.namelist() if m.endswith("diffusion_planner.param.json")]
            if len(pth_members) != 1 or len(param_members) != 1:
                raise RuntimeError(
                    f"expected one original DP .pth and .param.json in {weights_zip}; "
                    f"found pth={pth_members}, param={param_members}"
                )
            staged_model = stage / "original_dp_v5.pth"
            staged_args = stage / "args.json"
            with archive.open(pth_members[0]) as src, staged_model.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with archive.open(param_members[0]) as src, staged_args.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return staged_model, staged_args

    if model_path is None:
        raise ValueError("provide either --model_path or --weights_zip")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    args_candidates: list[Path] = []
    if args_path is not None:
        args_candidates.append(args_path)
    args_candidates.extend(
        [model_path.parent / "args.json", model_path.parent / "model_args.json"]
    )
    args_candidates.extend(sorted(model_path.parent.glob("*.param.json")))
    args_source = next((p for p in args_candidates if p.exists()), None)
    if args_source is None:
        raise FileNotFoundError(f"no args.json or *.param.json beside {model_path}")
    staged_model = stage / model_path.name
    staged_args = stage / "args.json"
    shutil.copy2(model_path, staged_model)
    shutil.copy2(args_source, staged_args)
    return staged_model, staged_args


def _load_config(path: Path) -> tuple[dict[str, Any], AWRRolloutConfig, RewardConfig]:
    raw = json.loads(path.read_text())
    awr_raw = dict(raw.get("awr", raw))
    noise_range = tuple(float(x) for x in awr_raw.get("noise_scale_range", (0.5, 2.0)))
    rollout = AWRRolloutConfig(
        n_trajectories=int(awr_raw.get("n_trajectories", awr_raw.get("K", 8))),
        sample_steps=int(awr_raw.get("sample_steps", 5)),
        noise_scale_range=noise_range,
        beta=float(awr_raw.get("beta", 0.75)),
        weight_clip=float(awr_raw.get("weight_clip", 20.0)),
        normalize_weights=bool(awr_raw.get("normalize_weights", True)),
        min_group_std=float(awr_raw.get("min_group_std", 1e-5)),
        safe_only=bool(awr_raw.get("safe_only", False)),
        structured_exploration=bool(awr_raw.get("structured_exploration", False)),
        deterministic_first=bool(awr_raw.get("deterministic_first", True)),
        hdp_trajectory_augmentation=bool(
            awr_raw.get("hdp_trajectory_augmentation", False)
        ),
        hdp_trajectory_augmentation_std=float(
            awr_raw.get("hdp_trajectory_augmentation_std", 0.5)
        ),
        positive_advantage_only=bool(awr_raw.get("positive_advantage_only", False)),
        positive_advantage_margin=float(
            awr_raw.get("positive_advantage_margin", 0.0)
        ),
    )
    reward_raw = dict(raw.get("reward", raw.get("reward_config", {})))
    reward_fields = {field.name for field in fields(RewardConfig)}
    reward_kwargs = {key: value for key, value in reward_raw.items() if key in reward_fields}
    reward = RewardConfig(**reward_kwargs)
    return raw, rollout, reward


def _choose_paths(
    path_list: Path,
    limit: int,
    seed: int,
    label: str,
    skip_filtered_scenes: bool = False,
    sidecar_root: Path | None = None,
) -> list[str]:
    paths = json.loads(path_list.read_text())
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{label} list is empty or not a JSON list: {path_list}")
    source_count = len(paths)
    if skip_filtered_scenes:
        paths = filter_scene_list(
            paths,
            sidecar_root=sidecar_root,
            enabled=True,
            label=f"{label}:{path_list.name}",
        )
        if not paths:
            raise ValueError(f"{label} is empty after is_skipped filtering: {path_list}")
        print(f"{label}: skip-filter kept {len(paths)} / {source_count}")
    if limit <= 0 or limit >= len(paths):
        selected = list(paths)
    else:
        rng = random.Random(seed)
        selected = [paths[i] for i in rng.sample(range(len(paths)), limit)]
    print(f"{label}: selected {len(selected)} / {len(paths)} scenes")
    return [str(path) for path in selected]


def _scene_selection_manifest(paths: list[str]) -> Any:
    """Keep full-corpus provenance small without losing reproducibility."""

    if len(paths) <= 100_000:
        return paths
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\n")
    return {
        "count": len(paths),
        "sha256_newline_joined_paths": digest.hexdigest(),
        "first_paths": paths[:3],
        "last_paths": paths[-3:],
        "note": "Full list omitted from the run directory; source JSON path is recorded in provenance.json.",
    }


def _configure_trainable_decoder(
    model: nn.Module, trainable_scope: str = "dit"
) -> list[nn.Parameter]:
    """Freeze the scene encoder and select a conservative DP denoiser scope.

    ``dit`` matches HDP's decoder-only fine-tuning.  ``last_block`` and
    ``output`` are useful stability ablations for the smaller original-DP
    checkpoint: they let AWR change the denoising output without rewriting
    every attention block, which is important when exploration targets are
    noisy or the local reward is only a proxy for full PDM.
    """

    planner = model.module if hasattr(model, "module") else model
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    scope = str(trainable_scope).lower()
    if scope == "dit":
        modules = [planner.decoder.dit]
    elif scope == "last_block":
        modules = [planner.decoder.dit.blocks[-1], planner.decoder.dit.final_layer]
    elif scope == "output":
        modules = [planner.decoder.dit.final_layer]
    else:
        raise ValueError(
            f"unknown trainable_scope={trainable_scope!r}; expected dit, last_block, or output"
        )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # No turn-indicator loss is used by AWR; keeping this head frozen avoids a
    # side objective whose labels are unrelated to the reward-weighted target.
    planner.decoder.turn_indicator_predictor.requires_grad_(False)
    planner.encoder.eval()
    planner.decoder.turn_indicator_predictor.eval()
    return [parameter for parameter in planner.parameters() if parameter.requires_grad]


def _make_optimizer(parameters: list[nn.Parameter], config: dict[str, Any], device: torch.device):
    weight_decay = float(config.get("weight_decay", 1e-4))
    decay, no_decay = [], []
    for parameter in parameters:
        if parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": float(config.get("learning_rate", 1e-5)), "betas": (0.9, 0.999)}
    if device.type == "cuda" and bool(config.get("fused_adamw", True)):
        try:
            optimizer = AdamW(groups, fused=True, **kwargs)
            print("Optimizer: fused AdamW")
            return optimizer
        except (TypeError, RuntimeError) as error:
            print(f"Fused AdamW unavailable ({error}); using standard AdamW")
    return AdamW(groups, **kwargs)


def _resolve_amp_dtype(requested: str, device: torch.device) -> str:
    requested = str(requested).lower()
    if requested in {"off", "none", "false"} or device.type != "cuda":
        return "off"
    if requested in {"fp16", "float16", "half"}:
        return "fp16"
    if requested in {"bf16", "bfloat16"}:
        return "bf16"
    if requested != "auto":
        raise ValueError(f"unsupported amp_dtype={requested}")
    capability = torch.cuda.get_device_capability(device)
    return "bf16" if capability[0] >= 8 else "fp16"


def _configure_cuda_runtime(device: torch.device) -> dict[str, Any]:
    """Enable the safe, high-throughput CUDA switches used by T4 training.

    The AWR loop has fixed tensor shapes (80 future steps and 321 agents), so
    the H100 can use TF32 for fp32 matmuls, bf16 autocast for the denoiser, and
    the compiler's shape-specialised kernels.  These flags do not alter the
    model architecture or the reward calculation.
    """

    if device.type != "cuda":
        return {"cuda": False, "tf32": False, "cudnn_benchmark": False}

    # The non-login experiment shell does not source the cluster CUDA setup,
    # while TorchInductor invokes nvcc by name when compiling a CUDA kernel.
    # Make the installed toolkit discoverable inside the training process so
    # --compile is reproducible from both an interactive shell and torchrun.
    cuda_bin = Path("/usr/local/cuda/bin")
    if cuda_bin.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        cuda_bin_str = str(cuda_bin)
        if cuda_bin_str not in path_entries:
            os.environ["PATH"] = os.pathsep.join(
                [cuda_bin_str, *[entry for entry in path_entries if entry]]
            )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The planner contains MultiheadAttention.  Keep the fused SDPA kernels
    # enabled; PyTorch will select the kernel supported by the active dtype and
    # sequence length.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    # Scene-wise AWR always reuses the same static geometry shapes.
    torch.backends.cudnn.benchmark = True
    return {
        "cuda": True,
        "device": str(device),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "nvcc": shutil.which("nvcc"),
        "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled())
        if hasattr(torch.backends.cuda, "flash_sdp_enabled")
        else None,
        "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
        else None,
    }


def _compile_planner_modules(
    model: nn.Module,
    compile_config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Compile fixed-shape encoder/DiT modules, leaving control flow eager.

    Compiling the whole planner would capture dictionaries, reward-side
    metadata, and the DPM solver's Python control flow.  Compiling only the
    tensor-heavy encoder and denoiser gives the useful Inductor/Triton speedup
    while preserving the original DP inference path and checkpoint format.
    """

    result: dict[str, Any] = {"label": label, "requested": bool(compile_config.get("enabled", True))}
    if not result["requested"]:
        result["enabled"] = False
        result["reason"] = "disabled_by_config"
        return result
    if not hasattr(torch, "compile"):
        result["enabled"] = False
        result["reason"] = "torch.compile_unavailable"
        return result

    planner = model.module if hasattr(model, "module") else model
    mode = str(compile_config.get("mode", "max-autotune"))
    backend = str(compile_config.get("backend", "inductor"))
    fullgraph = bool(compile_config.get("fullgraph", False))
    dynamic = bool(compile_config.get("dynamic", False))
    result.update({"enabled": True, "mode": mode, "backend": backend, "fullgraph": fullgraph, "dynamic": dynamic})

    targets: list[tuple[str, nn.Module, Any, str]] = []
    if bool(compile_config.get("encoder", True)):
        targets.append(("encoder", planner.encoder, planner, "encoder"))
    if bool(compile_config.get("decoder", True)):
        targets.append(("decoder_dit", planner.decoder.dit, planner.decoder, "dit"))

    modules: dict[str, Any] = {}
    for name, module, parent, attr in targets:
        if hasattr(module, "_orig_mod"):
            modules[name] = {"compiled": True, "already_compiled": True}
            continue
        try:
            compiled = torch.compile(
                module,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            setattr(parent, attr, compiled)
            modules[name] = {"compiled": True, "already_compiled": False}
        except Exception as error:  # pragma: no cover - depends on local compiler
            modules[name] = {
                "compiled": False,
                "error": f"{type(error).__name__}: {error}",
            }
    result["modules"] = modules
    return result


def _unwrap_compiled_modules(model: nn.Module) -> None:
    """Replace OptimizedModule wrappers with their original modules in-place."""

    planner = model.module if hasattr(model, "module") else model
    for parent, attr in ((planner, "encoder"), (planner.decoder, "dit")):
        module = getattr(parent, attr, None)
        original = getattr(module, "_orig_mod", None)
        if original is not None:
            setattr(parent, attr, original)


def _warmup_compiled_models(
    policy_model: nn.Module,
    behavior_model: nn.Module,
    model_args,
    scene_path: str | list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Trigger compilation once and record a post-compile timing sample."""

    data = load_scene(scene_path, device)
    start = time.perf_counter()
    try:
        # Warm both copies: the behavior copy is used for evaluation and the
        # policy copy is used by the AWR loss.  Keeping these calls here makes
        # the first reported metric a post-compile metric rather than a compile
        # plus inference measurement.
        rollout_and_score_scene(behavior_model, model_args, data, rollout_config, reward_config, device)
        rollout_and_score_scene(policy_model, model_args, data, rollout_config, reward_config, device)
        rollout_and_score_scene(behavior_model, model_args, data, rollout_config, reward_config, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return {"success": True, "elapsed_sec": elapsed}
    except Exception as error:  # pragma: no cover - compiler/backend dependent
        _unwrap_compiled_modules(policy_model)
        _unwrap_compiled_modules(behavior_model)
        return {"success": False, "elapsed_sec": time.perf_counter() - start, "error": f"{type(error).__name__}: {error}"}


def _trajectory_metrics(trajectory: torch.Tensor, data: dict[str, torch.Tensor]) -> dict[str, float]:
    trajectory = trajectory.detach().float()
    xy = trajectory[:, :2]
    path_len = float(torch.diff(xy, dim=0).norm(dim=-1).sum().item())
    result = {"path_len": path_len}
    gt = data.get("ego_agent_future")
    if gt is None:
        return result
    if gt.dim() == 3:
        gt = gt[0]
    gt_xy = gt[:, :2].float()
    valid = gt_xy.abs().sum(dim=-1) > 0.1
    if not bool(valid.any()):
        return result
    n = min(int(valid.sum().item()), trajectory.shape[0])
    gt_xy = gt_xy[valid][:n]
    pred_xy = xy[:n]
    result["ade"] = float((pred_xy - gt_xy).norm(dim=-1).mean().item())
    result["fde"] = float((pred_xy[-1] - gt_xy[-1]).norm().item())
    result["gt_path_len"] = float(torch.diff(gt_xy, dim=0).norm(dim=-1).sum().item()) if n > 1 else 0.0
    return result


def _candidate_diversity_metrics(
    trajectories: torch.Tensor, rewards: list[Any]
) -> dict[str, float]:
    """Measure useful K-sample spread; this is observation-only."""

    xy = trajectories.detach().float()[..., :2]
    K = int(xy.shape[0])
    totals = np.asarray([float(reward.total) for reward in rewards], dtype=np.float64)
    result = {
        "candidate_reward_std": float(np.nanstd(totals)),
        "best_vs_det_reward_gain": float(np.nanmax(totals) - totals[0]),
    }
    if K < 2:
        result.update(
            {
                "candidate_endpoint_spread_mean": 0.0,
                "candidate_endpoint_spread_max": 0.0,
                "candidate_pairwise_ade": 0.0,
                "candidate_temporal_spread": 0.0,
            }
        )
        return result
    pair = torch.triu_indices(K, K, offset=1, device=xy.device)
    pairwise_t = (xy[pair[0]] - xy[pair[1]]).norm(dim=-1)
    endpoint_pairwise = pairwise_t[:, -1]
    centroid = xy.mean(dim=0, keepdim=True)
    result.update(
        {
            "candidate_endpoint_spread_mean": float(endpoint_pairwise.mean().item()),
            "candidate_endpoint_spread_max": float(endpoint_pairwise.max().item()),
            "candidate_pairwise_ade": float(pairwise_t.mean(dim=-1).mean().item()),
            "candidate_temporal_spread": float(
                (xy - centroid).norm(dim=-1).mean().item()
            ),
        }
    )
    return result


@torch.no_grad()
def _slice_eval_scene_data(
    data: dict[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    """Keep one scene's loader batch dimension for per-scene metrics."""

    batch_size = int(data["ego_current_state"].shape[0])
    return {
        key: value[index : index + 1]
        if isinstance(value, torch.Tensor)
        and value.dim() > 0
        and value.shape[0] == batch_size
        else value
        for key, value in data.items()
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    model_args,
    scene_paths: list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
    output_dir: Path | None = None,
    epoch: int = 0,
    save_rollouts: int = 0,
    scene_batch_size: int = 1,
    scene_load_workers: int = 1,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate deterministic DP plus a fresh candidate group.

    Evaluation keeps exactly the same per-scene reward and metric definitions,
    but packs scenes into the same decoder batch used by full-corpus training.
    The old scene-at-a-time evaluator made a 2,048-scene validation set spend
    thousands of launches in Python and left seven GPUs idle behind rank 0.
    """

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    eval_batch_size = max(1, int(scene_batch_size))
    for batch_start in range(0, len(scene_paths), eval_batch_size):
        batch_paths = scene_paths[batch_start : batch_start + eval_batch_size]
        data = _load_scene_batch(
            batch_paths,
            device,
            workers=max(1, int(scene_load_workers)),
        )
        if len(batch_paths) == 1:
            batch_rollouts = [
                rollout_and_score_scene(
                    model, model_args, data, rollout_config, reward_config, device
                )
            ]
        else:
            batch_rollouts = rollout_and_score_scene_batch(
                model, model_args, data, rollout_config, reward_config, device
            )
        for offset, (scene_path, rollout) in enumerate(zip(batch_paths, batch_rollouts)):
            index = batch_start + offset
            scene_data = _slice_eval_scene_data(data, offset)
            det = rollout.rewards[0]
            best = max(rollout.rewards, key=lambda reward: reward.total)

            def configured_safe(reward: Any) -> bool:
                return bool(
                    reward.collision_step is None
                    and (not reward_config.rb_gate_enabled or not reward.rb_crossing)
                    and (not reward_config.lane_gate_enabled or not reward.lane_crossing)
                    and (
                        not reward_config.static_collision_enabled
                        or not reward_config.sc_gate_enabled
                        or not reward.static_crossing
                    )
                    and not reward.kinematic_violated
                    and reward.red_light >= -0.5
                )

            safe_rewards = [
                reward for reward in rollout.rewards if configured_safe(reward)
            ]
            best_safe = (
                max(safe_rewards, key=lambda reward: reward.total)
                if safe_rewards
                else None
            )
            traj_metrics = _trajectory_metrics(
                rollout.trajectories[0], scene_data
            )
            diversity_metrics = _candidate_diversity_metrics(
                rollout.trajectories, rollout.rewards
            )
            candidate_count = max(1, len(rollout.rewards))
            candidate_totals = np.asarray(
                [float(reward.total) for reward in rollout.rewards], dtype=np.float64
            )
            finite_candidate_totals = candidate_totals[
                np.isfinite(candidate_totals)
            ]
            zero_tol = 1e-8
            row: dict[str, Any] = {
                "scene_index": index,
                "scene_path": scene_path,
                "det_reward": det.total,
                "best_group_reward": best.total,
                "mean_group_reward": float(
                    np.mean([reward.total for reward in rollout.rewards])
                ),
                "det_collision": float(det.collision_step is not None),
                "det_rb_crossing": float(det.rb_crossing),
                "det_lane_crossing": float(det.lane_crossing),
                "det_kinematic_violation": float(det.kinematic_violated),
                "det_off_road_fraction": det.off_road_fraction,
                "det_progress": det.progress,
                "det_safety": det.safety,
                "det_centerline": det.centerline,
                "det_smoothness": det.smoothness,
                "candidate_count": float(len(rollout.rewards)),
                "safe_candidate_count": float(len(safe_rewards)),
                "candidate_collision_count": float(
                    sum(
                        reward.collision_step is not None
                        for reward in rollout.rewards
                    )
                ),
                "candidate_rb_crossing_count": float(
                    sum(bool(reward.rb_crossing) for reward in rollout.rewards)
                ),
                "candidate_kinematic_violation_count": float(
                    sum(
                        bool(reward.kinematic_violated)
                        for reward in rollout.rewards
                    )
                ),
                "safe_candidate_fraction": float(len(safe_rewards) / candidate_count),
                "candidate_collision_fraction": float(
                    sum(reward.collision_step is not None for reward in rollout.rewards)
                    / candidate_count
                ),
                "candidate_rb_crossing_fraction": float(
                    sum(bool(reward.rb_crossing) for reward in rollout.rewards)
                    / candidate_count
                ),
                "candidate_lane_crossing_fraction": float(
                    sum(bool(reward.lane_crossing) for reward in rollout.rewards)
                    / candidate_count
                ),
                "candidate_kinematic_violation_fraction": float(
                    sum(bool(reward.kinematic_violated) for reward in rollout.rewards)
                    / candidate_count
                ),
                "det_zero_reward": float(abs(float(det.total)) <= zero_tol),
                "zero_reward_candidate_fraction": float(
                    np.mean(np.abs(finite_candidate_totals) <= zero_tol)
                )
                if finite_candidate_totals.size
                else float("nan"),
                "all_zero_reward_group": float(
                    bool(finite_candidate_totals.size)
                    and bool(np.all(np.abs(finite_candidate_totals) <= zero_tol))
                ),
                "all_equal_reward_group": float(
                    bool(finite_candidate_totals.size)
                    and float(
                        np.max(finite_candidate_totals)
                        - np.min(finite_candidate_totals)
                    )
                    <= zero_tol
                ),
                "reward_unique_count": float(
                    np.unique(np.round(finite_candidate_totals, decimals=8)).size
                ),
                "det_zero_collision": float(
                    abs(float(det.total)) <= zero_tol
                    and det.collision_step is not None
                ),
                "det_zero_rb_crossing": float(
                    abs(float(det.total)) <= zero_tol and bool(det.rb_crossing)
                ),
                "det_zero_kinematic": float(
                    abs(float(det.total)) <= zero_tol
                    and bool(det.kinematic_violated)
                ),
                "det_zero_without_collision_rb_kinematic": float(
                    abs(float(det.total)) <= zero_tol
                    and det.collision_step is None
                    and not bool(det.rb_crossing)
                    and not bool(det.kinematic_violated)
                ),
                "best_safe_reward": (
                    float(best_safe.total)
                    if best_safe is not None
                    else float("nan")
                ),
                **traj_metrics,
                **diversity_metrics,
            }
            rows.append(row)
            if output_dir is not None and index < save_rollouts:
                stem = f"scene_{index:04d}"
                np.savez_compressed(
                    output_dir / f"{stem}.npz",
                    trajectories=rollout.trajectories.detach().cpu().numpy(),
                    noise_scales=rollout.noise_scales.detach().cpu().numpy(),
                    weights=rollout.weights.detach().cpu().numpy(),
                )
                _write_json(
                    output_dir / f"{stem}.json",
                    rollout_to_json(rollout, scene_path),
                )

    if not rows:
        return {"scene_count": 0.0}, rows
    keys = [
        "det_reward",
        "best_group_reward",
        "mean_group_reward",
        "det_collision",
        "det_rb_crossing",
        "det_lane_crossing",
        "det_kinematic_violation",
        "det_off_road_fraction",
        "det_progress",
        "det_safety",
        "det_centerline",
        "det_smoothness",
        "candidate_count",
        "safe_candidate_count",
        "candidate_collision_count",
        "candidate_rb_crossing_count",
        "candidate_kinematic_violation_count",
        "safe_candidate_fraction",
        "candidate_collision_fraction",
        "candidate_rb_crossing_fraction",
        "candidate_lane_crossing_fraction",
        "candidate_kinematic_violation_fraction",
        "det_zero_reward",
        "zero_reward_candidate_fraction",
        "all_zero_reward_group",
        "all_equal_reward_group",
        "reward_unique_count",
        "det_zero_collision",
        "det_zero_rb_crossing",
        "det_zero_kinematic",
        "det_zero_without_collision_rb_kinematic",
        "candidate_reward_std",
        "best_vs_det_reward_gain",
        "candidate_endpoint_spread_mean",
        "candidate_endpoint_spread_max",
        "candidate_pairwise_ade",
        "candidate_temporal_spread",
        "best_safe_reward",
        "path_len",
        "ade",
        "fde",
    ]
    summary: dict[str, float] = {"scene_count": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"p10_{key}"] = float(np.percentile(values, 10))
            summary[f"p50_{key}"] = float(np.median(values))
            summary[f"p90_{key}"] = float(np.percentile(values, 90))
    return summary, rows


def _save_rollout(output_dir: Path, scene_index: int, scene_path: str, rollout: AWRRollout) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene_{scene_index:04d}"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        trajectories=rollout.trajectories.detach().cpu().numpy(),
        noise_scales=rollout.noise_scales.detach().cpu().numpy(),
        weights=rollout.weights.detach().cpu().numpy(),
    )
    _write_json(output_dir / f"{stem}.json", rollout_to_json(rollout, scene_path))


def _rollout_to_cpu(rollout: AWRRollout) -> AWRRollout:
    """Detach one rollout so it can live in the HDP-style replay buffer."""

    return AWRRollout(
        trajectories=rollout.trajectories.detach().cpu(),
        noise_scales=rollout.noise_scales.detach().cpu(),
        rewards=rollout.rewards,
        weights=rollout.weights.detach().cpu(),
        reward_data={
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in rollout.reward_data.items()
        },
        scene_encoding=(
            rollout.scene_encoding.detach().cpu()
            if rollout.scene_encoding is not None
            else None
        ),
        diagnostics=dict(rollout.diagnostics),
    )


def _rollout_to_device(rollout: AWRRollout, device: torch.device) -> AWRRollout:
    """Move a replay record back to the rank-local device for regression."""

    return AWRRollout(
        trajectories=rollout.trajectories.to(device=device),
        noise_scales=rollout.noise_scales.to(device=device),
        rewards=rollout.rewards,
        weights=rollout.weights.to(device=device),
        reward_data={
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in rollout.reward_data.items()
        },
        scene_encoding=(
            rollout.scene_encoding.to(device=device)
            if rollout.scene_encoding is not None
            else None
        ),
        diagnostics=dict(rollout.diagnostics),
    )


def _merge_scene_data(datas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate loader outputs into one scene-batched observation."""

    if not datas:
        raise ValueError("cannot merge an empty scene batch")
    if len(datas) == 1:
        return datas[0]
    merged: dict[str, torch.Tensor] = {}
    for key in datas[0]:
        values = [data[key] for data in datas]
        first = values[0]
        if (
            isinstance(first, torch.Tensor)
            and first.dim() > 0
            and first.shape[0] == 1
            and all(isinstance(value, torch.Tensor) and value.shape == first.shape for value in values)
        ):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = first
    return merged


def _load_scene_batch(
    scene_paths: list[str],
    device: torch.device,
    workers: int = 1,
    replay_context_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a scene batch without serial GPU copies for every NPZ file.

    The original scene-wise path moves each tensor to CUDA while parsing each
    NPZ and then concatenates 192 one-scene CUDA tensors.  Full-corpus AWR is
    dominated by that Python/NPZ path between denoiser calls.  Parse a small
    batch concurrently on CPU, concatenate once, and perform one device copy
    per field.  The tensors and their ordering are unchanged; this is only an
    I/O/layout optimization.
    """

    if not scene_paths:
        raise ValueError("cannot load an empty scene batch")

    def load_one(path: str, target_device: torch.device):
        if not replay_context_only:
            return load_scene(path, target_device)

        # Frozen replay records already contain the exact behavior-policy
        # encoder output. The diffusion regression only consumes these three
        # observation tensors. Reading maps, lanes and 31-frame histories
        # from every compressed NPZ is otherwise pure I/O/allocation overhead.
        with np.load(str(path)) as loaded:
            required = {
                "ego_current_state",
                "neighbor_agents_past",
                "neighbor_agents_future",
            }
            missing = sorted(required.difference(loaded.files))
            if missing:
                raise KeyError(f"replay scene {path} is missing {missing}")
            ego = np.array(loaded["ego_current_state"], copy=True)
            # Preserve the canonical loss' ``[..., -1, :4]`` access while
            # avoiding the 30 unused historical frames.
            neighbor_current = np.array(
                loaded["neighbor_agents_past"][:, -1:, :], copy=True
            )
            neighbor_future = align_neighbor_future_numpy(
                np.array(loaded["neighbor_agents_future"], copy=True)
            )
        return {
            "ego_current_state": torch.from_numpy(ego)
            .unsqueeze(0)
            .to(target_device),
            "neighbor_agents_past": torch.from_numpy(neighbor_current)
            .unsqueeze(0)
            .to(target_device),
            "neighbor_agents_future": torch.from_numpy(neighbor_future)
            .unsqueeze(0)
            .to(target_device),
        }

    if len(scene_paths) == 1 or int(workers) <= 1:
        return (
            _merge_scene_data([load_one(scene_paths[0], device)])
            if len(scene_paths) == 1
            else _merge_scene_data(
                [load_one(path, device) for path in scene_paths]
            )
        )

    cpu = torch.device("cpu")
    worker_count = max(1, min(int(workers), len(scene_paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        cpu_datas = list(executor.map(lambda path: load_one(path, cpu), scene_paths))
    merged = _merge_scene_data(cpu_datas)
    if device.type == "cpu":
        return merged
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in merged.items()
    }


class _DiskReplayWriter:
    """Write an HDP-compatible replay epoch as rank-local NVMe memmaps.

    The released HDP agent keeps ``(token, rollout, gt, reward)`` tuples in
    RAM, while its encoder features are supplied by an offline cache.  A full
    T4 corpus is too large for the former representation, so this class keeps
    the same tuple contents in contiguous rank-local ``.npy`` arrays.  The
    rollout targets, group weights, rewards, and *frozen behavior encoder*
    are written once during the rollout epoch and are sampled with
    replacement during replay epochs.  No policy-dependent quantity is
    regenerated while replaying.

    Memmaps are intentionally uncompressed: the rollout pass is compute
    bound, and sequential writes plus direct random batch reads are much
    faster than millions of compressed NPZ files.  The arrays are float32 at
    rest, matching the loss/reference cache numerics; BF16 is still used for
    model forwards.
    """

    def __init__(
        self,
        root: Path,
        rank: int,
        expected_count: int,
        group_size: int,
        future_len: int,
    ) -> None:
        self.root = Path(root)
        self.rank = int(rank)
        self.rank_dir = self.root / f"rank_{self.rank:04d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        # A refresh overwrites the rank-local arrays. Revoke the previous
        # generation before touching any memmap so an interrupted epoch 11+
        # cannot expose partially-new arrays through a stale manifest.
        (self.rank_dir / "manifest.json").unlink(missing_ok=True)
        for stale_tmp in self.rank_dir.glob(".manifest.json.tmp.*"):
            stale_tmp.unlink(missing_ok=True)
        self.expected_count = int(expected_count)
        self.group_size = int(group_size)
        self.future_len = int(future_len)
        self.position = 0
        self._encoding_shape: tuple[int, ...] | None = None
        self._arrays: dict[str, np.memmap] = {}
        self._paths_path = self.rank_dir / "scene_paths.jsonl"
        # A refresh reuses the rank directory.  Truncate the path index before
        # appending this epoch; otherwise a second refresh would look like a
        # larger buffer than the newly-created memmaps.
        self._paths_path.write_text("")

        if self.expected_count <= 0:
            raise ValueError("a replay writer needs at least one scene")
        if self.group_size <= 0 or self.future_len <= 0:
            raise ValueError(
                f"invalid replay shape group_size={self.group_size}, "
                f"future_len={self.future_len}"
            )
        self._create_array(
            "trajectories",
            (self.expected_count, self.group_size, self.future_len, 4),
        )
        self._create_array("noise_scales", (self.expected_count, self.group_size))
        self._create_array("weights", (self.expected_count, self.group_size))
        self._create_array("rewards", (self.expected_count, self.group_size))

    def _create_array(self, name: str, shape: tuple[int, ...]) -> None:
        path = self.rank_dir / f"{name}.npy"
        self._arrays[name] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )

    @staticmethod
    def _to_numpy(value: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
        return (
            value.detach()
            .to(device="cpu", dtype=dtype)
            .contiguous()
            .numpy()
        )

    def append(
        self,
        scene_paths: list[str],
        rollout: AWRRollout,
        group_size: int,
    ) -> None:
        paths = [str(path) for path in scene_paths]
        batch_size = len(paths)
        if batch_size <= 0:
            return
        if int(group_size) != self.group_size:
            raise ValueError(
                f"replay group-size changed: expected {self.group_size}, got {group_size}"
            )
        expected_candidates = batch_size * self.group_size
        if rollout.trajectories.shape[0] != expected_candidates:
            raise ValueError(
                "combined rollout does not preserve scene-major groups: "
                f"paths={batch_size}, K={self.group_size}, "
                f"trajectories={tuple(rollout.trajectories.shape)}"
            )
        if self.position + batch_size > self.expected_count:
            raise ValueError("replay writer received more scenes than its manifest")

        trajectories = self._to_numpy(rollout.trajectories).reshape(
            batch_size, self.group_size, self.future_len, 4
        )
        noise_scales = self._to_numpy(rollout.noise_scales).reshape(
            batch_size, self.group_size
        )
        weights = self._to_numpy(rollout.weights).reshape(batch_size, self.group_size)
        if rollout.rewards and len(rollout.rewards) == expected_candidates:
            rewards = np.asarray(
                [float(reward.total) for reward in rollout.rewards], dtype=np.float32
            ).reshape(batch_size, self.group_size)
        else:
            rewards = np.full(
                (batch_size, self.group_size), np.nan, dtype=np.float32
            )

        encoding = rollout.scene_encoding
        if encoding is not None:
            encoding_np = self._to_numpy(encoding)
            if encoding_np.shape[0] != batch_size:
                raise ValueError(
                    "replay behavior encoding is not scene-batched: "
                    f"expected B={batch_size}, got {encoding_np.shape}"
                )
            if self._encoding_shape is None:
                self._encoding_shape = tuple(int(x) for x in encoding_np.shape[1:])
                self._create_array(
                    "scene_encoding",
                    (self.expected_count, *self._encoding_shape),
                )
            elif tuple(encoding_np.shape[1:]) != self._encoding_shape:
                raise ValueError(
                    "replay behavior encoding shape changed: "
                    f"expected {self._encoding_shape}, got {encoding_np.shape[1:]}"
                )
        elif self._encoding_shape is not None:
            raise ValueError("replay behavior encoding disappeared mid-epoch")

        start = self.position
        end = start + batch_size
        self._arrays["trajectories"][start:end] = trajectories
        self._arrays["noise_scales"][start:end] = noise_scales
        self._arrays["weights"][start:end] = weights
        self._arrays["rewards"][start:end] = rewards
        if encoding is not None:
            self._arrays["scene_encoding"][start:end] = encoding_np
        self.position = end
        with self._paths_path.open("a") as handle:
            for path in paths:
                handle.write(json.dumps(path) + "\n")

    def close(self) -> Path:
        if self.position != self.expected_count:
            raise RuntimeError(
                "faithful replay rollout was incomplete: "
                f"rank={self.rank}, wrote={self.position}, "
                f"expected={self.expected_count}"
            )
        for array in self._arrays.values():
            array.flush()
        manifest = {
            "version": 1,
            "rank": self.rank,
            "scene_count": self.position,
            "group_size": self.group_size,
            "future_len": self.future_len,
            "encoding_shape": list(self._encoding_shape)
            if self._encoding_shape is not None
            else None,
            "arrays": {
                name: str(path.name)
                for name, path in {
                    **{key: self.rank_dir / f"{key}.npy" for key in self._arrays}
                }.items()
            },
            "paths": self._paths_path.name,
        }
        manifest_path = self.rank_dir / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        return manifest_path


class _DiskReplayReader:
    """Read the frozen HDP replay epoch with group sampling semantics."""

    def __init__(self, root: Path, rank: int) -> None:
        self.rank = int(rank)
        self.rank_dir = Path(root) / f"rank_{self.rank:04d}"
        manifest_path = self.rank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        self.scene_count = int(self.manifest["scene_count"])
        self.group_size = int(self.manifest["group_size"])
        self.future_len = int(self.manifest["future_len"])
        self.trajectories = np.load(
            self.rank_dir / self.manifest["arrays"]["trajectories"], mmap_mode="r"
        )
        self.noise_scales = np.load(
            self.rank_dir / self.manifest["arrays"]["noise_scales"], mmap_mode="r"
        )
        self.weights = np.load(
            self.rank_dir / self.manifest["arrays"]["weights"], mmap_mode="r"
        )
        self.rewards = np.load(
            self.rank_dir / self.manifest["arrays"]["rewards"], mmap_mode="r"
        )
        encoding_name = self.manifest["arrays"].get("scene_encoding")
        self.scene_encoding = (
            np.load(self.rank_dir / encoding_name, mmap_mode="r")
            if encoding_name
            else None
        )
        with (self.rank_dir / self.manifest["paths"]).open() as handle:
            self.scene_paths = [json.loads(line) for line in handle]
        if len(self.scene_paths) != self.scene_count:
            raise RuntimeError(
                f"replay path count mismatch: {len(self.scene_paths)} != {self.scene_count}"
            )

    def __len__(self) -> int:
        return self.scene_count

    def close(self) -> None:
        """Release old generation mmaps before a refresh truncates them."""

        for array in (
            self.trajectories,
            self.noise_scales,
            self.weights,
            self.rewards,
            self.scene_encoding,
        ):
            mmap = getattr(array, "_mmap", None) if array is not None else None
            if mmap is not None:
                mmap.close()

    def batch_from_indices(self, indices: np.ndarray | list[int]):
        """Materialize one replay batch from buffer indices."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size <= 0:
            raise ValueError("replay indices must be a non-empty 1-D array")
        batch_size = int(indices.size)

        # Random AWR sampling is intentionally with replacement, but issuing
        # 192 large memmap reads in random physical order wastes NVMe queue
        # locality. Read the exact same indices in stable sorted order, then
        # restore the original RNG order before constructing tensors. This is
        # byte-exact, including when one index appears more than once.
        read_order = np.argsort(indices, kind="stable")
        sorted_indices = indices[read_order]
        restore_order = np.empty_like(read_order)
        restore_order[read_order] = np.arange(batch_size, dtype=read_order.dtype)

        def materialize(array: np.ndarray, *, copy: bool = True) -> np.ndarray:
            sorted_values = np.array(
                array[sorted_indices], dtype=np.float32, copy=copy
            )
            return sorted_values[restore_order]

        trajectories = torch.from_numpy(
            materialize(self.trajectories)
        ).reshape(batch_size * self.group_size, self.future_len, 4)
        noise_scales = torch.from_numpy(
            materialize(self.noise_scales)
        ).reshape(batch_size * self.group_size)
        weights = torch.from_numpy(
            materialize(self.weights)
        ).reshape(batch_size * self.group_size)
        if self.scene_encoding is not None:
            scene_encoding = torch.from_numpy(
                materialize(self.scene_encoding)
            )
        else:
            scene_encoding = None
        reward_array = materialize(self.rewards)
        finite_reward = np.isfinite(reward_array)
        zero_reward = finite_reward & (np.abs(reward_array) <= 1e-8)
        finite_counts = finite_reward.sum(axis=1)
        row_ranges = np.where(
            finite_counts > 0,
            np.nanmax(reward_array, axis=1) - np.nanmin(reward_array, axis=1),
            np.nan,
        )
        diagnostics = {
            # Per-scene K, matching on-policy/evaluation semantics.  The
            # enclosing row already carries scene_count=batch_size.
            "candidate_count": float(self.group_size),
            "valid_group": float(bool(np.isfinite(weights.numpy()).all())),
            "det_reward": float(np.nanmean(reward_array[:, 0]))
            if finite_reward[:, 0].any()
            else float("nan"),
            "best_reward": float(np.nanmean(np.nanmax(reward_array, axis=1)))
            if finite_reward.any()
            else float("nan"),
            "replay_sampled_with_replacement": 1.0,
            "det_zero_reward": float(np.mean(zero_reward[:, 0])),
            "zero_reward_candidate_fraction": float(
                zero_reward.sum() / max(1, finite_reward.sum())
            ),
            "all_zero_reward_group": float(
                np.mean((finite_counts > 0) & (zero_reward.sum(axis=1) == finite_counts))
            ),
            "all_equal_reward_group": float(np.nanmean(row_ranges <= 1e-8)),
            "reward_std": float(np.nanmean(np.nanstd(reward_array, axis=1))),
            "best_vs_det_reward_gain": float(
                np.nanmean(np.nanmax(reward_array, axis=1) - reward_array[:, 0])
            ),
        }
        return [self.scene_paths[int(index)] for index in indices], AWRRollout(
            trajectories=trajectories,
            noise_scales=noise_scales,
            rewards=[],
            weights=weights,
            reward_data={},
            scene_encoding=scene_encoding,
            diagnostics=diagnostics,
        )

    def iter_random_batches(
        self,
        batch_size: int,
        num_batches: int,
        seed: int,
    ):
        """Yield B scene groups, sampled with replacement like ReplayBuffer.get."""

        if self.scene_count <= 0:
            return
        batch_size = max(1, int(batch_size))
        num_batches = max(0, int(num_batches))
        rng = np.random.default_rng(int(seed))
        for _ in range(num_batches):
            indices = rng.integers(0, self.scene_count, size=batch_size, dtype=np.int64)
            yield self.batch_from_indices(indices)


def _iter_prefetched_replay_batches(
    replay_iterator,
    scene_load_workers: int,
    prefetch_batches: int,
    replay_context_only: bool,
):
    """Prepare replay memmaps and NPZ context while CUDA trains the prior batch.

    One background coordinator consumes the replay iterator in exactly the
    same order. It materializes the frozen memmaps and invokes the existing
    CPU scene loader; all CUDA copies and every model/optimizer operation stay
    on the main rank thread. Thus prefetch changes scheduling only, not RNG,
    sample order, tensor values, or update order.
    """

    depth = max(0, int(prefetch_batches))
    if depth == 0:
        for scenes, rollout in replay_iterator:
            yield scenes, rollout, None
        return

    source = iter(replay_iterator)
    cpu = torch.device("cpu")
    exhausted = object()

    def prepare_next():
        try:
            scenes, rollout = next(source)
        except StopIteration:
            return exhausted
        data = _load_scene_batch(
            scenes,
            cpu,
            workers=max(1, int(scene_load_workers)),
            replay_context_only=replay_context_only,
        )
        return scenes, rollout, data

    # A single coordinator keeps Python generator access strictly serial.
    # Its scene loader may still use the measured-optimal inner worker count.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque(executor.submit(prepare_next) for _ in range(depth))
        while pending:
            prepared = pending.popleft().result()
            if prepared is exhausted:
                break
            pending.append(executor.submit(prepare_next))
            yield prepared


def _salvage_partial_replay_manifest(root: Path, rank: int) -> dict[str, Any]:
    """Recover a fully-written prefix from an interrupted rollout cache.

    The disk writer preallocates every array to the padded rank-local corpus
    size.  If one packed scene batch fails, ``close()`` intentionally refuses
    to publish a manifest, but every earlier prefix is still complete and
    auditable through ``scene_paths.jsonl``.  This recovery path publishes
    only that common prefix; it never exposes the zero-filled tail and never
    mutates the large memmaps.
    """

    rank_dir = Path(root) / f"rank_{int(rank):04d}"
    paths_path = rank_dir / "scene_paths.jsonl"
    if not paths_path.exists():
        raise FileNotFoundError(paths_path)
    with paths_path.open() as handle:
        scene_count = sum(1 for line in handle if line.strip())
    if scene_count <= 0:
        raise RuntimeError(f"partial replay rank {rank} contains no scene paths")

    arrays: dict[str, np.memmap] = {}
    for name in [
        "trajectories",
        "noise_scales",
        "weights",
        "rewards",
        "scene_encoding",
    ]:
        array_path = rank_dir / f"{name}.npy"
        if not array_path.exists():
            raise FileNotFoundError(array_path)
        arrays[name] = np.load(array_path, mmap_mode="r")

    capacity = int(arrays["trajectories"].shape[0])
    if scene_count > capacity:
        raise RuntimeError(
            f"partial replay rank {rank} path count {scene_count} exceeds "
            f"array capacity {capacity}"
        )
    if any(int(array.shape[0]) != capacity for array in arrays.values()):
        shapes = {name: tuple(array.shape) for name, array in arrays.items()}
        raise RuntimeError(f"partial replay rank {rank} array capacities differ: {shapes}")

    trajectories = arrays["trajectories"]
    if trajectories.ndim != 4 or trajectories.shape[-1] != 4:
        raise RuntimeError(
            f"partial replay rank {rank} has invalid trajectory shape "
            f"{tuple(trajectories.shape)}"
        )
    group_size = int(trajectories.shape[1])
    future_len = int(trajectories.shape[2])
    expected_group_shapes = {
        "noise_scales": (capacity, group_size),
        "weights": (capacity, group_size),
        "rewards": (capacity, group_size),
    }
    for name, expected_shape in expected_group_shapes.items():
        if tuple(arrays[name].shape) != expected_shape:
            raise RuntimeError(
                f"partial replay rank {rank} {name} shape "
                f"{tuple(arrays[name].shape)} != {expected_shape}"
            )

    # Probe both ends of the published prefix.  This is cheap even for the
    # 300-GB encoding memmap and catches truncated/zero-tail publication.
    probe = np.asarray(sorted({0, scene_count - 1}), dtype=np.int64)
    for name, array in arrays.items():
        values = np.asarray(array[probe])
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"partial replay rank {rank} has non-finite {name} values "
                "inside the recoverable prefix"
            )
    if not bool((np.asarray(arrays["weights"][probe]) > 0).any(axis=1).all()):
        raise RuntimeError(
            f"partial replay rank {rank} has an all-zero AWR-weight probe"
        )

    manifest = {
        "version": 1,
        "rank": int(rank),
        "scene_count": int(scene_count),
        "group_size": group_size,
        "future_len": future_len,
        "encoding_shape": list(arrays["scene_encoding"].shape[1:]),
        "arrays": {name: f"{name}.npy" for name in arrays},
        "paths": paths_path.name,
        "salvaged_partial_prefix": True,
        "array_capacity": capacity,
        "discarded_tail_slots": capacity - scene_count,
    }
    _write_json_atomic(rank_dir / "manifest.json", manifest)
    return manifest


def _combine_rollouts(rollouts: list[AWRRollout], data: dict[str, torch.Tensor]) -> AWRRollout:
    """Pack per-scene groups for one batched diffusion-regression forward."""

    if not rollouts:
        raise ValueError("cannot combine an empty rollout list")
    if len(rollouts) == 1:
        return rollouts[0]
    diagnostics: dict[str, float] = {}
    keys = set().union(*(rollout.diagnostics for rollout in rollouts))
    for key in keys:
        values = [rollout.diagnostics[key] for rollout in rollouts if key in rollout.diagnostics]
        if values and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            diagnostics[key] = float(np.mean([float(value) for value in values]))
    # Diagnostics are per scene. Keep K rather than reporting B*K for a packed
    # rollout; the enclosing row separately records scene_count=B.
    diagnostics["candidate_count"] = float(
        np.mean([rollout.trajectories.shape[0] for rollout in rollouts])
    )
    diagnostics["valid_group"] = float(
        all(float(rollout.diagnostics.get("valid_group", 0.0)) > 0.5 for rollout in rollouts)
    )
    return AWRRollout(
        trajectories=torch.cat([rollout.trajectories for rollout in rollouts], dim=0),
        noise_scales=torch.cat([rollout.noise_scales for rollout in rollouts], dim=0),
        rewards=[reward for rollout in rollouts for reward in rollout.rewards],
        weights=torch.cat([rollout.weights for rollout in rollouts], dim=0),
        reward_data=data,
        scene_encoding=torch.cat([rollout.scene_encoding for rollout in rollouts], dim=0),
        diagnostics=diagnostics,
    )


def _expert_reward_is_hard_safe(reward: Any, reward_config: RewardConfig) -> bool:
    """Apply the exact configured hard gates to one logged-expert future."""

    return bool(
        reward.collision_step is None
        and (not reward_config.rb_gate_enabled or not reward.rb_crossing)
        and (not reward_config.lane_gate_enabled or not reward.lane_crossing)
        and (
            not reward_config.static_collision_enabled
            or not reward_config.sc_gate_enabled
            or not reward.static_crossing
        )
        and reward.red_light >= -0.5
        and not reward.kinematic_violated
    )


@torch.no_grad()
def _score_expert_anchor_batch(
    data: dict[str, torch.Tensor],
    future_len: int,
    reward_config: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build and independently hard-score one logged target per scene.

    Reward geometry cannot be broadcast across scenes because map, actors,
    wheelbase, and masks differ.  The model sampling/denoising path remains
    fully batched; this optional retention audit performs one additional
    reward call per scene and is disabled in the faithful HDP experiment.
    """

    future = data.get("ego_agent_future")
    if not isinstance(future, torch.Tensor):
        return None
    expert = heading_to_cos_sin(future)[:, : int(future_len), :4]
    batch_size = int(expert.shape[0])
    rewards = torch.empty(batch_size, device=expert.device, dtype=torch.float32)
    safe = torch.zeros(batch_size, device=expert.device, dtype=torch.bool)
    for scene_index in range(batch_size):
        scene_data = _slice_eval_scene_data(data, scene_index)
        scene_reward = compute_reward_batch(
            expert[scene_index : scene_index + 1],
            reward_compatible_data(scene_data),
            reward_config,
        )[0]
        rewards[scene_index] = float(scene_reward.total)
        safe[scene_index] = _expert_reward_is_hard_safe(
            scene_reward, reward_config
        )
    return expert, rewards, safe


def _inject_expert_anchor_batch(
    target_trajectories: torch.Tensor,
    target_weights: torch.Tensor,
    expert_trajectories: torch.Tensor,
    expert_safe: torch.Tensor,
    *,
    group_size: int,
    expert_weight: float,
    replace_worst: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert safe experts scene-major without mixing different groups."""

    batch_size = int(expert_trajectories.shape[0])
    group_size = int(group_size)
    expected = batch_size * group_size
    if target_trajectories.shape[0] != expected or target_weights.numel() != expected:
        raise ValueError(
            "expert-anchor group shape mismatch: "
            f"B={batch_size}, K={group_size}, "
            f"targets={tuple(target_trajectories.shape)}, weights={tuple(target_weights.shape)}"
        )
    if expert_safe.shape != (batch_size,):
        raise ValueError(
            f"expert safe mask must be [{batch_size}], got {tuple(expert_safe.shape)}"
        )
    trajectories = target_trajectories.reshape(
        batch_size, group_size, *target_trajectories.shape[1:]
    )
    weights = target_weights.reshape(batch_size, group_size)
    safe = expert_safe.to(device=weights.device, dtype=torch.bool)
    experts = expert_trajectories.to(device=trajectories.device, dtype=trajectories.dtype)
    if replace_worst:
        trajectories = trajectories.clone()
        weights = weights.clone()
        rows = torch.arange(batch_size, device=weights.device)[safe]
        if rows.numel() > 0:
            columns = torch.argmin(weights, dim=1)[safe]
            trajectories[rows, columns] = experts[safe]
            weights[rows, columns] = float(expert_weight)
    else:
        # Keep a fixed K+1 shape for compiled batched training.  Unsafe logged
        # targets occupy a zero-weight slot and therefore cannot override a
        # configured hard gate or change the weighted objective.
        anchor_weights = torch.where(
            safe,
            torch.full_like(safe, float(expert_weight), dtype=weights.dtype),
            torch.zeros_like(safe, dtype=weights.dtype),
        )
        trajectories = torch.cat([trajectories, experts[:, None]], dim=1)
        weights = torch.cat([weights, anchor_weights[:, None]], dim=1)
    return trajectories.flatten(0, 1), weights.flatten()


def _train_scene(
    model: nn.Module,
    behavior_model: nn.Module,
    model_args,
    optimizer,
    scene_path: str | list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    train_config: dict[str, Any],
    device: torch.device,
    amp_dtype: str,
    grad_scaler: Any | None = None,
    rollout_override: AWRRollout | None = None,
    replay_update: bool = False,
    optimizer_step: bool = True,
    zero_grad: bool = True,
    sync_gradients: bool = True,
    gradient_scale: float = 1.0,
    collect_only: bool = False,
    scene_load_workers: int = 1,
    data_override: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], AWRollout | None]:
    scene_paths = [scene_path] if isinstance(scene_path, str) else list(scene_path)
    if not scene_paths:
        raise ValueError("scene_path batch is empty")
    expert_anchor_weight = float(train_config.get("expert_anchor_weight", 0.0))
    replay_context_only = bool(
        replay_update
        and rollout_override is not None
        and rollout_override.scene_encoding is not None
        and expert_anchor_weight <= 0.0
    )
    if data_override is None:
        data = _load_scene_batch(
            scene_paths,
            device,
            workers=scene_load_workers,
            replay_context_only=replay_context_only,
        )
    else:
        data = {
            key: value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in data_override.items()
        }
    scoring_reward_config = reward_config
    # In the HDP/PDM profile lane robustness is the centerline-based reward
    # term.  The separate lane-departure polygon diagnostic is not a reward
    # or a hard gate in the faithful full-corpus config (all lane penalty
    # scales and lane_gate_enabled are off).  It is nevertheless expensive:
    # it evaluates ego perimeter containment against every nearby lane for
    # every one of K sampled trajectories.  Skip only that unused diagnostic
    # during rollout mining; the exact expert lane-change mask inside the HDP
    # lane reward remains active, and all validation/evaluation keeps the full
    # metric config.  Thus AWR weights and their hard semantics are unchanged.
    if (
        collect_only
        and bool(train_config.get("skip_unused_hdp_diagnostics", False))
        and reward_config.reward_profile in {"hdp_pdm", "hdp_multi"}
        and not reward_config.lane_gate_enabled
        and float(reward_config.lane_near_scale) == 0.0
        and float(reward_config.lane_wide_scale) == 0.0
        and float(reward_config.lane_cont_scale) == 0.0
    ):
        scoring_reward_config = copy.copy(reward_config)
        scoring_reward_config.enable_lane_departure = False
    if rollout_override is None:
        if len(scene_paths) == 1:
            rollout = rollout_and_score_scene(
                behavior_model, model_args, data, rollout_config, scoring_reward_config, device
            )
        else:
            rollout = _combine_rollouts(
                rollout_and_score_scene_batch(
                    behavior_model, model_args, data, rollout_config, scoring_reward_config, device
                ),
                data,
            )
    else:
        rollout = _rollout_to_device(rollout_override, device)
    result: dict[str, Any] = {
        "scene_path": scene_path,
        "scene_count": len(scene_paths),
        **rollout.diagnostics,
    }
    if replay_update:
        result["replay_update"] = 1.0
    expert_anchor_batch: torch.Tensor | None = None
    expert_anchor_safe: torch.Tensor | None = None
    if expert_anchor_weight > 0.0:
        scored_experts = _score_expert_anchor_batch(
            data, int(model_args.future_len), reward_config
        )
        if scored_experts is not None:
            expert_anchor_batch, expert_rewards, expert_anchor_safe = scored_experts
            safe_count = int(expert_anchor_safe.sum().item())
            result["expert_anchor_count"] = float(safe_count)
            result["expert_anchor_used"] = safe_count / max(1, len(scene_paths))
            if safe_count:
                result["expert_anchor_reward"] = float(
                    expert_rewards[expert_anchor_safe].mean().item()
                )
        else:
            result["expert_anchor_used"] = 0.0
    else:
        result["expert_anchor_used"] = 0.0

    if collect_only:
        # Faithful HDP phase: one epoch only mines a frozen-behavior replay
        # buffer; the following epochs train on those fixed groups. Keep
        # this before optimizer/gradient construction so the rollout epoch is
        # genuinely update-free rather than mixing on-policy and replay loss.
        result.update(
            {
                "skipped": 0.0,
                "loss": 0.0,
                "grad_norm": 0.0,
                "weight_sum": float(rollout.weights.sum().item()),
                "optimizer_step": 0.0,
                "rollout_only": 1.0,
            }
        )
        return result, rollout

    if (
        float(rollout.diagnostics.get("valid_group", 0.0)) < 0.5
        and not bool(
            expert_anchor_safe is not None and expert_anchor_safe.any().item()
        )
    ):
        # This is only reachable for a group with no finite reward at all.
        # Tied finite groups are valid in the released HDP implementation and
        # now receive unit weight for every candidate in compute_awr_weights.
        # Keep a finite DDP-safe fallback for malformed scenes; it is not used
        # for ordinary all-tied HDP groups.
        if hasattr(model, "no_sync"):
            target_trajectories = rollout.trajectories
            target_weights = torch.zeros_like(rollout.weights)
            target_weights[0] = 1.0
            result["fallback_nonfinite_update"] = 1.0
        else:
            result["skipped"] = 1.0
            result["skip_reason"] = "zero_or_nonfinite_group_std"
            return result, rollout
    else:
        target_trajectories = rollout.trajectories
        target_weights = rollout.weights

    planner = model.module if hasattr(model, "module") else model
    planner.train()
    planner.encoder.eval()
    planner.decoder.turn_indicator_predictor.eval()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    P = 1 + int(model_args.predicted_neighbor_num)
    T = int(model_args.future_len)
    n_steps = max(1, int(train_config.get("diffusion_k_steps", 4)))
    t_min, t_max = (float(x) for x in train_config.get("diffusion_t_range", (0.05, 0.95)))
    total_loss = 0.0
    if expert_anchor_batch is not None and expert_anchor_safe is not None:
        target_trajectories, target_weights = _inject_expert_anchor_batch(
            target_trajectories,
            target_weights,
            expert_anchor_batch,
            expert_anchor_safe,
            group_size=int(rollout_config.n_trajectories),
            expert_weight=expert_anchor_weight,
            replace_worst=bool(
                train_config.get("expert_anchor_replace_worst", False)
            ),
        )
    bc_weight = float(train_config.get("bc_weight", 0.0))
    if bc_weight > 0.0 and target_weights.numel() > 0:
        # HDP's BC term is a retention prior against exploration collapse.  In
        # the original DP path the deterministic zero-noise sample is the
        # frozen behavior/reference trajectory, so increasing its weight is
        # the exact same denoising regression without a second graph shape.
        target_weights = target_weights.clone()
        effective_group_size = int(target_weights.numel()) // len(scene_paths)
        target_weights[::effective_group_size] = (
            target_weights[::effective_group_size] + bc_weight
        )
        result["bc_weight"] = bc_weight
    for step_index in range(n_steps):
        # HDP samples each replay action with its own diffusion perturbation
        # and time.  Sharing one noise/t across a whole reward group is still
        # an unbiased Monte-Carlo estimate, but creates an avoidable strongly
        # correlated gradient and is not the released implementation.
        target_count = int(target_trajectories.shape[0])
        noise = torch.randn(target_count, P, T, 4, device=device)
        t_schedule = train_config.get("diffusion_t_schedule")
        if isinstance(t_schedule, (list, tuple)) and len(t_schedule) >= n_steps:
            # HDP's five-step DDIM/GRPO chain is represented by the same
            # descending log-SNR points used by the original DP solver.  A
            # fixed schedule trains every denoising decision instead of
            # repeatedly perturbing only the final low-noise output.
            t_value = float(t_schedule[step_index])
            t = torch.full(
                (target_count,), min(max(t_value, t_min), t_max), device=device
            )
        else:
            t = torch.rand(target_count, device=device) * (t_max - t_min) + t_min
        per_trajectory_loss = compute_batched_trajectory_losses(
            # Keep the DDP wrapper on the gradient path.  Calling
            # ``model.module`` here silently bypasses DDP's gradient
            # all-reduce, leaving every rank with a different policy while
            # rank 0 saves only its local update.
            model,
            data,
            target_trajectories,
            model_args,
            noise,
            t,
            device,
            neighbor_loss_weight=float(train_config.get("neighbor_loss_weight", 0.0)),
            cached_encoding=rollout.scene_encoding,
            amp_dtype=amp_dtype,
            hdp_hybrid_loss_weight=float(train_config.get("hdp_hybrid_loss_weight", 0.0)),
            hdp_hybrid_waypoint_weight=float(train_config.get("hdp_hybrid_waypoint_weight", 0.1)),
            hdp_hybrid_window=int(train_config.get("hdp_hybrid_window", 10)),
            use_prefix_mask=bool(train_config.get("awr_use_prefix_mask", True)),
            awr_loss_type=str(train_config.get("awr_loss_type", "dp_geometry")),
        )
        if not bool(torch.isfinite(per_trajectory_loss).all()):
            optimizer.zero_grad(set_to_none=True)
            result["skipped"] = 1.0
            result["skip_reason"] = "nonfinite_diffusion_loss"
            return result, rollout
        weights = target_weights.to(device=device, dtype=per_trajectory_loss.dtype)
        if str(train_config.get("awr_loss_reduction", "weighted_mean")).lower() in {
            "hdp_mean",
            "mean",
        }:
            # This is the reduction used by the released HDP code:
            # mean(exp(group-normalised reward) * per-sample diffusion loss).
            # Keep it opt-in because the original DP path normally normalises
            # by the finite candidate weight sum to make scenes with different
            # invalid-candidate counts comparable.
            weighted_loss = (per_trajectory_loss * weights).mean()
        else:
            weighted_loss = (per_trajectory_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        # HDP discounts denoising decisions from the fixed reference chain.
        # The original DP runner samples continuous t, so map t≈1 (first,
        # noisiest step) to s=0 and t≈0 (last, cleanest step) to the final
        # denoising index.  Apply this per candidate because the faithful HDP
        # path samples an independent t for every replay action.
        denoise_gamma = float(train_config.get("denoising_discount", 1.0))
        denoise_steps = max(1, int(train_config.get("denoising_steps", 5)))
        denoise_index = torch.round(
            (1.0 - t.detach()) * max(denoise_steps - 1, 0)
        ).clamp_min(0.0)
        denoise_weights = torch.pow(
            torch.as_tensor(denoise_gamma, device=device, dtype=per_trajectory_loss.dtype),
            denoise_index,
        )
        if str(train_config.get("awr_loss_reduction", "weighted_mean")).lower() in {
            "hdp_mean",
            "mean",
        }:
            weighted_loss = (per_trajectory_loss * weights * denoise_weights).mean()
        else:
            weighted_loss = (per_trajectory_loss * weights * denoise_weights).sum() / weights.sum().clamp_min(1e-6)
        # Scale by the accumulation window so the eventual update is the
        # mean scene loss, not a learning-rate multiplier tied to the window.
        loss_for_backward = weighted_loss / n_steps * float(gradient_scale)
        # The K denoising samples for one scene contribute to one optimizer
        # step.  Accumulate the first K-1 backward passes locally and let DDP
        # all-reduce only the final pass; this is mathematically equivalent
        # to synchronizing every pass but avoids K redundant full-model
        # collectives on the 8-GPU path.
        should_sync = bool(sync_gradients) and step_index == n_steps - 1
        sync_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and not should_sync
            else contextlib.nullcontext()
        )
        with sync_context:
            if grad_scaler is not None:
                grad_scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
        total_loss += float(weighted_loss.detach().item()) / n_steps

    if not optimizer_step:
        # Keep these gradients for the next scene in the accumulation window.
        # Clipping each scene separately would not be equivalent to a batch.
        result.update(
            {
                "skipped": 0.0,
                "loss": total_loss,
                "grad_norm": 0.0,
                "weight_sum": float(target_weights.sum().item()),
                "optimizer_step": 0.0,
            }
        )
        return result, rollout

    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in planner.parameters() if parameter.requires_grad],
        float(train_config.get("grad_clip_norm", 1.0)),
    )
    grad_norm_value = float(
        grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    )
    # A scene-wise AWR update can become numerically unstable after the policy
    # has drifted away from the behavior model.  Never let a non-finite update
    # contaminate the EMA behavior model or the next scene.
    if not math.isfinite(grad_norm_value):
        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None:
            grad_scaler.update()
        result.update(
            {
                "skipped": 1.0,
                "skip_reason": "nonfinite_grad_norm",
                "loss": total_loss,
                "grad_norm": grad_norm_value,
                "weight_sum": float(target_weights.sum().item()),
                "optimizer_step": 1.0,
            }
        )
        return result, rollout
    if grad_scaler is not None:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    result.update(
        {
            "skipped": 0.0,
            "loss": total_loss,
            "grad_norm": grad_norm_value,
            "weight_sum": float(target_weights.sum().item()),
            "optimizer_step": 1.0,
        }
    )
    return result, rollout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_path", type=Path)
    source.add_argument("--weights_zip", type=Path)
    parser.add_argument(
        "--args_path",
        type=Path,
        default=None,
        help="optional model args JSON, useful for an AWR epoch checkpoint",
    )
    parser.add_argument(
        "--use_policy_state",
        action="store_true",
        help="when loading an AWR checkpoint, evaluate its policy state instead of EMA",
    )
    parser.add_argument("--train_npz_list", type=Path, required=True)
    parser.add_argument("--valid_npz_list", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/awr_original_dp"))
    parser.add_argument("--exp_name", type=str, default="original_dp_awr_t4")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log rank-zero training/evaluation diagnostics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "original-dp-awr"),
    )
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument(
        "--resume_replay_root",
        type=Path,
        default=None,
        help=(
            "reuse an existing rank-local disk replay directory instead of "
            "re-mining epoch 1"
        ),
    )
    parser.add_argument(
        "--salvage_partial_replay",
        action="store_true",
        help=(
            "publish the fully-written scene_paths prefix of an interrupted "
            "preallocated disk replay cache"
        ),
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="first HDP lifecycle epoch to execute (use 2 when resuming epoch-1 replay)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_train_scenes", type=int, default=None)
    parser.add_argument("--max_valid_scenes", type=int, default=None)
    parser.add_argument(
        "--skip_filtered_scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop sidecar frames with is_skipped=true before sampling/training (default: on)",
    )
    parser.add_argument(
        "--sidecar_root",
        type=Path,
        default=None,
        help="optional sidecar root when <npz>.json is not next to each NPZ",
    )
    parser.add_argument(
        "--skip_filter_manifest",
        type=Path,
        default=None,
        help="record an explicit pre-filter manifest when the input lists were filtered offline",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--K", type=int, default=None, help="override the AWR/eval group size")
    parser.add_argument(
        "--noise_scale_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="override the initial-DP-noise scale range for rollout groups",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="override the stochastic original-DP rollout solver steps (HDP uses 5)",
    )
    parser.add_argument(
        "--awr_beta",
        type=float,
        default=None,
        help="override the group-normalised AWR inverse-temperature",
    )
    parser.add_argument(
        "--awr_weight_clip",
        type=float,
        default=None,
        help="override the positive AWR weight clip",
    )
    parser.add_argument(
        "--deterministic_first",
        action="store_true",
        help="prepend the zero-noise behavior trajectory as candidate 0",
    )
    parser.add_argument(
        "--hdp_trajectory_augmentation",
        action="store_true",
        help="apply HDP's per-trajectory N(0,0.5m) longitudinal/lateral rollout augmentation",
    )
    parser.add_argument("--hdp_trajectory_augmentation_std", type=float, default=None)
    parser.add_argument(
        "--positive_advantage_only",
        action="store_true",
        help="weight only sampled candidates that beat deterministic behavior",
    )
    parser.add_argument("--positive_advantage_margin", type=float, default=None)
    parser.add_argument("--diffusion_k_steps", type=int, default=None)
    parser.add_argument("--diffusion_t_range", type=float, nargs=2, default=None)
    parser.add_argument("--neighbor_loss_weight", type=float, default=None)
    parser.add_argument("--hdp_hybrid_loss_weight", type=float, default=None)
    parser.add_argument(
        "--awr_loss_type",
        choices=["dp_geometry", "plain_mse"],
        default=None,
        help="diffusion regression metric used for AWR targets",
    )
    parser.add_argument(
        "--trainable_scope",
        choices=["dit", "last_block", "output"],
        default=None,
        help="decoder scope to update; output/last_block are conservative AWR ablations",
    )
    parser.add_argument("--eval_k", type=int, default=None)
    parser.add_argument("--save_rollouts", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--bc_weight", type=float, default=None)
    parser.add_argument("--denoising_discount", type=float, default=None)
    parser.add_argument("--denoising_steps", type=int, default=None)
    parser.add_argument("--replay_buffer_size", type=int, default=None)
    parser.add_argument("--replay_updates_per_epoch", type=int, default=None)
    parser.add_argument(
        "--replay_storage",
        choices=["memory", "disk"],
        default=None,
        help=(
            "replay backend; disk is the full-corpus HDP backend and keeps "
            "frozen targets/encodings in NVMe memmaps"
        ),
    )
    parser.add_argument(
        "--replay_sampling",
        choices=["with_replacement", "shuffle"],
        default=None,
        help="sampling rule for disk replay (HDP uses with_replacement)",
    )
    parser.add_argument(
        "--hdp_rollout_interval",
        type=int,
        default=None,
        help=(
            "refresh/collect-only interval for the HDP replay cycle; 0 uses "
            "streaming on-policy AWR so every full-corpus scene is trained"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_scenes",
        type=int,
        default=None,
        help=(
            "mean this many scene groups before one optimizer step; this is "
            "the HDP-style large-batch stability knob"
        ),
    )
    parser.add_argument(
        "--scene_batch_size",
        type=int,
        default=None,
        help="pack this many scenes into one decoder/reward-regression batch",
    )
    parser.add_argument(
        "--rollout_scene_batch_size",
        type=int,
        default=None,
        help=(
            "pack this many scenes only during collect-only rollout; replay "
            "keeps scene_batch_size so optimizer semantics do not change"
        ),
    )
    parser.add_argument(
        "--scene_load_workers",
        type=int,
        default=None,
        help="CPU threads used to parse one packed scene batch",
    )
    parser.add_argument("--rollout_scene_load_workers", type=int, default=None)
    parser.add_argument("--replay_prefetch_batches", type=int, default=None)
    parser.add_argument(
        "--skip_unused_hdp_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "during HDP/PDM rollout mining, skip lane-departure diagnostics "
            "when they are not a configured reward/gate; evaluation remains full"
        ),
    )
    parser.add_argument(
        "--ema_per_epoch",
        action="store_true",
        help="keep the behavior/reference policy fixed during an epoch and refresh it once at epoch end",
    )
    parser.add_argument("--expert_anchor_weight", type=float, default=None)
    parser.add_argument(
        "--expert_anchor_replace_worst",
        action="store_true",
        help="replace the lowest-weight candidate with a safe logged-expert anchor (keeps K fixed)",
    )
    parser.add_argument(
        "--reward_mode",
        choices=["survival", "gate"],
        default=None,
        help="override reward aggregation for an ablation",
    )
    parser.add_argument("--reward_profile", type=str, default=None)
    parser.add_argument("--reward_horizon_steps", type=int, default=None)
    parser.add_argument("--pdm_comfort_scale", type=float, default=None)
    parser.add_argument("--hdp_risk_weight", type=float, default=None)
    parser.add_argument("--hdp_follow_weight", type=float, default=None)
    parser.add_argument("--hdp_lane_weight", type=float, default=None)
    parser.add_argument("--hdp_risk_use_clearance", action="store_true")
    parser.add_argument(
        "--safe_only",
        action="store_true",
        help=(
            "weight only candidates that pass the configured hard safety gates; "
            "lane departure is not hard unless lane_gate_enabled is true"
        ),
    )
    parser.add_argument(
        "--structured_exploration",
        action="store_true",
        help="add HDP-inspired smooth yield/hold and lateral-offset proposal candidates",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        choices=["auto", "off", "bf16", "fp16"],
        default=None,
        help="override acceleration.amp_dtype (H100 auto-selects bf16)",
    )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        help="disable torch.compile for a clean eager comparison",
    )
    parser.add_argument(
        "--enable_compile",
        action="store_true",
        help="force torch.compile even when the source config leaves it disabled",
    )
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument(
        "--compile_encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile the scene encoder too (decoder-only is faster to warm up)",
    )
    parser.add_argument(
        "--compile_decoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile the diffusion decoder/DiT module",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="expect torchrun; multi-GPU mode is also enabled automatically when WORLD_SIZE>1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size_env > 1
    if args.distributed and not distributed:
        raise RuntimeError("--distributed requires torchrun with WORLD_SIZE>1")
    rank = 0
    world_size = 1
    object_group = None
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed AWR requires CUDA/NCCL")
        dist.init_process_group(backend="nccl")
        # ``gather_object`` with a NCCL group stages serialized Python objects
        # on the current CUDA device.  On this PyTorch/CUDA combination its
        # size exchange can occasionally be interpreted as an absurd tensor
        # allocation (>1 EB) even though rows contain only JSON scalars.  Use
        # a CPU Gloo side group for metrics/checkpoint metadata; gradients and
        # model synchronization remain NCCL.
        object_group = dist.new_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"requested {args.device}, but CUDA is unavailable; pass --device cpu for a smoke test"
            )
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    is_main = rank == 0

    runtime_acceleration = _configure_cuda_runtime(device)

    # Keep scene ordering identical across ranks, but give each rank an
    # independent diffusion/noise stream. DDP averages the resulting AWR
    # gradients into one policy update.
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    raw_config, rollout_config, reward_config = _load_config(args.config)
    train_config = dict(raw_config.get("training", raw_config))
    acceleration_config = dict(raw_config.get("acceleration", {}))
    if args.amp_dtype is not None:
        acceleration_config["amp_dtype"] = args.amp_dtype
    if args.disable_compile:
        acceleration_config["compile"] = False
    if args.enable_compile:
        acceleration_config["compile"] = True
    if args.compile_mode is not None:
        acceleration_config["mode"] = args.compile_mode
    if args.compile_encoder is not None:
        acceleration_config["encoder"] = bool(args.compile_encoder)
    if args.compile_decoder is not None:
        acceleration_config["decoder"] = bool(args.compile_decoder)
    if args.epochs is not None:
        train_config["train_epochs"] = args.epochs
    if args.eval_k is not None:
        train_config["eval_k"] = args.eval_k
    if args.diffusion_k_steps is not None:
        train_config["diffusion_k_steps"] = args.diffusion_k_steps
    if args.diffusion_t_range is not None:
        train_config["diffusion_t_range"] = list(args.diffusion_t_range)
    if args.neighbor_loss_weight is not None:
        train_config["neighbor_loss_weight"] = args.neighbor_loss_weight
    if args.hdp_hybrid_loss_weight is not None:
        train_config["hdp_hybrid_loss_weight"] = args.hdp_hybrid_loss_weight
    if args.awr_loss_type is not None:
        train_config["awr_loss_type"] = args.awr_loss_type
    if args.trainable_scope is not None:
        train_config["trainable_scope"] = args.trainable_scope
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.ema_decay is not None:
        train_config["ema_decay"] = args.ema_decay
    if args.bc_weight is not None:
        train_config["bc_weight"] = args.bc_weight
    if args.denoising_discount is not None:
        train_config["denoising_discount"] = args.denoising_discount
    if args.denoising_steps is not None:
        train_config["denoising_steps"] = args.denoising_steps
    if args.replay_buffer_size is not None:
        train_config["replay_buffer_size"] = args.replay_buffer_size
    if args.replay_updates_per_epoch is not None:
        train_config["replay_updates_per_epoch"] = args.replay_updates_per_epoch
    if args.replay_storage is not None:
        train_config["replay_storage"] = args.replay_storage
    if args.replay_sampling is not None:
        train_config["replay_sampling"] = args.replay_sampling
    if args.hdp_rollout_interval is not None:
        train_config["hdp_rollout_interval"] = args.hdp_rollout_interval
    if args.gradient_accumulation_scenes is not None:
        train_config["gradient_accumulation_scenes"] = args.gradient_accumulation_scenes
    if args.scene_batch_size is not None:
        train_config["scene_batch_size"] = args.scene_batch_size
    if args.rollout_scene_batch_size is not None:
        train_config["rollout_scene_batch_size"] = args.rollout_scene_batch_size
    if args.scene_load_workers is not None:
        train_config["scene_load_workers"] = args.scene_load_workers
    if args.rollout_scene_load_workers is not None:
        train_config["rollout_scene_load_workers"] = args.rollout_scene_load_workers
    if args.replay_prefetch_batches is not None:
        train_config["replay_prefetch_batches"] = args.replay_prefetch_batches
    if args.skip_unused_hdp_diagnostics is not None:
        train_config["skip_unused_hdp_diagnostics"] = bool(
            args.skip_unused_hdp_diagnostics
        )
    if args.ema_per_epoch:
        train_config["ema_per_epoch"] = True
    if args.expert_anchor_weight is not None:
        train_config["expert_anchor_weight"] = args.expert_anchor_weight
    if args.expert_anchor_replace_worst:
        train_config["expert_anchor_replace_worst"] = True
    if args.reward_mode is not None:
        reward_config.reward_mode = args.reward_mode
    if args.reward_profile is not None:
        reward_config.reward_profile = args.reward_profile
    if args.reward_horizon_steps is not None:
        reward_config.reward_horizon_steps = args.reward_horizon_steps
    if args.pdm_comfort_scale is not None:
        reward_config.pdm_comfort_scale = args.pdm_comfort_scale
    if args.hdp_risk_weight is not None:
        reward_config.hdp_risk_weight = args.hdp_risk_weight
    if args.hdp_follow_weight is not None:
        reward_config.hdp_follow_weight = args.hdp_follow_weight
    if args.hdp_lane_weight is not None:
        reward_config.hdp_lane_weight = args.hdp_lane_weight
    if args.hdp_risk_use_clearance:
        reward_config.hdp_risk_use_clearance = True
    if args.K is not None:
        rollout_config.n_trajectories = args.K
    if args.noise_scale_range is not None:
        rollout_config.noise_scale_range = tuple(float(x) for x in args.noise_scale_range)
    if args.sample_steps is not None:
        rollout_config.sample_steps = args.sample_steps
    if args.awr_beta is not None:
        rollout_config.beta = args.awr_beta
    if args.awr_weight_clip is not None:
        rollout_config.weight_clip = args.awr_weight_clip
    if args.deterministic_first:
        rollout_config.deterministic_first = True
    if args.hdp_trajectory_augmentation:
        rollout_config.hdp_trajectory_augmentation = True
    if args.hdp_trajectory_augmentation_std is not None:
        rollout_config.hdp_trajectory_augmentation_std = args.hdp_trajectory_augmentation_std
    if args.positive_advantage_only:
        rollout_config.positive_advantage_only = True
    if args.positive_advantage_margin is not None:
        rollout_config.positive_advantage_margin = args.positive_advantage_margin
    if args.safe_only:
        rollout_config.safe_only = True
    if args.structured_exploration:
        rollout_config.structured_exploration = True
    save_rollouts = int(
        args.save_rollouts if args.save_rollouts is not None else train_config.get("save_rollouts", 8)
    )
    if is_main:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.output_dir / f"{stamp}_{args.exp_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        _copy_json_config(args.config, run_dir, raw_config)
        # ``awr_config.json`` is the immutable source file.  Keep a second
        # artifact with every command-line override applied; otherwise a
        # full-corpus run can appear to have the small smoke-test batch/K/lr
        # even though the effective process used different values.
        effective_config = dict(raw_config)
        effective_config["acceleration"] = dict(acceleration_config)
        effective_config["training"] = dict(train_config)
        effective_config["awr"] = asdict(rollout_config)
        effective_config["reward"] = asdict(reward_config)
        if reward_config.reward_profile == "hdp_pdm":
            # The HDP/PDM adapter uses the released scorer's fixed 5/5/2/2
            # quality composition.  The RewardConfig hdp_* fields belong to
            # the paper's separate risk/follow/lane profile and are inactive
            # here; record the effective objective explicitly for W&B readers.
            effective_config["reward"]["effective_quality_weights"] = {
                "ttc": 5.0,
                "ego_progress": 5.0,
                "lane_keeping": 2.0,
                "comfort": 2.0,
                "normalizer": 14.0,
            }
            effective_config["reward"]["hdp_multi_weight_fields_active"] = False
        _write_json(run_dir / "effective_config.json", effective_config)
        staged_model, staged_args = _prepare_checkpoint(
            args.model_path, args.weights_zip, run_dir, args.args_path
        )
        shutil.copy2(staged_args, run_dir / "model_args.json")
        _write_json(
            run_dir / "provenance.json",
            {
                "model_source": str(args.model_path or args.weights_zip),
                "staged_model": str(staged_model),
                "staged_args": str(staged_args),
                "architecture": "original Diffusion Planner, x_start, 4-channel ego/neighbor state",
                "neighbor_future_alignment_offset": get_neighbor_future_offset(),
                "neighbor_future_alignment_env": "DP_NEIGHBOR_FUTURE_OFFSET",
                "x2_legacy_ego_width_m": 2.29156,
                "train_npz_list": str(args.train_npz_list),
                "valid_npz_list": str(args.valid_npz_list),
                "skip_filtered_scenes": bool(args.skip_filtered_scenes),
                "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
                "skip_filter_manifest": str(args.skip_filter_manifest) if args.skip_filter_manifest else None,
                "device": str(device),
                "seed": args.seed,
                "distributed": bool(distributed),
                "world_size": world_size,
                "rank": rank,
                "awr_beta": rollout_config.beta,
                "awr_weight_clip": rollout_config.weight_clip,
                "resume_replay_root": str(args.resume_replay_root)
                if args.resume_replay_root
                else None,
                "salvage_partial_replay": bool(args.salvage_partial_replay),
                "start_epoch": int(args.start_epoch),
            },
        )
        run_metadata = [str(run_dir), str(staged_model), str(staged_args)]
    else:
        run_dir = None
        staged_model = None
        staged_args = None
        run_metadata = ["", "", ""]
    if distributed:
        dist.broadcast_object_list(run_metadata, src=0)
        run_dir = Path(run_metadata[0])
        staged_model = Path(run_metadata[1])
        staged_args = Path(run_metadata[2])
        dist.barrier()
    assert run_dir is not None and staged_model is not None and staged_args is not None

    if is_main:
        print(f"Run directory: {run_dir}")
        print(f"Device: {device}; rank={rank}/{world_size}")
    model, model_args = load_original_dp_checkpoint(
        staged_model, device, staged_args, use_ema=not args.use_policy_state
    )
    trainable_scope = str(train_config.get("trainable_scope", "dit"))
    trainable = _configure_trainable_decoder(model, trainable_scope)
    behavior_model = copy.deepcopy(model).to(device)
    for parameter in behavior_model.parameters():
        parameter.requires_grad_(False)
    behavior_model.eval()
    requested_amp = str(acceleration_config.get("amp_dtype", train_config.get("amp_dtype", "auto")))
    amp_dtype = _resolve_amp_dtype(requested_amp, device)
    # Use the same autocast mode for frozen behavior rollouts and for the
    # baseline/post-training evaluator.  Without this, training claimed bf16
    # while the expensive diffusion sampler still ran entirely in fp32.
    rollout_config.inference_amp_dtype = amp_dtype
    # ``effective_config.json`` is created before the model is loaded so the
    # launcher can record all CLI overrides early.  The selected autocast
    # dtype is only knowable after device capability resolution; refresh this
    # field here so the audit artifact describes the actual rollout numerics,
    # not the dataclass default ("off").
    if is_main:
        effective_config["awr"]["inference_amp_dtype"] = amp_dtype
        _write_json(run_dir / "effective_config.json", effective_config)
    if is_main:
        print(
            f"Original DP validated: agents={model_args.agent_num}, predicted_neighbors={model_args.predicted_neighbor_num}, "
            f"future={model_args.future_len}, DPM steps={model.decoder._sample_steps}"
        )
        print(
            f"Trainable decoder parameters: {sum(p.numel() for p in trainable):,}; scope={trainable_scope}; "
            f"amp={amp_dtype}; tf32={runtime_acceleration.get('tf32_matmul', False)}"
        )
    optimizer = _make_optimizer(trainable, train_config, device)

    # Compile after checkpoint loading and optimizer construction so the
    # checkpoint remains a normal DP state dict and optimizer parameters are
    # the original tensors.  Only tensor-heavy submodules are compiled.
    compile_enabled = bool(acceleration_config.get("compile", True)) and device.type == "cuda"
    compile_config = dict(acceleration_config)
    compile_config["enabled"] = compile_enabled
    compile_policy = _compile_planner_modules(model, compile_config, "policy")
    compile_behavior = _compile_planner_modules(behavior_model, compile_config, "behavior")
    if distributed:
        # Compile before wrapping so the checkpoint/state-dict format remains
        # the ordinary DP format. The optimizer still owns the same parameter
        # objects that DDP wraps, so all-reduced gradients update one policy.
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    acceleration_info: dict[str, Any] = {
        "runtime": runtime_acceleration,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "amp_requested": requested_amp,
        "amp_selected": amp_dtype,
        "compile": {"policy": compile_policy, "behavior": compile_behavior},
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
    }
    if is_main:
        _write_json(run_dir / "acceleration.json", acceleration_info)
        print("Acceleration:", json.dumps(_json_safe(acceleration_info), sort_keys=True))

    train_limit = int(train_config.get("max_train_scenes", 0))
    valid_limit = int(train_config.get("max_valid_scenes", 0))
    if args.max_train_scenes is not None:
        train_limit = args.max_train_scenes
    if args.max_valid_scenes is not None:
        valid_limit = args.max_valid_scenes
    train_paths = _choose_paths(
        args.train_npz_list,
        train_limit,
        args.seed,
        "train",
        skip_filtered_scenes=bool(args.skip_filtered_scenes),
        sidecar_root=args.sidecar_root,
    )
    valid_paths = _choose_paths(
        args.valid_npz_list,
        valid_limit,
        args.seed + 1,
        "valid",
        skip_filtered_scenes=bool(args.skip_filtered_scenes),
        sidecar_root=args.sidecar_root,
    )
    if is_main:
        _write_json(
            run_dir / "scene_selection.json",
            {
                "train": _scene_selection_manifest(train_paths),
                "valid": _scene_selection_manifest(valid_paths),
                "skip_filtered_scenes": bool(args.skip_filtered_scenes),
                "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
                "skip_filter_manifest": str(args.skip_filter_manifest) if args.skip_filter_manifest else None,
            },
        )

    wandb_run = None
    if is_main:
        wandb_run = _init_wandb(args, run_dir, effective_config)

    if compile_enabled and valid_paths:
        warmup = _warmup_compiled_models(
            model,
            behavior_model,
            model_args,
            valid_paths[0],
            rollout_config,
            reward_config,
            device,
        )
        if is_main:
            acceleration_info["compile_warmup"] = warmup
            _write_json(run_dir / "acceleration.json", acceleration_info)
            print("Compile warmup:", json.dumps(_json_safe(warmup), sort_keys=True))

    eval_deterministic_first = bool(
        train_config.get("eval_deterministic_first", True)
    )
    eval_rollout_config = AWRRolloutConfig(
        n_trajectories=int(train_config.get("eval_k", rollout_config.n_trajectories)),
        sample_steps=rollout_config.sample_steps,
        noise_scale_range=rollout_config.noise_scale_range,
        beta=rollout_config.beta,
        weight_clip=rollout_config.weight_clip,
        normalize_weights=rollout_config.normalize_weights,
        min_group_std=rollout_config.min_group_std,
        safe_only=rollout_config.safe_only,
        structured_exploration=rollout_config.structured_exploration,
        # Evaluation candidate 0 is always the deploy-time zero-noise DP
        # policy.  Training may use the faithful HDP all-stochastic group,
        # but logging a random candidate as "det" would make baseline/AWR
        # comparisons irreproducible.
        deterministic_first=eval_deterministic_first
        or rollout_config.deterministic_first,
        # Augmentation is a training-rollout exploration device in HDP, not
        # part of the deployed/evaluation policy.  Keeping it off here makes
        # baseline and post-training metrics directly comparable.
        hdp_trajectory_augmentation=False,
        hdp_trajectory_augmentation_std=rollout_config.hdp_trajectory_augmentation_std,
        positive_advantage_only=rollout_config.positive_advantage_only,
        positive_advantage_margin=rollout_config.positive_advantage_margin,
        inference_amp_dtype=rollout_config.inference_amp_dtype,
    )
    eval_dir = run_dir / "eval_epoch_000"
    if is_main:
        print("Evaluating the untouched original-DP EMA checkpoint...")
        base_summary, base_rows = evaluate_model(
            behavior_model,
            model_args,
            valid_paths,
            eval_rollout_config,
            reward_config,
            device,
            eval_dir / "rollouts",
            epoch=0,
            save_rollouts=save_rollouts,
            scene_batch_size=int(train_config.get("scene_batch_size", 1)),
            scene_load_workers=int(train_config.get("scene_load_workers", 1)),
        )
        _write_json(eval_dir / "summary.json", base_summary)
        _write_json(eval_dir / "scenes.json", base_rows)
        _append_jsonl(run_dir / "metrics.jsonl", {"phase": "eval", "epoch": 0, **base_summary})
        _wandb_log_summary(wandb_run, "eval", base_summary, epoch=0)
        if wandb_run is not None:
            for key, value in base_summary.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_path = _wandb_metric_path("baseline", key)
                    if metric_path is not None:
                        wandb_run.summary[metric_path] = float(value)
        print("Baseline:", json.dumps(_json_safe(base_summary), sort_keys=True))
    else:
        base_summary = {}
    if distributed:
        dist.barrier()

    epochs = max(0, int(train_config.get("train_epochs", 1)))
    ema_decay = float(train_config.get("ema_decay", 0.995))
    grad_scaler = None
    if amp_dtype == "fp16" and device.type == "cuda":
        grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
    replay_capacity = max(0, int(train_config.get("replay_buffer_size", 0)))
    replay_updates_per_epoch = int(train_config.get("replay_updates_per_epoch", 0))
    # In the faithful disk backend, ``-1`` means one replay DataLoader-sized
    # batch per rank-local mined-scene batch. This matches the released HDP
    # pattern: mine the buffer once, then sample a fresh group on each train
    # batch from that frozen buffer.
    if replay_updates_per_epoch < -1:
        replay_updates_per_epoch = -1
    hdp_rollout_interval = max(0, int(train_config.get("hdp_rollout_interval", 0)))
    replay_storage = str(train_config.get("replay_storage", "memory")).lower()
    replay_sampling = str(
        train_config.get("replay_sampling", "with_replacement")
    ).lower()
    faithful_disk_replay = replay_storage == "disk" and hdp_rollout_interval > 0
    if faithful_disk_replay and replay_sampling not in {"with_replacement", "shuffle"}:
        raise ValueError(
            "replay_sampling must be with_replacement or shuffle, got "
            f"{replay_sampling!r}"
        )
    gradient_accumulation_scenes = max(
        1, int(train_config.get("gradient_accumulation_scenes", 1))
    )
    scene_batch_size = max(1, int(train_config.get("scene_batch_size", 1)))
    rollout_scene_batch_size = max(
        1, int(train_config.get("rollout_scene_batch_size", scene_batch_size))
    )
    rollout_scene_load_workers = max(
        1,
        int(
            train_config.get(
                "rollout_scene_load_workers",
                train_config.get("scene_load_workers", 1),
            )
        ),
    )
    start_epoch = max(1, int(args.start_epoch))
    if start_epoch > epochs:
        raise ValueError(
            f"start_epoch={start_epoch} exceeds configured epochs={epochs}"
        )
    replay_buffer = deque(maxlen=replay_capacity or None)
    replay_root = (
        args.resume_replay_root.expanduser().resolve()
        if args.resume_replay_root is not None
        else run_dir / "replay_buffer"
    )
    disk_replay_reader: _DiskReplayReader | None = None
    if args.resume_replay_root is not None and not faithful_disk_replay:
        raise ValueError("--resume_replay_root requires the faithful disk replay backend")
    if args.resume_replay_root is not None and start_epoch <= 1:
        raise ValueError("resumed epoch-1 replay must start at epoch 2 or later")
    if args.resume_replay_root is not None and any(
        (epoch - 1) % hdp_rollout_interval == 0
        for epoch in range(start_epoch, epochs + 1)
    ):
        raise ValueError(
            "the requested resumed range contains a rollout-refresh epoch; "
            "finish this cache before starting a fresh refresh run"
        )
    if faithful_disk_replay:
        if args.resume_replay_root is not None:
            if not replay_root.exists():
                raise FileNotFoundError(replay_root)
            if args.salvage_partial_replay:
                local_manifest = _salvage_partial_replay_manifest(replay_root, rank)
            else:
                manifest_path = replay_root / f"rank_{rank:04d}" / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(
                        f"{manifest_path}; pass --salvage_partial_replay to recover "
                        "a complete prefix"
                    )
                local_manifest = json.loads(manifest_path.read_text())
            if distributed:
                dist.barrier()
            disk_replay_reader = _DiskReplayReader(replay_root, rank)
            local_count = torch.tensor(
                [len(disk_replay_reader)], device=device, dtype=torch.int64
            )
            if distributed:
                gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
                dist.all_gather(gathered_counts, local_count)
                replay_rank_counts = [int(value.item()) for value in gathered_counts]
            else:
                replay_rank_counts = [int(local_count.item())]
            if is_main:
                _write_json(
                    run_dir / "replay_resume.json",
                    {
                        "source_replay_root": str(replay_root),
                        "salvaged_partial_replay": bool(args.salvage_partial_replay),
                        "start_epoch": start_epoch,
                        "rank_scene_counts": replay_rank_counts,
                        "total_scene_groups": int(sum(replay_rank_counts)),
                        "group_size": int(local_manifest["group_size"]),
                        "scene_batch_size": scene_batch_size,
                        "sampling": replay_sampling,
                    },
                )
                print(
                    "Resuming frozen replay: "
                    f"{sum(replay_rank_counts):,} groups across {len(replay_rank_counts)} "
                    f"ranks; counts={replay_rank_counts}"
                )
        else:
            if is_main and replay_root.exists():
                shutil.rmtree(replay_root)
            if distributed:
                dist.barrier()
    best_eval_reward = float(base_summary.get("mean_det_reward", -math.inf))
    best_eval_epoch = 0
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        hdp_rollout_phase = (
            hdp_rollout_interval > 0
            and (epoch - 1) % hdp_rollout_interval == 0
        )
        active_scene_batch_size = (
            rollout_scene_batch_size if hdp_rollout_phase else scene_batch_size
        )
        if hdp_rollout_phase:
            # Start a fresh on-policy mining cycle, as in HDP's periodic
            # replay-buffer refresh.  Training epochs below consume these
            # frozen groups and do not re-rollout the scenes.
            replay_buffer.clear()
            order = list(train_paths)
            random.Random(args.seed + epoch).shuffle(order)
        elif hdp_rollout_interval > 0:
            order = []
        else:
            # Preserve the original online AWR behavior when the HDP replay
            # schedule is not requested by the config.
            order = list(train_paths)
            random.Random(args.seed + epoch).shuffle(order)
        if distributed:
            # DDP requires every rank to execute the same number of backward
            # passes. Pad only the final few slots; every real scene remains
            # present exactly once in the epoch.
            pad_multiple = world_size * active_scene_batch_size
            remainder = len(order) % pad_multiple
            if remainder:
                order.extend(order[: pad_multiple - remainder])
            local_order = order[rank::world_size]
        else:
            local_order = order
        disk_replay_writer: _DiskReplayWriter | None = None
        if faithful_disk_replay and hdp_rollout_phase:
            if disk_replay_reader is not None:
                disk_replay_reader.close()
                disk_replay_reader = None
            disk_replay_writer = _DiskReplayWriter(
                replay_root,
                rank=rank,
                expected_count=len(local_order),
                group_size=int(rollout_config.n_trajectories),
                future_len=int(model_args.future_len),
            )
        # The reference HDP agent applies its Gaussian trajectory
        # augmentation only during the first five *Lightning epochs* of the
        # rollout code.  With the ten-epoch refresh schedule this means the
        # first mining epoch only; later refreshes use the unchanged behavior
        # distribution.  Preserve the opt-in streaming behavior when no HDP
        # refresh schedule is requested.
        epoch_rollout_config = rollout_config
        if hdp_rollout_interval > 0 and hdp_rollout_phase:
            epoch_rollout_config = copy.copy(rollout_config)
            epoch_rollout_config.hdp_trajectory_augmentation = bool(epoch <= 5)
        train_rows: list[dict[str, Any]] = []
        if active_scene_batch_size > 1:
            batch_starts = range(0, len(local_order), active_scene_batch_size)
        else:
            batch_starts = range(len(local_order))
        for batch_start in batch_starts:
            if active_scene_batch_size > 1:
                scene_paths = local_order[
                    batch_start : batch_start + active_scene_batch_size
                ]
                local_index = batch_start // active_scene_batch_size
                scene_path: str | list[str] = scene_paths
                scene_index = rank + local_index * world_size * scene_batch_size
                accumulation_count = 1
                is_accumulation_end = True
                accumulation_start = local_index
            else:
                scene_path = local_order[batch_start]
                scene_paths = [scene_path]
                local_index = batch_start
                scene_index = rank + local_index * world_size
                accumulation_start = (
                    local_index // gradient_accumulation_scenes
                ) * gradient_accumulation_scenes
                accumulation_end = min(
                    accumulation_start + gradient_accumulation_scenes,
                    len(local_order),
                )
                accumulation_count = max(1, accumulation_end - accumulation_start)
                is_accumulation_end = local_index + 1 == accumulation_end
            rollout: AWRRollout | None = None
            try:
                row, rollout = _train_scene(
                    model,
                    behavior_model,
                    model_args,
                    optimizer,
                    scene_path,
                    epoch_rollout_config,
                    reward_config,
                    train_config,
                    device,
                    amp_dtype,
                    grad_scaler,
                    optimizer_step=is_accumulation_end,
                    zero_grad=local_index == accumulation_start,
                    sync_gradients=is_accumulation_end,
                    gradient_scale=1.0 / accumulation_count,
                    collect_only=hdp_rollout_phase,
                    scene_load_workers=(
                        rollout_scene_load_workers
                        if hdp_rollout_phase
                        else int(train_config.get("scene_load_workers", 1))
                    ),
                )
                if is_main and rollout is not None and len(scene_paths) == 1 and scene_index < save_rollouts:
                    _save_rollout(run_dir / f"train_epoch_{epoch:03d}" / "rollouts", scene_index, scene_path, rollout)
            except Exception as error:
                # Never silently turn a CUDA OOM into a partially trained
                # checkpoint.  Scene-level data issues may be skipped, but a
                # memory failure means the configured full-corpus batch is
                # not valid and must be fixed before trusting the run.
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                if (
                    faithful_disk_replay
                    and hdp_rollout_phase
                    and disk_replay_writer is not None
                    and len(scene_paths) > 1
                ):
                    # Isolate malformed NPZs without discarding the other 191
                    # valid scene groups in a packed rollout batch.  Successful
                    # single-scene retries are written immediately.  A truly
                    # bad scene still leaves the strict writer incomplete, so
                    # close() remains fail-closed and the valid prefix can be
                    # salvaged explicitly on resume.
                    print(
                        f"Packed rollout batch failed; retrying {len(scene_paths)} "
                        f"scenes individually: {type(error).__name__}: {error}"
                    )
                    for fallback_path in scene_paths:
                        try:
                            fallback_row, fallback_rollout = _train_scene(
                                model,
                                behavior_model,
                                model_args,
                                optimizer,
                                fallback_path,
                                epoch_rollout_config,
                                reward_config,
                                train_config,
                                device,
                                amp_dtype,
                                grad_scaler,
                                collect_only=True,
                                scene_load_workers=1,
                            )
                            train_rows.append(fallback_row)
                            disk_replay_writer.append(
                                [fallback_path],
                                fallback_rollout,
                                group_size=int(rollout_config.n_trajectories),
                            )
                        except Exception as fallback_error:
                            if isinstance(
                                fallback_error, torch.cuda.OutOfMemoryError
                            ):
                                raise
                            train_rows.append(
                                {
                                    "scene_path": fallback_path,
                                    "scene_count": 1,
                                    "skipped": 1.0,
                                    "skip_reason": (
                                        f"{type(fallback_error).__name__}: "
                                        f"{fallback_error}"
                                    ),
                                }
                            )
                            print(
                                f"Scene failed after isolated retry "
                                f"({fallback_path}): {fallback_error}"
                            )
                    continue
                row = {"scene_path": scene_path, "skipped": 1.0, "skip_reason": f"{type(error).__name__}: {error}"}
                print(f"Scene failed ({scene_path}): {error}")
            train_rows.append(row)
            if (
                faithful_disk_replay
                and hdp_rollout_phase
                and disk_replay_writer is not None
                and rollout is not None
            ):
                # Store the entire scene-major group, including the frozen
                # behavior encoder.  This is the original HDP cache contract:
                # replay does not re-rollout or re-encode with the changing
                # policy.  A failed scene is intentionally not silently
                # replaced by a bad target; close() will fail closed below.
                disk_replay_writer.append(
                    scene_paths,
                    rollout,
                    group_size=int(rollout_config.n_trajectories),
                )
            if (
                not faithful_disk_replay
                and
                replay_capacity > 0
                and rollout is not None
                and float(rollout.weights.sum().item()) > 0.0
                and float(rollout.diagnostics.get("valid_group", 0.0)) > 0.5
                and len(scene_paths) == 1
            ):
                # Keep targets/encodings on host memory; replay must not pin
                # one scene's CUDA graph or grow rank-local VRAM over an
                # 8-GPU epoch.
                replay_buffer.append((str(scene_path), _rollout_to_cpu(rollout)))
            if is_main and (local_index + 1) % max(1, int(train_config.get("log_interval", 10))) == 0:
                print(
                    f"epoch {epoch} rank0 scene {local_index + 1}/"
                    f"{max(1, len(local_order) // active_scene_batch_size)} "
                    f"loss={row.get('loss', float('nan')):.5f} "
                    f"Rdet={row.get('det_reward', float('nan')):+.2f} "
                    f"Rbest={row.get('best_reward', float('nan')):+.2f} "
                    f"valid={row.get('valid_group', 0):.0f}"
                )
            if (
                not hdp_rollout_phase
                and optimizer_step
                and not bool(train_config.get("ema_per_epoch", False))
            ):
                update_ema(behavior_model, model, ema_decay)

        if faithful_disk_replay and hdp_rollout_phase:
            if disk_replay_writer is None:
                raise RuntimeError("missing faithful replay writer in rollout epoch")
            manifest_path = disk_replay_writer.close()
            if is_main:
                print(
                    f"HDP replay refresh epoch {epoch}: "
                    f"{disk_replay_writer.position:,} frozen groups/rank "
                    f"written to {manifest_path.parent}"
                )
            if distributed:
                dist.barrier()
            disk_replay_reader = _DiskReplayReader(replay_root, rank)
            if distributed:
                replay_size = torch.tensor(
                    [len(disk_replay_reader)], device=device, dtype=torch.int64
                )
                replay_min = replay_size.clone()
                replay_max = replay_size.clone()
                dist.all_reduce(replay_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(replay_max, op=dist.ReduceOp.MAX)
                if int(replay_min.item()) != int(replay_max.item()):
                    raise RuntimeError(
                        "DDP ranks wrote different replay sizes: "
                        f"min={int(replay_min.item())}, max={int(replay_max.item())}"
                    )

        # HDP trains reward-weighted samples from a replay buffer after the
        # rollout pass.  The disk backend below follows the reference
        # ``ReplayBuffer.get(B)`` contract: each replay batch draws B groups
        # with replacement, then performs one reward-weighted diffusion
        # update.  The legacy memory backend remains available for small
        # ablations.
        replay_rows: list[dict[str, Any]] = []
        if faithful_disk_replay:
            if not hdp_rollout_phase:
                if disk_replay_reader is None:
                    raise RuntimeError(
                        "faithful replay training started without a replay cache"
                    )
                if replay_updates_per_epoch < 0:
                    # One replay DataLoader-sized update per rank-local
                    # buffer population, as in the reference trainer.  The
                    # groups *inside* each update are sampled with
                    # replacement, so this is not an accidental second
                    # deterministic pass over the mining order.
                    current_replay_updates = max(
                        1,
                        math.ceil(len(disk_replay_reader) / scene_batch_size),
                    )
                else:
                    current_replay_updates = max(0, replay_updates_per_epoch)
                if distributed:
                    # Salvaged prefixes may differ by a few failed packed
                    # batches.  DDP must still execute the same number of
                    # backwards on every rank.  With replacement, using the
                    # largest rank-local pass preserves all available update
                    # budget while smaller ranks simply resample a little
                    # more often.
                    update_count = torch.tensor(
                        [current_replay_updates], device=device, dtype=torch.int64
                    )
                    dist.all_reduce(update_count, op=dist.ReduceOp.MAX)
                    current_replay_updates = int(update_count.item())
                if replay_sampling == "with_replacement":
                    replay_iterator = disk_replay_reader.iter_random_batches(
                        scene_batch_size,
                        current_replay_updates,
                        seed=args.seed + epoch * 1000003 + rank,
                    )
                else:
                    # A deterministic shuffled pass is useful for IO
                    # benchmarks.  The faithful experiment uses the branch
                    # above; keep this option explicit rather than silently
                    # changing HDP's random-choice distribution.
                    permutation = np.random.default_rng(
                        args.seed + epoch * 1000003 + rank
                    ).permutation(len(disk_replay_reader))

                    def _iter_shuffled_replay():
                        for start in range(0, len(permutation), scene_batch_size):
                            indices = permutation[start : start + scene_batch_size]
                            if len(indices) < scene_batch_size:
                                indices = np.resize(indices, scene_batch_size)
                            yield disk_replay_reader.batch_from_indices(indices)

                    replay_iterator = _iter_shuffled_replay()
                replay_context_only = bool(
                    disk_replay_reader.scene_encoding is not None
                    and float(train_config.get("expert_anchor_weight", 0.0)) <= 0.0
                )
                prepared_replay_iterator = _iter_prefetched_replay_batches(
                    replay_iterator,
                    scene_load_workers=int(
                        train_config.get("scene_load_workers", 1)
                    ),
                    prefetch_batches=int(
                        train_config.get("replay_prefetch_batches", 0)
                    ),
                    replay_context_only=replay_context_only,
                )
                for replay_index, (
                    replay_scenes,
                    replay_rollout,
                    replay_data,
                ) in enumerate(prepared_replay_iterator):
                    replay_row, _ = _train_scene(
                        model,
                        behavior_model,
                        model_args,
                        optimizer,
                        replay_scenes,
                        rollout_config,
                        reward_config,
                        train_config,
                        device,
                        amp_dtype,
                        grad_scaler,
                        rollout_override=replay_rollout,
                        replay_update=True,
                        optimizer_step=True,
                        zero_grad=True,
                        sync_gradients=True,
                        gradient_scale=1.0,
                        scene_load_workers=int(
                            train_config.get("scene_load_workers", 1)
                        ),
                        data_override=replay_data,
                    )
                    replay_rows.append(replay_row)
                    if not bool(train_config.get("ema_per_epoch", False)):
                        # Reference HDP updates its EMA after every optimizer
                        # batch.  The faithful disk replay branch has no
                        # scene loop around this call, so perform the update
                        # explicitly here rather than silently reducing it to
                        # once per epoch.
                        update_ema(behavior_model, model, ema_decay)
                    if is_main and (
                        replay_index == 0
                        or (replay_index + 1)
                        % max(1, int(train_config.get("log_interval", 64)))
                        == 0
                    ):
                        print(
                            f"epoch {epoch} replay {replay_index + 1}/"
                            f"{current_replay_updates} "
                            f"loss={replay_row.get('loss', float('nan')):.5f} "
                            f"grad={replay_row.get('grad_norm', float('nan')):.3f}"
                        )
        else:
            if hdp_rollout_phase:
                current_replay_updates = 0
            elif replay_updates_per_epoch < 0:
                current_replay_updates = len(replay_buffer)
                if distributed:
                    replay_count_tensor = torch.tensor(
                        [current_replay_updates], device=device, dtype=torch.int64
                    )
                    dist.all_reduce(replay_count_tensor, op=dist.ReduceOp.MIN)
                    current_replay_updates = int(replay_count_tensor.item())
            else:
                current_replay_updates = replay_updates_per_epoch
            replay_ready = bool(replay_buffer) and current_replay_updates > 0
            if distributed:
                ready_tensor = torch.tensor(
                    [1 if replay_ready else 0], device=device, dtype=torch.int64
                )
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
                replay_ready = bool(ready_tensor.item())
            if replay_ready:
                for replay_index in range(current_replay_updates):
                    buffer_index = (
                        epoch * current_replay_updates + replay_index + rank
                    ) % len(replay_buffer)
                    replay_scene, replay_rollout = replay_buffer[buffer_index]
                    replay_start = (
                        replay_index // gradient_accumulation_scenes
                    ) * gradient_accumulation_scenes
                    replay_end = min(
                        replay_start + gradient_accumulation_scenes,
                        current_replay_updates,
                    )
                    replay_count = max(1, replay_end - replay_start)
                    replay_is_end = replay_index + 1 == replay_end
                    replay_row, _ = _train_scene(
                        model,
                        behavior_model,
                        model_args,
                        optimizer,
                        replay_scene,
                        rollout_config,
                        reward_config,
                        train_config,
                        device,
                        amp_dtype,
                        grad_scaler,
                        rollout_override=replay_rollout,
                        replay_update=True,
                        optimizer_step=replay_is_end,
                        zero_grad=replay_index == replay_start,
                        sync_gradients=replay_is_end,
                        gradient_scale=1.0 / replay_count,
                        scene_load_workers=int(
                            train_config.get("scene_load_workers", 1)
                        ),
                    )
                    replay_rows.append(replay_row)
                    if (
                        replay_is_end
                        and not bool(train_config.get("ema_per_epoch", False))
                    ):
                        update_ema(behavior_model, model, ema_decay)
        train_rows.extend(replay_rows)

        if bool(train_config.get("ema_per_epoch", False)):
            update_ema(behavior_model, model, ema_decay)

        if distributed:
            gathered_rows: list[list[dict[str, Any]] | None] | None = (
                [None] * world_size if is_main else None
            )
            dist.gather_object(train_rows, gathered_rows, dst=0, group=object_group)
            if is_main:
                assert gathered_rows is not None
                train_rows = [row for rank_rows in gathered_rows if rank_rows for row in rank_rows]

        if is_main:
            train_summary: dict[str, Any] = {
                "scene_count": int(sum(int(row.get("scene_count", 1)) for row in train_rows)),
                "epoch": epoch,
                "cycle": 1 + (epoch - 1) // max(1, hdp_rollout_interval),
                "rollout_refresh": float(hdp_rollout_phase),
                "replay_training": float(not hdp_rollout_phase),
            }
            numeric_keys = sorted(
                {
                    key
                    for row in train_rows
                    for key, value in row.items()
                    if key not in {"scene_count", "epoch"}
                    and isinstance(value, (int, float))
                }
            )
            for key in numeric_keys:
                values = [
                    float(row[key])
                    for row in train_rows
                    if key in row
                    and isinstance(row[key], (int, float))
                    and math.isfinite(float(row[key]))
                ]
                if values:
                    train_summary[f"mean_{key}"] = float(np.mean(values))
                    if key in {
                        "loss",
                        "grad_norm",
                        "det_reward",
                        "best_reward",
                        "reward_std",
                        "effective_sample_size",
                        "top1_weight_share",
                        "best_vs_det_reward_gain",
                    }:
                        train_summary[f"p50_{key}"] = float(np.median(values))
                        train_summary[f"p90_{key}"] = float(np.percentile(values, 90))
            train_summary["elapsed_sec"] = time.time() - epoch_start
            train_summary["scenes_per_sec"] = (
                float(train_summary["scene_count"])
                / max(float(train_summary["elapsed_sec"]), 1e-6)
            )
            train_summary["optimizer_steps"] = int(
                sum(float(row.get("optimizer_step", 0.0)) for row in train_rows)
                / max(1, world_size)
            )
            train_summary["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if device.type == "cuda":
                train_summary["gpu_peak_allocated_gib_rank0"] = float(
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
                train_summary["gpu_peak_reserved_gib_rank0"] = float(
                    torch.cuda.max_memory_reserved(device) / (1024**3)
                )
            train_dir = run_dir / f"train_epoch_{epoch:03d}"
            _write_json(train_dir / "summary.json", train_summary)
            _write_json(train_dir / "scenes.json", train_rows)
            _append_jsonl(run_dir / "metrics.jsonl", {"phase": "train", **train_summary})

            eval_dir = run_dir / f"eval_epoch_{epoch:03d}"
            # The behavior model is intentionally a frozen rollout/reference
            # policy for the whole HDP-style epoch and is updated only by a
            # slow EMA at the boundary.  Evaluating it here would hide almost
            # all of the current policy update (especially with decay=.995).
            # Validate the actual synchronized policy; the EMA is still saved
            # for the next rollout epoch and for deployment comparison.
            eval_policy = model.module if hasattr(model, "module") else model
            eval_summary, eval_rows = evaluate_model(
                eval_policy,
                model_args,
                valid_paths,
                eval_rollout_config,
                reward_config,
                device,
                eval_dir / "rollouts",
                epoch=epoch,
                save_rollouts=save_rollouts,
                scene_batch_size=int(train_config.get("scene_batch_size", 1)),
                scene_load_workers=int(train_config.get("scene_load_workers", 1)),
            )
            _write_json(eval_dir / "summary.json", eval_summary)
            _write_json(eval_dir / "scenes.json", eval_rows)
            _append_jsonl(run_dir / "metrics.jsonl", {"phase": "eval", "epoch": epoch, **eval_summary})
            _wandb_log_summary(wandb_run, "train", train_summary, epoch=epoch)
            _wandb_log_summary(
                wandb_run,
                "eval",
                eval_summary,
                epoch=epoch,
                baseline=base_summary,
            )
            current_eval_reward = float(eval_summary.get("mean_det_reward", -math.inf))
            if current_eval_reward > best_eval_reward:
                best_eval_reward = current_eval_reward
                best_eval_epoch = epoch
                if wandb_run is not None:
                    wandb_run.summary["best/epoch"] = int(best_eval_epoch)
                    wandb_run.summary["best/mean_det_reward"] = float(best_eval_reward)
            save_checkpoint_pair(
                model,
                behavior_model,
                run_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                {"train": train_summary, "eval": eval_summary},
            )
            print(
                f"Epoch {epoch} done in {train_summary['elapsed_sec']:.1f}s: "
                f"train loss={train_summary.get('mean_loss', float('nan')):.5f}, "
                f"eval det reward={eval_summary.get('mean_det_reward', float('nan')):+.3f}, "
                f"eval collision={eval_summary.get('mean_det_collision', float('nan')):.3f}"
            )
        if distributed:
            dist.barrier()

    if is_main:
        _write_json(
            run_dir / "final_summary.json",
            {
                "baseline": base_summary,
                "start_epoch": start_epoch,
                "epochs": epochs,
                "resumed_replay": args.resume_replay_root is not None,
                "best_eval_epoch": best_eval_epoch,
                "best_eval_reward": best_eval_reward,
            },
        )
        print(f"AWR run complete: {run_dir}")
        if wandb_run is not None:
            wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
