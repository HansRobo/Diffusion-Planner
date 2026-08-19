"""Run closed-loop evaluation for all groups.

Accepts the same ``ClosedLoopConfig`` fields as train.py. Example::

    python diffusion_planner/run_all_groups_closed_loop.py \\
        --closed_loop_npz_root override.json site.json \\
        --model_path /media/.../best_model.pth \\
        --out_root /media/.../cl_results
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from diffusion_planner.config.closed_loop_config import ClosedLoopConfig
from diffusion_planner.config.config_cli import build_config, build_parser, resolve_paths
from diffusion_planner.utils import ddp
from scenario_generation.wandb_closed_loop import build_groups_aggregate_log


def resolve_closed_loop_inputs(inputs: str | list[str]) -> dict[str, dict[str, list[str]]]:
    """Resolve input paths to ``{<json_name>: {<group_name>: [route_dirs]}}``.

    The top-level key is the JSON filename (without extension) or folder name;
    for folder or flat JSON (list) the inner key is ``"all"``; for grouped JSON
    (dict) the inner keys are the group's names from the JSON.
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    result: dict[str, dict[str, list[str]]] = {}

    for input_path in inputs:
        p = Path(input_path)

        if not p.exists():
            print(f"Warning: {input_path} does not exist, skipping", file=sys.stderr)
            continue

        if p.is_dir():
            # Folder: treat as one route directory
            name = p.name
            if name not in result:
                result[name] = {}
            if "all" not in result[name]:
                result[name]["all"] = []
            result[name]["all"].append(str(p))

        elif p.suffix == ".json":
            name = p.stem
            if name not in result:
                result[name] = {}

            with open(p, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Grouped JSON: {"g1": [...], "g2": [...]}
                for group_name, paths in data.items():
                    if group_name not in result[name]:
                        result[name][group_name] = []
                    for item in paths:
                        result[name][group_name].append(str(Path(item)))

            elif isinstance(data, list):
                # Flat JSON (list): ["path1", "path2", ...]
                if "all" not in result[name]:
                    result[name]["all"] = []
                for item in data:
                    result[name]["all"].append(str(Path(item)))

    return result


def run_one_group(
    model,  # always provided by the caller (train.py or main())
    model_args,  # model args from load_model (for closed-loop rollout)
    npz_root_list: list[str],
    out_dir: str | Path,
    cfg: ClosedLoopConfig,
    group_name: str | None = None,
    mode: str | None = None,
    render_media: bool = True,
) -> None:
    """Run closed-loop evaluation for a single group; writes ``summary.json`` + ``segments.jsonl``
    under ``out_dir``. Wandb logging is left to the caller.

    ``mode`` is passed explicitly so it doesn't have to be inferred from ``out_dir``
    (a json_name containing ``__noobj`` would silently mis-infer).
    """
    from scenario_generation.closed_loop_evaluation import (
        ClosedLoopEvalConfig,
        FullRouteClosedLoopEvaluation,
        RolloutParams,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode not in (None, "objects", "noobj"):
        raise ValueError(f"mode must be 'objects' or 'noobj', got {mode!r}")
    drop_objects = mode == "noobj"

    if len(npz_root_list) > 1:
        npz_root_arg = out_dir / "_npz_roots.json"
        npz_root_arg.write_text(json.dumps([str(p) for p in npz_root_list]))
    else:
        npz_root_arg = npz_root_list[0]

    ddp_rank = ddp.get_rank()
    ddp_world_size = ddp.get_world_size()

    evaluator = FullRouteClosedLoopEvaluation(
        model,
        model_args,
        ClosedLoopEvalConfig(
            out_dir=out_dir,
            params=RolloutParams(
                device=cfg.device,
                near_miss_thresh=cfg.closed_loop_near_miss_thresh,
                search_radius=cfg.closed_loop_search_radius,
                warmup_steps=cfg.closed_loop_warmup_steps,
                unstick_after=cfg.closed_loop_unstick_after,
                unstick_advance_m=cfg.closed_loop_unstick_advance_m,
                unstick_radius_mult=cfg.closed_loop_unstick_radius_mult,
                unstick_teleport_after=cfg.closed_loop_unstick_teleport_after,
                draw_every=cfg.closed_loop_draw_every if render_media else None,
                replan_interval=cfg.closed_loop_replan_interval,
                abort_deviation_m=cfg.closed_loop_abort_deviation_m,
                abort_after=cfg.closed_loop_abort_after,
                abort_max_snaps=cfg.closed_loop_abort_max_snaps,
                draw_workers=cfg.closed_loop_draw_workers,
                drop_objects=drop_objects,
            ),
            fps=float(cfg.closed_loop_fps),
            verbose=False,
        ),
        npz_root_arg,
        seg_len=cfg.closed_loop_seg_len,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    evaluator.run_distributed()


def _make_summary_key(json_name: str, group_name: str) -> str:
    """Build the summary key, e.g. 'override/departure' or 'site/all'."""
    if "/" in group_name or "/" in json_name:
        raise ValueError(
            f"json_name and group_name must not contain '/': got {json_name!r}, {group_name!r}"
        )
    return f"{json_name}/{group_name}"


def _load_group_results(out_dir: Path | str) -> dict[str, dict]:
    """Reload per-group results from ``out_dir`` (each ``<group_dir>/summary.json``
    augmented with rows from the matching ``segments.jsonl``). Groups missing
    ``segments.jsonl`` (partial run, manual deletion) are skipped with a warning."""
    out_dir = Path(out_dir)
    summaries: dict[str, dict] = {}
    for summary_file in out_dir.rglob("summary.json"):
        key = "/".join(summary_file.parent.relative_to(out_dir).parts)
        try:
            summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError as exc:
            print(
                f"Warning: skipping malformed summary at {summary_file}: {exc}",
                file=sys.stderr,
            )
            continue
        segments_jsonl = summary_file.parent / "segments.jsonl"
        if not segments_jsonl.is_file():
            print(
                f"Warning: {key} has no segments.jsonl, skipping",
                file=sys.stderr,
            )
            continue
        summary["segments"] = [
            json.loads(line) for line in segments_jsonl.read_text().splitlines() if line.strip()
        ]
        summaries[key] = summary
    return summaries


def _write_groups_manifest(out_dir: Path | str, summaries: dict[str, dict]) -> None:
    """Write ``<out_dir>/groups.json`` aggregating ``summaries`` via ``build_groups_aggregate_log``.

    The aggregate keys are prefixed with ``closed_loop_overview/`` internally;
    that prefix is stripped for the on-disk file so the manifest matches the
    shape consumed by ``wandb_closed_loop_workspace`` and the per-JSON helpers.
    """
    if summaries:
        aggregates = build_groups_aggregate_log(summaries, prefix="closed_loop_overview")
        manifest = {k.replace("closed_loop_overview/", ""): v for k, v in aggregates.items()}
    else:
        manifest = {}
    Path(out_dir, "groups.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def run_closed_loop_main(
    model,  # always provided by the caller
    model_args,  # model args from load_model (for closed-loop rollout)
    cfg: ClosedLoopConfig,
    out_root: str | Path | None = None,
    *,
    resolved: dict[str, dict[str, list[str]]] | None = None,
    wandb_run=None,  # Optional wandb.Run instance; if None, creates own session
    only_json: list[str] | None = None,
    render_media: bool | None = None,
) -> bool:
    """Unified entry point for closed-loop evaluation.

    Works both as:
    - Direct API call from train.py (model provided, wandb_run provided)
    - CLI entry point via main() (model provided directly)

    Either ``cfg.closed_loop_npz_root`` OR pre-computed ``resolved`` must be non-empty;
    the latter lets callers that already resolved (and filtered) the inputs
    skip the duplicate ``resolve_closed_loop_inputs()`` + ``only_json`` filter
    pass.

    Output directory structure:
        <out_root>/<json_name>/<group_name>          (objects mode)
        <out_root>/<json_name>__noobj/<group_name>   (noobj mode)

    Multi-GPU: launched via ``torch.distributed.run``. DDP rank is read from ``RANK``.
    All ranks evaluate all groups; per-group ``run_distributed()`` does its own barrier
    + rank-0 merge, so by the time we reach aggregation every rank's shards are flushed.
    Rank 0 then writes the manifest and logs to wandb.

    Writes:
        - ``<out_root>/groups.json`` (root aggregate)
        - ``<out_root>/<json_name>/groups.json`` and ``<out_root>/<json_name>__noobj/groups.json``
          (per-JSON aggregates)
        - W&B log payload if a run is provided (or one is created here)

    Returns True on success, False if no inputs were resolved (so CLI wrappers
    can map that to their own exit code).
    """
    if resolved is None:
        if not cfg.closed_loop_npz_root:
            print("No closed_loop_npz_root set, skipping closed-loop evaluation", file=sys.stderr)
            return False
        resolved = resolve_closed_loop_inputs(cfg.closed_loop_npz_root)
    if only_json:
        resolved = {k: v for k, v in resolved.items() if k in only_json}
    if not resolved:
        print(f"No inputs found under {cfg.closed_loop_npz_root}", file=sys.stderr)
        return False

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    modes = cfg.closed_loop_object_modes
    if render_media is None:
        render_media = cfg.render_media

    for json_name, groups in resolved.items():
        json_out_dir = out_root / json_name
        json_out_dir.mkdir(parents=True, exist_ok=True)

        for group_name, npz_paths in groups.items():
            for mode in modes:
                if mode == "objects":
                    mode_out_dir = json_out_dir / group_name
                    summary_key = _make_summary_key(json_name, group_name)
                else:  # noobj
                    json_mode_name = f"{json_name}__noobj"
                    mode_out_dir = out_root / json_mode_name / group_name
                    summary_key = _make_summary_key(json_mode_name, group_name)

                run_one_group(
                    model,
                    model_args,
                    npz_paths,
                    mode_out_dir,
                    cfg,
                    group_name=summary_key,
                    mode=mode,
                    render_media=render_media,
                )

    # rank-0 reads the per-group summaries; run_one_group already barriered + merged internally.
    if ddp.get_rank() == 0:
        all_summaries = _load_group_results(out_root)
        all_group_names = sorted(all_summaries.keys())

        _write_groups_manifest(out_root, all_summaries)

        for json_name in resolved:
            for mode in modes:
                json_label = json_name if mode == "objects" else f"{json_name}__noobj"
                json_prefix = f"{json_label}/"
                per_json_summaries = {
                    k: v for k, v in all_summaries.items() if k.startswith(json_prefix)
                }
                json_out_dir = out_root / json_label
                if json_out_dir.exists() and per_json_summaries:
                    _write_groups_manifest(json_out_dir, per_json_summaries)

        if all_group_names:
            _log_to_wandb(
                cfg, all_group_names, all_summaries, out_root, wandb_run, render_media=render_media
            )

    return True


def _build_parser() -> "argparse.ArgumentParser":
    parser = build_parser(ClosedLoopConfig, description=__doc__)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument(
        "--only_json",
        nargs="*",
        default=None,
        help="run only these JSON/folder names (e.g. override site)",
    )
    return parser


def main() -> int:
    from diffusion_planner.utils import ddp
    import torch

    from scenario_generation.simulate import load_model

    parser = _build_parser()
    args = parser.parse_args()
    resolve_paths(args, ClosedLoopConfig)

    cfg = build_config(ClosedLoopConfig, args)
    base_out_root = args.out_root

    # Init DDP when launched under torchrun (RANK/WORLD_SIZE present) so that
    # ddp.get_rank()/get_world_size() below reflect the real process group; without
    # this every rank would fall back to (0, 1) and silently re-run every route.
    global_rank, rank, world_size = ddp.ddp_setup_universal(False, cfg)
    print(f"{global_rank=}, {rank=}, {world_size=}")

    if cfg.device.startswith("cuda"):
        torch.cuda.set_device(rank)
        cfg.device = f"cuda:{rank}"

    model, model_args = load_model(args.model_path, cfg.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = base_out_root / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    ok = run_closed_loop_main(
        model=model,
        model_args=model_args,
        cfg=cfg,
        out_root=out_root,
        only_json=args.only_json,
    )
    return int(not ok)


def _log_to_wandb(
    cfg: ClosedLoopConfig,
    group_names: list[str],
    group_summaries: dict[str, dict],
    out_root: str | Path,
    run: "wandb.sdk.wandb_run.Run | None" = None,
    render_media: bool = True,
) -> None:
    """Push per-group closed-loop metrics to W&B and refresh the curated workspace view.

    Reuses ``run`` if given, else starts its own. The workspace view URL is written to
    ``run.summary["closed_loop/workspace_view_url"]`` so it's visible on the W&B run page.
    """
    import wandb

    from scenario_generation.wandb_closed_loop import build_groups_wandb_log

    if not group_summaries:
        return

    if run is None:
        # CLI entrypoint: own the wandb lifetime.
        run = wandb.init(project=cfg.wandb_project or None)
        own_run = True
    else:
        own_run = False

    try:
        log = build_groups_wandb_log(
            {key: group_summaries[key] for key in group_names},
            out_root=out_root,
            video_pick=cfg.closed_loop_wandb_video_pick,
            colormap_metrics=tuple(cfg.closed_loop_colormap_metrics or ()),
            near_miss_thresh_default=cfg.closed_loop_near_miss_thresh,
            render_media=render_media,
        )
        run.log(log)
        print(f"wandb: logged {len(group_summaries)} group(s) to run {run.id}")

        _refresh_workspace_view(run, cfg, group_names)
    finally:
        if own_run:
            wandb.finish()


def _refresh_workspace_view(
    run: "wandb.sdk.wandb_run.Run",
    args: object,
    group_names: list[str],
) -> None:
    """Rebuild the curated closed-loop workspace view for this run.

    Skipped when the user didn't opt-in (no ``wandb_project``), or when the workspace
    SDK isn't importable. API-side failures are logged and skipped (run metrics already
    landed); programming errors (empty list, etc.) still raise.
    """
    # Prefer explicit user opt-in; fall back to whatever the active run uses.
    project = getattr(args, "wandb_project", None) or run.project
    if not project:
        print(
            "wandb: skipping workspace refresh (no wandb_project set; "
            "pass --wandb_project to enable dashboard view).",
            file=sys.stderr,
        )
        return

    try:
        from scenario_generation.wandb_closed_loop_workspace import (
            build_closed_loop_workspace,
        )
    except ImportError as e:
        print(
            f"wandb: skipping workspace refresh (wandb_workspaces unavailable: {e})",
            file=sys.stderr,
        )
        return

    # Suffix ``run.id`` so two runs that share ``--exp_name`` don't upsert onto each
    # other's dashboard layout (wandb_workspaces upserts by name).
    base_name = run.name or run.id
    view_name = f"Closed-Loop / {base_name} ({run.id})"

    try:
        url = build_closed_loop_workspace(
            run.entity,
            project,
            group_names=list(group_names),
            name=view_name,
        )
    except ValueError as e:
        # Programming error — surface it.
        print(f"wandb: workspace build failed: {e}", file=sys.stderr)
        raise
    except Exception as e:
        # Network / W&B API error — log with traceback but don't crash; metrics landed.
        print(
            f"wandb: workspace refresh failed (run metrics still logged): {e}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc()
        return

    run.summary["closed_loop/workspace_view_url"] = url
    run.summary.update({"closed_loop/workspace_view_url": url})
    print(f"wandb: dashboard view saved → {url}")


if __name__ == "__main__":
    raise SystemExit(main())
