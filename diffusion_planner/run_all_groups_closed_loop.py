"""Run ``valid_predictor_closed_loop.py`` once per group.

``--groups_npz_root`` accepts:
- Folder path: treated as one route directory -> ``{folder_name: {"all": [paths]}}``
- Flat JSON (list): ``["/path/to/route1", ...]`` -> ``{json_stem: {"all": [paths]}}``
- Grouped JSON (dict): ``{"g1": [...], "g2": [...]}`` -> ``{json_stem: {g1: [paths], g2: [paths]}}``

Outputs land in ``<out_root>/<json_name>/<group_name>/`` (objects mode) or
``<out_root>/<json_name>__noobj/<group_name>/`` (noobj mode); per-json ``groups.json``
files aggregate one JSON, ``<out_root>/groups.json`` aggregates everything.

Example::

    python diffusion_planner/run_all_groups_closed_loop.py \\
        --groups_npz_root override.json site.json \\
        --model_path /media/.../best_model.pth \\
        --out_root /media/.../cl_results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
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
    model,  # PyTorch model (for train.py direct call) or None for subprocess
    npz_root_list: list[str],
    out_dir: str | Path,
    args: argparse.Namespace,
    group_name: str | None = None,
    mode: str | None = None,
) -> None:
    """Run closed-loop evaluation for a single group; writes ``summary.json`` + ``segments.jsonl``
    under ``out_dir``. Wandb logging is left to the caller (re-reads via :func:`_load_group_results`).

    ``mode`` is passed explicitly so it doesn't have to be inferred from ``out_dir`` (a json_name
    containing ``__noobj`` would silently mis-infer).
    """
    from scenario_generation.closed_loop_evaluation import (
        ClosedLoopEvalConfig,
        FullRouteClosedLoopEvaluation,
        RolloutParams,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = group_name or out_dir.name
    if mode not in (None, "objects", "noobj"):
        raise ValueError(f"mode must be 'objects' or 'noobj', got {mode!r}")
    drop_objects = mode == "noobj"

    # Write npz roots to JSON if multiple, otherwise use as-is
    if len(npz_root_list) > 1:
        npz_root_arg = out_dir / "_npz_roots.json"
        npz_root_arg.write_text(json.dumps([str(p) for p in npz_root_list]))
    else:
        npz_root_arg = npz_root_list[0]

    if model is not None:
        seg_len = args.closed_loop_seg_len
        fps = float(args.closed_loop_fps)

        evaluator = FullRouteClosedLoopEvaluation(
            model,
            args,
            ClosedLoopEvalConfig(
                out_dir=out_dir,
                params=RolloutParams(
                    device=args.device,
                    near_miss_thresh=args.closed_loop_near_miss_thresh,
                    search_radius=args.closed_loop_search_radius,
                    warmup_steps=args.closed_loop_warmup_steps,
                    unstick_after=args.closed_loop_unstick_after,
                    unstick_advance_m=args.closed_loop_unstick_advance_m,
                    unstick_radius_mult=args.closed_loop_unstick_radius_mult,
                    unstick_teleport_after=args.closed_loop_unstick_teleport_after,
                    draw_every=args.closed_loop_draw_every if args.render_media else None,
                    replan_interval=args.closed_loop_replan_interval,
                    abort_deviation_m=args.closed_loop_abort_deviation_m,
                    abort_after=args.closed_loop_abort_after,
                    abort_max_snaps=args.closed_loop_abort_max_snaps,
                    draw_workers=args.closed_loop_draw_workers,
                    drop_objects=drop_objects,
                ),
                fps=fps,
                verbose=False,
            ),
            npz_root_arg,
            seg_len=seg_len,
        )
        evaluator.run()
    else:
        cli_path = Path(__file__).resolve().parent / "valid_predictor_closed_loop.py"
        cmd = [
            sys.executable,
            str(cli_path),
            "--model_path",
            str(args.model_path),
            "--npz_root",
            str(npz_root_arg),
            "--out_dir",
            str(out_dir),
        ]
        if hasattr(args, "extra_args") and args.extra_args:
            cmd.extend(args.extra_args)
        cmd.extend(["--draw_workers", str(args.closed_loop_draw_workers)])
        if drop_objects:
            cmd.append("--drop_objects")

        for attempt in range(1, 3):
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError as e:
                print(f"  [{label}] attempt {attempt}/2 failed: {e}", file=sys.stderr)


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
            json.loads(line)
            for line in segments_jsonl.read_text().splitlines()
            if line.strip()
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
    Path(out_dir, "groups.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


def run_closed_loop_main(
    model,  # PyTorch model (for train.py direct call) or None for subprocess
    groups_npz_root: list[str] | None = None,
    args: argparse.Namespace = None,
    out_root: str | Path | None = None,
    *,
    resolved: dict[str, dict[str, list[str]]] | None = None,
    wandb_run=None,  # Optional wandb.Run instance; if None, creates own session
    only_json: list[str] | None = None,
    object_modes: list[str] | None = None,
    render_media: bool = True,
) -> bool:
    """Unified entry point for closed-loop evaluation.

    Works both as:
    - Direct API call from train.py (model provided, wandb_run provided)
    - CLI entry point via main() (model=None, wandb_run=None)

    Either ``groups_npz_root`` OR pre-computed ``resolved`` must be provided;
    the latter lets callers that already resolved (and filtered) the inputs
    skip the duplicate ``resolve_closed_loop_inputs()`` + ``only_json`` filter
    pass.

    Output directory structure:
        <out_root>/<json_name>/<group_name>          (objects mode)
        <out_root>/<json_name>__noobj/<group_name>   (noobj mode)

    Writes:
        - ``<out_root>/groups.json`` (root aggregate)
        - ``<out_root>/<json_name>/groups.json`` and ``<out_root>/<json_name>__noobj/groups.json``
          (per-JSON aggregates)
        - W&B log payload if a run is provided (or one is created here)

    Returns True on success, False if no inputs were resolved (so CLI wrappers
    can map that to their own exit code).
    """
    if resolved is None:
        if groups_npz_root is None:
            raise ValueError("run_closed_loop_main: pass either groups_npz_root or resolved")
        resolved = resolve_closed_loop_inputs(groups_npz_root)
    if only_json:
        resolved = {k: v for k, v in resolved.items() if k in only_json}
    if not resolved:
        print(f"No inputs found under {groups_npz_root}", file=sys.stderr)
        return False

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    modes = object_modes or ["objects", "noobj"]

    # Triple loop: json_name -> group_name -> mode
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

                print(f"=== [{summary_key}] npz={npz_paths} -> out={mode_out_dir} ===")

                run_one_group(
                    model,
                    npz_paths,
                    mode_out_dir,
                    args,
                    group_name=summary_key,
                    mode=mode,
                )

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
            args, all_group_names, all_summaries, out_root, wandb_run, render_media=render_media
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups_npz_root",
        required=True,
        nargs="+",
        help="Input paths: folder(s), flat JSON(s) (list of paths), or grouped JSON(s) (dict). "
        "Each JSON/folder becomes its own top-level namespace.",
    )
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument(
        "--only_json",
        nargs="*",
        default=None,
        help="run only these JSON/folder names (e.g. override site)",
    )
    parser.add_argument(
        "--object_modes",
        nargs="+",
        choices=("objects", "noobj"),
        default=["objects", "noobj"],
        help="'objects'=normal, 'noobj'=empty-world ablation (--drop_objects). "
        "Each group runs once per mode; noobj gets a '__noobj' suffix on the out_dir.",
    )
    parser.add_argument(
        "--closed_loop_draw_workers",
        type=int,
        default=4,
        help="parallelism for per-frame trajectory drawing (1=serial). Forwarded as --draw_workers "
        "in the subprocess CLI path; used directly in the in-process path.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="passed through verbatim to valid_predictor_closed_loop.py",
    )
    parser.add_argument(
        "--render_media",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render video/colormap artifacts during wandb logging (default: on)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="optional: log to wandb (one run, all groups + per-json aggregates)",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--closed_loop_wandb_video_pick",
        choices=("worst", "first", "longest"),
        default="worst",
        help="which episode gets its video uploaded per group",
    )
    parser.add_argument(
        "--closed_loop_colormap_metrics",
        nargs="*",
        default=[
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ],
        help="per-step metrics rendered as trajectory-colormap images for wandb",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_root = args.out_root / timestamp
    if args.out_root.exists() and any(args.out_root.iterdir()):
        print(
            f"Output directory {args.out_root} already exists and is not empty. "
            f"Specify a different --out_root to avoid data loss or remove the old directory.",
            file=sys.stderr,
        )
        return 1
    args.out_root.mkdir(parents=True, exist_ok=True)

    ok = run_closed_loop_main(
        model=None,  # CLI mode: subprocess
        groups_npz_root=args.groups_npz_root,
        args=args,
        out_root=args.out_root,
        only_json=args.only_json,
        object_modes=args.object_modes,
        render_media=args.render_media,
    )
    return int(not ok)


def _log_to_wandb(
    args: object,
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
        run = wandb.init(
            project=getattr(args, "wandb_project", None),
            name=getattr(args, "wandb_run_name", None),
        )
        own_run = True
    else:
        own_run = False

    try:
        log = build_groups_wandb_log(
            {key: group_summaries[key] for key in group_names},
            out_root=out_root,
            video_pick=args.closed_loop_wandb_video_pick,
            colormap_metrics=tuple(args.closed_loop_colormap_metrics or ()),
            # ``summary`` doesn't store ``near_miss_thresh``; pass trainer's value so
            # colormap matches the rollout's actual threshold.
            near_miss_thresh_default=getattr(args, "closed_loop_near_miss_thresh", 0.5),
            render_media=args.render_media,
        )
        run.log(log)
        print(f"wandb: logged {len(group_summaries)} group(s) to run {run.id}")

        _refresh_workspace_view(run, args, group_names)
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
    # ``run.summary[k] = v`` queues; explicit ``update()`` flushes so the URL is on the
    # run page even if outer ``_log_to_wandb`` returns without further logging.
    run.summary.update()
    print(f"wandb: dashboard view saved → {url}")


if __name__ == "__main__":
    raise SystemExit(main())
