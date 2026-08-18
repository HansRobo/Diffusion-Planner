"""Export a finished scenario_sim suite run into the layout the result viewer reads.

Post-hoc only: it reads a run directory and writes a second tree beside it::

    <out_root>/run.json  scenarios.json  cases.jsonl
              media/<scenario>/<case>.mp4  .rollout.jsonl  .<metric>.png

Anything a reader groups or filters by is a field rather than a directory, so a new grouping
costs no re-export. The layout lives only in :func:`write_viewer_tree`.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import aggregate
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps

# ``aggregate`` derives these from row keys this path never writes, so it returns 0.0 -- a
# mean over no samples -- which reads as a measured total failure. Drop them.
_UNMEASURED_SUMMARY_KEYS = ("mean_route_completion", "mean_gt_deviation_m")

# The viewer maps a sim step to a video second with this constant. The export re-times each
# mp4 so it holds.
VIEWER_STEPS_PER_VIDEO_SEC = 40

_VERDICT_KINDS = ("pass", "failure", "error", "undecided")

# The failure message names the condition that ended the run and the success conditions left
# unmet -- the scenario author's own criteria, which no metric in the row expresses.
_UNMET_RE = re.compile(r'^\s*-\s*"(.+?)"\s*$', re.M)

# The id is a uuid at a known place in the scenario path; anything else is a layout this
# cannot read, and saying so beats grouping every case under one wrong name.
_SCENARIO_ID_RE = re.compile(
    r"(?<![0-9a-fA-F-])([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})(?![0-9a-fA-F-])"
)


def sanitize(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``inf`` is in-band here (no finite sample), but ``json.dump`` writes it as the bare token
    ``Infinity``, which ``JSON.parse`` rejects -- a consumer outside Python cannot read it.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def _dump_json(path: Path, obj: Any) -> None:
    """Write sanitized JSON that a browser can parse."""
    path.write_text(
        json.dumps(sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_run_context(run_dir: Path) -> dict[str, str]:
    """The driver's ``key=value`` stamps. First wins; a run assembled by hand still exports."""
    ctx: dict[str, str] = {}
    try:
        text = (run_dir / "run_context.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ctx
    for match in re.finditer(r"(\w+)=(\S+)", text):
        ctx.setdefault(match.group(1), match.group(2))
    return ctx


def load_submitted(run_dir: Path) -> list[tuple[Path, Path]]:
    """``[(case directory, scenario file)]`` for every case the run *submitted*.

    The only honest denominator: a case that died before writing ``row.json`` leaves nothing
    on disk to count, so the rows can never supply it. Deriving one from them would report a
    run that lost cases as complete, which is the one thing this file exists to prevent.
    """
    manifest = run_dir / "work.json"
    try:
        pairs = json.loads(manifest.read_text(encoding="utf-8"))
        # A driver may record either absolute paths or paths relative to the run.
        return [(run_dir / Path(case).name, run_dir / osc) for case, osc in pairs]
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"viewer_export: unusable submitted-case list {manifest}: {exc}")


def read_scenario(xosc: Path) -> tuple[str | None, str | None, str | None, dict[str, str]]:
    """``(scenario id, version, description, parameters)`` from ``<id>/<version>/<file>.xosc``.

    The id says nothing to a reader and an index says nothing about what an expansion varied.
    Best effort: the suite need not be present when a run is exported.
    """
    sid, version = xosc.parent.parent.name, xosc.parent.name
    if not _SCENARIO_ID_RE.fullmatch(sid):
        return None, version, None, {}
    try:
        root = ET.parse(xosc).getroot()
    except (OSError, ET.ParseError):
        return sid, version, None, {}
    header = root.find("FileHeader")
    # First line only: authors put a tracker link on the lines that follow.
    text = (header.get("description") if header is not None else "") or ""
    params = {
        p.get("name") or "": p.get("value") or ""
        for p in root.iterfind("ParameterDeclarations/ParameterDeclaration")
    }
    return sid, version, text.strip().splitlines()[0].strip() if text.strip() else None, params


def load_sidecar(xosc: Path) -> dict[str, dict[str, str]]:
    """Scenario display names and category labels, from the file the suite carries.

    No name is known to this module, so a renamed or added category needs no code change. The
    suite root sits above a scenario's ``<id>/<version>/`` pair at a depth that varies.
    """
    for base in list(xosc.parents)[2:5]:
        try:
            loaded = json.loads((base / "scenario_names.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return {"scenarios": loaded.get("scenarios") or {},
                "categories": loaded.get("categories") or {}}
    return {"scenarios": {}, "categories": {}}


def read_verdict(case_dir: Path) -> dict[str, Any]:
    """The scenario's own verdict on a case, or a statement that it never reached one.

    ``result.junit.xml`` exists only once the storyboard resolves, so an absent file means the
    run was cut short. The row's ``result_kind`` cannot say that -- it is preset to a Timeout
    failure at configure time -- so undecided is its own state here.

    A file that exists but cannot be read is neither: the run was judged and the record is
    damaged. Reading it as a pass would invent the rarest and most consequential verdict.
    """
    verdict = case_dir / "osp_out" / "result.junit.xml"
    if not verdict.exists():
        return {"decided": False}
    try:
        case = ET.parse(verdict).getroot().find(".//testcase")
    except (OSError, ET.ParseError) as exc:
        case, damage = None, str(exc)
    else:
        damage = None if case is not None else "no testcase in the verdict"
    if damage is not None:
        return {"decided": True, "kind": "Error", "type": "MalformedVerdict",
                "trigger": damage, "unmet": []}
    node, kind = case.find("failure"), "Failure"
    if node is None:
        node, kind = case.find("error"), "Error"
    if node is None:
        return {"decided": True, "kind": "Pass"}
    message = node.get("message") or ""
    head, _, rest = message.partition("\nUnmet success conditions:")
    return {
        "decided": True, "kind": kind, "type": node.get("type"),
        # A configure-time error carries no triggering condition, only a message.
        "trigger": head.split("): ", 1)[-1].strip() or None,
        "unmet": _UNMET_RE.findall(rest),
    }


def verdict_reason(verdict: dict[str, Any]) -> str | None:
    """One line naming why a case reached no row, from the verdict it managed to write."""
    parts = [verdict.get("type") or verdict.get("kind"), verdict.get("trigger")]
    return ": ".join(p for p in parts if p) or None


def route_names(cases: list[dict]) -> list[str]:
    """One name per case, from the parameters its expansion varied.

    The parameter assignment is readable and survives a re-expansion, which an index does not.
    Falls back to the case key when it cannot be read or would not name cases uniquely.
    """
    varying = sorted(
        k for k in cases[0]["params"] if len({c["params"].get(k) for c in cases}) > 1
    )
    names = ["_".join(f"{k}{c['params'].get(k, '')}".replace(".", "p") for k in varying)
             for c in cases]
    if varying and len(set(names)) == len(names):
        return names
    return [c["dir"].name for c in cases]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst``, copying across filesystems.

    The raw run and the exported tree name the same file rather than storing it twice. Only a
    link across filesystems falls back to a copy; a destination that already exists means two
    cases collided, and the export is built in an empty tree so that cannot happen quietly.
    """
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)


def place_mp4(src: Path, dst: Path, scale: float) -> bool:
    """Put the video where the viewer looks, with its timestamps multiplied by ``scale``.

    The viewer's step-to-second constant cannot know how sparsely a run was drawn, so the
    export makes it true by re-timing the container -- a stream copy, no frame re-encoded.
    Publishing the source video when that fails would put a file the viewer mis-times under a
    name that promises otherwise, so nothing is published and ``False`` says so.
    """
    if abs(scale - 1.0) < 1e-9:
        _link_or_copy(src, dst)
        return True
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-itsscale", f"{scale:.9g}",
         "-i", str(src), "-c", "copy", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return True
    print(f"viewer_export: no video for {src.name}, re-timing failed: {proc.stderr.strip()}",
          file=sys.stderr)
    dst.unlink(missing_ok=True)
    return False


def collect_cases(run_dir: Path) -> dict[str, list[dict]]:
    """Every submitted case, keyed by the scenario it belongs to.

    Grouping by the submitted list rather than by the rows found is what lets a scenario whose
    every case failed still appear: it has cases, none of which produced a row.
    """
    by_scenario: dict[str, list[dict]] = {}
    for case_dir, xosc in load_submitted(run_dir):
        sid, version, description, params = read_scenario(xosc)
        try:
            row = json.loads((case_dir / "row.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            row = None
        by_scenario.setdefault(sid or "unknown_scenario", []).append({
            "dir": case_dir, "row": row, "params": params,
            "version": version, "description": description,
            "verdict": read_verdict(case_dir),
        })
    return by_scenario


def write_viewer_tree(
    out_root: Path,
    by_scenario: dict[str, list[dict]],
    *,
    meta: dict,
    names: dict[str, str],
    categories: dict[str, str],
    scale: float,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> dict[str, dict[str, int]]:
    """Write the whole export into an empty ``out_root``. **This function owns the layout.**

    Returns ``{scenario: {"rows": n, "mp4": n, "traces": n, "colormaps": n}}``.
    """
    counts: dict[str, dict[str, int]] = {}
    scenarios: dict[str, Any] = {}
    case_lines: list[str] = []

    for scenario, cases in sorted(by_scenario.items()):
        tally = {"rows": 0, "mp4": 0, "traces": 0, "colormaps": 0}
        rows = [c["row"] for c in cases if c["row"] is not None]
        missing = [c for c in cases if c["row"] is None]
        if rows:
            (out_root / "media" / scenario).mkdir(parents=True)
        display = names.get(scenario)
        letter = display[0] if display and re.match(r"[A-Z]-", display) else None
        scenario_dir = out_root / "media" / scenario

        for case, route in zip(cases, route_names(cases)):
            if case["row"] is None:
                continue
            case_dir, row = case["dir"], case["row"]
            near_miss = float(row.get("object", {}).get("miss_thresh_m") or 1.0)
            strong_brake = float(row.get("strong_brake", {}).get("thresh_mps2") or -2.5)

            # Both drivers name the video after the case directory they were handed.
            mp4 = case_dir / f"{case_dir.name}.mp4"
            if mp4.is_file():
                tally["mp4"] += place_mp4(mp4, scenario_dir / f"{route}.mp4", scale)
            trace = case_dir / "rollout.jsonl"
            if trace.is_file():
                _link_or_copy(trace, scenario_dir / f"{route}.rollout.jsonl")
                tally["traces"] += 1
                # Drawn from the trace: the run deletes its PNGs after encoding, and an
                # unobserved metric is skipped rather than drawn as a measured "no event".
                rendered = render_trajectory_colormaps(
                    case_dir, scenario_dir, route, metrics=colormap_metrics,
                    near_miss_thresh=near_miss, strong_brake_mps2=strong_brake,
                    title=f"{display or scenario} {route}",
                )
                for metric, drawn in rendered.items():
                    shutil.move(str(drawn), scenario_dir / f"{route}.{metric}.png")
                tally["colormaps"] += len(rendered)

            row["case_key"] = case_dir.name
            row["route"] = route
            row["scenario"] = scenario
            case_lines.append(json.dumps(
                sanitize({**row, "verdict": case["verdict"]}), ensure_ascii=False, allow_nan=False
            ))
            tally["rows"] += 1

        summary = aggregate(rows, float(rows[0].get("object", {}).get("miss_thresh_m") or 1.0),
                            strong_brake_mps2=float(
                                rows[0].get("strong_brake", {}).get("thresh_mps2") or -2.5)
                            ) if rows else None
        unmeasured = []
        if summary is not None:
            unmeasured = [k for k in _UNMEASURED_SUMMARY_KEYS if summary.pop(k, None) is not None]
            # The rollup drops the row's ``measured`` flag, so an unobserved family aggregates
            # to a zero that reads as "checked, found nothing". Carry the flag up.
            block = summary.get("red_light_violation")
            if isinstance(block, dict) and not any(
                r.get("red_light_violation", {}).get("measured", True) for r in rows
            ):
                block["measured"] = False
                unmeasured.append("red_light_violation")

        scenarios[scenario] = {
            "name": display,
            "category": letter,
            "category_name": categories.get(letter or "", None),
            "description": next((c["description"] for c in cases if c["description"]), None),
            "map": Path(rows[0]["map_path"]).parts[-3] if rows and rows[0].get("map_path") else None,
            "version": next((c["version"] for c in cases if c["version"]), None),
            "n_cases": len(rows),
            "verdicts": {
                k: sum(1 for c in cases if c["row"] is not None
                       and (c["verdict"]["kind"].lower() if c["verdict"]["decided"] else "undecided") == k)
                for k in _VERDICT_KINDS
            },
            "error": (
                f"{len(missing)} of this scenario's case(s) produced no row: "
                f"{', '.join(c['dir'].name for c in missing[:5])}"
            ) if missing else None,
            "unmeasured_keys": unmeasured + ["reproducer"],
            "summary": sanitize(summary),
        }
        counts[scenario] = tally

    meta["verdicts"] = {
        k: sum(e["verdicts"][k] for e in scenarios.values()) for k in _VERDICT_KINDS
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cases.jsonl").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    _dump_json(out_root / "scenarios.json", scenarios)
    _dump_json(out_root / "run.json", meta)
    return counts


def _report(out_root: Path, counts: dict[str, dict[str, int]], submitted: int,
            missing: list[str]) -> str:
    """Render the count report. A failure is an absent row, not an absent file.

    ``rows``/``mp4``/``traces`` all descend from the same rows, so their agreeing says nothing
    about a case that never produced one. Only the submitted count does.
    """
    lines = [f"scenario{'':<20}rows   mp4  trace  cmaps"]
    total = {"rows": 0, "mp4": 0, "traces": 0, "colormaps": 0}
    for scenario, t in sorted(counts.items()):
        lines.append(f"{scenario:<28}{t['rows']:>4}  {t['mp4']:>4}  {t['traces']:>5}  {t['colormaps']:>5}")
        for k in total:
            total[k] += t[k]
    lines += [
        f"{'TOTAL':<28}{total['rows']:>4}  {total['mp4']:>4}  {total['traces']:>5}  {total['colormaps']:>5}",
        "",
        f"submitted cases : {submitted or 'unknown (no work list)'}",
        f"rows exported   : {total['rows']}",
        f"MISSING rows    : {len(missing)}",
    ]
    lines += [f"  - {k}" for k in missing[:20]]
    if len(missing) > 20:
        lines.append(f"  ... and {len(missing) - 20} more")
    text = "\n".join(lines) + "\n"
    (out_root / "export_report.txt").write_text(text, encoding="utf-8")
    return text


def export(
    run_dir: Path,
    out_root: Path,
    *,
    fps: float = 10.0,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> dict[str, dict[str, int]]:
    """Export one scenario_sim run directory into ``out_root``."""
    run_dir, out_root = Path(run_dir), Path(out_root)
    # Publishing renames out_root aside and deletes it. Overlapping the run would destroy the
    # very artifacts this reads, so it is refused before anything is written.
    source, destination = run_dir.resolve(), out_root.resolve()
    if (source == destination or destination.is_relative_to(source)
            or source.is_relative_to(destination)):
        raise SystemExit(f"viewer_export: --out_root {out_root} overlaps --run_dir {run_dir}")
    ctx = parse_run_context(run_dir)
    submitted = load_submitted(run_dir)
    by_scenario = collect_cases(run_dir)
    if not by_scenario:
        raise SystemExit(f"viewer_export: no submitted case under {run_dir}")
    # Resolving no id at all is a wrong assumption about the run's layout, not an empty run.
    if set(by_scenario) == {"unknown_scenario"}:
        raise SystemExit(f"viewer_export: no scenario id in any case path under {run_dir}")

    sidecar = load_sidecar(submitted[0][1])
    # The driver stamps ``off`` when it drew nothing, so this is not always a number. Nothing
    # drawn means nothing to re-time, and the raw value still goes out for a reader to see.
    drawn_every = ctx.get("draw_every") or ""
    draw_every = int(drawn_every) if drawn_every.isdigit() else 0
    meta = {
        "run_dir": str(run_dir),
        "fps": fps,
        # Frame n of the video is trace step n * draw_every; a consumer that syncs a plot to
        # the video needs both numbers.
        "draw_every": ctx.get("draw_every"),
        "scenario_root": ctx.get("scenario_root"),
        "ckpt": ctx.get("ckpt"),
        "dp_commit": ctx.get("dp_commit"),
        "branch": ctx.get("branch"),
        "max_steps": ctx.get("max_steps"),
        "suite_name": None,
        "submitted_cases": len(submitted),
        # A key alone says a case vanished; the verdict it wrote before dying says why.
        "missing_rows": [
            {"case_key": c["dir"].name, "reason": verdict_reason(c["verdict"])}
            for cases in by_scenario.values() for c in cases if c["row"] is None
        ],
    }

    # Built beside the published tree and swapped in whole: a run that dies partway leaves the
    # previous export intact rather than pairing its metadata with this run's media.
    out_root.parent.mkdir(parents=True, exist_ok=True)
    staging = out_root.with_name(f"{out_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        counts = write_viewer_tree(
            staging,
            by_scenario,
            meta=meta,
            names=sidecar["scenarios"],
            categories=sidecar["categories"],
            scale=(fps * draw_every / VIEWER_STEPS_PER_VIDEO_SEC) if draw_every and fps else 1.0,
            colormap_metrics=colormap_metrics,
        )
        report = _report(staging, counts, len(submitted),
                         [m["case_key"] for m in meta["missing_rows"]])
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    previous = out_root.with_name(f"{out_root.name}.previous-{os.getpid()}")
    if out_root.exists():
        out_root.rename(previous)
    staging.rename(out_root)
    shutil.rmtree(previous, ignore_errors=True)
    print(report, end="")
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True, type=Path, help="a finished suite run directory")
    p.add_argument("--out_root", required=True, type=Path, help="viewer tree to write")
    p.add_argument(
        "--fps", type=float, default=10.0,
        help="tick rate the mp4s were encoded at; the driver does not stamp it",
    )
    p.add_argument("--colormap_metrics", nargs="*", default=list(METRIC_CHOICES))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    export(a.run_dir, a.out_root, fps=a.fps, colormap_metrics=tuple(a.colormap_metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
