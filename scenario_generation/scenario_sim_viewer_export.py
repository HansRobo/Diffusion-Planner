"""Export a finished scenario_sim suite run into the layout the result viewer reads.

Post-hoc only: it reads a run directory the driver already wrote and produces a second tree
beside it, never running inside a rollout and never changing what a worker writes.

The viewer's ``closed_loop_scenario`` dataset is three levels deep::

    <out_root>/                                  # one evaluation run
      groups_summary.json
      <group>/                                   # the map that was evaluated
        <scenario>/                              # the scenario id, shared across runs
          summary.json
          segments.jsonl
          <stem>.mp4
          <stem>_trajcolormap_<metric>.png
          <stem>/rollout.jsonl

A run is flat instead -- one directory per case -- so this module derives the two grouping
levels and re-links the artifacts. The output *shape* lives only in
:func:`write_viewer_tree`; everything else derives values and must not encode the layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import aggregate
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps

# ``aggregate`` derives these from row keys this path never writes, so it returns the
# neutral value of the reduction -- a mean over no samples is 0.0 -- which reads as a
# measured total failure. Drop them. Add a key here only when the path cannot measure it.
_UNMEASURED_SUMMARY_KEYS = ("mean_route_completion", "mean_gt_deviation_m")

# An absence is not a statement: a consumer that defaults a missing number to 0 prints the
# same misleading value, and null is no better (in JS ``null * 100`` is 0). Say it as data.
_UNMEASURED_MARKER_KEY = "unmeasured_keys"

# A case directory is the scenario path with separators replaced. Two producers disagree on
# the separator, so a run is matched against both rather than by inverting one.
_KEY_SEPARATORS = ("_", "__")

# The viewer maps a sim step to a video second with this constant for the
# closed_loop_scenario dataset. The export re-times each mp4 so the constant holds.
VIEWER_STEPS_PER_VIDEO_SEC = 40

_MAP_ID_RE = re.compile(r"/map/[^/]+/(\d+)/")

# What ``site_of`` returns when it could derive nothing. A run made entirely of these means
# the mode does not fit the suite.
_PLACEHOLDER_SITES = frozenset({"unknown_map", "unknown_scenario"})

# What ``site_of`` accepts.
_SITE_MODES = ("scenario_id", "map", "category", "flat")

# The suite's own scenario names, written beside it when the suite is pulled. A category is
# the leading component of a name like ``C-01-31500_case01_dp``. The names travel with the
# suite because the credentials that would fetch them are deliberately not on the cluster.
_NAMES_SIDECAR = "scenario_names.json"
_CATEGORY_RE = re.compile(r"^([A-Z])-")

# The id is a uuid somewhere in the scenario's path, but not at a fixed depth: the suite
# root is an expansion whose shape belongs to whoever expanded it. Match by form, not index.
_SCENARIO_ID_RE = re.compile(
    r"(?<![0-9a-fA-F-])([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})(?![0-9a-fA-F-])"
)


def sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``inf`` is a legitimate in-band value here (no finite clearance sample, no braking
    event), but ``json.dump`` writes it as the bare token ``Infinity``, which is not JSON
    and which ``JSON.parse`` rejects. A consumer outside Python cannot read a file that
    carries one, so the whole tree is sanitized on the way out.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def _dump_json(path: Path, obj: Any) -> None:
    """Write sanitized JSON that a browser can parse."""
    path.write_text(
        json.dumps(sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_run_context(run_dir: Path) -> dict[str, str]:
    """Pull the driver's ``key=value`` stamps out of ``run_context.txt``.

    First occurrence wins; a missing file yields an empty dict rather than an error, since
    a run assembled by hand still exports.
    """
    ctx: dict[str, str] = {}
    path = run_dir / "run_context.txt"
    if not path.is_file():
        return ctx
    for match in re.finditer(r"(\w+)=(\S+)", path.read_text(encoding="utf-8", errors="replace")):
        ctx.setdefault(match.group(1), match.group(2))
    return ctx


def key_aliases(rel: str) -> tuple[str, ...]:
    """Every case-directory name ``rel`` could have been written as.

    One rel is one submitted case; the aliases exist only to recognise the directory a
    producer chose to name it, so they must never be counted as separate cases.
    """
    stem = rel[: -len(".xosc")] if rel.endswith(".xosc") else rel
    return tuple(dict.fromkeys(stem.replace("/", sep) for sep in _KEY_SEPARATORS))


def load_submitted(run_dir: Path) -> list[tuple[str, str]]:
    """``[(case key, scenario path)]`` for every case the run *submitted*.

    This is the only honest denominator: a case that died before writing ``row.json``
    leaves no directory to count, so the artifacts on disk cannot reveal it.

    Two drivers record it differently. The per-case one lists scenario paths relative to
    the suite root and derives the directory name from them; the pooled one hands each
    worker the pair outright, so the key is read rather than derived.
    """
    work_json = run_dir / "work.json"
    if work_json.is_file():
        try:
            pairs = json.loads(work_json.read_text(encoding="utf-8"))
        except ValueError:
            pairs = []
        # Entries are resolved against the manifest's own directory, so a driver may record
        # either absolute paths or paths relative to the run. Absolute ones pass through.
        return [(Path(out).name, str(work_json.parent / osc)) for out, osc in pairs]

    work_tsv = run_dir / "work.tsv"
    if not work_tsv.is_file():
        return []
    out = []
    for line in work_tsv.read_text(encoding="utf-8").splitlines():
        _, _, rel = line.partition("\t")
        rel = rel.strip()
        if rel:
            out.append((key_aliases(rel)[0], rel))
    return out


def load_case_rels(run_dir: Path) -> dict[str, str]:
    """``{case directory name: scenario path}`` for every alias a case could be under."""
    return {
        alias: rel
        for key, rel in load_submitted(run_dir)
        for alias in dict.fromkeys((key, *key_aliases(rel)))
    }


def load_scenario_names(rel: str | None) -> dict[str, str]:
    """``{scenario id: display name}`` from the sidecar beside the suite.

    ``rel`` is any scenario path; the suite root is two levels above its
    ``<scenario id>/<version>/`` pair. A missing sidecar yields an empty map.
    """
    if not rel:
        return {}
    # The suite root is above the scenario's ``<id>/<version>/`` pair, but whether the
    # scenarios live directly under it varies, so look up a few levels rather than assume.
    for base in list(Path(rel).parents)[2:5]:
        try:
            return json.loads((base / _NAMES_SIDECAR).read_text(encoding="utf-8"))["scenarios"]
        except (OSError, ValueError, KeyError):
            continue
    return {}


def site_of(
    case_key: str, row: dict, rel: str | None, mode: str, names: dict[str, str] | None = None
) -> str:
    """Derive one of the viewer's two grouping levels for a case.

    ``scenario_id`` groups a scenario's parameter expansions together under the id the
    scenario is known by elsewhere, which is the identity a reader needs to line results up
    across runs. ``category`` is the leading component of the scenario's name, which is what
    a reader browses by. ``map`` is the location that was evaluated, and ``flat`` puts
    everything in one bucket.
    """
    if mode == "flat":
        return "all"
    if mode == "map":
        match = _MAP_ID_RE.search(str(row.get("map_path", "")))
        return match.group(1) if match else "unknown_map"
    scenario = _SCENARIO_ID_RE.search(rel or case_key)
    if mode == "category":
        display = (names or {}).get(scenario.group(1) if scenario else "", "")
        match = _CATEGORY_RE.match(display)
        return match.group(1) if match else "uncategorized"
    return scenario.group(1) if scenario else "unknown_scenario"


def scenario_description(rel: str | None, scenario_root: str | None = None) -> str | None:
    """The scenario's authored description, or ``None`` when it cannot be read.

    A scenario id is unique and stable but says nothing to a reader, so the one human-
    readable name the scenario carries is passed through. One driver records the scenario
    path outright and the other records it relative to the suite root, so an absolute path
    is used as-is. Best effort: the suite need not be present when a run is exported.
    """
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        if not scenario_root:
            return None
        path = Path(scenario_root) / path
    try:
        import xml.etree.ElementTree as ET

        for _, elem in ET.iterparse(path, events=("start",)):
            if elem.tag == "FileHeader":
                # First line only: authors put a tracker link on the following lines, and
                # this is a label in a list.
                text = (elem.get("description") or "").strip()
                return text.splitlines()[0].strip() if text else None
    except Exception:  # noqa: BLE001 - a label is never worth failing an export over
        return None
    return None


def scenario_parameters(rel: str | None) -> dict[str, str]:
    """The scenario's top-level parameter assignment, or ``{}`` when it cannot be read."""
    if not rel:
        return {}
    try:
        import xml.etree.ElementTree as ET

        out: dict[str, str] = {}
        for _, elem in ET.iterparse(rel, events=("start",)):
            if elem.tag == "ParameterDeclaration":
                out[elem.get("name") or ""] = elem.get("value") or ""
            elif elem.tag == "Storyboard":
                break
        return out
    except Exception:  # noqa: BLE001 - a name is never worth failing an export over
        return {}


def route_names(cases: list[tuple[str, str | None]]) -> dict[str, str]:
    """``{case key: route}`` naming each case by what its expansion actually varied.

    A positional index says nothing to a reader and does not survive a re-expansion, whereas
    the parameter assignment is both readable and the thing compared across runs. Falls back
    to the case key when the parameters cannot be read or would not name cases uniquely.
    """
    fallback = {key: key for key, _ in cases}
    params = {key: scenario_parameters(rel) for key, rel in cases}
    keys = set().union(*params.values()) if params else set()
    varying = sorted(k for k in keys if len({p.get(k) for p in params.values()}) > 1)
    if not varying:
        return fallback
    named = {
        key: "_".join(f"{k}{params[key].get(k, '')}".replace(".", "p") for k in varying)
        for key, _ in cases
    }
    # A collision would silently overwrite one case's media with another's.
    return named if len(set(named.values())) == len(named) else fallback


def _version_of(rel: str | None) -> str | None:
    """The scenario version directory from a path shaped ``<id>/<version>/<file>``."""
    parents = Path(rel).parents if rel else []
    return parents[0].name if len(parents) >= 2 else None


def find_mp4(case_dir: Path, stem: str) -> Path | None:
    """Locate a case's video.

    Two producers disagree on where it goes -- the single-scenario worker encodes into the
    case directory, the suite evaluator into the run root -- so both are tried rather than
    assuming whichever one wrote this run.
    """
    for candidate in (case_dir / f"{stem}.mp4", case_dir.parent / f"{stem}.mp4"):
        if candidate.is_file():
            return candidate
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst``, falling back to a copy across filesystems.

    Hardlinking keeps the exported tree free: the raw run and the viewer tree name the same
    mp4 rather than storing it twice, which matters at suite scale.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def retime_mp4(src: Path, dst: Path, scale: float) -> bool:
    """Copy ``src`` to ``dst`` with its timestamps multiplied by ``scale``.

    The viewer maps a sim step to a video time with a constant per dataset; it cannot know
    how sparsely a given run was drawn. Rather than ask it to, the export makes the constant
    true by re-timing the container. Stream copy, so no frame is re-encoded.

    Returns False when nothing was written, leaving the caller to link the file unchanged.
    """
    if abs(scale - 1.0) < 1e-9:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-itsscale", f"{scale:.9g}",
         "-i", str(src), "-c", "copy", str(dst)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"viewer_export: retime failed for {src.name}: {proc.stderr.strip()}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return False
    return True


def collect_cases(
    run_dir: Path, site_mode: str, group_mode: str
) -> tuple[dict[tuple[str, str], list[dict]], list[str]]:
    """Read every case that produced a row, keyed by ``(group, scenario)``.

    Returns ``({(group, scenario): [row, ...]}, [case_key of every submitted case with no
    row])``. Each row gains ``route`` (its case key) when the producer did not stamp one,
    because that is what names the row's artifacts.
    """
    rels = load_case_rels(run_dir)
    submitted = load_submitted(run_dir)
    names = load_scenario_names(next(iter(rels.values()), None))
    by_site: dict[tuple[str, str], list[dict]] = {}
    found: set[str] = set()
    for row_path in sorted(run_dir.glob("*/row.json")):
        case_dir = row_path.parent
        case_key = case_dir.name
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"viewer_export: unreadable row for {case_key}: {exc}", file=sys.stderr)
            continue
        row["case_key"] = case_key
        row.setdefault("route", case_key)
        row["_case_dir"] = str(case_dir)
        row["_rel"] = rels.get(case_key)
        found.add(case_key)
        rel = rels.get(case_key)
        cell = (
            site_of(case_key, row, rel, group_mode, names),
            site_of(case_key, row, rel, site_mode, names),
        )
        by_site.setdefault(cell, []).append(row)
    # One entry per submitted case, so two spellings of one directory never count twice.
    missing = [
        key
        for key, rel in submitted
        if not any(a in found for a in dict.fromkeys((key, *key_aliases(rel))))
    ]
    return by_site, missing


def write_viewer_tree(
    out_root: Path,
    by_site: dict[tuple[str, str], list[dict]],
    *,
    meta: dict,
    site_errors: dict[str, str],
    names: dict[str, str] | None = None,
    site_labels: dict[str, str | None] | None = None,
    run_error: str | None = None,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> dict[str, dict[str, int]]:
    """Write the whole viewer tree. **This function owns the output layout.**

    ``<out_root>/groups_summary.json`` plus ``<out_root>/<group>/<scenario>/`` holding that
    scenario's summary, rows and media. Returns
    ``{"<group>/<scenario>": {"rows": n, "mp4": n, "traces": n, "colormaps": n}}`` so the
    caller can report counts it did not itself construct.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    sub_groups: dict[str, Any] = {}

    # A partial re-export must not drop groups written by an earlier one.
    manifest_path = out_root / "groups_summary.json"
    if manifest_path.is_file():
        try:
            sub_groups = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "sub_groups", {}
            )
        except (ValueError, AttributeError):
            sub_groups = {}

    # The time base is a property of the run, but the viewer holds it as a per-dataset
    # constant, so the video is moved to the constant instead.
    draw_every = int(meta.get("draw_every") or 0)
    fps = float(meta.get("fps") or 0)
    scale = (fps * draw_every / VIEWER_STEPS_PER_VIDEO_SEC) if draw_every and fps else 1.0

    for (group, site), rows in sorted(by_site.items()):
        site_dir = out_root / group / site
        site_dir.mkdir(parents=True, exist_ok=True)
        sub_groups.setdefault(group, {"overview": {}})
        tally = {"rows": 0, "mp4": 0, "traces": 0, "colormaps": 0}

        near_miss = float(rows[0].get("object", {}).get("miss_thresh_m") or 1.0)
        strong_brake = float(rows[0].get("strong_brake", {}).get("thresh_mps2") or -2.5)

        first_rel = rows[0].get("_rel")
        first_map = site_of("", rows[0], None, "map")
        # Named per scenario, because what varies between its expansions is only knowable
        # by comparing them.
        routes = route_names([(str(r["case_key"]), r.get("_rel")) for r in rows])

        clean_rows = []
        for row in rows:
            case_dir = Path(row.pop("_case_dir"))
            row.pop("_rel", None)
            row["route"] = routes.get(str(row["case_key"]), str(row["route"]))
            stem = str(row["route"])

            mp4 = find_mp4(case_dir, str(row["case_key"]))
            if mp4 is not None:
                dst = site_dir / f"{stem}.mp4"
                if not retime_mp4(mp4, dst, scale):
                    _link_or_copy(mp4, dst)
                tally["mp4"] += 1

            trace = case_dir / "rollout.jsonl"
            if trace.is_file():
                _link_or_copy(trace, site_dir / stem / "rollout.jsonl")
                tally["traces"] += 1
                # Drawn from the trace because the run deletes its PNGs after encoding.
                # Unobserved metrics are skipped rather than drawn as a measured "no event".
                rendered = render_trajectory_colormaps(
                    site_dir / stem,
                    site_dir,
                    stem,
                    metrics=colormap_metrics,
                    near_miss_thresh=near_miss,
                    strong_brake_mps2=strong_brake,
                    title=f"{site} {stem}",
                )
                tally["colormaps"] += len(rendered)

            clean_rows.append(row)
            tally["rows"] += 1

        # Named, never globbed: a DDP run leaves merged and per-shard files side by side.
        with (site_dir / "segments.jsonl").open("w", encoding="utf-8") as f:
            for row in clean_rows:
                f.write(json.dumps(sanitize(row), ensure_ascii=False, allow_nan=False) + "\n")

        summary = aggregate(clean_rows, near_miss, strong_brake_mps2=strong_brake)
        unmeasured = []
        for key in _UNMEASURED_SUMMARY_KEYS:
            if summary.pop(key, None) is not None:
                unmeasured.append(key)
        # The rollup drops the row's ``measured`` flag, so an unobserved family aggregates to
        # a zero that reads as "checked, found nothing". Carry the flag up.
        for family in ("red_light_violation",):
            block = summary.get(family)
            if isinstance(block, dict) and not any(
                r.get(family, {}).get("measured", True) for r in clean_rows
            ):
                block["measured"] = False
                unmeasured.append(family)
        # ``reproducer`` counts a mechanism this path does not have -- there is no recorded
        # drive to snap back to -- so its zeros are structural rather than observations.
        summary[_UNMEASURED_MARKER_KEY] = unmeasured + ["reproducer"]
        # The shortfall travels in the scenario's own summary because that is the only file
        # the viewer reads per scenario; a count kept only in the export report is invisible.
        # The map and the scenario's version are properties of the scenario, not levels of
        # the tree, and the viewer has only three.
        summary.update(
            {
                "n_scenarios": len(clean_rows),
                "map": first_map,
                "version": _version_of(first_rel),
                "scenario_name": (names or {}).get(site),
                "draw_every": meta.get("draw_every"),
                "fps": meta.get("fps"),
                "description": (site_labels or {}).get(site),
                "error": " / ".join(e for e in (site_errors.get(site), run_error) if e) or None,
            }
        )
        _dump_json(site_dir / "summary.json", summary)
        counts[f"{group}/{site}"] = tally

    _dump_json(manifest_path, {"overview": {}, "sub_groups": sub_groups})
    _dump_json(out_root / "export_meta.json", meta)
    return counts


def _report(
    out_root: Path,
    counts: dict[str, dict[str, int]],
    submitted: int,
    missing: list[str],
) -> str:
    """Render the count report. Failures are counted as absent rows, not as absent files.

    ``rows``/``mp4``/``traces`` all descend from the same set of ``row.json`` files, so
    agreeing with each other says nothing about cases that never produced one -- only the
    submitted count does.
    """
    lines = [f"site{'':<24}rows   mp4  trace  cmaps"]
    total = {"rows": 0, "mp4": 0, "traces": 0, "colormaps": 0}
    for site, t in sorted(counts.items()):
        lines.append(
            f"{site:<28}{t['rows']:>4}  {t['mp4']:>4}  {t['traces']:>5}  {t['colormaps']:>5}"
        )
        for k in total:
            total[k] += t[k]
    lines.append(
        f"{'TOTAL':<28}{total['rows']:>4}  {total['mp4']:>4}  "
        f"{total['traces']:>5}  {total['colormaps']:>5}"
    )
    lines.append("")
    lines.append(f"submitted cases : {submitted if submitted else 'unknown (no work.tsv)'}")
    lines.append(f"rows exported   : {total['rows']}")
    if missing:
        lines.append(f"MISSING rows    : {len(missing)}")
        lines += [f"  - {k}" for k in missing[:20]]
        if len(missing) > 20:
            lines.append(f"  ... and {len(missing) - 20} more")
    elif submitted:
        lines.append("MISSING rows    : 0")
    text = "\n".join(lines) + "\n"
    (out_root / "export_report.txt").write_text(text, encoding="utf-8")
    return text


def export(
    run_dir: Path,
    out_root: Path,
    *,
    site_mode: str = "scenario_id",
    group_mode: str = "map",
    fps: float = 10.0,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> dict[str, dict[str, int]]:
    """Export one scenario_sim run directory into ``out_root``."""
    ctx = parse_run_context(run_dir)
    by_site, missing = collect_cases(run_dir, site_mode, group_mode)
    if not by_site:
        raise SystemExit(f"viewer_export: no */row.json under {run_dir}")
    # Resolving nothing is a wrong assumption about the layout, not an empty run.
    if {site for _, site in by_site} <= _PLACEHOLDER_SITES:
        sample = next(iter(next(iter(by_site.values()))), {}).get("route")
        raise SystemExit(
            f"viewer_export: --site_from {site_mode} resolved no site for any case "
            f"(sample case: {sample})"
        )

    site_labels = {
        site: scenario_description(rows[0].get("_rel"), ctx.get("scenario_root"))
        for (_, site), rows in by_site.items()
    }

    rels = load_case_rels(run_dir)
    submitted = load_submitted(run_dir)
    names = load_scenario_names(next(iter(rels.values()), None))
    # A case with no row can only be blamed on a site when the site comes from the scenario's
    # path. What cannot be attributed still has to reach the manifest, as a run-level fact.
    site_errors: dict[str, str] = {}
    run_error: str | None = None
    if missing:
        if site_mode != "map":
            per_site: dict[str, list[str]] = {}
            for key in missing:
                per_site.setdefault(
                    site_of(key, {}, rels.get(key), site_mode, names), []
                ).append(key)
            site_errors = {
                s: f"{len(k)} of this site's case(s) produced no row: {', '.join(k[:5])}"
                for s, k in per_site.items()
            }
        else:
            run_error = (
                f"run incomplete: {len(missing)} of {len(submitted)} submitted case(s) produced "
                f"no row ({', '.join(missing[:5])}); not attributable to a site under "
                f"site_from={site_mode}"
            )

    meta = {
        "run_dir": str(run_dir),
        "site_mode": site_mode,
        "group_mode": group_mode,
        "fps": fps,
        # Frame n of the video is trace step n * draw_every; a consumer that syncs a plot to
        # the video needs both numbers.
        "draw_every": ctx.get("draw_every"),
        "scenario_root": ctx.get("scenario_root"),
        "ckpt": ctx.get("ckpt"),
        "dp_commit": ctx.get("dp_commit"),
        "branch": ctx.get("branch"),
        "max_steps": ctx.get("max_steps"),
        "submitted_cases": len(submitted),
        "missing_rows": missing,
    }

    counts = write_viewer_tree(
        out_root,
        by_site,
        meta=meta,
        site_errors=site_errors,
        names=names,
        site_labels=site_labels,
        run_error=run_error,
        colormap_metrics=colormap_metrics,
    )
    print(_report(out_root, counts, len(submitted), missing), end="")
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True, type=Path, help="a finished suite run directory")
    p.add_argument("--out_root", required=True, type=Path, help="viewer tree to write")
    p.add_argument(
        "--site_from",
        default="scenario_id",
        help="scenario_id | map | flat -- what becomes the viewer's scenario",
    )
    p.add_argument(
        "--group_from",
        default="map",
        help="what becomes the viewer's group, the level above the scenario (same choices)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="tick rate the mp4s were encoded at; the driver does not stamp it",
    )
    p.add_argument("--colormap_metrics", nargs="*", default=list(METRIC_CHOICES))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    for flag, value in (("--site_from", a.site_from), ("--group_from", a.group_from)):
        if value not in _SITE_MODES:
            raise SystemExit(f"{flag} must be one of {', '.join(_SITE_MODES)}, got {value!r}")
    export(
        a.run_dir,
        a.out_root,
        site_mode=a.site_from,
        group_mode=a.group_from,
        fps=a.fps,
        colormap_metrics=tuple(a.colormap_metrics),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
