"""Export a finished scenario_sim suite run into the layout the result viewer reads.

Post-hoc only: it reads a run directory the driver already wrote and produces a second tree
beside it, never running inside a rollout and never changing what a worker writes.

The listing is three files at the run root and the bulk sits under ``media/``::

    <out_root>/
      run.json            provenance, counts, what was not measured
      scenarios.json      one entry per scenario: name, category, map, version, summary
      cases.jsonl         one line per case
      media/<scenario>/<case>.mp4
                        <case>.rollout.jsonl
                        <case>.<metric>.png

Anything a reader groups or filters by is a field rather than a directory, so a new
grouping costs no re-export. The output *shape* lives only in :func:`write_viewer_tree`;
everything else derives values and must not encode the layout.
"""

from __future__ import annotations

import argparse
import errno
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import aggregate
from scenario_generation.scene_trace import asset_dir as scene_asset_dir_for
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

# The interpreter's failure message names the condition that ended the run and lists the
# success conditions left unmet. Those names are the scenario author's own criteria, so they
# say what a case was judged against -- which no metric in the row does.
_TRIGGER_RE = re.compile(r"\):\s*(.*?)(?:\nUnmet success conditions:|\Z)", re.S)
_UNMET_RE = re.compile(r'^\s*-\s*"(.+?)"\s*$', re.M)

# What ``site_of`` returns when it could derive nothing. A run made entirely of these means
# the mode does not fit the suite.
_PLACEHOLDER_SITES = frozenset({"unknown_map", "unknown_scenario"})

# What ``site_of`` accepts.
_SITE_MODES = ("scenario_id", "map", "flat")

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


def load_sidecar(rel: str | None) -> dict[str, dict[str, str]]:
    """The suite's sidecar: scenario display names and the category labels they use.

    The names live with the suite because the credentials that would fetch them are
    deliberately not on the cluster. Nothing here is known to this module -- both maps come
    from the file, so a renamed or added category needs no code change.

    ``rel`` is any scenario path; the suite root is above its ``<id>/<version>/`` pair, but
    whether the scenarios sit directly under it varies, so a few levels are tried.
    """
    if not rel:
        return {"scenarios": {}, "categories": {}}
    for base in list(Path(rel).parents)[2:5]:
        try:
            loaded = json.loads((base / _NAMES_SIDECAR).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return {
            "scenarios": loaded.get("scenarios") or {},
            "categories": loaded.get("categories") or {},
        }
    return {"scenarios": {}, "categories": {}}


def category_of(display_name: str | None) -> str | None:
    """The category letter a scenario name leads with, or ``None`` when it has none."""
    match = _CATEGORY_RE.match(display_name or "")
    return match.group(1) if match else None


def site_of(
    case_key: str, row: dict, rel: str | None, mode: str, names: dict[str, str] | None = None
) -> str:
    """Derive one of the viewer's two grouping levels for a case.

    ``scenario_id`` is the id the scenario is known by elsewhere, which is the identity a
    reader needs to line results up across runs. ``map`` is the location that was evaluated,
    and ``flat`` puts everything in one bucket.
    """
    if mode == "flat":
        return "all"
    if mode == "map":
        match = _MAP_ID_RE.search(str(row.get("map_path", "")))
        return match.group(1) if match else "unknown_map"
    scenario = _SCENARIO_ID_RE.search(rel or case_key)
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
    """Hardlink ``src`` to ``dst``, copying only across filesystems.

    The raw run and the viewer tree name the same file rather than storing it twice. ``EXDEV``
    is the one failure a copy answers; any other says the destination is wrong. An existing
    ``dst`` means two cases resolved to one name, which must not read as a successful export.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except FileExistsError:
        raise SystemExit(f"viewer_export: two cases claim {dst}") from None
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
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


def read_verdict(case_dir: Path) -> dict[str, Any]:
    """The scenario's own verdict on a case, or a statement that it never reached one.

    The interpreter writes ``result.junit.xml`` when and only when the storyboard resolves,
    so an absent file means the rollout hit the step limit before the scenario decided. The
    row's ``result_kind`` cannot express that: it is preset to a Timeout failure at configure
    time, so a case that never decided still reads ``Failure``. Undecided is its own state.
    """
    path = case_dir / "osp_out" / "result.junit.xml"
    if not path.is_file():
        return {"decided": False}
    try:
        case = ET.parse(path).getroot().find(".//testcase")
    except (OSError, ET.ParseError) as exc:
        print(f"viewer_export: unreadable verdict for {case_dir.name}: {exc}", file=sys.stderr)
        return {"decided": False}
    if case is None:
        return {"decided": False}
    node, kind = case.find("failure"), "Failure"
    if node is None:
        node, kind = case.find("error"), "Error"
    if node is None:
        return {"decided": True, "kind": "Pass"}
    message = node.get("message") or ""
    trigger = _TRIGGER_RE.search(message)
    return {
        "decided": True,
        "kind": kind,
        "type": node.get("type"),
        # A configure-time error carries no triggering condition, only a message; that message
        # is the whole of what it has to say, so it stands in as the trigger.
        "trigger": trigger.group(1).strip() if trigger else (message.strip() or None),
        "unmet": _UNMET_RE.findall(message),
    }


def verdict_reason(case_dir: Path) -> str | None:
    """One line naming why a case reached no row, from its verdict if it wrote one."""
    verdict = read_verdict(case_dir)
    if not verdict.get("decided"):
        return None
    parts = [verdict.get("type") or verdict.get("kind"), verdict.get("trigger")]
    return ": ".join(p for p in parts if p) or None


def _tally_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    """Three decided counts and the undecided one, so no consumer has to subtract."""
    out = {"pass": 0, "failure": 0, "error": 0, "undecided": 0}
    for verdict in verdicts:
        out[verdict["kind"].lower() if verdict.get("decided") else "undecided"] += 1
    return out


def collect_cases(
    run_dir: Path, site_mode: str
) -> tuple[dict[str, list[dict]], list[str]]:
    """Read every case that produced a row, keyed by scenario.

    Returns ``({scenario: [row, ...]}, [case_key of every submitted case with no row])``.
    """
    rels = load_case_rels(run_dir)
    submitted = load_submitted(run_dir)
    by_scenario: dict[str, list[dict]] = {}
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
        scenario = site_of(case_key, row, rels.get(case_key), site_mode)
        by_scenario.setdefault(scenario, []).append(row)
    # One entry per submitted case, so two spellings of one directory never count twice.
    missing = [
        key
        for key, rel in submitted
        if not any(a in found for a in dict.fromkeys((key, *key_aliases(rel))))
    ]
    return by_scenario, missing


def _as_number(value) -> float:
    """``value`` as a float, or 0.0 when it does not name one.

    Run metadata is whatever the driver exported, so a switch that is off arrives as a word.
    0.0 means "no time base", which is what an absent value already means.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_viewer_tree(
    out_root: Path,
    by_scenario: dict[str, list[dict]],
    *,
    meta: dict,
    scenario_errors: dict[str, str],
    names: dict[str, str] | None = None,
    categories: dict[str, str] | None = None,
    run_error: str | None = None,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    include_legacy_media: bool = False,
) -> dict[str, dict[str, int]]:
    """Write the whole export. **This function owns the output layout.**

    The listing is three files at the run root, so opening it costs three reads rather than
    two per scenario. Everything a reader might group or filter by -- category, map, version
    -- is a field on the scenario rather than a directory, so a new grouping needs no
    re-export. Media stays per scenario, where the bulk is::

        <out_root>/
          run.json            provenance, counts, what was not measured
          scenarios.json      one entry per scenario: name, category, map, version, summary
          cases.jsonl         one line per case
          maps/<map_ref>.json.gz
          media/<scenario>/<case>.scene.jsonl.gz
                            <case>.rollout.jsonl

    Returns ``{scenario: {"rows": n, "mp4": n, "traces": n, "colormaps": n}}``.
    """
    # Laid over an earlier export, the index files are replaced but its media stays: one run
    # described, two contained.
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"viewer_export: {out_root} is not empty -- export into a new tree")
    out_root.mkdir(parents=True, exist_ok=True)
    media_root = out_root / "media"
    counts: dict[str, dict[str, int]] = {}
    scenarios: dict[str, Any] = {}
    case_lines: list[str] = []

    # Legacy videos only; a scene trace carries simulation steps and needs no retiming. The run
    # reports these as configured, so neither is guaranteed numeric.
    draw_every = _as_number(meta.get("draw_every"))
    fps = _as_number(meta.get("fps"))
    scale = (fps * draw_every / VIEWER_STEPS_PER_VIDEO_SEC) if draw_every and fps else 1.0

    for scenario, rows in sorted(by_scenario.items()):
        scenario_dir = media_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        tally = {"rows": 0, "mp4": 0, "traces": 0, "colormaps": 0}

        near_miss = float(rows[0].get("object", {}).get("miss_thresh_m") or 1.0)
        strong_brake = float(rows[0].get("strong_brake", {}).get("thresh_mps2") or -2.5)
        first_rel = rows[0].get("_rel")
        display = (names or {}).get(scenario)
        letter = category_of(display)

        # Named per scenario, because what varies between its expansions is only knowable
        # by comparing them.
        routes = route_names([(str(r["case_key"]), r.get("_rel")) for r in rows])

        clean_rows = []
        case_verdicts = []
        for row in rows:
            case_dir = Path(row.pop("_case_dir"))
            row.pop("_rel", None)
            case = routes.get(str(row["case_key"]), str(row["route"]))
            row["route"] = case
            row["scenario"] = scenario

            scene_trace = case_dir / "scene_trace.jsonl.gz"
            if scene_trace.is_file():
                trace_dst = scenario_dir / f"{case}.scene.jsonl.gz"
                _link_or_copy(scene_trace, trace_dst)
                map_ref = None
                try:
                    with gzip.open(scene_trace, "rt", encoding="utf-8") as trace_file:
                        for line in trace_file:
                            candidate = json.loads(line)
                            if candidate.get("event") == "header":
                                map_ref = candidate.get("map_ref")
                                break
                except (OSError, ValueError):
                    map_ref = None
                row["scene_trace"] = f"media/{scenario}/{case}.scene.jsonl.gz"
                if map_ref:
                    source_map = scene_asset_dir_for(case_dir.parent) / f"{map_ref}.json.gz"
                    if source_map.is_file():
                        map_dst = out_root / "maps" / source_map.name
                        map_dst.parent.mkdir(parents=True, exist_ok=True)
                        # The first case to reach it places it. A name already taken is the
                        # sharing working: the name is a digest of the contents.
                        if not map_dst.exists():
                            _link_or_copy(source_map, map_dst)
                        row["map_asset"] = f"maps/{source_map.name}"

            mp4 = find_mp4(case_dir, str(row["case_key"])) if include_legacy_media else None
            if mp4 is not None:
                dst = scenario_dir / f"{case}.mp4"
                if not retime_mp4(mp4, dst, scale):
                    _link_or_copy(mp4, dst)
                tally["mp4"] += 1

            trace = case_dir / "rollout.jsonl"
            if trace.is_file():
                _link_or_copy(trace, scenario_dir / f"{case}.rollout.jsonl")
                tally["traces"] += 1
            if include_legacy_media and trace.is_file():
                # Drawn from the trace because the run deletes its PNGs after encoding.
                # Unobserved metrics are skipped rather than drawn as a measured "no event".
                # The renderer reads the trace as ``<dir>/rollout.jsonl``, so it gets a
                # directory of its own rather than the layout growing one per case.
                with tempfile.TemporaryDirectory() as staging:
                    _link_or_copy(trace, Path(staging) / "rollout.jsonl")
                    rendered = render_trajectory_colormaps(
                        Path(staging),
                        Path(staging),
                        case,
                        metrics=colormap_metrics,
                        near_miss_thresh=near_miss,
                        strong_brake_mps2=strong_brake,
                        title=f"{display or scenario} {case}",
                    )
                    for metric, drawn in rendered.items():
                        shutil.move(str(drawn), scenario_dir / f"{case}.{metric}.png")
                tally["colormaps"] += len(rendered)

            clean_rows.append(row)
            verdict = read_verdict(case_dir)
            case_verdicts.append(verdict)
            # Kept out of the aggregated rows: ``aggregate`` rolls a row up by key prefix and
            # has no business seeing a block it cannot reduce.
            case_lines.append(
                json.dumps(
                    sanitize({**row, "verdict": verdict}), ensure_ascii=False, allow_nan=False
                )
            )
            tally["rows"] += 1

        summary = aggregate(clean_rows, near_miss, strong_brake_mps2=strong_brake)
        unmeasured = [k for k in _UNMEASURED_SUMMARY_KEYS if summary.pop(k, None) is not None]
        # The rollup drops the row's ``measured`` flag, so an unobserved family aggregates to
        # a zero that reads as "checked, found nothing". Carry the flag up.
        block = summary.get("red_light_violation")
        if isinstance(block, dict) and not any(
            r.get("red_light_violation", {}).get("measured", True) for r in clean_rows
        ):
            block["measured"] = False
            unmeasured.append("red_light_violation")

        scenarios[scenario] = {
            "name": display,
            "category": letter,
            "category_name": (categories or {}).get(letter or "", None),
            "description": scenario_description(first_rel, meta.get("scenario_root")),
            "map": site_of("", rows[0], None, "map"),
            "version": _version_of(first_rel),
            "n_cases": len(clean_rows),
            "verdicts": _tally_verdicts(case_verdicts),
            "error": " / ".join(
                e for e in (scenario_errors.get(scenario), run_error) if e
            )
            or None,
            "unmeasured_keys": unmeasured + ["reproducer"],
            "summary": sanitize(summary),
        }
        counts[scenario] = tally

    # A scenario whose every case failed writes no row, and would simply not appear -- a run
    # that lost cases would look complete to anyone reading this file alone. Say it with an
    # entry that has no cases. The counts stay per case present in ``cases.jsonl``, so a
    # scenario with none of them carries its story in ``error``.
    for scenario, message in scenario_errors.items():
        if scenario in scenarios:
            continue
        display = (names or {}).get(scenario)
        letter = category_of(display)
        scenarios[scenario] = {
            "name": display,
            "category": letter,
            "category_name": (categories or {}).get(letter or "", None),
            "description": None,
            "map": None,
            "version": None,
            "n_cases": 0,
            "verdicts": {"pass": 0, "failure": 0, "error": 0, "undecided": 0},
            "error": message,
            "unmeasured_keys": [],
            "summary": None,
        }

    meta["verdicts"] = {
        key: sum(e["verdicts"][key] for e in scenarios.values())
        for key in ("pass", "failure", "error", "undecided")
    }

    (out_root / "cases.jsonl").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    _dump_json(out_root / "scenarios.json", scenarios)
    _dump_json(out_root / "run.json", meta)
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
    fps: float = 10.0,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    include_legacy_media: bool = False,
) -> dict[str, dict[str, int]]:
    """Export one scenario_sim run directory into ``out_root``."""
    ctx = parse_run_context(run_dir)
    by_scenario, missing = collect_cases(run_dir, site_mode)
    if not by_scenario:
        raise SystemExit(f"viewer_export: no */row.json under {run_dir}")
    # Resolving nothing is a wrong assumption about the layout, not an empty run.
    if set(by_scenario) <= _PLACEHOLDER_SITES:
        sample = next(iter(next(iter(by_scenario.values()))), {}).get("route")
        raise SystemExit(
            f"viewer_export: --site_from {site_mode} resolved nothing for any case "
            f"(sample case: {sample})"
        )

    rels = load_case_rels(run_dir)
    submitted = load_submitted(run_dir)
    sidecar = load_sidecar(next(iter(rels.values()), None))

    # A case with no row can still be blamed on its scenario, because the submitted list
    # carries the path. What cannot be attributed becomes a run-level statement instead.
    scenario_errors: dict[str, str] = {}
    run_error: str | None = None
    missing_rows: list[dict[str, Any]] = []
    if missing:
        if site_mode != "map":
            per_scenario: dict[str, list[str]] = {}
            for key in missing:
                per_scenario.setdefault(
                    site_of(key, {}, rels.get(key), site_mode), []
                ).append(key)
            scenario_errors = {
                s: f"{len(k)} of this scenario's case(s) produced no row: {', '.join(k[:5])}"
                for s, k in per_scenario.items()
            }
        else:
            run_error = (
                f"run incomplete: {len(missing)} of {len(submitted)} submitted case(s) "
                f"produced no row ({', '.join(missing[:5])})"
            )
        # A key alone says a case vanished; the verdict it managed to write says why.
        missing_rows = [
            {"case_key": key, "reason": verdict_reason(run_dir / key)} for key in missing
        ]

    meta = {
        "run_dir": str(run_dir),
        "site_mode": site_mode,
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
        "missing_rows": missing_rows,
    }

    counts = write_viewer_tree(
        out_root,
        by_scenario,
        meta=meta,
        scenario_errors=scenario_errors,
        names=sidecar["scenarios"],
        categories=sidecar["categories"],
        run_error=run_error,
        colormap_metrics=colormap_metrics,
        include_legacy_media=include_legacy_media,
    )
    print(_report(out_root, counts, len(submitted), missing), end="")
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True, type=Path, help="a finished suite run directory")
    p.add_argument("--out_root", required=True, type=Path, help="viewer tree to write")
    p.add_argument("--include_legacy_media", action="store_true", help="also copy MP4 and draw legacy PNG colormaps")
    p.add_argument(
        "--site_from",
        default="scenario_id",
        help="scenario_id | map | flat -- what a case is grouped under",
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
    if a.site_from not in _SITE_MODES:
        raise SystemExit(
            f"--site_from must be one of {', '.join(_SITE_MODES)}, got {a.site_from!r}"
        )
    export(
        a.run_dir,
        a.out_root,
        site_mode=a.site_from,
        fps=a.fps,
        colormap_metrics=tuple(a.colormap_metrics),
        include_legacy_media=a.include_legacy_media,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
