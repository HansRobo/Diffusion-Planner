#!/usr/bin/env python3
"""Build a self-contained, hash-verifiable bundle of every meeting visual.

The authoritative file list comes from ``visual_asset_catalog.csv``.  The
bundle also contains each visual's sidecar/manifest, its generator, the final
paired-visual pipeline, current diagnostic source JSON, all three Japanese
pages, the local Japanese font, and the audit/index files.  Raster media are
stored without recompression so rebuilding the bundle remains cheap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "docs/t4_conference_assets/visual_asset_catalog.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs/t4_conference_assets/t4_awr_visual_assets_bundle.zip"
ALREADY_COMPRESSED = {".gif", ".jpg", ".jpeg", ".png", ".webp", ".zip"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing bundle and its embedded manifest",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_name(path: Path, shared_root: Path) -> str:
    return path.resolve().relative_to(shared_root.resolve()).as_posix()


def _split_paths(value: str) -> list[str]:
    return [part for part in value.split(";") if part]


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def _collect(root: Path, catalog: Path) -> tuple[dict[Path, set[str]], list[dict[str, str]]]:
    files: dict[Path, set[str]] = {}

    def add(path: Path, role: str) -> None:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundle input ({role}): {path}")
        files.setdefault(path, set()).add(role)

    rows = list(csv.DictReader(catalog.open(newline="")))
    for row in rows:
        add(_resolve(root, row["path"]), "visual")
        for value in _split_paths(row["companion_data"]):
            add(_resolve(root, value), "scene_data_or_manifest")
        for value in _split_paths(row["generator"]):
            add(_resolve(root, value), "generator")

    fixed = {
        "docs/t4_dp_awr_material_ja.html": "report",
        "docs/t4_dp_awr_presentation_ja.html": "presentation",
        "docs/t4_dp_awr_visual_assets_ja.html": "gallery",
        "docs/t4_conference_assets/README.md": "visual_contract",
        "docs/t4_conference_assets/VISUAL_ASSET_INDEX.md": "visual_index",
        "docs/t4_conference_assets/visual_asset_catalog.csv": "catalog",
        "docs/t4_conference_assets/visual_asset_audit.json": "audit",
        "docs/t4_conference_assets/visual_asset_checksums.sha256": "checksums",
        "docs/audit_t4_visual_assets.py": "audit_generator",
        "docs/package_t4_visual_assets.py": "bundle_generator",
        "docs/assets/fonts/NotoSansJP-VF.ttf": "font",
        "docs/assets/fonts/OFL.txt": "font_license",
        "reference/external/awr-hdp-original-dp-proposal/build_full_validation_assets.py": "final_visual_generator",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/finalize_current_awr_full_validation.sh": "final_visual_orchestrator",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/dump_full_paired_trajectories.py": "final_visual_generator",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/render_gate_comparison.py": "final_visual_generator",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/awr_viz.py": "visual_geometry_library",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/diag_anchor_multimodality.py": "visual_generator_entrypoint",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/select_awr_finalists.py": "paper_aligned_checkpoint_selector",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/test_awr_viz_neighbor_focus.py": "visual_regression_test",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/test_select_awr_finalists.py": "selector_regression_test",
        "../Diffusion-Planner-t4-main/rlvr/autoresearch/tools/test_render_gate_selection.py": "visual_regression_test",
        "../Diffusion-Planner-t4-main/outputs/awr_t4_full_sequence_filtered/20260717-004516_full_sequence_filtered_hdp_plain_mse_replay_resume_single_checkpoint/effective_config.json": "formal_reward_config",
        "../Diffusion-Planner-t4-main/outputs/awr_t4_full_sequence_filtered/20260717-004516_full_sequence_filtered_hdp_plain_mse_replay_resume_single_checkpoint/paired_eval_summary.json": "training_diagnostic_source",
        "../Diffusion-Planner-t4-main/outputs/awr_t4_full_sequence_filtered/lane_change_semantic_audit_monitor2048.json": "semantic_audit_source",
        "../Diffusion-Planner-t4-main/outputs/awr_t4_full_sequence_filtered/lane_change_eval_slice_epoch008_monitor2048.json": "training_diagnostic_source",
    }
    for value, role in fixed.items():
        add(_resolve(root, value), role)
    return files, rows


def _manifest(
    files: dict[Path, set[str]],
    rows: list[dict[str, str]],
    shared_root: Path,
) -> dict[str, object]:
    entries = []
    for path in sorted(files, key=lambda item: _archive_name(item, shared_root)):
        entries.append(
            {
                "archive_path": _archive_name(path, shared_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "roles": sorted(files[path]),
            }
        )
    tiers = {row["evidence_tier"] for row in rows}
    final_visuals = sum(row["evidence_tier"] == "D_final_paired" for row in rows)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "final_full_validation" if final_visuals else "training_diagnostic",
        "claim_boundary": (
            "Includes saved-EMA paired full-validation visuals. Read each asset's evidence tier."
            if final_visuals
            else "No tier-D saved-EMA full-validation visual is present; do not claim checkpoint promotion."
        ),
        "catalog_visual_count": len(rows),
        "dynamic_gif_count": sum(row["format"] == "GIF" for row in rows),
        "evidence_tiers": sorted(tiers),
        "file_count": len(entries),
        "files": entries,
    }


def _verify(output: Path) -> dict[str, object]:
    with zipfile.ZipFile(output) as bundle:
        manifest = json.loads(bundle.read("BUNDLE_MANIFEST.json"))
        names = set(bundle.namelist())
        for item in manifest["files"]:
            name = item["archive_path"]
            if name not in names:
                raise RuntimeError(f"Bundle entry missing: {name}")
            digest = hashlib.sha256(bundle.read(name)).hexdigest()
            if digest != item["sha256"]:
                raise RuntimeError(f"Bundle hash mismatch: {name}")
        if len(manifest["files"]) != manifest["file_count"]:
            raise RuntimeError("Bundle manifest file_count mismatch")
    return manifest


def main() -> None:
    args = _args()
    root = args.root.resolve()
    catalog = args.catalog.resolve()
    output = args.output.resolve()
    if args.verify_only:
        manifest = _verify(output)
        print(json.dumps({"verified": str(output), **manifest}, ensure_ascii=False))
        return

    files, rows = _collect(root, catalog)
    shared_root = root.parent
    manifest = _manifest(files, rows, shared_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as bundle:
            bundle.writestr(
                "BUNDLE_MANIFEST.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            for path in sorted(files, key=lambda item: _archive_name(item, shared_root)):
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix.lower() in ALREADY_COMPRESSED
                    else zipfile.ZIP_DEFLATED
                )
                bundle.write(path, _archive_name(path, shared_root), compress_type=compression)
        os.replace(temporary, output)
        output.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)

    verified = _verify(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{_sha256(output)}  {output.name}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256_file": str(checksum_path),
                "status": verified["status"],
                "visuals": verified["catalog_visual_count"],
                "gifs": verified["dynamic_gif_count"],
                "files": verified["file_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
