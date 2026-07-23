#!/usr/bin/env python3
"""Audit every local raster image used by the T4 AWR report, deck, and gallery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageChops


VISUAL_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


DEFAULT_PAGES = (
    "docs/t4_dp_awr_material_ja.html",
    "docs/t4_dp_awr_presentation_ja.html",
    "docs/t4_dp_awr_visual_assets_ja.html",
)


class _ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[dict[str, str]] = []
        self._section_stack: list[str] = []
        self._section_context = "page"
        self._slide_number = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "section":
            self._section_stack.append(self._section_context)
            classes = set(str(values.get("class") or "").split())
            if "slide" in classes:
                self._slide_number += 1
                title = str(values.get("data-title") or "untitled")
                self._section_context = f"slide-{self._slide_number:02d}:{title}"
            else:
                self._section_context = f"section:{values.get('id') or 'unnamed'}"
        if tag == "img" and values.get("src"):
            self.references.append(
                {
                    "source": str(values["src"]),
                    "alt": values.get("alt") or "",
                    "reference_role": "displayed",
                    "context": self._section_context,
                }
            )
        if tag == "a" and values.get("href"):
            source = str(values["href"])
            if Path(urlparse(source).path).suffix.lower() in VISUAL_SUFFIXES:
                self.references.append(
                    {
                        "source": source,
                        "alt": "",
                        "reference_role": "linked",
                        "context": self._section_context,
                    }
                )

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self._section_stack:
            self._section_context = self._section_stack.pop()


def _evidence_contract(path: Path) -> dict[str, str]:
    """Attach the claim boundary readers should use for each rendered asset."""
    value = path.as_posix()
    if "/final_validation/" in value:
        return {
            "evidence_tier": "D_final_paired",
            "claim_boundary": "saved-EMA source/AWR paired deployment-policy evidence",
        }
    if "/formal_replay_safety/" in value or "/full_replay_signal/" in value:
        return {
            "evidence_tier": "A_formal_replay",
            "claim_boundary": "exact frozen replay targets, rewards, and weights; not checkpoint improvement",
        }
    if "/awr_colleague_" in value or "/awr_multimodality_compare/" in value:
        return {
            "evidence_tier": "B_source_policy_coverage",
            "claim_boundary": "source-policy candidate coverage; not post-training evidence",
        }
    if "/lane_mask_audit/" in value:
        return {
            "evidence_tier": "C_metric_audit",
            "claim_boundary": "real-validation metric/mask diagnostic; independent label is not ground truth",
        }
    if "/training_monitor/" in value:
        return {
            "evidence_tier": "C_training_monitor",
            "claim_boundary": "2,048-scene current-policy K=10/5-step learning diagnostic; not saved-EMA promotion evidence",
        }
    if "/t4_rl_assets/t4_scene_" in value:
        return {
            "evidence_tier": "C_counterfactual_probe",
            "claim_boundary": "real-scene geometry with designed trajectory probes; not sampled policy output",
        }
    if "/t4_rl_assets/" in value:
        return {
            "evidence_tier": "context_or_historical",
            "claim_boundary": "background or historical diagnostic; not the formal Original-DP AWR result",
        }
    if "/reference/external/" in value:
        return {
            "evidence_tier": "paper_reference",
            "claim_boundary": "figure from the cited paper/reference implementation",
        }
    return {
        "evidence_tier": "uncategorized",
        "claim_boundary": "must be reviewed before use",
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pages", nargs="*", default=list(DEFAULT_PAGES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "t4_conference_assets" / "visual_asset_audit.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_existing(root: Path, candidates: list[Path]) -> list[str]:
    """Return stable root-relative paths for existing handoff companions."""

    result: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            value = os.path.relpath(candidate, root)
            if value not in result:
                result.append(value)
    return result


def _companion_data(root: Path, path: Path) -> list[str]:
    """Find the scene sidecar and directory manifest that explain a visual."""

    candidates = [path.with_suffix(".json"), path.parent / "manifest.json"]
    if path.stem.endswith("_multimodality"):
        candidates.append(path.with_name(path.stem.removesuffix("_multimodality") + ".json"))
    if "/t4_rl_assets/" in path.as_posix():
        candidates.append(path.parent / "manifest.json")
    return _relative_existing(root, candidates)


def _generator_paths(root: Path, path: Path) -> list[str]:
    """Map each evidence family to the code that produces or imports it."""

    value = path.as_posix()
    t4_root = root.parent / "Diffusion-Planner-t4-main"
    candidates: list[Path]
    if "/final_validation/" in value:
        candidates = [
            t4_root / "rlvr/autoresearch/finalize_current_awr_full_validation.sh",
            root
            / "reference/external/awr-hdp-original-dp-proposal/build_full_validation_assets.py",
            t4_root / "rlvr/autoresearch/tools/dump_full_paired_trajectories.py",
            t4_root / "rlvr/autoresearch/tools/render_gate_comparison.py",
            t4_root / "rlvr/autoresearch/tools/grpo_viz.py",
        ]
    elif "/formal_replay_safety/" in value or "/awr_colleague_" in value or "/awr_multimodality_compare/" in value:
        candidates = [t4_root / "rlvr/autoresearch/tools/grpo_viz.py"]
    elif "/full_replay_signal/" in value:
        candidates = [t4_root / "rlvr/autoresearch/tools/analyze_full_replay_signal.py"]
    elif "/lane_mask_audit/" in value:
        candidates = [t4_root / "rlvr/autoresearch/tools/render_lane_mask_audit.py"]
    elif "/training_monitor/" in value:
        candidates = [root / "docs/generate_t4_awr_final_results.py"]
    elif "/t4_rl_assets/t4_scene_" in value:
        candidates = [root / "docs/generate_t4_scene_reward_visualizations.py"]
    elif "/t4_rl_assets/" in value:
        candidates = [root / "docs/generate_t4_rl_visualizations.py"]
    else:
        candidates = []
    return _relative_existing(root, candidates)


def _write_handoff_catalogs(root: Path, output: Path, payload: dict[str, object]) -> None:
    """Write human- and machine-readable inventories for every used visual."""

    output_dir = output.parent
    csv_path = output_dir / "visual_asset_catalog.csv"
    checksum_path = output_dir / "visual_asset_checksums.sha256"
    markdown_path = output_dir / "VISUAL_ASSET_INDEX.md"
    rows: list[dict[str, str]] = []
    assets = list(payload["assets"])
    for index, asset in enumerate(assets, start=1):
        path = root / str(asset["path"])
        references = list(asset["referenced_by"])
        displayed_in = sorted(
            {str(item["page"]) for item in references if item["reference_role"] == "displayed"}
        )
        linked_in = sorted(
            {str(item["page"]) for item in references if item["reference_role"] == "linked"}
        )
        rows.append(
            {
                "asset_id": f"V{index:03d}",
                "path": str(asset["path"]),
                "evidence_tier": str(asset["evidence_tier"]),
                "format": str(asset["format"]),
                "width_px": str(asset["width_px"]),
                "height_px": str(asset["height_px"]),
                "frame_count": str(asset["frame_count"]),
                "displayed_in": ";".join(displayed_in),
                "linked_in": ";".join(linked_in),
                "usage_contexts": ";".join(
                    sorted(
                        {
                            f'{item["page"]}#{item.get("context", "page")}'
                            for item in references
                        }
                    )
                ),
                "companion_data": ";".join(_companion_data(root, path)),
                "generator": ";".join(_generator_paths(root, path)),
                "claim_boundary": str(asset["claim_boundary"]),
                "sha256": str(asset["sha256"]),
            }
        )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checksum_path.write_text(
        "".join(f'{row["sha256"]}  {row["path"]}\n' for row in rows)
    )

    tier_order = (
        "A_formal_replay",
        "B_source_policy_coverage",
        "C_metric_audit",
        "C_training_monitor",
        "C_counterfactual_probe",
        "D_final_paired",
        "context_or_historical",
        "paper_reference",
        "uncategorized",
    )
    claim_boundary_ja = {
        "A_formal_replay": "frozen replayのtarget / reward / weightは正式値。checkpoint改善の証明ではない",
        "B_source_policy_coverage": "source policyの候補coverage。post-training改善の証明ではない",
        "C_metric_audit": "実valid scene上のmetric / mask診断。独立labelは完全なGTではない",
        "C_training_monitor": "2,048 scene・current policy K=10/5-step診断。saved EMAの採用証拠ではない",
        "C_counterfactual_probe": "実scene geometry上の設計軌跡probe。policy sampleではない",
        "D_final_paired": "saved EMAとsourceの同一scene・deployment設定によるfull paired証拠",
        "context_or_historical": "背景・過去診断。今回の正式Original-DP AWR結果ではない",
        "paper_reference": "引用論文・reference implementationの原図",
        "uncategorized": "使用前にevidence boundaryの再確認が必要",
    }
    lines = [
        "# 会議資料・全可視化素材の引継ぎ台帳",
        "",
        "日本語詳細資料、16:9 presentation、証拠galleryが表示または原図として直接linkする全visualの台帳です。`docs/audit_t4_visual_assets.py` から自動生成するため、手編集しません。",
        "",
        f"- 全visual: {payload['unique_asset_count']}（ページ内表示 {payload['displayed_asset_count']}、原図linkのみ {payload['linked_only_asset_count']}）",
        f"- 動的GIF: {payload['gif_asset_count']}",
        f"- missing / bad GIF / low-resolution: {payload['missing_asset_count']} / {payload['bad_gif_count']} / {payload['low_resolution_count']}",
        "- 機械可読台帳: [`visual_asset_catalog.csv`](visual_asset_catalog.csv)",
        "- 全frame/hash監査: [`visual_asset_audit.json`](visual_asset_audit.json)",
        "- コピー検証用hash: [`visual_asset_checksums.sha256`](visual_asset_checksums.sha256)",
        "- 一括交付: [`t4_awr_visual_assets_bundle.zip`](t4_awr_visual_assets_bundle.zip) / [`bundle SHA-256`](t4_awr_visual_assets_bundle.zip.sha256)",
        "",
        "各行から元visual、対応するscene JSON/manifest、生成コードへ移動できます。Evidence tier Dは46,068件full validation後のsource対saved-EMA成対結果にだけ使用します。",
        "",
    ]
    for tier in tier_order:
        tier_rows = [row for row in rows if row["evidence_tier"] == tier]
        if not tier_rows:
            continue
        lines.extend(
            [
                f"## {tier}",
                "",
                "| ID | Visual | 解像度 / frames | 使用箇所 | 対応data | 生成コード | 主張境界 |",
                "|---|---|---:|---|---|---|---|",
            ]
        )
        for row in tier_rows:
            visual = f'[{Path(row["path"]).name}](../../{row["path"]})'
            data_links = "<br>".join(
                f'[{Path(item).name}](../../{item})'
                for item in row["companion_data"].split(";")
                if item
            ) or "—"
            generator_links = "<br>".join(
                f'[{Path(item).name}](../../{item})'
                for item in row["generator"].split(";")
                if item
            ) or "paper / source原図"
            shape = f'{row["width_px"]}×{row["height_px"]} / {row["frame_count"]}'
            usage = "<br>".join(
                item.replace("docs/t4_dp_awr_presentation_ja.html#", "deck ")
                .replace("docs/t4_dp_awr_material_ja.html#", "report ")
                .replace("docs/t4_dp_awr_visual_assets_ja.html#", "gallery ")
                for item in row["usage_contexts"].split(";")
                if item
            )
            lines.append(
                f'| {row["asset_id"]} | {visual} | {shape} | {usage} | {data_links} | {generator_links} | {claim_boundary_ja.get(tier, row["claim_boundary"])} |'
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines))


def _inspect_image(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".svg":
        root = ET.parse(path).getroot()
        view_box = root.attrib.get("viewBox", "").replace(",", " ").split()

        def _number(value: str | None) -> float:
            if not value:
                return 0.0
            match = re.match(r"[-+0-9.eE]+", value)
            return float(match.group(0)) if match else 0.0

        width = _number(root.attrib.get("width"))
        height = _number(root.attrib.get("height"))
        if (not width or not height) and len(view_box) == 4:
            width, height = float(view_box[2]), float(view_box[3])
        return {
            "format": "SVG",
            "width_px": width,
            "height_px": height,
            "frame_count": 1,
        }
    with Image.open(path) as image:
        result: dict[str, object] = {
            "format": image.format,
            "width_px": image.width,
            "height_px": image.height,
            "frame_count": int(getattr(image, "n_frames", 1)),
        }
        if result["frame_count"] > 1:
            frame_hashes: list[str] = []
            content_roi_hashes: list[str] = []
            content_roi_changed_fractions: list[float] = []
            durations_ms: list[int] = []
            previous_roi = None
            for index in range(int(result["frame_count"])):
                image.seek(index)
                frame = image.convert("RGB")
                frame_hashes.append(hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest())
                # Exclude the top 15% where a changing timestamp/title could
                # make an otherwise static plot look animated.  The remaining
                # ROI contains map, trajectories, actors, OBBs, and event zoom.
                roi = frame.crop((0, int(frame.height * 0.15), frame.width, frame.height))
                content_roi_hashes.append(
                    hashlib.blake2b(roi.tobytes(), digest_size=16).hexdigest()
                )
                if previous_roi is not None:
                    histogram = ImageChops.difference(previous_roi, roi).convert("L").histogram()
                    changed_pixels = sum(histogram[1:])
                    content_roi_changed_fractions.append(
                        changed_pixels / max(1, roi.width * roi.height)
                    )
                previous_roi = roi
                durations_ms.append(int(image.info.get("duration", 0)))
            result.update(
                {
                    "unique_frame_count": len(set(frame_hashes)),
                    "adjacent_changed_count": sum(
                        before != after for before, after in zip(frame_hashes, frame_hashes[1:])
                    ),
                    "content_roi": "full width, lower 85%; excludes changing title/timestamp band",
                    "content_roi_unique_frame_count": len(set(content_roi_hashes)),
                    "content_roi_adjacent_changed_count": sum(
                        before != after
                        for before, after in zip(content_roi_hashes, content_roi_hashes[1:])
                    ),
                    "content_roi_changed_pixel_fraction_min": min(
                        content_roi_changed_fractions
                    ),
                    "content_roi_changed_pixel_fraction_mean": sum(
                        content_roi_changed_fractions
                    )
                    / len(content_roi_changed_fractions),
                    "duration_ms_min": min(durations_ms),
                    "duration_ms_max": max(durations_ms),
                }
            )
        return result


def main() -> None:
    args = _args()
    root = args.root.resolve()
    referenced_by: dict[Path, list[dict[str, str]]] = defaultdict(list)
    missing: list[dict[str, str]] = []

    for page_value in args.pages:
        page = (root / page_value).resolve()
        parser = _ImageParser()
        parser.feed(page.read_text())
        for image in parser.references:
            source = image["source"]
            parsed = urlparse(source)
            if parsed.scheme or source.startswith("data:"):
                continue
            path = (page.parent / unquote(parsed.path)).resolve()
            if not path.exists():
                missing.append({"page": str(page.relative_to(root)), "source": source})
                continue
            referenced_by[path].append(
                {
                    "page": str(page.relative_to(root)),
                    "source": source,
                    "alt": image["alt"],
                    "reference_role": image["reference_role"],
                    "context": image.get("context", "page"),
                }
            )

    assets: list[dict[str, object]] = []
    for path in sorted(referenced_by, key=lambda item: str(item)):
        item: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "referenced_by": referenced_by[path],
        }
        item.update(_evidence_contract(path))
        item.update(_inspect_image(path))
        item["minimum_resolution_pass"] = bool(
            item["format"] == "SVG"
            or (int(item["width_px"]) >= 700 and int(item["height_px"]) >= 600)
        )
        assets.append(item)

    gif_assets = [asset for asset in assets if int(asset["frame_count"]) > 1]
    bad_gifs = [
        str(asset["path"])
        for asset in gif_assets
        if int(asset.get("unique_frame_count", 0)) != int(asset["frame_count"])
        or int(asset.get("adjacent_changed_count", -1)) != int(asset["frame_count"]) - 1
        or int(asset.get("content_roi_unique_frame_count", 0)) != int(asset["frame_count"])
        or int(asset.get("content_roi_adjacent_changed_count", -1))
        != int(asset["frame_count"]) - 1
    ]
    low_resolution = [
        str(asset["path"]) for asset in assets if not bool(asset["minimum_resolution_pass"])
    ]
    empty_alt_references = [
        {"path": str(asset["path"]), "page": ref["page"], "source": ref["source"]}
        for asset in assets
        for ref in asset["referenced_by"]
        if ref.get("reference_role") == "displayed" and not str(ref.get("alt", "")).strip()
    ]
    hash_paths: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        hash_paths[str(asset["sha256"])].append(str(asset["path"]))
    duplicate_hash_groups = [paths for paths in hash_paths.values() if len(paths) > 1]
    payload = {
        "contract": {
            "scope": "all local displayed and linked visual assets used by the Japanese report, deck, and evidence gallery",
            "gif_pass": "every decoded frame and lower-85% content ROI is unique; every adjacent content ROI changes",
            "resolution_pass": "at least 700 x 600 source pixels",
            "evidence_contract": "every raster is assigned a claim boundary; only tier D may prove checkpoint improvement",
        },
        "pages": list(args.pages),
        "unique_asset_count": len(assets),
        "displayed_asset_count": sum(
            any(ref["reference_role"] == "displayed" for ref in asset["referenced_by"])
            for asset in assets
        ),
        "linked_only_asset_count": sum(
            all(ref["reference_role"] == "linked" for ref in asset["referenced_by"])
            for asset in assets
        ),
        "vector_asset_count": sum(asset["format"] == "SVG" for asset in assets),
        "raster_asset_count": sum(asset["format"] != "SVG" for asset in assets),
        "gif_asset_count": len(gif_assets),
        "missing_asset_count": len(missing),
        "bad_gif_count": len(bad_gifs),
        "low_resolution_count": len(low_resolution),
        "empty_alt_reference_count": len(empty_alt_references),
        "duplicate_hash_group_count": len(duplicate_hash_groups),
        "missing": missing,
        "bad_gifs": bad_gifs,
        "low_resolution": low_resolution,
        "empty_alt_references": empty_alt_references,
        "duplicate_hash_groups": duplicate_hash_groups,
        "assets": assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_handoff_catalogs(root, args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "assets": len(assets),
                "displayed": payload["displayed_asset_count"],
                "linked_only": payload["linked_only_asset_count"],
                "vectors": payload["vector_asset_count"],
                "gifs": len(gif_assets),
                "missing": len(missing),
                "bad_gifs": len(bad_gifs),
                "low_resolution": len(low_resolution),
                "empty_alt_references": len(empty_alt_references),
                "duplicate_hash_groups": len(duplicate_hash_groups),
            }
        )
    )
    if missing or bad_gifs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
