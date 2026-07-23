#!/usr/bin/env python3
"""Targeted visual/content repair for the manually edited T4 AWR draft.

This is intentionally *not* a full-page publisher.  It requires the reviewed
remote body hash, replaces only named image+caption blocks and exact stale
sentences, uploads only the referenced assets, and checks the body hash again
immediately before the update.  Manual Confluence edits outside those anchors
are preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from publish_t4_awr_confluence import Confluence, attachment_name


ROOT = Path(__file__).resolve().parents[1]


def _macro(path: Path, alt: str) -> str:
    name = attachment_name(path)
    return (
        f'<ac:image ac:align="center" ac:layout="center" ac:width="1200" '
        f'ac:alt="{html.escape(alt, quote=True)}">'
        f'<ri:attachment ri:filename="{html.escape(name, quote=True)}" />'
        "</ac:image>"
    )


def _caption(value: str) -> str:
    return f"<p><em>{value}</em></p>"


def _replace_figure(storage: str, old_name: str, replacement: str) -> str:
    pattern = re.compile(
        r'<ac:image\b[^>]*>\s*<ri:attachment\b[^>]*ri:filename="'
        + re.escape(old_name)
        + r'"[^>]*/>\s*</ac:image>\s*<p\b[^>]*>\s*<em>.*?</em>\s*</p>',
        re.DOTALL,
    )
    updated, count = pattern.subn(replacement, storage)
    if count != 1:
        raise RuntimeError(f"expected exactly one figure block for {old_name}, got {count}")
    return updated


def _replace_caption(storage: str, filename: str, caption: str) -> str:
    pattern = re.compile(
        r'(?P<image><ac:image\b[^>]*>\s*<ri:attachment\b[^>]*ri:filename="'
        + re.escape(filename)
        + r'"[^>]*/>\s*</ac:image>)\s*<p\b[^>]*>\s*<em>.*?</em>\s*</p>',
        re.DOTALL,
    )
    updated, count = pattern.subn(lambda match: match.group("image") + _caption(caption), storage)
    if count != 1:
        raise RuntimeError(f"expected exactly one caption for {filename}, got {count}")
    return updated


def _replace_exact(storage: str, old: str, new: str, label: str) -> str:
    count = storage.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, got {count}")
    return storage.replace(old, new, 1)


def _repair(storage: str, paths: dict[str, Path]) -> str:
    stopped_static = _macro(paths["stopped_static"], "formal stopped-leader reward explainer with three counterfactual actions and OBBs")
    stopped_gif = _macro(paths["stopped_gif"], "dynamic formal stopped-leader reward unit test with ego and leader OBBs")
    stopped_block = (
        stopped_static
        + _caption(
            "<strong>停止先行車を正式rewardで読み解く：</strong>46,068-scene validationから、同方向・同一corridor・4秒で0.18 mしか動かない実先行車sceneを再抽出。初期間隔は5.0 m。"
            "Safe holdはR=1.000、safe crawlはR=0.994。Progress pushは1.6 sでOBB overlapし、Collision gate=0のためR=0。"
            "logged expertが5 m未満しか動かないsceneではEPは中立値1となるため、この図が示すのは「常に前進」ではなく「可能なら安全に前進」。"
            "実map/actorを使ったcounterfactual evaluator単体試験であり、model outputではない。"
        )
        + stopped_gif
        + _caption(
            "<strong>動的OBB確認：</strong>左から安全停止・安全な微速前進・強行前進。同じrecorded leader future（raw[t+1]→aligned[t]）に対し、"
            "Ego boxはscene固有wheelbaseとrear-axle conventionで描画する。"
        )
    )
    storage = _replace_figure(
        storage,
        "t4-awr-docs-t4_rl_assets-t4_scene_stopped_leader.gif",
        stopped_block,
    )

    paired_static = _macro(paths["paired_static"], "full-validation actual original-DP versus AWR road-border recovery")
    paired_gif = _macro(paths["paired_gif"], "dynamic full-validation actual original-DP versus AWR road-border recovery")
    paired_block = (
        paired_static
        + _caption(
            "<strong>実checkpoint出力のroad-border recovery：</strong>これはprobeではなく、full validation上のzero-noise K=1 deployment trajectory。"
            "Source Original DPは0.1 sで車体OBBがborder marginを破りR=0、同一sceneのAWR checkpointは通過してR=0.999。"
            "RB clearanceは0.19→0.25 m。一方ADE/FDEは各+0.08 mであり、targeted safety gainとimitation trade-offを同時に公開する。"
        )
        + paired_gif
        + _caption(
            "<strong>動的比較：</strong>左=Source Original DP、中央=AWR checkpoint、右=同一時刻のOBB event camera。"
            "両者は同一scene、同一scale、K=1、zero noise、10 solver steps、+1-aligned neighbor future。"
        )
    )
    storage = _replace_figure(
        storage,
        "t4-awr-docs-t4_rl_assets-t4_scene_tight_right_turn.gif",
        paired_block,
    )

    storage = _replace_caption(
        storage,
        "t4-awr-docs-t4_conference_assets-formal_replay_safety-scene_002.gif",
        "<strong>Collision / TTC（formal replay evidence）：</strong>epoch-1 replayに実際に保存された同一sceneのK=10を時系列表示。"
        "各色は一本のcomplete future、矩形は時刻ごとのEgo/neighbor OBB、右下はevent camera。"
        "この図で見るのは「Kのうち安全候補を高く、3.9 sで衝突する候補を低く順位付けできるか」であり、checkpoint改善のbefore/afterではない。",
    )
    storage = _replace_caption(
        storage,
        "t4-awr-docs-t4_conference_assets-formal_replay_safety-scene_000.gif",
        "<strong>Road border（formal replay evidence）：</strong>epoch-1 replayの同一K=10中、8候補は車体OBBがmargin内、2候補はcrossing。"
        "中心線だけでなく車体四隅で判定するため、event cameraでborderとbboxの接触を読む。checkpoint改善のbefore/afterではなく、training candidateのranking evidence。",
    )
    storage = _replace_caption(
        storage,
        "t4-awr-docs-t4_conference_assets-lane_mask_audit-lane_mask_audit.gif",
        "<strong>Lane maskの四象限監査：</strong>A/Dが期待動作、B=本来maskすべきlane changeの見逃し、C=lane changeでないsceneの過剰免除。"
        "独立labelは完全なGTではないため診断扱い。今回のformal runではlane hard gateはOFFで、maskが変えるのはquality内のLane weight 2/14だけ；通常lane changeだけでtotalが0にはならない。",
    )
    storage = _replace_caption(
        storage,
        "t4-awr-docs-t4_conference_assets-awr_multimodality_compare-scene_001_multimodality.gif",
        "<strong>多様性の動的診断（formal training resultではない）：</strong>同一seed・同一実sceneでraw K=16とHDP-style augmentation K=16を比較。"
        "endpoint pairwise meanは1.51→1.83 m、横方向stdは0.05→0.43 mへ増えるが、2 m thresholdでは両方1 mode。augmentationは幾何spreadを増やすが、別maneuverを保証しない。",
    )
    storage = _replace_caption(
        storage,
        "t4-awr-docs-t4_conference_assets-awr_multimodality_compare-scene_003_multimodality.png",
        "<strong>急右折のmode監査：</strong>raw/augmentationとも16本は同じ右折maneuverに留まり、2 m endpoint componentは1 mode。"
        "augmentationでpairwise path distanceは0.35→0.88 mへ増えるが、これは同一maneuver内のvariationであり、stop/goや左右分岐のsemantic multimodalityではない。"
        "図中のring/componentは幾何clusterで、行動ラベルではない。",
    )

    full_chart = _macro(paths["full_chart"], "full 46068 scene deterministic paired validation summary")
    storage = _replace_figure(
        storage,
        "t4-awr-docs-t4_conference_assets-training_monitor-checkpoint_paired_sweep.png",
        full_chart
        + _caption(
            "<strong>最終full validation（46,068 scenes）：</strong>単一EMA epoch-7 checkpointをSource Original DPと同一scene pathでK=1 / zero-noise / 10-step比較。"
            "Mean rewardは0.9303→0.9411（+0.0108、+1.16%）、road-borderは940→266 scenes（−71.7%）。"
            "同時にADE +0.112 m、FDE +0.297 m、collision 65→72、kinematic 55→68も隠さず表示する。すなわち平均formal rewardの改善はfull dataで証明されたが、全submetricの一様改善ではない。"
        ),
    )
    transition_chart = _macro(paths["transition_chart"], "full validation recovered and newly introduced hard event transitions")
    storage = _replace_figure(
        storage,
        "t4-awr-docs-t4_conference_assets-training_monitor-hard_event_transition_audit.png",
        transition_chart
        + _caption(
            "<strong>net値の内訳：</strong>collisionは20回収 / 27新規（net +7）、road-borderは735回収 / 61新規（net −674）、kinematicは21回収 / 34新規（net +13）。"
            "AWR failures = Source failures − recovered + newly introduced を各panelで確認できる。平均値だけでregressionを隠さない。"
        ),
    )

    storage = _replace_exact(
        storage,
        '<p local-id="c1203515f709">下図は固定2,048 validation scenesを、各epochの<strong>current policy・K=10・5 solver steps</strong>で成対比較した中間monitor。epoch 2&ndash;10はすべてsource比で平均scene scoreの95% CI下限が0より大きい。</p>',
        '<p local-id="c1203515f709">学習中は固定2,048-scene monitorでepochを追跡し、最終的に<strong>epoch-7の単一EMA checkpoint</strong>を固定して、is_skipped filtering後の全46,068 validation scenesを成対評価した。下図は中間monitorではなく、deployment設定（K=1 / zero noise / 10 solver steps）の最終結果。</p>',
        "stale experiment introduction",
    )
    storage = _replace_exact(
        storage,
        "epoch 2&ndash;10をK=1 / zero-noise / 10-stepで比較予定。",
        "epoch 2&ndash;10をscreenし、epoch 7の単一EMA checkpointをfull validationへ固定。",
        "saved EMA pending text",
    )
    storage = _replace_exact(
        storage,
        "<strong>最終結論。現在実行中。</strong>",
        "<strong>完了。</strong>46,068 scenesでmean reward +1.16%、road-border −71.7%。ADE/FDE、collision、kinematicの回帰も上図の通り併記。",
        "full validation pending text",
    )

    # Explain the denominator directly inside the table: it is a normalized
    # quality weight, not a pass count or number of scenes.
    storage, count_5 = re.subn(
        r'(<p\b[^>]*>)5/14(</p>)',
        r'\1weight 5/14（quality内35.7%；hard gateではない）\2',
        storage,
    )
    storage, count_2 = re.subn(
        r'(<p\b[^>]*>)2/14(</p>)',
        r'\1weight 2/14（quality内14.3%；hard gateではない）\2',
        storage,
    )
    if count_5 != 2 or count_2 != 2:
        raise RuntimeError(f"unexpected reward-weight cell counts: 5/14={count_5}, 2/14={count_2}")

    wrapped = (
        '<root xmlns:ac="http://atlassian.com/content" '
        'xmlns:ri="http://atlassian.com/resource/identifier">'
        + storage
        + "</root>"
    )
    # Existing Confluence storage may legitimately retain HTML named entities
    # (for example ``&ndash;``).  They are accepted by the editor but are not
    # predefined XML entities.  Escape only those entities in the validation
    # copy; keep the remote storage byte-for-byte unchanged outside our anchors.
    validation_copy = re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)",
        "&amp;",
        wrapped,
    )
    ET.fromstring(validation_copy)
    return storage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-id", default="5392827425")
    parser.add_argument("--expected-remote-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = {
        "stopped_static": ROOT / "docs/t4_conference_assets/reward_explainers/stopped_leader_formal_reward_explainer.png",
        "stopped_gif": ROOT / "docs/t4_conference_assets/reward_explainers/stopped_leader_formal_reward_explainer.gif",
        "paired_static": ROOT / "docs/t4_conference_assets/final_validation/scenes/scene_002_paired.png",
        "paired_gif": ROOT / "docs/t4_conference_assets/final_validation/scenes/scene_002_paired.gif",
        "lane_gif": ROOT / "docs/t4_conference_assets/lane_mask_audit/lane_mask_audit.gif",
        "full_chart": ROOT / "docs/t4_conference_assets/final_validation/charts/full_validation_paired.png",
        "transition_chart": ROOT / "docs/t4_conference_assets/final_validation/charts/full_validation_hard_event_transitions.png",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing assets: {missing}")

    confluence = Confluence()
    page = confluence.get_page(args.page_id)
    storage = (((page.get("body") or {}).get("storage") or {}).get("value") or "")
    actual_hash = hashlib.sha256(storage.encode()).hexdigest()
    expected_hash = args.expected_remote_sha256.strip().lower()
    if page.get("status") != "draft":
        raise SystemExit(f"refusing non-draft page: {page.get('status')}")
    if actual_hash != expected_hash:
        raise SystemExit(f"remote body changed; refusing targeted update: actual_sha256={actual_hash}")

    repaired = _repair(storage, paths)
    print(
        {
            "old_sha256": actual_hash,
            "new_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
            "old_chars": len(storage),
            "new_chars": len(repaired),
            "changed_chars": sum(a != b for a, b in zip(storage, repaired)) + abs(len(storage) - len(repaired)),
            "assets": {key: attachment_name(path) for key, path in paths.items()},
        }
    )
    if args.dry_run:
        return 0

    existing = confluence.list_attachments(args.page_id)
    for key, path in paths.items():
        name = attachment_name(path)
        confluence.upload_attachment(args.page_id, path, name, existing.get(name))
        print(f"uploaded {key}: {name}")

    # A human may edit while GIFs upload.  Check the live body one more time.
    latest = confluence.get_page(args.page_id)
    latest_storage = (((latest.get("body") or {}).get("storage") or {}).get("value") or "")
    latest_hash = hashlib.sha256(latest_storage.encode()).hexdigest()
    if latest_hash != expected_hash:
        raise SystemExit(
            "remote body changed during attachment upload; attachments are safe but body update was refused: "
            f"actual_sha256={latest_hash}"
        )
    updated = confluence.update_draft(latest, repaired)
    verify = confluence.get_page(args.page_id)
    verify_storage = (((verify.get("body") or {}).get("storage") or {}).get("value") or "")
    for old in (
        "t4-awr-docs-t4_rl_assets-t4_scene_stopped_leader.gif",
        "t4-awr-docs-t4_rl_assets-t4_scene_tight_right_turn.gif",
        "t4-awr-docs-t4_conference_assets-training_monitor-checkpoint_paired_sweep.png",
        "t4-awr-docs-t4_conference_assets-training_monitor-hard_event_transition_audit.png",
    ):
        if old in verify_storage:
            raise RuntimeError(f"stale attachment remains in remote body: {old}")
    for key in ("stopped_static", "stopped_gif", "paired_static", "paired_gif", "full_chart", "transition_chart"):
        name = attachment_name(paths[key])
        if verify_storage.count(name) != 1:
            raise RuntimeError(f"remote body does not contain exactly one {key}: {name}")
    if "46,068 scenesでmean reward +1.16%" not in verify_storage:
        raise RuntimeError("full-validation conclusion was not persisted")
    print(
        {
            "updated": True,
            "id": updated.get("id"),
            "status": updated.get("status"),
            "verified_sha256": hashlib.sha256(verify_storage.encode()).hexdigest(),
            "manual_content_preserved_outside_targeted_anchors": True,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
