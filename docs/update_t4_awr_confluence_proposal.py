#!/usr/bin/env python3
"""Replace only the lane-change explainer and final-proposal section of the T4 draft."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import xml.etree.ElementTree as ET

from publish_t4_awr_confluence import Confluence


LANE_START = '<h3 local-id="ae23cbad1890">正常なlane changeはどう扱うか</h3>'
SECTION_6 = '<h2 local-id="4b3e7a4dd0e1">6. AWRが学べる条件：候補の多様性</h2>'
SECTION_8 = '<h2 local-id="7d3c8e95d1b0">8. 最終提案</h2>'
SECTION_9 = '<h2 local-id="a249dd664e67">9. 参考文献</h2>'


LANE_REPLACEMENT = (
    '<h3 local-id="ae23cbad1890">正常なlane changeの扱い</h3>'
    '<p local-id="c6141ed54fbb"><strong>Lane Keepingはhard gateではない。</strong>'
    'HDPと同様、通常はcenterlineへの近さをquality weightとして使い、logged expertがoff-laneまたはlane-changeを示すsceneでは'
    'Lane項をneutralにして意図的な横移動を罰しない。今回のformal runも<code>lane_gate_enabled=false</code>であり、'
    'hard gateはCollision / Road-border / Kinematic、Laneはquality内の<code>2/14</code>である。</p>'
)


PROPOSAL = r'''<h2 local-id="7d3c8e95d1b0">8. 最終提案：T4 AWR Post-Training Framework</h2><ac:structured-macro ac:name="info" ac:schema-version="1"><ac:rich-text-body><p><strong>提案を一文で：</strong>Original DPを初期policyとし、全量clean data上で<strong>HDP-style group rollout → reward ranking → AWR replay → EMA policy refresh</strong>を反復する。通常sceneを保持しながら「現在のtrajectoryより良い候補が存在するscene」を重点学習し、K候補に有効解がないsceneだけR2LPL repairで補完する。最終成果物は既存Original DPをそのまま置き換えられる<strong>単一checkpoint</strong>とする。</p></ac:rich-text-body></ac:structured-macro><p><strong>この節は昨晩の実験継続計画ではなく、今後の標準post-training方式の提案である。</strong>第7節の10-epoch結果はcore loopが全量dataで動作し得ることを示す初期証拠であり、以下のframeworkそのものとは区別する。</p><h3>8.1 目標と設計原則</h3><table data-layout="wide"><tbody><tr><th><p>項目</p></th><th><p>提案</p></th><th><p>理由</p></th></tr><tr><td><p>最終成果</p></td><td><p>Original DP互換の単一EMA checkpoint</p></td><td><p>ensemble、reranker、別SFT modelを車載時に追加しない</p></td></tr><tr><td><p>最適化対象</p></td><td><p>全量clean distribution上のdeployment reward</p></td><td><p>ADE/FDE最小化の延長ではなく、運転結果を直接post-trainする</p></td></tr><tr><td><p>RL backbone</p></td><td><p>HDPのKL-regularized AWRとpolicy iteration</p></td><td><p>rewardを微分せず、既存diffusion lossを高reward trajectoryで再重み付けできる</p></td></tr><tr><td><p>変更範囲</p></td><td><p>Original DPのarchitectureと推論interfaceは維持</p></td><td><p>HDP固有modelへの移植ではなく、Original DPにAWRを適用する</p></td></tr></tbody></table><h3>8.2 Core loop：全量dataで反復するpolicy improvement</h3><table data-layout="wide"><tbody><tr><th><p>Step</p></th><th><p>実行内容</p></th><th><p>学習上の意味</p></th></tr><tr><td><p>1 · Clean corpus</p></td><td><p><code>is_skipped=false</code>でfull-sequence dataを構成し、既知の旧dataではneighbor futureを<code>+1 frame</code> alignmentして採点する</p></td><td><p>汚れたlabelと時刻ずれをrewardへ混入させない</p></td></tr><tr><td><p>2 · Group rollout</p></td><td><p>最新EMA policyから各sceneにつきK=10 trajectoryを一括sampleする。early cycleではHDPの局所trajectory augmentationを候補探索に加える</p></td><td><p>同じscene内に比較可能な複数actionを作る</p></td></tr><tr><td><p>3 · Fixed reward</p></td><td><p>Collision / Road-border / Kinematicをhard gate、TTC / Progress / Lane / Comfortをqualityとして同一versionのrewardで採点する。Laneはexpert lane-change時にneutral化する</p></td><td><p>安全制約と運転品質を混同せず、途中で指標を足し引きしない</p></td></tr><tr><td><p>4 · AWR target</p></td><td><p>scene内で<code>Aᵢ=(Rᵢ−mean(R))/(std(R)+ε)</code>、<code>wᵢ=exp(βAᵢ)</code>を計算し、<code>L=mean(wᵢ·Ldiffusion(τᵢ))</code>で更新する</p></td><td><p>高reward候補を強く、低reward候補を弱く模倣する。reward自体のgradientは不要</p></td></tr><tr><td><p>5 · Replay</p></td><td><p>trajectory / reward / weight / scene contextをrollout時にfreezeし、次のrefreshまでweighted diffusion regressionを反復する</p></td><td><p>毎epochのK-samplingと幾何reward計算を省き、RL固有の高コスト部分を再利用する</p></td></tr><tr><td><p>6 · EMA refresh</p></td><td><p>saved EMAを固定deployment screenで比較し、best policyから次のrollout cacheを再生成する</p></td><td><p>古いpolicyの候補だけを学び続けず、改善後に到達する新しいaction/stateへ更新する</p></td></tr></tbody></table><p><strong>同点group：</strong>全K候補のrewardが同一ならAWRの選好情報がないため、HDP原論文に合わせてRL updateから除外する。training lossの低下だけではpolicy improvementと判定しない。</p><h3>8.3 Data curriculum：全体性能とhard sceneを同時に改善する</h3><p>最初のcycleはfailure-onlyにせず、全量clean distributionを一様にminingする。次cycle以降は、各sceneをreward結果から三つに分ける。</p><table data-layout="wide"><tbody><tr><th><p>Scene group</p></th><th><p>判定</p></th><th><p>処理</p></th></tr><tr><td><p>通常 / retention</p></td><td><p>deployment trajectoryが高score、または候補差が小さい</p></td><td><p>全量distributionから継続sampleし、特定failureだけへの過適合を防ぐ</p></td></tr><tr><td><p>Learnable hard</p></td><td><p>現在trajectoryは低scoreだが、K内にhard-safeかつ高reward候補がある</p></td><td><p><code>Rbest−Rdet</code>とhard-eventをpriorityにして重点replayする。AWRが最も直接改善できる層</p></td></tr><tr><td><p>Unrecoverable</p></td><td><p>all-zero、全候補が同じ違反、または必要なsemantic modeがK内にない</p></td><td><p>通常AWRを反復せず、candidate coverageの改善または限定的R2LPL repairへ送る</p></td></tr></tbody></table><p><strong>次cycleの初期sampling proposal：</strong>50%をfull clean corpusから一様sampleし、50%をLearnable hard poolからpriority sampleする。これを100% uniform AWRと成対ablationし、hard gainと全体retentionの実測で比率を決める。BC/SFT branchは初期proposalに追加せず、忘却が実測された場合だけ別ablationとする。</p><h3>8.4 Candidate coverage：AWRで解けないsceneだけR2LPLを吸収する</h3><p>AWRはK候補の順位を学ぶため、K本すべてが同じ失敗なら正しいtargetを作れない。noiseを大きくするだけでsemantic modeが増えない場合は、次の順でcoverageを広げる。</p><ol><li><p>native diffusion samplingとHDP local augmentationで候補を作る。</p></li><li><p>hard sceneだけKまたはsampling条件を増やし、stop/go、yield/go、左右回避などのsemantic mode数を確認する。</p></li><li><p>それでもhard-safe candidateがないall-zero / override sceneだけ、R2LPLのmistake mining、retrieval、feasible repairまたはmorph candidateを使う。</p></li><li><p>repairを別modelへSFTするのではなく、同じcandidate poolへ投入し、同じhard gateとrewardで再採点して同じAWR objectiveへ戻す。</p></li></ol><ac:image ac:align="center" ac:layout="center" ac:original-height="1419" ac:original-width="3854" ac:custom-width="true" ac:local-id="d9c36d1ffd68" ac:alt="R2LPL as a fallback candidate generator for unrecoverable AWR scenes" ac:width="900"><ri:attachment ri:filename="t4-awr-reference-external-learning-from-mistakes-rollout-retrieval-lifelong-policy-learning-for-autonomous-driving-docs-assets-r2lpl_framework.png" ri:version-at-save="1" /></ac:image><p local-id="b9af9b32d268"><em>位置づけ：R2LPLはtraining backboneではない。通常のAWR candidate setに正解が存在しないsceneへ、hard-safeな追加候補を供給するfallbackである。</em></p><h3>8.5 単一checkpointの選定protocol</h3><ol><li><p><strong>Saved-EMA screen：</strong>全epochを同じscene paths、K=1、zero-noise、同じsolver stepsで比較する。</p></li><li><p><strong>Full paired validation：</strong>screen上位だけを全validationへ進め、同じsceneごとの差分とconfidence intervalを計算する。primary criterionはformal deployment rewardであり、ADE/FDEと各eventは原因を読むdiagnosticとして併記する。全指標の同時非劣化を機械的vetoにはしない。</p></li><li><p><strong>Reactive / closed-loop：</strong>open-loopで選んだ単一checkpointをreactive simulation、closed-loop scenario、実車smokeの順に確認する。ここで新しいcritical behaviorが見つかればsceneを次cycleのhard poolへ戻す。</p></li><li><p><strong>Best-not-last：</strong>最後のepochや最小training lossではなく、上記protocolで最高のEMAだけをreplacement artifactとして固定する。</p></li></ol><h3>8.6 必須ablationと完了条件</h3><table data-layout="wide"><tbody><tr><th><p>比較</p></th><th><p>確認する問い</p></th></tr><tr><td><p>Original DP vs full-uniform AWR</p></td><td><p>post-training coreだけで全体rewardが再現性をもって上がるか</p></td></tr><tr><td><p>Uniform vs 50:50 priority curriculum</p></td><td><p>全体分布を保持しながらhard scene改善を増幅できるか</p></td></tr><tr><td><p>HDP augmentation ON/OFF</p></td><td><p>局所候補spreadが実際のAWR gainへ寄与するか</p></td></tr><tr><td><p>K / semantic diversity sweep</p></td><td><p>Kを増やす価値があるsceneと、mode collapseしているsceneを分離できるか</p></td></tr><tr><td><p>R2LPL fallback ON/OFF</p></td><td><p>all-zero / override sceneだけで追加candidateが有効か</p></td></tr><tr><td><p>Refresh cycle / training length</p></td><td><p>何cycleで改善が飽和し、いつbest checkpointを固定すべきか</p></td></tr></tbody></table><p><strong>完了成果物：</strong>単一replacement checkpoint、固定reward specification、再現可能なfull paired evaluation、代表sceneのtrajectory / OBB / dynamic GIF evidence、reactive・closed-loop・実車smoke結果を一組として提出する。</p><p><strong>現在の証拠境界：</strong>第7節の10-epoch pilotは、全量data上のOriginal DP AWRでopen-loop formal rewardを改善できる初期証拠である。一方、最適なrefresh回数、priority curriculum、R2LPL fallback、reactive / closed-loop / 実車性能はこのproposalに基づく次の検証対象であり、昨晩の結果だけで証明済みとは主張しない。</p>'''


def _replace_between(storage: str, start: str, end: str, replacement: str, label: str) -> str:
    if storage.count(start) != 1:
        raise RuntimeError(f"expected one {label} start, got {storage.count(start)}")
    if storage.count(end) != 1:
        raise RuntimeError(f"expected one {label} end, got {storage.count(end)}")
    begin = storage.index(start)
    finish = storage.index(end, begin)
    return storage[:begin] + replacement + storage[finish:]


def _validate_xml(storage: str) -> None:
    wrapped = (
        '<root xmlns:ac="http://atlassian.com/content" '
        'xmlns:ri="http://atlassian.com/resource/identifier">'
        + storage
        + "</root>"
    )
    wrapped = re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", wrapped
    )
    ET.fromstring(wrapped)


def _repair(storage: str) -> str:
    repaired = _replace_between(
        storage, LANE_START, SECTION_6, LANE_REPLACEMENT, "lane explainer"
    )
    repaired = _replace_between(repaired, SECTION_8, SECTION_9, PROPOSAL, "proposal")
    _validate_xml(repaired)
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-id", default="5392827425")
    parser.add_argument("--expected-remote-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    confluence = Confluence()
    page = confluence.get_page(args.page_id)
    storage = (((page.get("body") or {}).get("storage") or {}).get("value") or "")
    actual_hash = hashlib.sha256(storage.encode()).hexdigest()
    expected_hash = args.expected_remote_sha256.strip().lower()
    if page.get("status") != "draft":
        raise SystemExit(f"refusing non-draft page: {page.get('status')}")
    if actual_hash != expected_hash:
        raise SystemExit(f"remote body changed; refusing update: {actual_hash}")

    repaired = _repair(storage)
    new_hash = hashlib.sha256(repaired.encode()).hexdigest()
    checks = {
        "proposal_title": "8. 最終提案：T4 AWR Post-Training Framework" in repaired,
        "single_checkpoint": "単一checkpoint" in repaired,
        "lane_audit_removed": "Lane maskの四象限監査" not in repaired,
        "lane_gif_removed": "lane_mask_audit.gif" not in repaired,
        "lane_role_correct": "Lane Keepingはhard gateではない" in repaired,
        "full_data_loop": "Core loop：全量dataで反復するpolicy improvement" in repaired,
        "r2lpl_fallback": "training backboneではない" in repaired,
        "experiment_not_proposal_removed": "coverageを閉じる" not in repaired,
        "references_preserved": SECTION_9 in repaired,
    }
    print(
        {
            "old_sha256": actual_hash,
            "new_sha256": new_hash,
            "old_chars": len(storage),
            "new_chars": len(repaired),
            "checks": checks,
        }
    )
    if not all(checks.values()):
        raise RuntimeError("local verification failed")
    if args.dry_run:
        return 0

    latest = confluence.get_page(args.page_id)
    latest_storage = (((latest.get("body") or {}).get("storage") or {}).get("value") or "")
    if hashlib.sha256(latest_storage.encode()).hexdigest() != expected_hash:
        raise SystemExit("remote body changed before update; refusing write")
    confluence.update_draft(latest, repaired)

    verify = confluence.get_page(args.page_id)
    value = (((verify.get("body") or {}).get("storage") or {}).get("value") or "")
    remote_checks = {
        "status_draft": verify.get("status") == "draft",
        "proposal_title": value.count("8. 最終提案：T4 AWR Post-Training Framework") == 1,
        "single_checkpoint": "単一checkpoint" in value,
        "lane_audit_removed": "Lane maskの四象限監査" not in value,
        "lane_gif_removed": "lane_mask_audit.gif" not in value,
        "lane_role_correct": "Lane Keepingはhard gateではない" in value,
        "r2lpl_image_once": value.count("r2lpl_framework.png") == 1,
        "section_9_once": value.count(SECTION_9) == 1,
    }
    print(
        {
            "updated": True,
            "sha256": hashlib.sha256(value.encode()).hexdigest(),
            "chars": len(value),
            "checks": remote_checks,
        }
    )
    if not all(remote_checks.values()):
        raise RuntimeError("remote verification failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
