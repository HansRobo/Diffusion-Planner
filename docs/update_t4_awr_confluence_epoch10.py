#!/usr/bin/env python3
"""Targeted epoch-10 completion update for the manually edited T4 AWR draft."""

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
CHART = ROOT / "docs/t4_conference_assets/checkpoint_selection/epoch_2_10_checkpoint_selection.png"
DATA_RUNTIME = ROOT / "docs/t4_conference_assets/data_runtime/full_data_replay_and_runtime.png"
HDP_MULTIMODAL = ROOT / "docs/t4_conference_assets/hdp_multimodality/hdp_native_multimodality_explainer.png"


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, got {count}")
    return value.replace(old, new, 1)


def _image(path: Path, alt: str) -> str:
    return (
        '<ac:image ac:align="center" ac:layout="center" ac:width="1200" '
        f'ac:alt="{html.escape(alt, quote=True)}">'
        f'<ri:attachment ri:filename="{html.escape(attachment_name(path), quote=True)}" />'
        "</ac:image>"
    )


def _repair(storage: str) -> str:
    storage = _replace_once(
        storage,
        "固定2,048 sceneの中間monitorに加え、選定した単一EMA checkpointは全46,068-scene paired validationでmean rewardを0.9303&rarr;0.9411（+1.16%）、road-border eventを940&rarr;266へ改善した。",
        "全量dataの10-epoch pilot（1 mining + 9 replay-training pass）が完了した。saved EMA epoch 2&ndash;10を同じ2,048 scenesでscreenし、epoch 7を選定。全46,068-scene paired validationではmean rewardを0.9303&rarr;0.9411（+1.16%）、road-border eventを940&rarr;266へ改善した。ただしHDP論文はRL総epoch数を公開しておらず、必要な学習長は次cycle以降の実験で決める。",
        "executive epoch status",
    )

    awr_anchor = (
        '<p local-id="1742b277b72e"><strong>K=10の意味：</strong>10回のPython逐次推論ではない。'
        'B&times;Kを一つのbatchとして5回のdiffusion solver stepを通す。10本は独立Gaussian noiseから始まるため、'
        '同じsceneでも異なる候補を出せる。</p>'
    )
    awr_explainer = awr_anchor + (
        '<h3>高reward trajectoryはどう学習に使われるか</h3>'
        '<p><strong>AWRはrewardを微分しない。</strong>旧policyが生成したK本をrewardで順位付けし、'
        '高reward trajectoryをより大きな重みでdiffusion regressionする。</p>'
        '<p><strong>1 &middot; scene内の相対評価：</strong>'
        '<code>zᵢ = (Rᵢ &minus; mean(R)) / (sample_std(R) + 10⁻⁶)</code>、'
        '<code>wᵢ = exp(&beta;zᵢ)</code>（今回&beta;=1）。高得点は<code>w&gt;1</code>、低得点は'
        '<code>0&lt;w&lt;1</code>。finiteな同点groupは全候補<code>w=1</code>で保持する。</p>'
        '<p><strong>2 &middot; 候補をdiffusion target化：</strong>各候補τᵢごとに独立な'
        '<code>tᵢ ~ U(0.001, 0.999)</code>と<code>εᵢ ~ N(0,I)</code>を引き、'
        '<code>τt,ᵢ = α(tᵢ)τᵢ + σ(tᵢ)εᵢ</code>を作る。Original DPはnoisy trajectoryから'
        '80-stepの<code>[x,y,cos θ,sin θ]</code> targetを復元する。</p>'
        '<p><strong>3 &middot; weighted update：</strong>'
        '<code>L = mean(wᵢ &times; MSE(fθ(τt,ᵢ,tᵢ,s), τᵢ))</code>。'
        'したがって高reward候補は大きなgradient、低reward候補は小さなgradientを作る。'
        '今回のformal runは<code>safe_only=false</code>なので低得点候補も完全削除せず、'
        '<code>bc_weight=0</code> / expert anchor=0なのでhuman GTを追加していない。</p>'
        '<h3>Rollout&ndash;replay cache：AWR固有の加速</h3>'
        '<p>高コストなのは、各sceneでK本を生成し、OBB / road geometry / TTCなどを計算してrewardとweightを作るrollout phase。'
        'HDP-style scheduleはこれを毎gradient epochで繰り返さず、trajectory、reward、weight、policy contextを一度freezeしてreplay bufferへ置く。'
        '後続epochは固定pseudo-targetへのweighted diffusion regressionだけを反復するため、K=10 samplingとreward geometryを再実行しない。'
        'ただしpolicyが改善するとcacheは古くなるため、一定周期で最新policyからrolloutを再生成する。今回の1-based scheduleはepoch 1でmining、'
        'epoch 2&ndash;10でreplay、次のrefreshはepoch 11である。</p>'
    )
    storage = _replace_once(storage, awr_anchor, awr_explainer, "AWR training and replay explainer")

    storage = _replace_once(
        storage,
        '<tr ac:local-id="49ad32f526b2"><td ac:local-id="b7cdfd422f7c"><p local-id="fb9852124a2b">Kinematic</p></td><td ac:local-id="60a485bd5a67"><p local-id="50aba5f0184c">急なyaw、加速度、jerkで実行不能な軌跡</p></td><td ac:local-id="edae288c0ac9"><p local-id="9ee46c2e8675">車両として実行可能</p></td><td ac:local-id="f7ce540bafc2"><p local-id="a5ca2ab89cde">hard gate</p></td></tr>',
        '<tr ac:local-id="49ad32f526b2"><td ac:local-id="b7cdfd422f7c"><p local-id="fb9852124a2b">Kinematic</p></td><td ac:local-id="60a485bd5a67"><p local-id="50aba5f0184c">急なyaw、低速での旋回、車両のsteering/曲率上限を超える軌跡</p></td><td ac:local-id="edae288c0ac9"><p local-id="9ee46c2e8675">全時刻でyaw-rateとbicycle-model上限を満たす</p></td><td ac:local-id="f7ce540bafc2"><p local-id="a5ca2ab89cde">hard gate</p></td></tr>',
        "kinematic metric row",
    )
    storage = _replace_once(
        storage,
        '</table><h3 local-id="81cac607ad16">実sceneで見る</h3>',
        '</table><p><strong>Kinematic rateの正確な意味：</strong>sceneごとのdeployment trajectoryをSG平滑化し、'
        '<code>|wrap(θt+1&minus;θt)| / 0.1 &le; 1.0 rad/s</code>、かつ'
        '<code>yaw_rate &le; [2.5&times;tan(0.64)/wheelbase]&times;max(speed,0.1)</code>を全時刻で確認する。'
        'どこか1 stepでも超えればそのsceneはkinematic failureでR=0。rateは失敗scene数/Nであり、失敗timestep率ではない。'
        '加速度・横加速度は別のcontinuous feasibility/comfort項で、kinematic hard-failure countには含めない。'
        '全46,068 scenesではOriginal DP 55（0.1194%）&rarr;AWR epoch 7 68（0.1476%）、net +13。'
        'paired-bootstrap 95% CIは0を跨ぐため、現時点ではsystematicな増加を確定できない。</p>'
        '<h3 local-id="81cac607ad16">実sceneで見る</h3>',
        "kinematic definition",
    )

    diversity_intro = (
        '<p local-id="6e1b35199924">AWRは既存のK候補を並べ替える方法であり、全候補が同じ失敗をすると直せない。'
        'したがって「K本あるか」ではなく、stop / go、速度、横方向、曲率などの<strong>意味のあるmodeが含まれるか</strong>を確認する。</p>'
    )
    diversity_expanded = diversity_intro + _image(
        HDP_MULTIMODAL,
        "HDP paper data scaling and learned native diffusion multimodality",
    ) + (
        '<p><em><strong>HDP論文のsemantic multimodality：</strong>100K規模では同一sceneのsampleが一つにcollapseするが、'
        '20M規模では複数の挙動へ分岐する。論文はこれを大規模real-driving dataによる学習能力として説明し、'
        'Trajectory Divergence（同一scene sample間の平均pairwise距離）で測る。anchor/goal conditioningや下記augmentationとは別の主張である。</em></p>'
        '<h3>HDP-style trajectory augmentationは何をしているか</h3>'
        '<p>release codeのRL rolloutでは、生成後の各candidateに一組の'
        '<code>a,b ~ N(0,0.5²)</code>を与え、全horizonで'
        '<code>x′t = xt + a cos θt &minus; b sin θt</code>、'
        '<code>y′t = yt + a sin θt + b cos θt</code>と局所的に平行移動する。headingは変更しない。'
        'つまりcandidate周辺の縦/横方向を探索し、その増広後trajectoryを同じrewardで採点してAWR targetにする。'
        'これは<strong>局所的な幾何diversity</strong>を増やす方法であり、stop/goやyield/goを生成するsemantic mode generatorではない。'
        'referenceは最初の5 epochだけ適用し、今回の10-epoch cycleでは初回miningにのみ適用した。</p>'
    )
    storage = _replace_once(storage, diversity_intro, diversity_expanded, "HDP multimodality distinction")
    storage = _replace_once(
        storage,
        '<p local-id="e9a8df767e6c"><em><strong>多様性の動的診断（formal training resultではない）：</strong>同一seed・同一実sceneでraw K=16とHDP-style augmentation K=16を比較。endpoint pairwise meanは1.51&rarr;1.83 m、横方向stdは0.05&rarr;0.43 mへ増えるが、2 m thresholdでは両方1 mode。augmentationは幾何spreadを増やすが、別maneuverを保証しない。</em></p>',
        '<p local-id="e9a8df767e6c"><em><strong>局所幾何diversityの動的診断：</strong>同一checkpoint・scene・seedでraw K=16と、上記座標augmentation後のK=16を比較。endpoint pairwise meanは1.51&rarr;1.83 m、横方向stdは0.05&rarr;0.43 mへ増える。一方2 m thresholdでは両方1 modeのまま。したがって図が証明するのはlocal spreadの増加であり、学習済みsemantic multimodalityの改善ではない。</em></p>',
        "augmentation caption",
    )

    old_intro = (
        '<p local-id="c1203515f709">学習中は固定2,048-scene monitorでepochを追跡し、最終的に'
        '<strong>epoch-7の単一EMA checkpoint</strong>を固定して、is_skipped filtering後の全46,068 validation scenesを成対評価した。'
        '下図は中間monitorではなく、deployment設定（K=1 / zero noise / 10 solver steps）の最終結果。</p>'
    )
    new_intro = (
        '<p local-id="c1203515f709"><strong>全量dataの10-epoch pilotが完了。</strong>'
        'これはepoch 1のrollout/miningとepoch 2&ndash;10の9 replay-training passからなる最初の1 cycleであり、HDP論文と同量のRL trainingを意味しない。'
        '論文はILの10 epochs（64&times;H20）は明記するが、RLの総epoch数は公開していないため、必要な学習長は次cycle以降のdeployment evalで決める。'
        '最後のcheckpointを自動採用せず、同一2,048-scene deployment screenでsaved EMA epoch 2&ndash;10を比較した。'
        'epoch 7がscoreとpaired-bootstrap 95% CI下限の両方で最高となり、epoch 10はbaselineより有意に高いがepoch 7を下回った。'
        'そのためepoch 5/6/7だけをis_skipped filtering後の全46,068 scenesへ進め、full mean rewardが最も高いepoch 7を最終checkpointとした。</p>'
        + _image(CHART, "ten epoch early AWR pilot saved EMA checkpoint selection evidence")
        + '<p><em><strong>10-epoch pilot / checkpoint selection：</strong>'
        '2,048-scene screenはbaseline 92.582、epoch 7は94.052（gain +1.470 pt、95% CI lower +0.846 pt）、'
        'epoch 10は93.876（gain +1.294 pt、95% CI lower +0.631 pt）。'
        'Replay train lossはepoch 2の0.001056からepoch 10の0.000703まで低下したが、deployment scoreはepoch 7でpeak。'
        'したがってloss最小またはlast epochではなく、saved-EMA deployment scoreでcheckpointを選ぶ。'
        'Full-validation finalistのmean rewardはepoch 5=0.940505、epoch 6=0.940917、epoch 7=0.941053。</em></p>'
        + _image(DATA_RUNTIME, "HDP style rollout replay decoupling full data selection and measured runtime")
        + '<p><em><strong>Data selection / RL-specific cache：</strong>raw train 5,092,382 scenesからis_skipped 514,346を除外し、'
        'clean target 4,578,036をfailure-onlyに絞らず全量miningした。今回実際に学習へ使ったfrozen replayは4,501,440 scene-groups。'
        'easy/normal scenesは最初から含まれ、同点groupは全K候補w=1なのでold-policy self-distillationとして残るが、human-GT rehearsalではない。'
        'mining/replay buildは約6.1 h、9 replay passは約7.0 h（46.6 min/pass）、mining開始からepoch-10 checkpointまで約13.3 h。'
        '同じcostが続く仮定では20/50/100 epochsは約27/67/133 hだが、性能は線形外挿せず、各refresh cycleのscreenで継続/停止を決める。</em></p>'
    )
    storage = _replace_once(storage, old_intro, new_intro, "experiment intro")

    storage = _replace_once(
        storage,
        '<p local-id="0f796665da59"><strong>方法は有望。</strong>平均scene scoreは統計的に改善。</p>',
        '<p local-id="0f796665da59"><strong>10-epoch pilot完了。</strong>最初の1 mining + 9 replay passと固定evalを完走。lossは最適化の進行確認にのみ使い、checkpoint選定には使わない。必要な総epoch数は未確定。</p>',
        "monitor evidence row",
    )
    storage = _replace_once(
        storage,
        '<p local-id="45ca16ee9deb">epoch 2&ndash;10をscreenし、epoch 7の単一EMA checkpointをfull validationへ固定。</p>',
        '<p local-id="45ca16ee9deb"><strong>完了。</strong>epoch 2&ndash;10を同じ2,048 scenesでscreen。epoch 7 score=94.052 / gain=+1.470 pt / CI lower=+0.846 ptで首位。epoch 10もbaselineより改善したが93.876のため採用しない。</p>',
        "saved EMA evidence row",
    )
    storage = _replace_once(
        storage,
        '<p local-id="76821e6580f6"><strong>checkpoint選定基準：</strong>同じ46,068 sceneでsourceとsaved EMAを比較し、<code>100 &times; mean(Rema &minus; Rsource)</code>のpaired-bootstrap 95% CI下限が最大のepochを選ぶ。下限が0より大きければ平均score改善と判定する。ADE/FDE、collision、road-border、kinematic、lane-changeは変化の理由を示す診断として併記するが、すべて同時改善を要求する独立vetoにはしない。</p>',
        '<p local-id="76821e6580f6"><strong>checkpoint選定基準（二段階）：</strong>'
        '第1段階は全saved EMA epoch 2&ndash;10を<strong>同じ2,048 scene paths</strong>、K=1 / zero noise / 10 solver stepsで比較し、'
        '<code>100 &times; mean(Rema &minus; Rsource)</code>のpaired-bootstrap 95% CI下限を主順位、mean gainをtie-breakerとする。'
        '第2段階はfinalist epoch 5/6/7を全46,068 scenesで再評価し、full mean reward最大のepoch 7を固定する。'
        'ADE/FDE、collision、road-border、kinematic、lane-changeは変化理由を示す診断として併記するが、すべて同時改善を要求する独立vetoにはしない。</p>',
        "checkpoint selection method",
    )
    storage = _replace_once(
        storage,
        '<p local-id="da50464f4717"><strong>Faithful baselineを完走：</strong>現在のHDP-style AWRを途中変更せず、saved EMAをfull validationする。</p>',
        '<p local-id="da50464f4717"><strong>第1 rollout&ndash;replay cycleは完了：</strong>10-epoch pilot、saved-EMA screen、epoch 5/6/7のfull validationを完了。次は最新policyでepoch 11 refreshを行い、改善が続くかを同じprotocolで測る。</p>',
        "completed baseline proposal item",
    )
    storage = _replace_once(
        storage,
        '<p local-id="efe8986f0635"><strong>単一checkpointを選ぶ：</strong>full mean-score CIで最良のEMAだけをreplacement候補にする。</p>',
        '<p local-id="efe8986f0635"><strong>単一checkpointを固定：</strong>epoch 10ではなく、screenとfull validationの双方で首位を維持したepoch 7 EMAをreplacement候補にする。</p>',
        "selected checkpoint proposal item",
    )

    wrapped = (
        '<root xmlns:ac="http://atlassian.com/content" '
        'xmlns:ri="http://atlassian.com/resource/identifier">' + storage + "</root>"
    )
    validation = re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", wrapped
    )
    ET.fromstring(validation)
    return storage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-id", default="5392827425")
    parser.add_argument("--expected-remote-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    attachments = [CHART, DATA_RUNTIME, HDP_MULTIMODAL]
    for path in attachments:
        if not path.is_file():
            raise SystemExit(f"missing attachment: {path}")

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
    print({"old_sha256": actual_hash, "new_sha256": new_hash, "chart": attachment_name(CHART)})
    if args.dry_run:
        return 0

    existing = confluence.list_attachments(args.page_id)
    chart_name = attachment_name(CHART)
    for path in attachments:
        name = attachment_name(path)
        confluence.upload_attachment(args.page_id, path, name, existing.get(name))

    latest = confluence.get_page(args.page_id)
    latest_storage = (((latest.get("body") or {}).get("storage") or {}).get("value") or "")
    if hashlib.sha256(latest_storage.encode()).hexdigest() != expected_hash:
        raise SystemExit("remote body changed during upload; body update refused")
    confluence.update_draft(latest, repaired)

    verify = confluence.get_page(args.page_id)
    value = (((verify.get("body") or {}).get("storage") or {}).get("value") or "")
    checks = {
        "status_draft": verify.get("status") == "draft",
        "chart_once": value.count(chart_name) == 1,
        "runtime_chart_once": value.count(attachment_name(DATA_RUNTIME)) == 1,
        "multimodal_chart_once": value.count(attachment_name(HDP_MULTIMODAL)) == 1,
        "epoch_10_pilot": "全量dataの10-epoch pilotが完了" in value,
        "epoch_7_selected": "epoch 10ではなく" in value,
        "two_stage": "checkpoint選定基準（二段階）" in value,
        "awr_formula": "weighted diffusion regression" in value,
        "kinematic_formula": "Kinematic rateの正確な意味" in value,
        "augmentation_boundary": "局所的な幾何diversity" in value,
        "old_selection_removed": "同じ46,068 sceneでsourceとsaved EMAを比較し" not in value,
    }
    print({"updated": True, "checks": checks, "sha256": hashlib.sha256(value.encode()).hexdigest()})
    if not all(checks.values()):
        raise RuntimeError("remote verification failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
