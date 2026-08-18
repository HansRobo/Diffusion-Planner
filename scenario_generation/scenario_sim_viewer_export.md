# closed_loop_scenario データ形式

OpenSCENARIO スイートを scenario_simulator_v2 で closed-loop 実行した評価結果です。
共有ルートの `<dataset root>/closed_loop_scenario/` に置きます。

## 配置

```text
closed_loop_scenario/
  <run>/                        # 評価実行 1 回。dataset 直下の 1 段だけ
    run.json                    # 実行の素性・員数
    scenarios.json              # シナリオ 82 件のメタと集計
    cases.jsonl                 # ケース 452 行（一覧の行）
    media/
      <scenario_id>/            # Web.Auto のシナリオ ID（uuid）
        <case>.mp4
        <case>.rollout.jsonl
        <case>.<metric>.png
```

一覧に必要なのは **`scenarios.json` と `cases.jsonl` の 2 ファイルだけ**です。
`media/` を読まずに一覧とサマリを組み立てられます。

## 概念

| 語 | 意味 |
|---|---|
| run | スイート実行 1 回。ckpt とコミットが 1 つに決まる単位 |
| scenario | Web.Auto のシナリオ。`scenario_id`(uuid) が識別子 |
| case | シナリオのパラメータ展開 1 件 = ロールアウト 1 回。**一覧の 1 行** |

**同一シーンの突き合わせキーは `(scenario, route)`** です。`route` はパラメータの値から作って
いるので、シナリオが改版されても同じ条件のケース同士が対応します。

npz 経路と違い、**記録走行という基準がありません**。そのため区間(`segment`)の概念が無く、
1 シナリオ展開 = 1 ロールアウトです。

## run.json

```json
{
  "run_dir": "/mnt/.../full_run/verify-final2-3989/run",
  "ckpt": "/mnt/.../epoch0080/best_model.pth",
  "dp_commit": "da3a0270...", "branch": "final-v1",
  "draw_every": "4", "fps": 10.0, "max_steps": "1700",
  "submitted_cases": 464,
  "verdicts": { "pass": 21, "failure": 218, "error": 1, "undecided": 212 },
  "missing_rows": [{ "case_key": "<case_key>", "reason": "SyntaxError: ..." }]
}
```

`verdicts` がラン全体の成否です（`scenarios.json` の合計）。

`submitted_cases` が投入数、`cases.jsonl` の行数が成功数です。**差が失敗**で、
`missing_rows` にその一覧と理由が入ります。走らなかったケースもディレクトリは残るので、
ファイルを数えても検出できません。

## scenarios.json

`scenario_id` をキーにしたオブジェクト。

```json
{
  "08a38cfe-...": {
    "name": "N-01-10000_case01_dp",
    "category": "N",
    "category_name": "死角",
    "description": "路駐車両による死角からの飛び出し",
    "map": "2231",
    "version": "14",
    "n_cases": 12,
    "verdicts": { "pass": 0, "failure": 7, "error": 0, "undecided": 5 },
    "error": null,
    "unmeasured_keys": ["mean_route_completion", "mean_gt_deviation_m",
                        "red_light_violation", "reproducer"],
    "summary": { "n_segments": 12, "total_steps": 9580, "object": {...}, ... }
  }
}
```

| フィールド | 用途 |
|---|---|
| `name` / `description` | 表示名。uuid のままでは読めないので |
| `category` / `category_name` | 絞り込み。`A`〜`Z` と日本語名 |
| `map` / `version` | 絞り込み。**別の地図の結果は比較できない**ので区別が要る |
| `verdicts` | `pass` / `failure` / `error` / `undecided` の 4 件数。合計は `n_cases` |
| `error` | そのシナリオで行を書けなかったケース数と理由。null なら完全 |
| `unmeasured_keys` | **測っていない**項目名。0 と区別するため |
| | 1 ケースも行が出なかったシナリオも、`n_cases: 0` と `error` を持つエントリとして現れます |
| `summary` | `aggregate()` の出力。npz 経路の `summary.json` と同じ形 |

### 測っていない項目（重要）

この経路は基準走行を持たないため、次は**測定していません**。0 ではなく「無い」です。

| 項目 | 理由 |
|---|---|
| `route_completion` / `mean_gt_deviation_m` | 基準となる記録走行が無い |
| `red_light_violation` | 信号を読んでいない（`measured: false` が付く） |
| `reproducer`（`snap_count` 等） | 再現器そのものが無く、snap が構造的に起き得ない |

`summary` からはキーごと省いてあり、`unmeasured_keys` に名前が並びます。
**`?? 0` で補うと「完走率 0%」「Stuck 0 件」と表示され、実測と区別が付かなくなります。**
一覧では空欄・`—` 等にしていただけると助かります。

## cases.jsonl

1 行 1 ケース。行のスキーマは npz 経路の `segments.jsonl` と同じで、
`scenario` / `route` / `case_key` が加わります。

```json
{
  "scenario": "08a38cfe-...",
  "route": "ego_speed8p3333_pedestrian_speed1p3889_startpoint40",
  "case_key": "08a38cfe-..._14_scenario_0",
  "n_steps_run": 199, "terminated": "max_steps", "result_kind": "Failure",
  "progress_m": 160.6,
  "object": { "collision_count": 1, ... },
  "road_border": {...}, "red_light_violation": {...},
  "strong_brake": {...}, "reproducer": {...}
}
```

- `route` は**そのケースで変えたパラメータの値**から作った名前です。同じシナリオ内で一意で、
  メディアのファイル名でもあります
- `case_key` は元のラン上のディレクトリ名。追跡用で、表示には使いません
- `segment` はありません

### ケースの成否 — この経路で最も重要な値

`verdict` が**シナリオ自身の判定**です。

```json
"verdict": {
  "decided": true,
  "kind": "Failure",
  "type": "SimulationFailure",
  "trigger": "Is any of [ego] colliding with another given entity Npc1?",
  "unmet": ["goal_position", "scenarioCheck_npc_cross"]
}
```

| キー | 意味 |
|---|---|
| `decided` | シナリオが判定に到達したか。**`false` は評価側の歩数上限による打ち切り** |
| `kind` | `Pass` / `Failure` / `Error`。`decided: false` のとき**このキーはありません** |
| `type` | `SimulationFailure` / `SyntaxError` など |
| `trigger` | 判定を引いた条件の文 |
| `unmet` | 満たせなかった success condition 名。シナリオ作者が付けた名前です |

**同じ行にある `result_kind` は判定ではありません。** インタプリタが configure 時に
`Failure("Timeout", …)` を初期値として置くため、**判定に到達しなかったケースでも `Failure` と
読めてしまいます**。生の値として残してありますが、成否は必ず `verdict` を見てください。

実測 452 ケースのうち判定に到達したのは 240 件で、`Pass` 21 / `Failure` 218 / `Error` 1。
残る 212 件は `decided: false` です。**この 212 件を失敗に数えないでください。**

未判定が出るのは、**評価側の歩数上限がシナリオ自身のタイムアウト条件より手前にある**ためです。
シナリオはいずれも `SimulationTimeCondition` を持っていて、期限に達すると `exitFailure` で
判定に到達します（実測 76 件がこれ）。このランは `max_steps: 1700` = 169.5 s で、未判定 212 件の
シナリオはすべて 180〜200 s の条件を持つため、その手前で切れています。
**上限を 2010 以上にすれば、全ケースがシナリオ自身の判定に到達します。**

`Pass` は success condition の**すべて**を満たすことを要求します。実測で満たせなかった条件は
`goal_position` 170 件・`ego_speed` 121 件・方向指示器 64 件・`ego_stop` 32 件で、
走行の安全性とは別の条件も含みます。**Pass 率は走行品質の指標ではありません。**

`terminated` は走行が止まった理由（`sim_terminated` / `max_steps`）で、`verdict.decided` と
対応します。

## media

```text
media/<scenario_id>/<route>.mp4
                    <route>.rollout.jsonl
                    <route>.<metric>.png
```

`<metric>` は `clearance` / `collision` / `near_miss` / `speed` / `road_border` /
`strong_brake`。**測っていないメトリクスの画像は出しません**（平坦な「異常なし」を
描かないため）。`red_light` は常にありません。

### 動画時刻

```
video_time[s] = step / 40
```

4 sim step ごとに 1 フレーム、10fps。間引き幅は実行ごとに変えられますが、
**この関係が成り立つように mp4 のタイムスタンプを揃えて出力**しているので、
読む側は定数として扱って構いません。`run.json` の `draw_every` / `fps` に実際の値があります。

### rollout.jsonl

1 行 1 sim step。npz 経路の `rollout.jsonl` と同じスキーマです。

```json
{"k": 12, "ego": [x, y], "yaw": 0.1, "dist_goal": 30.2,
 "speed": 2.57, "clearance_m": null, "collision": false, "rb_dist_m": 0.52}
```

末尾に `{"event": "terminated", "k": 760, "reason": "goal"}` が付きます。
`gt_deviation_m` / `red_light_violation` / `snap_count` は**ありません**（未計測）。

## JSON の非有限値について

`clearance_min_m` などは「有限サンプルが無い」を `inf` で表しますが、
**出力時に `null` へ変換しています**。`Infinity` は JSON の仕様外で `JSON.parse` が
拒否するためです（npz 経路の `summary.json` には現在 `Infinity` が入っています）。

## 生成方法

```bash
python3 -m scenario_generation.scenario_sim_viewer_export \
  --run_dir  <スイートのラン出力> \
  --out_root <dataset>/closed_loop_scenario/<run 名>
```

`<run 名>` は 1 階層にまとめます（例 `20260817-230105_final-v1-3989`）。

シナリオの表示名とカテゴリ名は、スイートに同梱した `scenario_names.json`
（`scenario_id → display_name` と `カテゴリ文字 → 名前`）から読んでいます。
Web.Auto の suite 定義由来で、コードには持たせていません。
