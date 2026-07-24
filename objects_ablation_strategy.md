# 戦略: 物体あり/なし closed-loop 評価 (objects ablation)

## ゴール

各拠点(お台場など)を **物体あり / 物体なし** の両方で周回する closed-loop 評価を行い、結果を
HTML レポートと W&B に **並べて比較** できるようにする。

「物体なし」= 地図とルートはそのまま、他の交通(動的エージェント + 静的物体)が一切いない
空の世界を ego 単独で走らせるモード。目的は **交通への反応と、ルート/地図追従そのものを切り分ける
ablation 診断**:
- 物体ありで破綻、物体なしで完走 → 破綻原因は「交通への反応」
- 両方で破綻 → 「ルート/地図追従」自体の問題

## 確定した方針(ユーザー承認済み)

1. **実装方針 = ② メモリ上で除去するフラグ**。新データセット/converter は作らない(ディスク二重化で無駄)。
   共通コアのリファクタもしない(フラグが実質それの軽量版)。
2. **除去対象 = 動的 + 静的の両方**。すなわち `neighbor_agents_past` と `static_objects` の両方をゼロ化。
3. **適用範囲 = スタンドアロン評価 と 学習時の両方**。
   - スタンドアロン(`run_all_sites_closed_loop.py`): 各拠点を objects / no-objects 両方実行。
   - **学習時(`closed_loop_validate` in `train.py`): 数エポックごと(既存の `save_utd` 間隔)に、各拠点を
     objects と no-objects の両方走らせる**。同じ run に loss と一緒に時系列でlog → 1画面で追える。
   - トレードオフ: 学習時の closed-loop コストが約 2 倍。ユーザー承認済み(比較が目的なので許容)。
4. **ダッシュボード = 案B(1 ビュー・パネルごとにキー出し分け)**。W&B は複数ビューの同時表示/インポートを
   持たない(ビューは切替式)ため、1 run の loss + objects + no-objects を一画面で見るには 1 ビューに
   セクションを並べるしかない。その 1 ビュー内で:
   - **両モードで意味のある指標**(route_completion / curb_hits / gt_deviation / snaps)→ objects と
     no-objects を**同じグラフに重ねて直接比較**。
   - **物体ありでしか意味がない指標**(collision / near_miss)→ **objects ラベルのキーだけ**を描画
     (no-objects は常に 0 なので出さない)。

## データ上の「物体」の定義(調査済み)

npz のモデル入力キーのうち:

| キー | 分類 | 物体なしモードで |
|---|---|---|
| `neighbor_agents_past` (320,31,11) | 動的エージェント(他車/歩行者)。PerceptionReproducer が再生する本体 | **ゼロ化** |
| `static_objects` (N,10) | 駐車車両・コーン等の静的物体 | **ゼロ化** |
| `lanes` / `route_lanes` | 車線・ルート(地図) | 残す |
| `line_strings` (停止線/境界) | 地図 | 残す(縁石メトリクスに必要) |
| `polygons` (交差点領域) | 地図 | 残す |

`neighbors_live` (320,11) = `neighbor_agents_past[0,:,-1,:]` はスコアリング(衝突/クリアランス)に使われる。
これもゼロ化することで、valid マスク(`abs(neighbors_live[:,:6]).sum > 0`)が全て偽になり、
clearance=inf / collision=False になる(=ぶつかる相手がいない、正しい挙動)。

## 「perception reproducer なのに物体を扱わない」名前問題への整理

PerceptionReproducer 本体(ego 走行位置 → 記録フレームを対応づけるカーソル)は **一切変更しない**。
変えるのは「対応づけた記録フレームの neighbor/static チャンネルをモデル入力に流すか」だけ。
reproducer は従来通り仕事をし、rollout/eval 層が「空世界 variant」を要求する、という切り分け。
フラグ名は rollout/eval 層の `drop_objects` とし、reproducer の意味を汚さない。

---

## 実装サーフェス(ファイル別・具体)

### 1. `scenario_generation/reproducer_rollout.py` — `render_segment()`

**(a) シグネチャに追加**(既存の `abort_max_snaps: int = 0` の後、`) -> dict | SegmentResult:` の直前、
キーワード専用ブロック `*,` 以降):

```python
    drop_objects: bool = False,
```

**(b) docstring に一文追記**: `drop_objects=True` のとき neighbor_agents_past と static_objects を
毎ステップゼロ化し、モデル入力・描画・スコアリングすべてを「物体なし(空世界)」にする。地図
(lanes/route_lanes/line_strings/polygons)は残す。ablation 用。

**(c) ループ内の choke point 1 箇所**: `np_dict, neighbors_live, idx, slot_uuids, _wbu = pre`
(現状 L1538 付近)の **直後** に挿入:

```python
        if drop_objects:
            # Empty-world ablation: no other traffic (dynamic neighbors + static objects), map
            # kept. Zeroing makes every neighbor/static slot fail its validity mask, so the model
            # sees an empty scene, the PNG/video render empty, and scoring finds nothing to hit
            # (clearance inf, collision 0) — consistent across model input, draw, and scoring.
            np_dict["neighbor_agents_past"] = np.zeros_like(np_dict["neighbor_agents_past"])
            if "static_objects" in np_dict:
                np_dict["static_objects"] = np.zeros_like(np_dict["static_objects"])
            neighbors_live = np.zeros_like(neighbors_live)
```

- 挿入位置は abort 判定 / `_score_into` / `_draw_step` / replan の **すべてより前**。これ 1 箇所で
  モデル入力・描画・衝突スコアリングが一貫して「物体なし」になる。
- render_segment は非 gpu パス(`_pre_step(s)`、gpu_transform=False)なので、この choke point だけで
  full-route eval は網羅できる。gpu バッチ mining パスは今回のスコープ外(触らない)。

### 2. `scenario_generation/closed_loop_evaluation.py` — `RolloutParams`

- フィールド追加(`abort_max_snaps: int = 0` の後):
  ```python
      drop_objects: bool = False  # empty-world ablation (no dynamic/static objects; map kept)
  ```
- `render_kwargs()` の返す dict に追加:
  ```python
      "drop_objects": self.drop_objects,
  ```

### 3. `scenario_generation/closed_loop_cli.py`

- `add_rollout_args()` に引数追加:
  ```python
      parser.add_argument(
          "--drop_objects",
          action="store_true",
          help="empty-world ablation: run with NO other traffic (dynamic neighbors + static "
          "objects zeroed each step); map/route kept. collision/near-miss go to 0 by design.",
      )
  ```
- `rollout_params_from_args()` の `RolloutParams(...)` に追加:
  ```python
      drop_objects=args.drop_objects,
  ```

### 4. `diffusion_planner/run_all_sites_closed_loop.py` — 両モード実行 + 複合ラベル

**方針**: 各拠点を「モード」ごとに実行し、物体なしは **out_dir とサイト名にサフィックス `__noobj`** を付ける。
これで下流(sites_summary / HTML / W&B)は `odaiba` と `odaiba__noobj` を **別サイト扱い** し、既存の
「拠点別」表示機構にそのまま載る(W&B scores は同じ指標グラフに両者が重なる = 直接比較)。

- `parse_args()` に追加:
  ```python
      parser.add_argument(
          "--object_modes",
          nargs="+",
          choices=("objects", "noobj"),
          default=["objects", "noobj"],
          help="run each site once per mode: 'objects'=normal, 'noobj'=empty-world ablation "
          "(--drop_objects). Output/site label for noobj gets a '__noobj' suffix so both show "
          "up as separate sites (overlaid per metric in W&B, side-by-side rows in the report).",
      )
  ```
- `main()` のサイトループを **(site × mode)** の二重ループに変更。各実行で:
  - `label = site_name if mode == "objects" else f"{site_name}__noobj"`
  - `site_out_dir = args.out_root / label`
  - `cmd` は既存 + `mode == "noobj"` のとき `--drop_objects` を追加
  - `results[label] = {...}`(キーを複合ラベルに)
- `all_site_names` は `results`/`merged` のキーからそのまま拾えば両モードを含む(既存ロジックのまま)。
- HTML(`build_html_report`)/ W&B(`_log_to_wandb`)は **変更不要**。複合ラベルを「サイト名」として
  受け取り、既存の per-site 表示・オーバーレイがそのまま効く。
- 注意: `--only_sites` を使う場合のフィルタは元のサイト名(サフィックスなし)に対して効くよう、
  `discover_sites` の結果に対してフィルタ → その後にモード展開、の順にする。

### 5. `diffusion_planner/diffusion_planner/train.py` — `closed_loop_validate()` 学習時の両モード

**現状**: `closed_loop_sites_root` 指定時、`discover_sites()` の各拠点を `run_one(npz_root, out_dir, site_name)`
で 1 回ずつ実行し、`build_full_closed_loop_wandb_log(..., site=site_name)` のキーをマージ。
`run_one` 内の `RolloutParams(...)` は `drop_objects` を渡していない(=常に objects)。

**変更**: 各拠点を **objects と no-objects の 2 回** 実行し、複合ラベルでログする。

- `run_one` に引数 `drop_objects: bool` を追加し、`RolloutParams(...)` に
  `drop_objects=drop_objects` を渡す(§2 で追加したフィールド)。`site_label`/print も
  ラベルを反映(例 `[odaiba]` / `[odaiba__noobj]`)。
- サイトループ(現状 L216 付近 `for site_name, npz_root in sites.items():`)を
  **(site × mode)** に展開:
  ```python
  MODES = (("objects", False), ("noobj", True))  # (suffix_tag, drop_objects)
  for site_name, npz_root in sites.items():
      for tag, drop in MODES:
          label = site_name if tag == "objects" else f"{site_name}__noobj"
          site_out_dir = os.path.join(out_dir, label)
          site_log, summary = run_one(str(npz_root), site_out_dir, label, drop_objects=drop)
          ...  # 既存の grouped/full 分岐・log.update・episode_data.append をそのまま label ベースで
  ```
  - `site_summaries[label] = summary` / `episode_data.append((label, rows, site_out_dir))` も label ベースに。
  - 既存の combined table・`build_sites_aggregate_log` はそのまま(label が「サイト」として並ぶだけ)。
- 非 sites-root ブランチ(`args.closed_loop_npz_root` 単一)も同様に 2 モード回すか、単一のままにするかは
  実装者判断。基本は sites-root 経路が主なのでそちらを優先。
- **ゲート**: 学習時 2 モードでコスト倍。既定で両方実行(ユーザー承認済み)。将来オフにしたい場合に備え、
  `TrainConfig` に `closed_loop_object_modes: list[str] = ["objects","noobj"]` 相当の knob を足しても良い
  (任意・今回は必須ではない)。

### 6. `scenario_generation/wandb_closed_loop_workspace.py` — 案B ダッシュボード

**目的**: 1 ビューに loss(既存の自動 or 別セクション)+ closed-loop の両モードを並べ、指標の性質で出し分ける。

- **スコアキーを 2 群に分割**:
  - **比較群(objects/no-obj を重ねる)**: `mean_route_completion`, `total_curb_hits`, `total_snaps`
    (縁石=地図・stuck・走破率は両モードで意味がある)。
    → 従来どおり `metric_regex=rf"^closed_loop_scores/{metric}/.*$"`(両モードのラベルにマッチ=重なる)。
  - **objects 限定群**: `total_collision_events`(将来 near_miss を出すならそれも)。
    → **regex は使わず**、objects ラベルのみを明示列挙:
      `y=[f"closed_loop_scores/{metric}/{name}" for name in objects_site_names]`。
      no-objects ラベル(`__noobj`)は含めない(常に 0 の無意味な線を出さない)。
      ※ W&B の metric_regex は否定先読みの可搬性が不明なので、明示 y 指定が確実。
- **入力サイト名の扱い**: `--site_names` には **ベース名**(サフィックス無し, 例 `odaiba takanawa`)を渡す運用にする。
  ビルダ内で:
  - `objects_labels = list(site_names)`、`noobj_labels = [f"{n}__noobj" for n in site_names]`
  - 比較群パネル: regex で両方に自動マッチ(ラベル列挙不要)
  - objects 限定群パネル: `y` に `objects_labels` のみ
  - media(trajectory overlay / video)・episodes: `objects_labels + noobj_labels` の全ラベルでギャラリー/行を作る
  - `--include_noobj`(bool, default True)を足し、False の時は noobj を一切出さない(objects 専用ダッシュボードにも流用可)。
- セクション構成(案B):
  1. `Overview`(pinned): 比較群の overview(両モード) + objects 限定の overview(collision 系は objects のみ)
  2. `Scores: comparison (objects vs no-obj)`: 比較群、各パネルに 2 本重なる
  3. `Scores: objects-only (collision / near-miss)`: objects ラベルのみ
  4. `Trajectory Overlay` / `Video`: (site × mode) ギャラリー
  5. `Episodes`: 全ラベルの結合テーブル
  - loss は W&B が同 run に自動生成する(別セクション)。1 ビューなので loss と closed-loop が一画面に共存する。

---

## メトリクスへの影響(想定・要周知)

物体なしモードでは:
- **collision 回数 / near-miss → 常に 0**(相手がいない)。これは ablation の狙い通りで正常。
  「物体なしなのに collision>0」なら実装バグ(neighbors_live ゼロ化漏れ等)を疑う検算になる。
- **route_completion / curb_hits(縁石は地図なので残る) / gt_deviation / snaps → 引き続き有効**。

診断の読み方:
- あり=破綻 / なし=完走 → 交通反応が原因
- あり=破綻 / なし=破綻 → ルート/地図追従の問題

**overview 合算の注意**: `build_sites_aggregate_log` は全ラベル(objects + noobj)を「サイト」として
合算する。collision 系の overview は no-obj の 0 が混じって薄まるため、**collision/near_miss の
overview は objects ラベルのみで合算**するのが望ましい(§6 のダッシュボードでも objects 限定表示)。
実装上は `build_sites_aggregate_log` に objects/noobj を区別させるか、collision 系 overview を
objects サブセットで別途計算する。route_completion/curb/snaps は両モード合算で問題ない
(ただし「モード混在の平均」の意味は薄いので、ダッシュボードでは §6 のとおりモード別に重ねて見るのが主)。

## 表示(HTML / W&B)

複合ラベル(`odaiba` / `odaiba__noobj`)を既存の per-site 軸に載せるだけ:
- **W&B scores(比較群)**: `closed_loop_scores/route_completion/odaiba` と `.../odaiba__noobj` が同じ指標
  ラインプロットに重なり、物体あり/なしを直接比較できる(既存 metric_regex オーバーレイがそのまま)。
- **W&B scores(objects 限定群)**: collision/near_miss は objects ラベルのみ描画(§6)。
- **W&B media / episodes**: 複合ラベルごとにギャラリー・テーブル行が並ぶ。
- **HTML**: サマリ行・エピソードカードがモード別に並び、既存の拠点フィルタで絞れる。
  (HTML 側は複合ラベルを「サイト名」として渡すだけで既存機構がそのまま効く — build_html_report は変更不要。)
- **ダッシュボード**(`wandb_closed_loop_workspace.py`): `--site_names` には**ベース名**を渡す(§6 参照)。
  ビルダが内部で objects/noobj ラベルへ展開する。

## 検証計画

1. 単体: 小さい拠点(smoke の odaiba)で `--drop_objects` 付き single-site eval を実行し、
   `segments.jsonl` の `n_collision_events == 0` / `n_curb_hits` は非ゼロあり得る、を確認。
   動画・trajectory overlay が「他車なし」で描画されることを目視。
2. 両モード: `run_all_sites_closed_loop.py --object_modes objects noobj` で odaiba/takanawa を実行し、
   `sites_summary.json` に `odaiba` と `odaiba__noobj` が並ぶこと、HTML に両方の行/カードが出ること、
   route_completion 等が両モードで比較可能なことを確認。
3. W&B(スタンドアロン): `_log_to_wandb` 経由で両ラベルがログされ、ダッシュボードの比較群グラフに
   2 本(objects / noobj)が重なり、collision グラフは objects のみであることを確認。
4. 学習時: 既存の smoke 手順(`train_predictor.py` 直接実行, `--resume_model_path best_model.pth`,
   空 train/valid, `--train_epochs 110 --save_utd 10`)で、各 firing が **各拠点 objects と noobj の
   2 回** 走ること、同 run に `closed_loop_scores/*/odaiba` と `.../odaiba__noobj` が時系列でlog
   されることを確認。loss と両モードが 1 ビューに共存するのを目視。

## スコープ外(今回やらない)

- gpu バッチ mining パス(`run_segments_batched`)への drop_objects 対応。
- 新 converter / 物体なし npz データセットの生成。
- 「動的のみ除去(static は残す)」モード。今回は動的+静的の一括除去のみ。

（学習時の両モード実行は §3・§5 のとおり **今回スコープ内**。）
