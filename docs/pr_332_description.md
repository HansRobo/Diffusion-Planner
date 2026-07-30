# PR: トークン重要度 / Attention分析ツールの追加

**base**: `tier4-main` ← **head**: `feat/token-importance-eval`

## 概要

Diffusion Planner Encoderのトークン構成（最大564トークン）が適切かを定量評価するため、以下を一括で実行・可視化する分析ツールを追加します。

- 入力クラス単位のablationによるFeature importance
- neighbors / lanes / line stringsの近い順Top-K評価
- neighbors / lanesの距離カットオフ評価
- Fusion Encoder内部のAttention分析
- クラス別の有効トークン数とスロット使用率
- FDE / ADEを含む数値表とグラフのHTMLレポート

モデル本体および学習処理への変更はありません。保存済みcheckpointと評価データを指定して使用するオフライン分析ツールです。

## 追加・変更ファイル

| ファイル | 内容 |
|---|---|
| `scripts/token_importance.py` | 入力ablation、近い順Top-K、距離カットオフを評価し、FDE / ADE / min FDE / min ADEとbaseline差分をTSVへ出力 |
| `scripts/attention_analysis.py` | クラス別Attention、selectivity、value-weighted share、層別share、距離ビン、旋回・直進比較、有効トークン統計をJSONへ出力 |
| `scripts/token_occupancy_scan.py` | NPZデータセットを単独走査する場合のトークンスロット占有率集計 |
| `run_token_analysis.sh` | PyTorch分析2本、日本語・英語Notebook実行、ポータブルHTML生成を一括実行 |
| `notebook/token_analysis.ipynb` | 実験方法、指標、全数値結果、グラフ、解釈方法を日本語で掲載 |
| `notebook/token_analysis_en.ipynb` | 日本語版と同じ解析内容を英語で掲載 |
| `notebook/generate_english_notebook.py` | 日本語版とコードセルを同期した英語版Notebookの生成スクリプト |
| `notebook/portable_html.j2` | 外部CSS・JavaScriptを必要としない単一HTML用テンプレート |
| `test_scripts/test_token_analysis_helpers.py` | neighbor有効判定、ONNX出力device、有効トークン統計の回帰テスト |

## 分析内容

### 1. Feature importance（入力ablation）

同じ評価シーンに対して、入力を変更しないbaselineと、特定入力をゼロへ置換した結果を比較します。

```text
importance = ablation時の誤差 - baselineの誤差
```

- 正の値: 入力を隠すと悪化したため、その入力情報を利用している可能性が高い
- 0付近: このデータと評価指標では明確な寄与が確認できない
- 負の値: 入力を隠すと改善した。ただし、入力が不要とは直ちに判断せず、標本誤差や分布外入力の影響も確認する

可変長入力（neighbors / maps）は、ゼロ化するとpadding maskによってAttention対象から除外されます。

goal pose / turn indicatorsは固定長トークンのため、トークン自体の完全除去ではなく、入力情報を一定値へ置換するablationとして解釈します。

### 2. 近い順Top-K

自車に近いK個だけを残して評価します。

```text
nbr_top:16   # 近いneighbor 16台だけを残す
lane_top:40  # 近いlane 40本だけを残す
ls_top:10    # 近いline string 10本だけを残す
```

Kを増やしたときにFDEとADEがbaseline付近へ安定する最小Kを、トークン上限削減の候補として確認します。

### 3. 距離カットオフ

指定距離以内の入力だけを残します。

```text
nbr_within:50
nbr_within:100
lane_within:50
lane_within:100
```

Top-Kは個数上限、距離カットオフは物理的な入力範囲を検討するための別の評価です。

### 4. Attention分析

Fusion EncoderのAttention重みをクラス別に集計します。

| 指標 | 意味 |
|---|---|
| count share | 全有効トークンのうち、そのクラスが占める割合 |
| ego-query share | 自車トークンが各クラスへ向けたAttentionの合計 |
| all-query share | 全有効queryから各クラスが受け取ったAttentionの平均 |
| selectivity | `Attention share / count share` |
| value-weighted share | Attentionにvalueベクトルの大きさを掛けた寄与の近似値 |

selectivityの目安:

- `1.0`: トークン数にほぼ比例
- `> 1.0`: トークン数の割合以上に選好
- `< 1.0`: トークン数の割合に対して参照が少ない

Attentionは「どこを参照したか」を示す値であり、最終予測への因果的な重要度そのものではありません。Feature importanceと組み合わせて判断します。

### 5. 有効トークン数と使用率

Attention分析時のpadding maskから、同じ評価サンプルにおける有効トークン数を集計し、Attention JSONへ保存します。

- 最大スロット数
- 平均
- p50
- p95
- p99
- 最大値
- 平均使用率
- p95 / p99を収容できる整数の容量候補

p95 / p99はスロット上限削減の初期候補として使用できますが、残りの高密度シーンを切り捨てて安全であることは意味しません。Top-K評価と閉ループ評価も必要です。

## 評価指標

| 指標 | 意味 |
|---|---|
| FDE | 最終予測地点と正解最終地点の距離 |
| ADE | 全予測時刻における位置誤差の平均 |
| min FDE / min ADE | 複数候補中で正解に最も近い候補の誤差 |
| Δ | 各指標のablation結果とbaselineの差 |

単一trajectory出力の場合、top指標とmin指標は同じ値になります。

FDEは最終到達点、ADEは途中経路を含む軌跡全体を見るため、レポートでは両方を表示します。

## 事前準備

モデルディレクトリには、少なくとも以下が必要です。

```text
best_model/
├── args.json
└── best_model.pth
```

Notebook実行とHTML変換に必要なパッケージが未導入の場合は、次を一度実行します。

```bash
UV_CACHE_DIR=/tmp/diffusion-planner-uv-cache \
uv pip install --python .venv/bin/python nbconvert ipykernel
```

## 推奨実行方法

リポジトリルートから`run_token_analysis.sh`を実行します。

```bash
MODEL_DIR=best_models/20260730/best_model \
DATADIR=/path/to/dataset \
N_SAMPLES=1024 \
BATCH_SIZE=64 \
DEVICE=cuda \
./run_token_analysis.sh
```

`DATADIR/path_list_valid.json`が存在する場合は、そのリストを使用します。

特定の評価リストを明示する場合:

```bash
MODEL_DIR=best_models/20260730/best_model \
DATADIR=/path/to/dataset \
VALID_LIST=/path/to/path_list_valid.json \
N_SAMPLES=1024 \
BATCH_SIZE=64 \
DEVICE=cuda \
./run_token_analysis.sh
```

`path_list_valid.json`が存在せず、`VALID_LIST`も指定されていない場合は、`DATADIR`以下の全NPZからリストを生成します。この場合、ディレクトリ構成によってはtrain / validationが混在する可能性があるため、正式評価では`VALID_LIST`の明示を推奨します。

## 実行パラメータ

| 環境変数 | 既定値 | 説明 |
|---|---:|---|
| `MODEL_DIR` | `best_models/20260730/best_model` | `args.json`と`best_model.pth`を含むディレクトリ |
| `DATADIR` | mini dataset | 評価データのルート |
| `VALID_LIST` | `$DATADIR/path_list_valid.json` | 評価対象NPZのリスト |
| `N_SAMPLES` | `128` | 評価する移動シーンの最大数 |
| `BATCH_SIZE` | `32` | 推論batch size |
| `DEVICE` | `cuda` | `cuda`または`cpu` |
| `MOVE_MIN_M` | `5.0` | 終端移動距離がこの値以上のシーンを評価 |
| `TURN_DEG` | `15.0` | 旋回シーン判定に使用する終端方位角 |
| `OUT_DIR` | dataset名から自動生成 | 分析結果とレポートの出力先 |
| `PYTHON_BIN` | `.venv/bin/python` | 実行に使用するPython |

小さい`N_SAMPLES`で動作確認した後、正式評価では512〜1024以上へ増やすことを推奨します。

## 生成物

既定では`MODEL_DIR`の隣に`eval_<dataset名>/`を作成します。

```text
eval_<dataset名>/
├── token_importance_n1024.tsv
├── token_importance_n1024.log
├── attention_n1024.json
├── attention_analysis_n1024.log
├── token_analysis_n1024.executed.ipynb
├── token_analysis_n1024.html
├── token_analysis_en_n1024.executed.ipynb
└── token_analysis_en_n1024.html
```

日本語版・英語版ともに、HTMLには画像とスタイルを埋め込んでいます。TSV / JSONやリポジトリがない環境でも、HTML単体で閲覧できます。

## Notebook / HTMLに掲載する内容

- 実行モデル、データリスト、サンプル数、batch size、device、各閾値
- 全24構成のFDE / ADE / min FDE / min ADEと全baseline差分
- クラス別のΔ FDE / Δ ADE
- FDE / ADEそれぞれのTop-K曲線
- 距離カットオフのFDE / ADE
- 全クラスのcount / ego-query / all-query / value-weighted share / selectivity
- 全Fusion層のego-query Attention
- 旋回・直進別route share
- lane / neighborの全距離ビン
- クラス別・全体の有効トークン数とスロット使用率
- 各結果の解釈方法と判断時のチェックリスト

## ONNXについて

`scripts/token_importance.py`単体ではONNXモデルも使用できます。

```bash
.venv/bin/python scripts/token_importance.py \
  --onnx /path/to/diffusion_planner.onnx \
  --args_json /path/to/args.json \
  --valid_set_list /path/to/path_list_valid.json \
  --n_samples 128 \
  --batch_size 8 \
  --device cpu \
  --out_tsv token_importance_onnx.tsv
```

ONNXではFusion Encoder内部のAttention重みへアクセスできないため、`attention_analysis.py`はPyTorch checkpoint専用です。統合ShellもPyTorch checkpointを使用します。

## 解釈上の注意

- 測定対象は「現在のcheckpointが、選択した評価データで何を利用しているか」です
- Feature importanceが小さくても、安全上重要な少数シーンへの効果が平均値に現れない場合があります
- Top-Kやゼロ埋めは学習時と異なる入力分布を作る可能性があります
- Attentionが大きいことと、最終予測への因果的寄与が大きいことは同義ではありません
- 使用率はスロットが埋まっている割合であり、トークンの有用性ではありません
- 実際の入力削減には、再学習、複数split、個別シーン確認、閉ループ安全評価が必要です

## 検証

- Python構文チェック
- Ruff lint / format
- 回帰テスト
- Notebook JSON検証
- 全Notebookコードセルの先頭からの順次実行
- 画像埋め込み済みHTMLへの変換
- 外部CSS / JavaScript / CDN参照がないことを確認
