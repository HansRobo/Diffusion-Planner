# diffusion_planner_data_tools

rosbag から diffusion planner のモデル入力を直接構築する Python バインディング。
前処理は推論時と同一の C++ コード
(`autoware::diffusion_planner::preprocess::create_input_data_map`) を共有する。

## 使い方

```python
import diffusion_planner_data_tools as dpt

# DataLoader worker ごとに 1 インスタンス (worker_init_fn で生成)
cache = dpt.FrameDataCache(reader_capacity=16, map_capacity=4)
spec = dpt.VehicleSpec(
    base_link_to_front=3.55,
    vehicle_length=4.65,
    vehicle_width=1.85)

frame = cache.create_frame_data(
    bag_path="path/to/bag_dir",
    map_path="path/to/lanelet2_map.osm",
    frame_time_ns=frame_time_ns,          # 基準時刻 (epoch ns)
    vehicle_spec=spec)                    # -> dict[str, np.ndarray] | None
```

モデル入力と学習ラベルを 1 つの dict で返す（単一バッチ・正規化なし）。
フレームが使用不能（ego 欠損、route なし、未来 8 秒をカバーしない等）の
場合は None。

入力キー: `ego_agent_past`, `neighbor_agents_past`, `lanes*`, `route_lanes*`,
`lane/route_traffic_light_past`, `polygons`, `stop_lines`, `road_borders`, `goal_pose`,
`ego_shape`, `turn_indicators`
（`sampled_trajectories` (推論専用乱数) と正規化は学習側で行う）

`ego_shape` のshapeは `(3,)`、要素順は
`[base_link_to_front, vehicle_length, vehicle_width]`。すべて単位はメートル。

地図線要素は種類ごとに独立した座標テンソルとして出力する。type one-hot は持たない。

| キー           | shape       | 内容                                     |
| -------------- | ----------- | ---------------------------------------- |
| `stop_lines`   | (30, 2, 2)  | 最大30本、左右端点2点の `[x, y]`         |
| `road_borders` | (30, 20, 2) | 最大30本、各20点に再サンプリングした座標 |

ラベルキー（未来グリッドは frame_time + (i+1)×0.1s, i = 0..79。座標系は入力と
同じ frame_time 時点の ego frame）:

| キー                     | shape        | 内容                                                                                                                                           |
| ------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `ego_agent_future`       | (80, 6)      | [x, y, cos, sin, velocity, yaw_rate]、補間                                                                                                     |
| `neighbor_agents_future` | (320, 80, 4) | [x, y, cos, sin]、ZOH。**行順は`neighbor_agents_past`と同一**。観測がないステップは全ゼロ=無効 (有効なら cos²+sin²=1 なので全ゼロにはならない) |
| `turn_indicators_future` | (80,)        | report値のZOH                                                                                                                                  |

## Parquet frame index

学習フレームの候補一覧（0.1s格子）を bag 1パスで作成する。メッセージ本体は
コピーせず、DataLoader は index の (bag_path, frame_time_ns) 行を
`FrameDataCache` に渡すだけ。

```python
import pyarrow as pa
import pyarrow.parquet as pq

index = dpt.scan_bag_index(bag_path)   # C++で1パススキャン (numpy配列のdict)
n = len(index["frame_time_ns"])
table = pa.table({"bag_path": [bag_path] * n, "map_path": [map_path] * n, **index})
pq.write_table(table, "frame_index.parquet")   # 書き込みはPython側
```

`scan_bag_index(bag_path, topics=config, frame_interval_s=0.1,
traffic_light_timeout_s=0.2)`。`frame_interval_s` はフレーム行の生成間隔 [s]
で、index のサンプル密度を変えるだけ（モデル内部の 0.1s グリッドには影響
しない）。

index には **valid なフレームのみ**が含まれる（過去3.6秒を ego/turn/objects
がカバー、ego が未来8秒をカバー、route がフレーム時刻までに存在、をすべて
満たすフレーム）。

| カラム                              | 内容                              |
| ----------------------------------- | --------------------------------- |
| `bag_path`, `map_path`              | データ取得先（絶対パス）          |
| `frame_time_ns`                     | フレーム基準時刻 (epoch ns)       |
| `ego_speed_mps`, `ego_yaw_rate_rps` | キュレーション用 (ZOH)            |
| `turn_indicator`                    | report 値 (ZOH)                   |
| `num_objects`                       | 追跡オブジェクト数 (ZOH)          |
| `traffic_signal_fresh`              | 信号メッセージが 0.2 秒以内に存在 |

停車フレームのダウンサンプルや右左折・信号ありフレームのオーバーサンプルは、
この統計カラムに対する pandas/pyarrow フィルタで bag を開かずに行える。
行は (bag, 時刻) 順で格納されるため、そのまま局所性のあるアクセス順になる。

## 設計

- **FrameDataCache**: bag reader と地図コンテキスト (`LaneSegmentContext`) の
  LRU キャッシュ。地図の変換は map_path ごとに 1 回だけ実行される
- **BagFrameReader**: 時刻順アクセスに最適化。連続する frame_time では bag を
  読み進めるだけ (~1 ms/frame)。過去への跳び戻りは seek + バッファ再構築 (数 ms)
- frame_time より新しい stamp のメッセージは入力に含まれない
  (リアルタイム推論と同じ因果性)
- 必要な過去データは `HISTORY_WINDOW_S` (= 3.6 s)。route は bag 全体から
  事前スキャンして保持
- 計算中は GIL を解放するため、マルチプロセス DataLoader と併用可能

## 性能の前提

数千 bag 規模では、DataLoader 側で
「bag 単位で worker に割当 → worker 内は時刻順に処理 → シャッフルバッファで
ランダム性を回復」という局所性のあるサンプリングを行うこと。
これにより reader cache は実質常にヒットする。
