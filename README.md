# Diffusion Planner

This is the project for the diffusion planner, which is used for the Autoware.

## Workspace layout

This repository is managed as a uv workspace with one shared environment and
lockfile.

```text
configs/                             # Hydra configuration tree, split by purpose
├── dataset/
└── train/
scripts/                             # command line entry points, split by purpose
├── dataset/
└── train/
packages/
├── diffusion_planner/
│   ├── pyproject.toml
│   ├── src/diffusion_planner/
│   └── tests/
└── diffusion_planner_dashboard/
    ├── pyproject.toml
    └── src/diffusion_planner_dashboard/
```

Synchronize all workspace packages from the repository root:

```bash
uv sync
```

## Frame dashboard

Start the Streamlit dashboard:

```bash
uv run diffusion-planner-dashboard
```

Configure either a `frames.h5` file or the generated frame-index Parquet from
the dashboard sidebar. Generated Parquet indexes contain absolute paths to their
H5 shards, so no separate H5 root configuration is needed.

The H5 shards and Parquet index are generated directly from rosbags:

```bash
source ros2_ws/install/setup.bash
uv run python scripts/dataset/create_h5_dataset.py \
  root=/data/rosbags_from_label \
  output_root=/data/diffusion_planner_h5 \
  split=train
```

The script writes one `frames.h5` per rosbag while preserving the source
directory hierarchy. Complete H5 shards can be reused when rebuilding an
interrupted Parquet index.

## Training

The entry point is driven by Hydra; its configuration lives in `configs/train/`.

```bash
source ros2_ws/install/setup.bash
uv run --package diffusion-planner python scripts/train/train.py \
  dataloader.dataset.parquet_path=/data/diffusion_planner_h5/indexes/train.parquet
```

The tensors already contain the vehicle shape generated from
`configs/dataset/vehicles/`, so training performs no rosbag or map preprocessing.

Benchmark training-style dataset loading:

```bash
uv run --package diffusion-planner python scripts/dataset/check_dataset.py \
  /data/diffusion_planner_h5/indexes/train.parquet \
  --jobs 32
```
