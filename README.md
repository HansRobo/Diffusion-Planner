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

Build and source the ROS 2 workspace so that `diffusion_planner_data_tools` is
available, then start the Streamlit dashboard:

```bash
source ros2_ws/install/setup.bash
uv run diffusion-planner-dashboard
```

Configure the frame-index Parquet path, vehicle dimensions, and visualization
layers from the dashboard sidebar. The Parquet file can be created with
`scripts/dataset/create_dataset_index.py`, which is driven by Hydra:

```bash
source ros2_ws/install/setup.bash
uv run --package diffusion-planner python scripts/dataset/create_dataset_index.py \
  root=/data/rosbags_from_label output=/data/parquet/train.parquet split=train
```

## Training

The entry point is driven by Hydra; its configuration lives in `configs/train/`.

```bash
source ros2_ws/install/setup.bash
uv run --package diffusion-planner python scripts/train/train.py \
  dataloader.dataset.parquet_path=/data/parquet/train.parquet
```

The index carries the ego dimensions of each row, stamped in at scan time from
`configs/dataset/vehicles/`, so training needs no separate vehicle configuration.

Before training, verify that every frame of an index really loads:

```bash
uv run --package diffusion-planner python scripts/dataset/check_dataset.py \
  /data/parquet/train.parquet --jobs 32
```
