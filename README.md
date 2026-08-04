# Diffusion Planner

This is the project for the diffusion planner, which is used for the Autoware.

## Workspace layout

This repository is managed as a uv workspace with one shared environment and
lockfile.

```text
packages/
├── diffusion_planner/
│   ├── pyproject.toml
│   ├── scripts/
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
`packages/diffusion_planner/scripts/create_parquet.py`.
