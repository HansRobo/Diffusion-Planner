# Diffusion Planner

This is the project for the diffusion planner, which is used for the Autoware.

## Frame dashboard

Build and source the ROS 2 workspace so that `diffusion_planner_data_tools` is
available, then start the Streamlit dashboard:

```bash
source ros2_ws/install/setup.bash
uv run diffusion-planner-dashboard
```

Configure the frame-index Parquet path, vehicle dimensions, and visualization
layers from the dashboard sidebar. The Parquet file can be created with
`scripts/create_parquet.py`.
