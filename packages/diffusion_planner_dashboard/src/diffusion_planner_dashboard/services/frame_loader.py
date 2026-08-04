"""Bridge from frame-index rows to diffusion planner frame tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .frame_index import FrameIndexRow


@dataclass(frozen=True)
class VehicleParameters:
    """Primitive parameters accepted by ``diffusion_planner_data_tools.VehicleSpec``."""

    base_link_to_front: float = 5.71111
    vehicle_length: float = 7.2369
    vehicle_width: float = 2.42741


class FrameLoader:
    """Keep the native bag/map caches alive across dashboard frame selections."""

    def __init__(self, reader_capacity: int = 16, map_capacity: int = 4) -> None:
        try:
            import diffusion_planner_data_tools as dpt
        except ImportError as error:
            raise RuntimeError(
                "diffusion_planner_data_tools is unavailable. Build the ROS 2 workspace and "
                "source ros2_ws/install/setup.bash before starting the dashboard."
            ) from error
        self._dpt = dpt
        self._cache = dpt.FrameDataCache(
            reader_capacity=reader_capacity,
            map_capacity=map_capacity,
        )

    def load(self, row: FrameIndexRow, vehicle: VehicleParameters) -> dict[str, Any] | None:
        """Build one model-ready frame from a selected index row."""
        spec = self._dpt.VehicleSpec(
            base_link_to_front=vehicle.base_link_to_front,
            vehicle_length=vehicle.vehicle_length,
            vehicle_width=vehicle.vehicle_width,
        )
        return self._cache.create_frame_data(
            bag_path=row.bag_path,
            map_path=row.map_path,
            frame_time_ns=row.frame_time_ns,
            vehicle_spec=spec,
        )
