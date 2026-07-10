import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_ONE_HOT_DIM,
)
from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list, traffic_light_mask_list=None):
        self.data_list = openjson(data_list)
        self.traffic_light_mask_paths = (
            set(openjson(traffic_light_mask_list)) if traffic_light_mask_list else set()
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        data = np.load(path, allow_pickle=True)
        data = dict(data)  # npz to dict
        if path in self.traffic_light_mask_paths:
            self._mask_traffic_lights(data)
        for key, value in data.items():
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.unsignedinteger)
                and value.dtype != np.uint8
            ):
                data[key] = value.astype(np.int64, copy=False)
        return data

    @staticmethod
    def _mask_traffic_lights(data):
        for key in ("lanes", "route_lanes"):
            if key not in data:
                continue
            lanes = data[key].copy()
            valid = np.any(np.abs(lanes[..., :TRAFFIC_LIGHT]) > 0, axis=-1)
            lanes[..., TRAFFIC_LIGHT : TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM] = 0.0
            lanes[..., TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT] = valid.astype(lanes.dtype)
            data[key] = lanes
