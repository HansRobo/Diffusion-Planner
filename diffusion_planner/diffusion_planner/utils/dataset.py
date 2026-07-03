import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        self.data_list = openjson(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        for key, value in data.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.unsignedinteger) and value.dtype != np.uint8:
                data[key] = value.astype(np.int64, copy=False)
        return data
