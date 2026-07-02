from torch.utils.data import Dataset

from diffusion_planner.utils.path_key import data_path_to_rel
from diffusion_planner.utils.path_list import (
    build_frame_index,
    load_frame,
    load_path_list,
)


class DiffusionPlannerData(Dataset):
    """Frame-level dataset over a path-list JSON.

    Supports both the legacy flat format (one sample per ``.npz``) and the packed manifest
    format (one sequence per ``.npz``, expanded to one sample per valid frame). The unit of
    indexing is a single frame, so ``len()`` is the number of trainable samples and
    ``__getitem__`` returns the same per-sample dict shape in both formats.

    ``frame_index`` -- the flat ``[(entry_idx, frame_idx), ...]`` list -- is the subsampling
    handle: slice it (e.g. ``ds.frame_index = ds.frame_index[::step]``) to thin the dataset.
    """

    def __init__(self, data_list):
        self.entries = load_path_list(data_list)
        self.frame_index = build_frame_index(self.entries)

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        entry_idx, frame_idx = self.frame_index[idx]
        return load_frame(self.entries[entry_idx]["path"], frame_idx)

    def path_for_index(self, idx):
        """Absolute ``.npz`` path backing sample ``idx`` (shared across frames of a sequence)."""
        entry_idx, _ = self.frame_index[idx]
        return self.entries[entry_idx]["path"]

    def rel_for_index(self, idx):
        """Unique relative output path for sample ``idx``.

        Mirrors the input hierarchy (see ``data_path_to_rel``); for packed sequences a
        ``_fNNNNNN`` frame suffix is appended so frames of one file never collide on save.
        """
        entry_idx, frame_idx = self.frame_index[idx]
        entry = self.entries[entry_idx]
        rel = data_path_to_rel(entry["path"])
        is_packed = entry["num_frames"] > 1 or entry["valid"] is not None
        if is_packed:
            rel = rel.with_name(f"{rel.name}_f{frame_idx:06d}")
        return rel
