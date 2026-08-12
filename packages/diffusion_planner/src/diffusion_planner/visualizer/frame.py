"""Adapters for model-ready diffusion planner frame dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schema import FRAME_KEY_NDIMS, REQUIRED_FRAME_KEYS


def _to_numpy(value: Any) -> NDArray[np.generic]:
    """Convert common array-like values without retaining gradient state."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class FrameData:
    """Validated, unbatched view of one diffusion planner frame."""

    arrays: Mapping[str, NDArray[np.generic]]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], batch_index: int = 0) -> FrameData:
        """Create a frame view, selecting an optional leading batch dimension.

        Args:
            data: Model input and optional training-label arrays.
            batch_index: Batch item selected from arrays that include a batch axis.

        Raises:
            KeyError: If a required visualization key is absent.
            ValueError: If a known array has an unexpected number of dimensions.
            IndexError: If ``batch_index`` does not exist.
        """
        missing = sorted(REQUIRED_FRAME_KEYS.difference(data))
        if missing:
            raise KeyError(f"Missing required frame keys: {', '.join(missing)}")

        arrays: dict[str, NDArray[np.generic]] = {}
        for key, value in data.items():
            array = _to_numpy(value)
            expected_ndim = FRAME_KEY_NDIMS.get(key)
            if expected_ndim is None:
                arrays[key] = array
                continue
            if array.ndim == expected_ndim + 1:
                if not 0 <= batch_index < array.shape[0]:
                    raise IndexError(
                        f"batch_index {batch_index} is out of range for {key} with shape {array.shape}"
                    )
                array = array[batch_index]
            elif array.ndim != expected_ndim:
                raise ValueError(
                    f"{key} must have rank {expected_ndim} (or {expected_ndim + 1} with batch), "
                    f"got shape {array.shape}"
                )
            arrays[key] = array
        return cls(arrays)

    def __getitem__(self, key: str) -> NDArray[np.generic]:
        return self.arrays[key]

    def get(self, key: str) -> NDArray[np.generic] | None:
        """Return an array if the frame contains it."""
        return self.arrays.get(key)

    @staticmethod
    def valid_rows(array: NDArray[np.generic]) -> NDArray[np.bool_]:
        """Return a mask for non-padding rows along the first axis."""
        if array.ndim < 2:
            raise ValueError(
                f"Expected an array with at least two dimensions, got {array.shape}"
            )
        axes = tuple(range(1, array.ndim))
        return np.any(array != 0, axis=axes)

    @staticmethod
    def valid_steps(array: NDArray[np.generic]) -> NDArray[np.bool_]:
        """Return a mask for valid pose steps using cos/sin as validity fields."""
        if array.shape[-1] < 4:
            raise ValueError(
                f"Expected pose data with at least four fields, got {array.shape}"
            )
        return np.square(array[..., 2]) + np.square(array[..., 3]) > 0.5
