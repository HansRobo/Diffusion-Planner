"""Tests for planner dataset filtering and batching, without the native bag reader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from diffusion_planner.data import PlannerDataset, VehicleParameters, collate_frames

TAXI = VehicleParameters(base_link_to_front=3.79, vehicle_length=4.77, vehicle_width=1.75)
BUS = VehicleParameters(base_link_to_front=5.71, vehicle_length=7.24, vehicle_width=2.43)


def write_index(path: Path) -> None:
    """Write a small single-split frame index covering two projects and two vehicles."""
    pq.write_table(
        pa.table(
            {
                "bag_path": ["/bags/a", "/bags/a", "/bags/b", "/bags/c"],
                "map_path": ["/m/a.osm", "/m/a.osm", "/m/b.osm", "/m/c.osm"],
                "frame_time_ns": [1, 2, 3, 4],
                "project_id": ["x2_dev", "x2_dev", "prd_jt", "x2_dev"],
                "split": ["train", "train", "train", "train"],
                "base_link_to_front": [3.79, 3.79, 5.71, 3.79],
                "vehicle_length": [4.77, 4.77, 7.24, 4.77],
                "vehicle_width": [1.75, 1.75, 2.43, 1.75],
            }
        ),
        path,
    )


class PlannerDatasetTest(unittest.TestCase):
    """The index is validated up front, so it must not need the native module."""

    def setUp(self) -> None:
        """Write a fresh index for each test."""
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "index.parquet"
        write_index(self.path)

    def tearDown(self) -> None:
        """Remove the temporary index."""
        self._dir.cleanup()

    def test_loads_every_row_of_the_index(self) -> None:
        """One dataset covers the whole index it is given."""
        self.assertEqual(len(PlannerDataset(self.path)), 4)

    def test_reads_the_vehicle_of_each_row(self) -> None:
        """Rows carry their own ego dimensions, so an index may mix vehicles."""
        dataset = PlannerDataset(self.path)
        self.assertEqual(dataset.vehicle(0), TAXI)
        self.assertEqual(dataset.vehicle(2), BUS)

    def test_rejects_an_index_without_vehicle_columns(self) -> None:
        """An index written before the dimensions were stamped in is named as such."""
        path = Path(self._dir.name) / "old.parquet"
        pq.write_table(
            pa.table(
                {
                    "bag_path": ["/bags/a"],
                    "map_path": ["/m/a.osm"],
                    "frame_time_ns": [1],
                }
            ),
            path,
        )
        with self.assertRaisesRegex(ValueError, "vehicle_length"):
            PlannerDataset(path)

    def test_rejects_a_missing_column(self) -> None:
        """A file without the required columns is rejected."""
        path = Path(self._dir.name) / "bare.parquet"
        pq.write_table(pa.table({"bag_path": ["/bags/a"]}), path)
        with self.assertRaises(ValueError):
            PlannerDataset(path)


class CollateFramesTest(unittest.TestCase):
    """Frames that fail to build are dropped instead of aborting the batch."""

    @staticmethod
    def frame(value: float) -> dict[str, torch.Tensor]:
        """Return one frame filled with a constant."""
        return {"ego_agent_past": torch.full((3, 6), value)}

    def test_stacks_frames(self) -> None:
        """Frames are stacked along a new batch dimension."""
        batch = collate_frames([self.frame(0.0), self.frame(1.0)])
        assert batch is not None
        self.assertEqual(batch["ego_agent_past"].shape, (2, 3, 6))

    def test_drops_failed_frames(self) -> None:
        """Failed frames shrink the batch instead of breaking it."""
        batch = collate_frames([self.frame(0.0), None])
        assert batch is not None
        self.assertEqual(batch["ego_agent_past"].shape, (1, 3, 6))

    def test_returns_none_when_every_frame_failed(self) -> None:
        """An all-failed batch collates to None."""
        self.assertIsNone(collate_frames([None, None]))


if __name__ == "__main__":
    unittest.main()
