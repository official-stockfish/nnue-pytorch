import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.dataset import FixedNumBatchesDataset
from data_loader.stream import _get_ddp_rank_and_world_size


class TestFixedNumBatchesDeferral(unittest.TestCase):
    """Verify that FixedNumBatchesDataset defers iter() until first __getitem__."""

    def test_init_does_not_call_iter(self):
        mock_dataset = MagicMock()
        ds = FixedNumBatchesDataset(mock_dataset, num_batches=10)

        mock_dataset.__iter__.assert_not_called()
        self.assertIsNone(ds.iter)

    def test_getitem_triggers_iter(self):
        mock_dataset = MagicMock()
        mock_iter = MagicMock()
        mock_iter.__next__ = MagicMock(return_value="batch_data")
        mock_dataset.__iter__ = MagicMock(return_value=mock_iter)

        ds = FixedNumBatchesDataset(mock_dataset, num_batches=10)
        mock_dataset.__iter__.assert_not_called()

        # First __getitem__ should trigger iter() and prefetching
        item = ds[0]
        mock_dataset.__iter__.assert_called_once()
        self.assertEqual(item, "batch_data")

    def test_iter_called_only_once(self):
        items = iter(["batch1", "batch2", "batch3"])
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=items)

        ds = FixedNumBatchesDataset(mock_dataset, num_batches=3)
        _ = ds[0]
        _ = ds[1]
        _ = ds[2]

        mock_dataset.__iter__.assert_called_once()


class TestGetDDPRankAndWorldSize(unittest.TestCase):
    """Verify _get_ddp_rank_and_world_size uses torch.distributed when available."""

    @patch("data_loader.stream.os.environ", {})
    def test_no_dist_returns_defaults(self):
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=False):
            rank, world_size = _get_ddp_rank_and_world_size()

        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)

    @patch("data_loader.stream.os.environ", {})
    def test_dist_initialized_returns_dist_values(self):
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=3), \
             patch("torch.distributed.get_world_size", return_value=4):
            rank, world_size = _get_ddp_rank_and_world_size()

        self.assertEqual(rank, 3)
        self.assertEqual(world_size, 4)

    @patch("data_loader.stream.os.environ", {})
    def test_dist_not_available_returns_defaults(self):
        with patch("torch.distributed.is_available", return_value=False):
            rank, world_size = _get_ddp_rank_and_world_size()

        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)


if __name__ == "__main__":
    unittest.main()
