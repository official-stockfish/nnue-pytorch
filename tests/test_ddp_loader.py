import unittest
from unittest.mock import MagicMock, patch

from data_loader.dataset import DevicePrefetcher
from data_loader.stream import _get_ddp_rank_and_world_size


class TestDevicePrefetcherDeferral(unittest.TestCase):
    """Verify that DevicePrefetcher defers iter() until the first batch."""

    def test_init_does_not_call_iter(self):
        mock_dataset = MagicMock()
        ds = DevicePrefetcher(mock_dataset, num_batches=10)

        mock_dataset.__iter__.assert_not_called()
        self.assertIsNone(ds.iter)

    def test_first_batch_triggers_iter(self):
        mock_dataset = MagicMock()
        mock_iter = MagicMock()
        mock_iter.__next__ = MagicMock(return_value="batch_data")
        mock_dataset.__iter__ = MagicMock(return_value=mock_iter)

        ds = DevicePrefetcher(mock_dataset, num_batches=10)
        mock_dataset.__iter__.assert_not_called()

        # The first batch should trigger iter() and prefetching
        item = ds.next_batch()
        mock_dataset.__iter__.assert_called_once()
        self.assertEqual(item, "batch_data")

    def test_iter_called_only_once(self):
        items = iter(["batch1", "batch2", "batch3"])
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=items)

        ds = DevicePrefetcher(mock_dataset, num_batches=3)
        self.assertEqual(list(ds), ["batch1", "batch2", "batch3"])

        mock_dataset.__iter__.assert_called_once()


class TestGetDDPRankAndWorldSize(unittest.TestCase):
    """Verify _get_ddp_rank_and_world_size uses torch.distributed when available."""

    @patch("data_loader.stream.os.environ", {})
    def test_no_dist_returns_defaults(self):
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            rank, world_size = _get_ddp_rank_and_world_size()

        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)

    @patch("data_loader.stream.os.environ", {})
    def test_dist_initialized_returns_dist_values(self):
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=3),
            patch("torch.distributed.get_world_size", return_value=4),
        ):
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
