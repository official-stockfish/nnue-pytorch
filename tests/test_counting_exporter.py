import unittest

import torch

import model as M
from model.config import NNUEConfig
from model.utils.serialize import CountingExporter, NNUEWriter


class TestCountingExporter(unittest.TestCase):
    """The CountingExporter must tally exactly the bytes the default writer writes."""

    def _roundtrip(self, features: str, l1: int, l2: int, l3: int, description=None):
        config = NNUEConfig()
        config.features = features
        config.model_config.L1 = l1
        config.model_config.L2 = l2
        config.model_config.L3 = l3

        torch.manual_seed(42)
        nnue = M.NNUE(config=config)

        # the default path: writer.buf works as before
        writer = NNUEWriter(nnue.model, description, verbose=False)
        actual = len(writer.buf)

        # the counting path
        counting = CountingExporter()
        NNUEWriter(nnue.model, description, verbose=False, exporter=counting)

        self.assertEqual(
            actual,
            counting.size,
            f"counting exporter ({counting.size}) != actual ({actual}) "
            f"for features={features} l1={l1} l2={l2} l3={l3}",
        )
        return counting.size

    def test_default_config(self):
        """The production config (Full_Threats+PP_3Wide+K32H3^, 1024/32/32)."""
        size = self._roundtrip("Full_Threats+PP_3Wide+K32H3^", 1024, 32, 32)
        self.assertGreater(size, 100_000_000)

    def test_small_config(self):
        """A small config still matches exactly."""
        size = self._roundtrip("Full_Threats+PP_3Wide+K32H3^", 128, 16, 16)
        self.assertGreater(size, 0)

    def test_custom_description(self):
        """A non-default description changes both equally."""
        self._roundtrip(
            "Full_Threats+PP_3Wide+K32H3^",
            128,
            16,
            16,
            description="A test description",
        )

    def test_buf_compatibility(self):
        """The default exporter's buf is accessible via the writer property."""
        config = NNUEConfig()
        config.features = "Full_Threats+PP_3Wide+K32H3^"
        config.model_config.L1 = 128
        config.model_config.L2 = 16
        config.model_config.L3 = 16

        torch.manual_seed(42)
        nnue = M.NNUE(config=config)
        writer = NNUEWriter(nnue.model, verbose=False)
        self.assertIsInstance(writer.buf, bytearray)
        self.assertGreater(len(writer.buf), 0)


if __name__ == "__main__":
    unittest.main()
