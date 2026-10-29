import unittest

import torch

import model as M
from model.config import NNUEConfig
from model.utils.serialize import NNUEWriter, estimate_nnue_size


class TestNnueSizeEstimation(unittest.TestCase):
    """The config-only estimate must exactly match the actual serialized size."""

    def _roundtrip(self, features: str, l1: int, l2: int, l3: int, description=None):
        config = NNUEConfig()
        config.features = features
        config.model_config.L1 = l1
        config.model_config.L2 = l2
        config.model_config.L3 = l3

        estimated = estimate_nnue_size(config, description)

        torch.manual_seed(42)
        nnue = M.NNUE(config=config)
        writer = NNUEWriter(nnue.model, description, verbose=False)

        self.assertEqual(
            estimated,
            len(writer.buf),
            f"estimate {estimated} != actual {len(writer.buf)} "
            f"for features={features} l1={l1} l2={l2} l3={l3}",
        )
        return estimated

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

    def test_estimate_purely_config(self):
        """The estimate requires no model instantiation (no torch memory)."""
        config = NNUEConfig()
        config.features = "Full_Threats+PP_3Wide+K32H3^"
        size = estimate_nnue_size(config)
        self.assertGreater(size, 100_000_000)

    def test_max_nnue_size_cap(self):
        """A cap below the estimate must be detectable (train.py aborts)."""
        config = NNUEConfig()
        config.features = "Full_Threats+PP_3Wide+K32H3^"
        size = estimate_nnue_size(config)
        # the default cap (200 MiB) passes for the production config
        self.assertLess(size, 200 * 2**20)
        # but an artificially low cap (e.g. 50 MiB) would abort
        self.assertGreater(size, 50 * 2**20)


if __name__ == "__main__":
    unittest.main()
