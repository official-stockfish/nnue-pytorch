import time
import unittest

import torch

from trainer.callbacks import _tensor_to_bytes


class TestTensorToBytes(unittest.TestCase):
    def test_correctness(self):
        t = torch.randint(0, 255, (1 << 20,), dtype=torch.uint8)
        self.assertEqual(_tensor_to_bytes(t), bytes(t.tolist()))

    def test_speed(self):
        t = torch.randint(0, 255, (1 << 20,), dtype=torch.uint8)
        t0 = time.perf_counter()
        for _ in range(3):
            _tensor_to_bytes(t)
        dt = (time.perf_counter() - t0) / 3
        # bytes(tensor) takes ~900 ms; allow generous CI headroom.
        self.assertLess(dt, 0.1, f"tensor->bytes took {dt*1e3:.1f} ms")


if __name__ == "__main__":
    unittest.main()
