import os
import shutil
import tempfile
import threading
import time
import unittest

import torch

import trainer.checkpoint_writer as cw
from trainer.checkpoint_writer import (
    AsyncCheckpointWriter,
    _atomic_save,
    cpu_snapshot,
)

_ORIG_SAVE = torch.save
_ORIG_ATOMIC = cw._atomic_save


def _ckpt(tag, n=1000):
    return {
        "epoch": tag,
        "state_dict": {"w": torch.arange(n, dtype=torch.float32)},
        "blob": b"\x01" * 32,
        "nested": [{"x": torch.ones(4)}, (torch.zeros(2), 3)],
    }


class TestAsyncCheckpointWriter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt_writer_test_")
        self.path = os.path.join(self.dir, "last.ckpt")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip(self):
        w = AsyncCheckpointWriter()
        ck = _ckpt(7)
        w.submit(ck, self.path)
        w.flush()
        loaded = torch.load(self.path, weights_only=False)
        self.assertEqual(loaded["epoch"], 7)
        self.assertTrue(torch.equal(loaded["state_dict"]["w"], ck["state_dict"]["w"]))
        self.assertEqual(loaded["blob"], ck["blob"])
        self.assertTrue(torch.equal(loaded["nested"][0]["x"], torch.ones(4)))
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_pool_reuse_and_isolation(self):
        """Repeated submits reuse the same buffers without cross-talk."""
        w = AsyncCheckpointWriter()
        w.submit(_ckpt(1), self.path)
        w.flush()
        loaded = torch.load(self.path, weights_only=False)
        ref = loaded["state_dict"]["w"].clone()
        # Mutate the source between epochs; the pooled snapshot must be
        # independent of it.
        src = _ckpt(2)
        src["state_dict"]["w"] = ref * 2
        w.submit(src, self.path)
        w.flush()
        loaded = torch.load(self.path, weights_only=False)
        self.assertTrue(torch.equal(loaded["state_dict"]["w"], ref * 2))
        self.assertEqual(len(w._pool), 1, "pool must not grow for same-shaped ckpts")
        self.assertEqual(w._free, {0})

    def test_coalescing_same_path(self):
        writes = []
        entered = threading.Event()
        release = threading.Event()

        def gated_atomic(ckpt, path):
            writes.append(os.path.basename(path))
            entered.set()
            release.wait(5)
            _ORIG_ATOMIC(ckpt, path)

        cw._atomic_save = gated_atomic
        try:
            w = AsyncCheckpointWriter()
            other = os.path.join(self.dir, "epoch=19.ckpt")
            w.submit(_ckpt(19), other)
            self.assertTrue(entered.wait(5))  # writer inside the slow write
            # First last.ckpt submit queues; the second must replace it
            # (same path) and reuse the freed buffer set.
            w.submit(_ckpt(1), self.path)
            w.submit(_ckpt(2), self.path)
            release.set()
            w.flush()
        finally:
            cw._atomic_save = _ORIG_ATOMIC
        self.assertEqual(
            torch.load(self.path, weights_only=False)["epoch"], 2,
            "the newest checkpoint must win",
        )
        last_writes = [p for p in writes if p.startswith("last")]
        self.assertLessEqual(len(last_writes), 2, f"too many last.ckpt writes: {writes}")
        self.assertEqual(
            torch.load(other, weights_only=False)["epoch"], 19,
            "distinct paths must not be dropped by coalescing",
        )
        self.assertLessEqual(len(w._pool), 2, "pool must stay bounded")

    def test_distinct_paths_not_dropped(self):
        w = AsyncCheckpointWriter()
        p2 = os.path.join(self.dir, "epoch=19-step=20480.ckpt")
        w.submit(_ckpt(19), p2)
        w.submit(_ckpt(20), self.path)
        w.flush()
        self.assertEqual(torch.load(p2, weights_only=False)["epoch"], 19)
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 20)

    def test_atomic_on_failure(self):
        _atomic_save(_ckpt(1), self.path)
        with open(self.path, "rb") as f:
            good = f.read()

        def boom(obj, path):
            with open(path, "wb") as fh:
                fh.write(b"partial")
            raise RuntimeError("disk full")

        cw.torch.save = boom
        try:
            w = AsyncCheckpointWriter()
            w.submit(_ckpt(2), self.path)
            with self.assertRaises(RuntimeError):
                w.flush()
        finally:
            cw.torch.save = _ORIG_SAVE
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), good, "old file must survive")
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_fallback_to_sync_after_failure(self):
        def boom(obj, path):
            raise RuntimeError("fail")

        cw._atomic_save = boom
        try:
            w = AsyncCheckpointWriter()
            w.submit(_ckpt(1), self.path)
            time.sleep(0.2)
            cw._atomic_save = _ORIG_ATOMIC
            w.submit(_ckpt(2), self.path)  # must write synchronously
        finally:
            cw._atomic_save = _ORIG_ATOMIC
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 2)
        with self.assertRaises(RuntimeError):
            w.flush()  # recorded error still surfaces

    def test_cpu_snapshot_isolation(self):
        if torch.cuda.is_available():
            t = torch.ones(64, device="cuda")
            snap = cpu_snapshot({"t": t})
            t.add_(1.0)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(snap["t"], torch.ones(64)))
            self.assertEqual(snap["t"].device.type, "cpu")
        else:
            t = torch.ones(64)
            snap = cpu_snapshot({"t": t})
            t.add_(1.0)
            self.assertTrue(torch.equal(snap["t"], torch.ones(64)))


if __name__ == "__main__":
    unittest.main()
