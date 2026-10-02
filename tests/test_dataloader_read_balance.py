import concurrent.futures
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import stream
from data_loader.config import (
    DataloaderDDPConfig,
    DataloaderIOConfig,
    DataloaderSkipConfig,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_test_binpack.cpp")
LIB_INCLUDE = os.path.join(REPO_ROOT, "data_loader", "cpp", "lib")

# Fixture layout: 8 files, sizes roughly 40:36:33:30:27:25:24:24 MiB. Every
# file is >= 24 MiB so that (a) the per-file balance-window allocation spans
# multiple 1 MiB chunks and (b) DDP rank partitioning (world_size=4) is
# meaningful at all.
FIXTURE_POSITIONS = [14_600_000, 13_100_000, 12_000_000, 10_900_000, 9_800_000, 9_100_000, 8_700_000, 8_700_000]
BATCH_SIZE = 16384
CONCURRENCY = 8

SKIP_ALL_OFF = DataloaderSkipConfig(
    filtered=False,
    wld_filtered=False,
    random_fen_skipping=0,
    early_fen_skipping=-1,
    soft_early_fen_skipping=0,
    simple_eval_skipping=-1,
)


def _fixture_dir():
    # Plain temp dir: the tests' assertions are all relative (shares, ratios,
    # spreads, counts), so tmpfs is not required.
    d = os.path.join(tempfile.gettempdir(), "nnue_loader_test_fixtures")
    os.makedirs(d, exist_ok=True)
    return d


def _build_generator():
    with open(GENERATOR_SRC, "rb") as f:
        tag = hashlib.sha1(f.read()).hexdigest()[:12]
    exe = os.path.join(tempfile.gettempdir(), f"make_test_binpack_{tag}")
    if not os.path.exists(exe):
        subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++20",
                "-march=native",
                f"-I{LIB_INCLUDE}",
                GENERATOR_SRC,
                "-o",
                exe,
            ],
            check=True,
        )
    return exe


def _fixtures():
    """Return the list of fixture files, generating them once if needed."""
    generator = _build_generator()
    fixture_dir = _fixture_dir()
    files = [os.path.join(fixture_dir, f"test_{i}.binpack") for i in range(len(FIXTURE_POSITIONS))]
    missing = [
        (path, npos, 1000 + i)
        for i, (path, npos) in enumerate(zip(files, FIXTURE_POSITIONS))
        if not os.path.exists(path) or os.path.getsize(path) < 1024 * 1024
    ]
    for path, _, _ in missing:
        if os.path.exists(path):
            os.remove(path)  # writer appends to existing files
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(missing)) as pool:
            for result in pool.map(
                lambda job: subprocess.run(
                    [generator, job[0], str(job[1]), str(job[2])], check=True
                ),
                missing,
            ):
                pass
    return files


def _make_stream(files, ddp=None, batch_size=BATCH_SIZE, cyclic=True):
    if ddp is None:
        ddp = DataloaderDDPConfig(0, 1)
    return stream.create_sparse_batch_stream(
        b"HalfKAv2_hm",
        CONCURRENCY,
        files,
        batch_size,
        cyclic,
        SKIP_ALL_OFF,
        ddp,
        None,
        DataloaderIOConfig(balance_window_mb=32),
    )


def _consume(st, seconds=None, batches=None):
    """Consume batches; returns (num_batches, per_file_stats)."""
    n = 0
    deadline = time.time() + (seconds or 0)
    while True:
        if seconds is not None and time.time() >= deadline:
            break
        if batches is not None and n >= batches:
            break
        b = stream.fetch_next_sparse_batch(st)
        if not b:
            break
        stream.destroy_sparse_batch(b)
        n += 1
    return n, stream.get_io_stats(st)


@unittest.skipUnless(
    os.path.exists(os.path.join(REPO_ROOT, "data_loader", "cpp", "build", "libtraining_data_loader.so")),
    "data loader library not built",
)
class TestReadBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = _fixtures()
        cls.sizes = [os.path.getsize(f) for f in cls.files]

    def test_size_proportional_ratios(self):
        """Healthy case: bytes read per file are proportional to file size."""
        st = _make_stream(self.files)
        try:
            _, stats = _consume(st, seconds=12)
        finally:
            stream.destroy_sparse_batch_stream(st)

        self.assertEqual(len(stats), len(self.files))
        total_size = sum(self.sizes)
        total_read = sum(s["bytes_read"] for s in stats)
        self.assertGreater(total_read, 100 * 1024 * 1024)
        for i, s in enumerate(stats):
            expected = total_read * self.sizes[i] / total_size
            tolerance = max(2.0 * 1024 * 1024, 0.08 * expected)
            self.assertLess(
                abs(s["bytes_read"] - expected),
                tolerance,
                f"file {i} ({self.sizes[i]} bytes): {s['bytes_read']} bytes read, "
                f"expected ~{expected:.0f}",
            )

    def test_slow_file_does_not_block_others(self):
        """One file with a very slow OST must not stall the stream."""
        st = _make_stream(self.files)
        try:
            control_batches, control_stats = _consume(st, seconds=12)
        finally:
            stream.destroy_sparse_batch_stream(st)

        os.environ["NNUE_LOADER_SIM_SLOW"] = "0:3000"
        try:
            slow = _make_stream(self.files)
            try:
                slow_batches, slow_stats = _consume(slow, seconds=12)
            finally:
                stream.destroy_sparse_batch_stream(slow)
        finally:
            del os.environ["NNUE_LOADER_SIM_SLOW"]

        # Throughput must not collapse.
        self.assertGreaterEqual(
            slow_batches,
            0.8 * control_batches,
            f"throughput collapsed: {slow_batches} vs {control_batches} batches",
        )
        # The slow file must have actually been slow (hook applied).
        self.assertGreaterEqual(slow_stats[0]["read_ns_max"], 2_000_000_000)
        # The slow file must have contributed almost nothing.
        self.assertLessEqual(slow_stats[0]["chunks_read"], 12)
        # Every healthy file must have kept reading (at least 70% of its
        # control run - they also absorb the slow file's share).
        for i in range(1, len(self.files)):
            self.assertGreaterEqual(
                slow_stats[i]["chunks_read"],
                0.7 * control_stats[i]["chunks_read"],
                f"healthy file {i} stalled",
            )

    def test_ddp_chunk_partition(self):
        """4 ranks with world_size=4 read equal shares of every file."""
        streams = []
        try:
            for rank in range(4):
                streams.append(
                    _make_stream(self.files, ddp=DataloaderDDPConfig(rank, 4))
                )
            per_rank = []
            for st in streams:
                _, stats = _consume(st, seconds=10)
                per_rank.append([s["chunks_read"] for s in stats])
        finally:
            for st in streams:
                stream.destroy_sparse_batch_stream(st)

        for i in range(len(self.files)):
            counts = [per_rank[r][i] for r in range(4)]
            mean = sum(counts) / 4
            if mean == 0:
                self.fail(f"file {i}: no chunks read by any rank")
            spread = max(counts) - min(counts)
            self.assertLessEqual(
                spread,
                max(3.0, 0.30 * mean),
                f"file {i}: uneven partition across ranks: {counts}",
            )

    def test_noncyclic_exhaustion(self):
        """Non-cyclic streams deliver all data and terminate cleanly.

        Regression test for a pre-existing deadlock: the shared-chunk-queue
        stop predicate called is_empty() while take() held the queue mutex,
        self-deadlocking workers racing to drain the final chunks after the
        readers finished.
        """
        generator = _build_generator()
        fixture_dir = _fixture_dir()
        files = []
        for i in range(8):
            path = os.path.join(fixture_dir, f"tiny_{i}.binpack")
            if not os.path.exists(path) or os.path.getsize(path) < 1024:
                if os.path.exists(path):
                    os.remove(path)  # writer appends to existing files
                subprocess.run([generator, path, "20000", str(200 + i)], check=True)
            files.append(path)

        for _ in range(3):
            st = stream.create_sparse_batch_stream(
                b"HalfKAv2_hm",
                CONCURRENCY,
                files,
                4096,
                False,  # cyclic=False: stream must end
                SKIP_ALL_OFF,
                DataloaderDDPConfig(0, 1),
                None,
                DataloaderIOConfig(balance_window_mb=32),
            )
            result = {}

            def consume(st=st, result=result):
                positions = 0
                while True:
                    b = stream.fetch_next_sparse_batch(st)
                    if not b:
                        break
                    positions += b.contents.size
                    stream.destroy_sparse_batch(b)
                result["positions"] = positions

            t = threading.Thread(target=consume, daemon=True)
            t.start()
            t.join(timeout=60.0)
            self.assertFalse(t.is_alive(), "non-cyclic stream did not terminate (deadlock?)")
            stream.destroy_sparse_batch_stream(st)
            # 8 files x 20000 positions, no skipping.
            self.assertEqual(result.get("positions"), 8 * 20000)

    def test_destroy_with_slow_read_in_flight(self):
        """Stream destruction must not hang on an in-flight (slow) read."""
        os.environ["NNUE_LOADER_SIM_SLOW"] = "0:400"
        try:
            st = _make_stream(self.files)
            _consume(st, batches=2)
            t0 = time.time()
            stream.destroy_sparse_batch_stream(st)
            elapsed = time.time() - t0
        finally:
            del os.environ["NNUE_LOADER_SIM_SLOW"]
        self.assertLess(elapsed, 15.0, f"destroy took {elapsed:.1f}s")


if __name__ == "__main__":
    unittest.main()
