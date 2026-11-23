"""Several prefetch threads (DevicePrefetcher num_threads): every batch arrives once, the end
of the data is signalled only after the last batch, and the bottleneck report averages the threads."""

import gc
import threading
import time
import weakref
from types import SimpleNamespace

import pytest
import torch

from data_loader.dataset import DevicePrefetcher
from trainer.callbacks import DataLoaderBottleneckMonitor


class _Generated:
    """A generator-backed dataset: advancing it from two threads at once raises."""

    def __init__(self, n, delay=0.0):
        self.n = n
        self.delay = delay

    def __iter__(self):
        for i in range(self.n):
            time.sleep(self.delay)
            yield (torch.full((4,), float(i)),)


class _ConcurrentBatches:
    """Declares concurrent next() like the native provider; counts how many
    calls overlap and spends ``delay`` outside any lock (like the transport)."""

    concurrent_next = True

    def __init__(self, n, delay):
        self.n = n
        self.delay = delay
        self.native_fetch_s = 0.0
        self._next = 0
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._next >= self.n:
                raise StopIteration
            i = self._next
            self._next += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return (torch.full((4,), float(i)),)


def _drain(ds):
    out = []
    while True:
        try:
            out.append(ds.next_batch()[0][0].item())
        except StopIteration:
            return out


@pytest.mark.parametrize("num_threads", [1, 2, 4])
def test_every_batch_arrives_once_then_the_end(num_threads):
    ds = DevicePrefetcher(
        _Generated(37), num_batches=37, queue_size_limit=2, num_threads=num_threads
    )
    assert sorted(_drain(ds)) == [float(i) for i in range(37)]


def test_concurrent_iterator_is_advanced_in_parallel():
    batches = _ConcurrentBatches(24, delay=0.02)
    ds = DevicePrefetcher(batches, 24, queue_size_limit=8, num_threads=4)
    t0 = time.perf_counter()
    assert sorted(_drain(ds)) == [float(i) for i in range(24)]
    elapsed = time.perf_counter() - t0
    assert batches.max_active > 1
    assert elapsed < 24 * 0.02  # one thread would need at least this long


def test_num_threads_must_be_positive():
    with pytest.raises(ValueError):
        DevicePrefetcher(_Generated(1), 1, num_threads=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_device_batches_from_several_threads():
    seen = []
    lock = threading.Lock()

    def hook(batch):
        with lock:
            seen.append(batch[0].is_cuda)

    ds = DevicePrefetcher(
        _Generated(16),
        16,
        device="cuda",
        prefetch_hook=hook,
        queue_size_limit=2,
        num_threads=3,
    )
    assert sorted(_drain(ds)) == [float(i) for i in range(16)]
    assert seen == [True] * 16


def test_report_averages_producer_time_over_threads():
    """Four threads that each spend all their time in transport are fully busy, not 400% busy,
    and the verdict names the thread count."""
    n = 40
    ds = DevicePrefetcher(
        _ConcurrentBatches(10**6, delay=0.02), n, queue_size_limit=2, num_threads=4
    )
    ds.next_batch()
    monitor = DataLoaderBottleneckMonitor(ds, num_workers=8)
    monitor.cpus = 8
    trainer = SimpleNamespace(
        rank=0,
        world_size=1,
        current_epoch=0,
        global_step=0,
        device=torch.device("cpu"),
        callback_metrics={},
        _log_metrics=lambda *a, **k: None,
    )
    monitor.on_train_epoch_start(trainer)
    for _ in range(n):
        ds.next_batch()
    monitor.on_train_epoch_end(trainer)
    report = monitor.last_report
    assert 0.5 < report["transport"] <= 1.0
    assert report["verdict"].startswith("TRANSPORT BOUND")
    assert "--data-loader-threads (now 4)" in report["verdict"]


def test_prefetcher_is_collectable_while_its_threads_run():
    """The threads must not keep the prefetcher alive: dropping it stops them."""
    ds = DevicePrefetcher(
        _ConcurrentBatches(10**9, delay=0.001), 10, queue_size_limit=2, num_threads=3
    )
    ds.next_batch()
    threads = list(ds._prefetch_threads)
    ref = weakref.ref(ds)
    del ds
    gc.collect()
    assert ref() is None
    deadline = time.perf_counter() + 5
    while any(t.is_alive() for t in threads) and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert not any(t.is_alive() for t in threads)
