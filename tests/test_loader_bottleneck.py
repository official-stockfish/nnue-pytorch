"""DataLoaderBottleneckMonitor: the loader's timing counters tell a slow loader from a slow step."""

import time
from types import SimpleNamespace

import torch

from data_loader.dataset import FixedNumBatchesDataset
from trainer.callbacks import DataLoaderBottleneckMonitor


class _TimedBatches:
    """Stands in for the native provider: ``native`` decides whether the
    delay counts as waiting for the C++ decoders or as transport."""

    def __init__(self, delay, native=True):
        self.delay = delay
        self.native = native
        self.native_fetch_s = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        time.sleep(self.delay)
        if self.native:
            self.native_fetch_s += time.perf_counter() - t0
        return (torch.zeros(4),)


def _fake_trainer():
    return SimpleNamespace(
        rank=0,
        world_size=1,
        current_epoch=0,
        global_step=0,
        device=torch.device("cpu"),
        callback_metrics={},
        _log_metrics=lambda *a, **k: None,
    )


def _run_epoch(produce_delay, step_delay, n=40, num_workers=1, native=True):
    ds = FixedNumBatchesDataset(
        _TimedBatches(produce_delay, native), n, queue_size_limit=2
    )
    ds[0]  # startup batch; excluded from the epoch numbers
    monitor = DataLoaderBottleneckMonitor(ds, num_workers=num_workers)
    monitor.cpus = 8
    trainer = _fake_trainer()
    monitor.on_train_epoch_start(trainer)
    for i in range(n):
        ds[i]
        time.sleep(step_delay)
    monitor.on_train_epoch_end(trainer)
    return monitor.last_report, trainer


def test_slow_loader_is_reported_as_bottleneck():
    report, trainer = _run_epoch(produce_delay=0.01, step_delay=0.001)
    assert report["wait"] > 0.5
    assert report["idle"] < 0.1
    assert report["verdict"].startswith("DATA-LOADER BOUND: raise --num-workers")
    assert trainer.callback_metrics["loader_wait_frac"] == report["wait"]


def test_slow_transport_is_not_blamed_on_workers():
    report, _ = _run_epoch(produce_delay=0.01, step_delay=0.001, native=False)
    assert report["wait"] > 0.5
    assert report["transport"] > report["native"]
    assert report["verdict"].startswith("TRANSPORT BOUND")


def test_slow_step_is_not_loader_bound():
    report, _ = _run_epoch(produce_delay=0.001, step_delay=0.01)
    assert report["wait"] < 0.05
    assert report["idle"] > 0.5
    assert report["verdict"].startswith("not loader bound")


def test_verdicts():
    m = DataLoaderBottleneckMonitor(None, num_workers=24)
    m.cpus = 24
    assert "all 24 CPUs" in m._verdict(wait=0.3, idle=0.0, native=0.9, transport=0.1)
    m.cpus = 32
    assert "now 24, 32 CPUs" in m._verdict(0.3, 0.0, native=0.9, transport=0.1)
    assert m._verdict(0.3, 0.0, native=0.1, transport=0.9).startswith("TRANSPORT")
    assert m._verdict(0.3, 0.4, native=0.5, transport=0.1).startswith("BURSTY")
    assert m._verdict(0.01, 0.02, native=0.9, transport=0.1).startswith(
        "NEAR CAPACITY: the loader"
    )
    assert m._verdict(0.01, 0.5, native=0.4, transport=0.1).startswith(
        "not loader bound"
    )
