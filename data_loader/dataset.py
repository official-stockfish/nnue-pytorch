"""Batches from the C++ loader (SparseBatchProvider, FenBatchProvider) and the device-side
prefetching that feeds them to training (DevicePrefetcher)."""

import queue
import threading
import time
import warnings
from dataclasses import dataclass

import torch

from . import stream
from .config import (
    DEFAULT_LS_KEYS,
    DataloaderDDPConfig,
    DataloaderHllConfig,
    DataloaderIOConfig,
    DataloaderSkipConfig,
)


class FenBatchProvider:
    """Batches of FEN strings from one binpack."""

    def __init__(
        self,
        filename: str,
        cyclic: bool,
        num_workers: int,
        batch_size: int,
        config: DataloaderSkipConfig | None = None,
        ddp_config: DataloaderDDPConfig | None = None,
        io_config: DataloaderIOConfig | None = None,
    ):
        self.stream = stream.create_fen_batch_stream(
            num_workers,
            [filename],
            batch_size,
            cyclic,
            config or DataloaderSkipConfig(),
            ddp_config,
            None,
            io_config or DataloaderIOConfig(),
        )

    def __iter__(self):
        return self

    def __next__(self) -> list[str]:
        v = stream.fetch_next_fen_batch(self.stream)
        if not v:
            raise StopIteration
        fens = v.contents.get_fens()
        stream.destroy_fen_batch(v)
        return fens

    def __del__(self):
        if getattr(self, "stream", None):
            stream.destroy_fen_batch_stream(self.stream)


class SparseBatchProvider:
    """Training batches from binpacks, as tensor tuples on ``device``.

    The C++ stream starts with the first batch. It owns a whole decoder pipeline, so consumers of
    the same data share one provider; ``next()`` is thread-safe.
    """

    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size: int,
        cyclic: bool = True,
        num_workers: int = 1,
        config: DataloaderSkipConfig | None = None,
        ddp_config: DataloaderDDPConfig | None = None,
        hll_config: DataloaderHllConfig | None = None,
        io_config: DataloaderIOConfig | None = None,
        use_pinned_memory: bool = False,
        device: str | torch.device = "cpu",
        ls_keys: str = DEFAULT_LS_KEYS,
    ):
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.config = config or DataloaderSkipConfig()
        self.ddp_config = ddp_config
        self.hll_config = hll_config or DataloaderHllConfig()
        self.io_config = io_config or DataloaderIOConfig()
        self.use_pinned_memory = use_pinned_memory
        self.device = device
        self.ls_keys = ls_keys
        # Time spent waiting for the C++ decoder workers, summed over all callers.
        self.native_fetch_s = 0.0
        self._stream = None
        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        return self._stream is not None

    def _get_stream(self):
        with self._lock:
            if self._stream is None:
                self._stream = stream.create_sparse_batch_stream(
                    self.feature_set.encode("utf-8"),
                    self.num_workers,
                    self.filenames,
                    self.batch_size,
                    self.cyclic,
                    self.config,
                    self.ddp_config,
                    self.hll_config,
                    self.io_config,
                    ls_keys=self.ls_keys,
                )
            return self._stream

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, ...]:
        handle = self._get_stream()
        t0 = time.perf_counter()
        v = stream.fetch_next_sparse_batch(handle)
        elapsed = time.perf_counter() - t0
        with self._lock:
            self.native_fetch_s += elapsed
        if not v:
            raise StopIteration
        try:
            # Copies everything off the native batch before it is freed.
            return v.contents.get_tensors(
                self.device, use_pinned_memory=self.use_pinned_memory
            )
        finally:
            stream.destroy_sparse_batch(v)

    def set_initial_hll(self, hll_bytes: bytes, total: int, preskip: int = 0) -> None:
        """Unique-position counting state to resume from; before the first batch only."""
        if self.started:
            raise RuntimeError("Cannot set initial HLL after the stream has started")
        self.hll_config.initial_hll = hll_bytes if hll_bytes else None
        self.hll_config.initial_total = total
        self.hll_config.initial_preskip = preskip

    def get_unique_stats(self) -> tuple[int, int, int]:
        """(preskip, total, unique); zeros before the first batch."""
        if not self.started:
            return 0, 0, 0
        return stream.get_unique_position_stats(self._stream)

    def get_hll_state(self) -> bytes:
        """The unique-position counting state, for checkpoints."""
        return stream.get_hll_state(self._stream) if self.started else b""

    def get_io_stats(self) -> list[dict]:
        """Per-file I/O statistics (read balancing)."""
        return stream.get_io_stats(self._stream) if self.started else []

    def __del__(self):
        if getattr(self, "_stream", None):
            stream.destroy_sparse_batch_stream(self._stream)


def _safe_put(stop_event, q, item):
    """Helper to ensure we don't hang on shutdown if queue is full."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=1.0)
            break
        except queue.Full:
            continue


class LoaderStats:
    """Cumulative timing counters of a DevicePrefetcher.

    Each side writes only its own fields (the prefetch thread the producer
    fields, the training thread the consumer fields), so no lock is needed;
    readers take a ``snapshot()`` and diff two of them over an interval.

    - ``consumer_wait_s``: time the training step spent blocked waiting for a
      batch. Time here is time the step could not run, i.e. the loader is
      what limits training speed.
    - ``producer_put_wait_s``: time the prefetch thread spent blocked on a
      full queue: the loader had a batch ready but training was not yet
      asking for it (spare loader capacity).
    - ``producer_fetch_s``: time the prefetch thread spent getting the next
      batch from the C++ loader (waiting for decoder workers included).
    - ``startup_wait_s``: the consumer wait for the very first batch, kept out
      of ``consumer_wait_s`` because it is pipeline startup, not throughput.
    """

    __slots__ = (
        "consumer_batches",
        "consumer_empty_gets",
        "consumer_fill_sum",
        "consumer_wait_s",
        "producer_batches",
        "producer_fetch_s",
        "producer_put_wait_s",
        "startup_wait_s",
    )

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, 0)

    def snapshot(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass
class _ReadyBatch:
    """A batch on the device and the event its copies and preparation complete on."""

    tensors: tuple
    ready_event: torch.cuda.Event


class DevicePrefetcher:
    """Yields ``num_batches`` batches per iteration (an epoch), drawn from ``source`` ahead of
    time by a prefetch thread into a queue. The thread starts with the first batch.

    For a CUDA ``device`` the thread moves batches to the device on its own CUDA stream, which the
    consumer's stream then waits on. ``prefetch_hook(batch)``, when given, runs there too for
    every batch (see NNUE.prefetch_hook). It must not modify the batch; if it raises, it is
    disabled with a warning.
    """

    def __init__(
        self,
        source,
        num_batches: int,
        queue_size_limit: int = 2,
        device: str | torch.device | None = None,
        prefetch_hook=None,
    ):
        self.source = source
        self.iter = None
        self.num_batches = num_batches
        self.device = torch.device(device) if device is not None else None
        self.prefetch_hook = prefetch_hook

        self._prefetch_queue = queue.Queue(maxsize=queue_size_limit)
        self._prefetch_thread = None
        self._prefetch_device = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()
        self._stats = LoaderStats()
        self._first_get_done = False

    def _resolve_prefetch_device(self):
        if self.device is None or self.device.type != "cuda":
            return None
        if self.device.index is not None:
            return self.device
        return torch.device("cuda", torch.cuda.current_device())

    @staticmethod
    def _prefetch_worker(
        stop_event,
        iterator,
        prefetch_queue,
        prefetch_device,
        prefetch_hook,
        stats,
    ):
        def put(item):
            t0 = time.perf_counter()
            _safe_put(stop_event, prefetch_queue, item)
            stats.producer_put_wait_s += time.perf_counter() - t0

        try:
            prefetch_stream = None
            if prefetch_device is not None:
                torch.cuda.set_device(prefetch_device)
                # A hook puts GPU work of its own on the prefetch stream. At the higher priority
                # (-1; the training stream has 0) that work is scheduled ahead of the step's
                # kernels, so the batch and what the hook prepared for it are ready when the
                # step asks for them.
                prefetch_stream = torch.cuda.Stream(
                    device=prefetch_device,
                    priority=-1 if prefetch_hook is not None else 0,
                )

            while not stop_event.is_set():
                try:
                    if prefetch_stream is not None:
                        with torch.cuda.stream(prefetch_stream):
                            t0 = time.perf_counter()
                            item = next(iterator)
                            stats.producer_fetch_s += time.perf_counter() - t0
                            if item is None:
                                _safe_put(stop_event, prefetch_queue, None)
                                break
                            item = tuple(
                                t.to(prefetch_device, non_blocking=True) for t in item
                            )
                            if prefetch_hook is not None:
                                try:
                                    prefetch_hook(item)
                                except Exception as e:  # noqa: BLE001 - the hook is optional work
                                    warnings.warn(
                                        "data loader: the prefetch hook failed and is disabled "
                                        f"for the rest of the run: {e!r}",
                                        RuntimeWarning,
                                        stacklevel=1,
                                    )
                                    prefetch_hook = None
                            ready_event = torch.cuda.Event()
                            ready_event.record(prefetch_stream)
                        item = _ReadyBatch(tensors=item, ready_event=ready_event)
                    else:
                        t0 = time.perf_counter()
                        item = next(iterator)
                        stats.producer_fetch_s += time.perf_counter() - t0
                        if item is None:
                            _safe_put(stop_event, prefetch_queue, None)
                            break
                    put(item)
                    stats.producer_batches += 1
                except StopIteration:
                    _safe_put(stop_event, prefetch_queue, None)
                    break
        except Exception as e:  # noqa: BLE001
            _safe_put(stop_event, prefetch_queue, e)

    def _start_prefetching(self):
        with self._lock:
            if self._prefetch_started:
                return
            # Resolve the concrete CUDA device on the consumer thread after the
            # training process has selected its rank-local device.
            self._prefetch_device = self._resolve_prefetch_device()
            self.iter = iter(self.source)
            self._prefetch_thread = threading.Thread(
                target=DevicePrefetcher._prefetch_worker,
                args=(
                    self._stop_prefetching,
                    self.iter,
                    self._prefetch_queue,
                    self._prefetch_device,
                    self.prefetch_hook,
                    self._stats,
                ),
                daemon=True,
            )
            self._prefetch_thread.start()
            self._prefetch_started = True

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            try:
                yield self.next_batch()
            except StopIteration:
                return

    def next_batch(self):
        """The next batch; StopIteration once a finite source is exhausted."""
        self._start_prefetching()

        stats = self._stats
        fill = self._prefetch_queue.qsize()
        t0 = time.perf_counter()
        try:
            item = self._prefetch_queue.get(timeout=300.0)
        except queue.Empty:
            raise RuntimeError("Prefetch timeout - no data available")
        waited = time.perf_counter() - t0
        if self._first_get_done:
            stats.consumer_wait_s += waited
            stats.consumer_batches += 1
            stats.consumer_fill_sum += fill
            stats.consumer_empty_gets += fill == 0
        else:
            stats.startup_wait_s += waited
            self._first_get_done = True

        if item is None:
            raise StopIteration("End of dataset reached")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, _ReadyBatch):
            current_stream = torch.cuda.current_stream(device=self._prefetch_device)
            current_stream.wait_event(item.ready_event)
            for t in item.tensors:
                t.record_stream(current_stream)
            return item.tensors
        return item

    @property
    def queue_capacity(self) -> int:
        return self._prefetch_queue.maxsize

    def get_loader_stats(self) -> dict:
        """Cumulative pipeline timing counters (see LoaderStats); diff two
        snapshots to get the numbers for an interval."""
        snap = self._stats.snapshot()
        # Part of producer_fetch_s spent inside the C++ loader (the rest is
        # host-to-device transport); only this part scales with workers.
        snap["native_fetch_s"] = getattr(self.iter, "native_fetch_s", 0.0)
        return snap

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if getattr(self, "_prefetch_thread", None):
            self._prefetch_thread.join(timeout=1.0)
