import queue
import threading
import time
import warnings
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from . import stream
from ._transport import compact_transport_available
from .config import (
    DEFAULT_LS_KEYS,
    DataloaderDDPConfig,
    DataloaderHllConfig,
    DataloaderIOConfig,
    DataloaderSkipConfig,
)


def _recursive_pin(obj):
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory()
    elif isinstance(obj, dict):
        return {k: _recursive_pin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_recursive_pin(v) for v in obj)
    return obj


def _recursive_to_device(obj, device, non_blocking=False):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=non_blocking)
    elif isinstance(obj, dict):
        return {
            k: _recursive_to_device(v, device, non_blocking) for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_recursive_to_device(v, device, non_blocking) for v in obj)
    return obj


def _recursive_record_stream(obj, stream):
    if isinstance(obj, torch.Tensor):
        try:
            obj.record_stream(stream)
        except RuntimeError:
            pass
    elif isinstance(obj, dict):
        for value in obj.values():
            _recursive_record_stream(value, stream)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _recursive_record_stream(value, stream)


@dataclass
class _CudaPrefetchedItem:
    item: object
    ready_event: object


class FenBatchProvider:
    def __init__(
        self,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        config: DataloaderSkipConfig | None = None,
        ddp_config: DataloaderDDPConfig = None,
        io_config: DataloaderIOConfig | None = None,
    ):
        self.filename = filename
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        if config is None:
            config = DataloaderSkipConfig()
        self.config = config
        if io_config is None:
            io_config = DataloaderIOConfig()

        if batch_size:
            self.stream = stream.create_fen_batch_stream(
                self.num_workers,
                [self.filename],
                batch_size,
                cyclic,
                config,
                ddp_config,
                None,
                io_config,
            )
        else:
            # doesnt work yet
            assert False
            # self.stream = make_fen_batch_stream(
            #     self.num_workers,
            #     [self.filename],
            #     cyclic,
            #     config=config
            # )

    def __iter__(self):
        return self

    def __next__(self):
        v = stream.fetch_next_fen_batch(self.stream)

        if v:
            fens = v.contents.get_fens()
            stream.destroy_fen_batch(v)
            return fens
        else:
            raise StopIteration

    def __del__(self):
        stream.destroy_fen_batch_stream(self.stream)


class TrainingDataProvider:
    def __init__(
        self,
        feature_set: str,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filenames: list[str],
        cyclic,
        num_workers,
        batch_size=None,
        config: DataloaderSkipConfig | None = None,
        ddp_config: DataloaderDDPConfig = None,
        hll_config: DataloaderHllConfig | None = None,
        io_config: DataloaderIOConfig | None = None,
        use_pinned_memory=False,
        device="cpu",
        ls_keys: str = DEFAULT_LS_KEYS,
    ):
        self.feature_set = feature_set.encode("utf-8")
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filenames = filenames
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        if config is None:
            config = DataloaderSkipConfig()
        self.config = config
        if hll_config is None:
            hll_config = DataloaderHllConfig()
        self.hll_config = hll_config
        if io_config is None:
            io_config = DataloaderIOConfig()
        self.io_config = io_config
        self.use_pinned_memory = use_pinned_memory
        self.device = device
        self.native_fetch_s = 0.0

        if batch_size:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                batch_size,
                cyclic,
                config,
                ddp_config,
                hll_config,
                io_config,
                ls_keys=ls_keys,
                # Let the C++ workers build the packed transport layout when
                # batches will take that path, instead of this thread.
                prepack_indices=compact_transport_available(device, use_pinned_memory),
            )
        else:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                cyclic,
                config,
                ddp_config,
                hll_config,
                io_config,
            )

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        v = self.fetch_next(self.stream)
        # Time spent waiting for the C++ decoder workers, as opposed to moving
        # the batch to the device afterwards.
        self.native_fetch_s += time.perf_counter() - t0

        if v:
            tensors = v.contents.get_tensors(
                self.device, use_pinned_memory=self.use_pinned_memory
            )
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def get_unique_stats(self) -> tuple[int, int, int]:
        """Return (preskip, total, unique)."""
        return stream.get_unique_position_stats(self.stream)

    def get_hll_state(self) -> bytes:
        """Serialize the HLL state for checkpoint storage."""
        return stream.get_hll_state(self.stream)

    def get_io_stats(self) -> list[dict]:
        """Return per-file I/O statistics (read balancing)."""
        return stream.get_io_stats(self.stream)

    def __del__(self):
        self.destroy_stream(self.stream)


class SparseBatchProvider(TrainingDataProvider):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig | None = None,
        ddp_config: DataloaderDDPConfig = None,
        hll_config: DataloaderHllConfig | None = None,
        io_config: DataloaderIOConfig | None = None,
        use_pinned_memory=False,
        device="cpu",
        ls_keys: str = DEFAULT_LS_KEYS,
    ):
        super().__init__(
            feature_set,
            stream.create_sparse_batch_stream,
            stream.destroy_sparse_batch_stream,
            stream.fetch_next_sparse_batch,
            stream.destroy_sparse_batch,
            filenames,
            cyclic,
            num_workers,
            batch_size,
            config,
            ddp_config,
            hll_config,
            io_config,
            use_pinned_memory,
            device,
            ls_keys,
        )


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig | None = None,
        ddp_config: DataloaderDDPConfig = None,
        hll_config: DataloaderHllConfig | None = None,
        io_config: DataloaderIOConfig | None = None,
        use_pinned_memory=False,
        ls_keys: str = DEFAULT_LS_KEYS,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.ls_keys = ls_keys
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        if config is None:
            config = DataloaderSkipConfig()
        self.config = config
        self.ddp_config = ddp_config
        if hll_config is None:
            hll_config = DataloaderHllConfig()
        self.hll_config = hll_config
        if io_config is None:
            io_config = DataloaderIOConfig()
        self.io_config = io_config
        self.use_pinned_memory = use_pinned_memory
        self.device = "cpu"
        self._cached_iter = None
        self._iter_lock = threading.Lock()

    def _new_provider(self):
        return SparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            config=self.config,
            ddp_config=self.ddp_config,
            hll_config=self.hll_config,
            io_config=self.io_config,
            use_pinned_memory=self.use_pinned_memory,
            device=self.device,
            ls_keys=self.ls_keys,
        )

    def __iter__(self):
        # A cyclic dataset can serve any number of (concurrent) consumers from
        # a single C++ stream: FeaturedBatchStream::next() is thread-safe.
        # Reusing the stream matters because every provider owns a complete
        # decoder pipeline (per-worker shuffle buffers, chunk queue, batch
        # deque), so e.g. a validation loader drawing from the training data
        # would otherwise duplicate the whole pipeline at full memory cost for
        # the rest of the run. Non-cyclic (finite) streams are one-shot and
        # get a fresh provider per iteration.
        if not self.cyclic:
            return self._new_provider()
        with self._iter_lock:
            if self._cached_iter is None:
                self._cached_iter = self._new_provider()
            else:
                # Align the shared stream with the dataset's current transport
                # settings (a fresh provider would pick these up on creation).
                self._cached_iter.use_pinned_memory = self.use_pinned_memory
                self._cached_iter.device = self.device
            return self._cached_iter


def _safe_put(stop_event, q, item):
    """Helper to ensure we don't hang on shutdown if queue is full."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=1.0)
            break
        except queue.Full:
            continue


class LoaderStats:
    """Cumulative timing counters of a FixedNumBatchesDataset pipeline.

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


class FixedNumBatchesDataset(Dataset):
    def __init__(
        self,
        dataset,
        num_batches,
        pin_memory=False,
        queue_size_limit=None,
        device=None,
        prefetch_hook=None,
    ):
        """``prefetch_hook(batch)``, when given, runs on the prefetch thread for every batch copied
        to a CUDA ``device``, on the prefetch stream, before the batch is handed out: index-only work
        the model can do ahead of the step (see NNUE.prefetch_hook). It must not modify the batch;
        if it raises, a warning reports the error and the hook is no longer called."""
        super().__init__()
        self.dataset = dataset
        self.iter = None  # Deferred to _start_prefetching
        self.num_batches = num_batches
        self.device = torch.device(device) if device is not None else None
        self.prefetch_hook = prefetch_hook
        self.prefetch_to_device = self.device is not None and self.device.type == "cuda"
        # Async H2D copies require pinned host memory.
        self.pin_memory = pin_memory or self.prefetch_to_device
        if queue_size_limit is None:
            queue_size_limit = 10 if self.pin_memory else 100

        self._prefetch_queue = queue.Queue(maxsize=queue_size_limit)
        self._prefetch_thread = None
        self._prefetch_device = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()
        self._stats = LoaderStats()
        self._first_get_done = False

    def _resolve_prefetch_device(self):
        if not self.prefetch_to_device:
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
        stats=None,
    ):
        if stats is None:
            stats = LoaderStats()

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
                            item = _recursive_to_device(
                                item, prefetch_device, non_blocking=True
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
                        item = _CudaPrefetchedItem(item=item, ready_event=ready_event)
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
            if not self._prefetch_started:
                # Resolve the concrete CUDA device on the consumer thread after the
                # training process has selected its rank-local device.
                self._prefetch_device = self._resolve_prefetch_device()
                if hasattr(self.dataset, "use_pinned_memory"):
                    self.dataset.use_pinned_memory = self.pin_memory
                if (
                    hasattr(self.dataset, "device")
                    and self._prefetch_device is not None
                ):
                    self.dataset.device = self._prefetch_device
                self.iter = iter(self.dataset)
                self._prefetch_thread = threading.Thread(
                    target=FixedNumBatchesDataset._prefetch_worker,
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

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        self._start_prefetching()

        try:
            stats = self._stats
            fill = self._prefetch_queue.qsize()
            t0 = time.perf_counter()
            item = self._prefetch_queue.get(timeout=300.0)  # 300 second timeout
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
            elif isinstance(item, Exception):
                raise item
            elif isinstance(item, _CudaPrefetchedItem):
                prefetch_device = self._prefetch_device
                current_stream = torch.cuda.current_stream(device=prefetch_device)
                current_stream.wait_event(item.ready_event)
                _recursive_record_stream(item.item, current_stream)
                return item.item

            return item

        except queue.Empty:
            raise RuntimeError("Prefetch timeout - no data available")

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

    def get_unique_stats(self) -> tuple[int, int, int]:
        """Return (preskip, total, unique) from the underlying data provider.
        Available after prefetching has started (i.e. after the first batch)."""
        if self.iter is not None and hasattr(self.iter, "get_unique_stats"):
            return self.iter.get_unique_stats()
        return 0, 0, 0

    def get_hll_state(self) -> bytes:
        """Serialize the HLL state for checkpoint storage.
        Available after prefetching has started."""
        if self.iter is not None and hasattr(self.iter, "get_hll_state"):
            return self.iter.get_hll_state()
        return b""

    def get_io_stats(self) -> list[dict]:
        """Return per-file I/O statistics (read balancing).
        Available after prefetching has started (i.e. after the first batch)."""
        if self.iter is not None and hasattr(self.iter, "get_io_stats"):
            return self.iter.get_io_stats()
        return []

    def set_initial_hll(self, hll_bytes: bytes, total: int, preskip: int = 0) -> None:
        """Set the initial HLL state, total count, and preskip count for
        restart from checkpoint.  Must be called before the first
        __getitem__ (before prefetching starts)."""
        if self._prefetch_started:
            raise RuntimeError("Cannot set initial HLL after prefetching has started")
        # FixedNumBatchesDataset wraps an inner SparseBatchDataset (self.dataset).
        # Only the inner dataset has hll_config; propagate there.
        if hasattr(self.dataset, "hll_config"):
            self.dataset.hll_config.initial_hll = hll_bytes if hll_bytes else None
            self.dataset.hll_config.initial_total = total
            self.dataset.hll_config.initial_preskip = preskip

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
