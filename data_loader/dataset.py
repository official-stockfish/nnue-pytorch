import threading
import queue
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from . import stream
from .config import DataloaderSkipConfig, DataloaderDDPConfig


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
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
    ):
        self.filename = filename
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config

        if batch_size:
            self.stream = stream.create_fen_batch_stream(
                self.num_workers,
                [self.filename],
                batch_size,
                cyclic,
                config,
                ddp_config,
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
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
        use_pinned_memory=False,
        device="cpu",
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
        self.config = config
        self.use_pinned_memory = use_pinned_memory
        self.device = device

        if batch_size:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                batch_size,
                cyclic,
                config,
                ddp_config,
            )
        else:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                cyclic,
                config,
                ddp_config,
            )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors(self.device, use_pinned_memory=self.use_pinned_memory)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

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
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
        use_pinned_memory=False,
        device="cpu",
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
            use_pinned_memory,
            device,
        )


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
        use_pinned_memory=False,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.config = config
        self.ddp_config = ddp_config
        self.use_pinned_memory = use_pinned_memory
        self.device = "cpu"

    def __iter__(self):
        return SparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            config=self.config,
            ddp_config=self.ddp_config,
            use_pinned_memory=self.use_pinned_memory,
            device=self.device,
        )


def _safe_put(stop_event, q, item):
    """Helper to ensure we don't hang on shutdown if queue is full."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=1.0)
            break
        except queue.Full:
            continue


class FixedNumBatchesDataset(Dataset):
    def __init__(
        self,
        dataset,
        num_batches,
        pin_memory=False,
        queue_size_limit=None,
        device=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.iter = None  # Deferred to _start_prefetching
        self.num_batches = num_batches
        self.device = torch.device(device) if device is not None else None
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
        prefetch_to_device,
    ):
        try:
            prefetch_stream = None
            if prefetch_device is not None:
                torch.cuda.set_device(prefetch_device)
                prefetch_stream = torch.cuda.Stream(device=prefetch_device)

            while not stop_event.is_set():
                try:
                    if prefetch_stream is not None:
                        with torch.cuda.stream(prefetch_stream):
                            item = next(iterator)
                            if item is None:
                                _safe_put(stop_event, prefetch_queue, None)
                                break
                            item = _recursive_to_device(
                                item, prefetch_device, non_blocking=True
                            )
                            ready_event = torch.cuda.Event()
                            ready_event.record(prefetch_stream)
                        item = _CudaPrefetchedItem(item=item, ready_event=ready_event)
                    else:
                        item = next(iterator)
                        if item is None:
                            _safe_put(stop_event, prefetch_queue, None)
                            break
                    _safe_put(stop_event, prefetch_queue, item)
                except StopIteration:
                    _safe_put(stop_event, prefetch_queue, None)
                    break
        except Exception as e:
            _safe_put(stop_event, prefetch_queue, e)

    def _start_prefetching(self):
        with self._lock:
            if not self._prefetch_started:
                # Resolve the concrete CUDA device on the consumer thread after the
                # training process has selected its rank-local device.
                self._prefetch_device = self._resolve_prefetch_device()
                if hasattr(self.dataset, "use_pinned_memory"):
                    self.dataset.use_pinned_memory = self.pin_memory
                if hasattr(self.dataset, "device") and self._prefetch_device is not None:
                    self.dataset.device = self._prefetch_device
                self.iter = iter(self.dataset)
                self._prefetch_thread = threading.Thread(
                    target=FixedNumBatchesDataset._prefetch_worker,
                    args=(
                        self._stop_prefetching,
                        self.iter,
                        self._prefetch_queue,
                        self._prefetch_device,
                        self.prefetch_to_device,
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
            item = self._prefetch_queue.get(timeout=300.0)  # 300 second timeout

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

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
