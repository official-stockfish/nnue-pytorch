import threading
import queue

import torch
from torch.utils.data import Dataset

from . import stream
from .config import DataloaderSkipConfig, DataloaderDDPConfig


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
                self.num_workers, [self.filename], batch_size, cyclic, config, ddp_config
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
                self.feature_set, self.num_workers, self.filenames, cyclic, config, ddp_config
            )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors("cpu")
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
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.config = config
        self.ddp_config = ddp_config

    def __iter__(self):
        return SparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            config=self.config,
            ddp_config=self.ddp_config,
        )


class FixedNumBatchesDataset(Dataset):
    def __init__(self, dataset, num_batches):
        super().__init__()
        self.dataset = dataset
        self.iter = None  # Deferred to _start_prefetching
        self.num_batches = num_batches

        self._prefetch_queue = queue.Queue(maxsize=100)
        self._prefetch_thread = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()

    def _prefetch_worker(self):
        try:
            while not self._stop_prefetching.is_set():
                try:
                    item = next(self.iter)
                    self._prefetch_queue.put(item)
                except StopIteration:
                    self._prefetch_queue.put(None)
                    break
                except queue.Full:
                    continue
        except Exception as e:
            self._prefetch_queue.put(e)

    def _start_prefetching(self):
        with self._lock:
            if not self._prefetch_started:
                self.iter = iter(self.dataset)
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker, daemon=True
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

            return item

        except queue.Empty:
            raise RuntimeError("Prefetch timeout - no data available")

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
