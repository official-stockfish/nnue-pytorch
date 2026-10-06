"""Asynchronous checkpoint writing for SimpleTrainer.

Checkpoints are snapshotted into persistent host buffers (pinned when
the platform allows it) and serialized on a background thread.  Writes
are atomic (``.tmp`` + ``os.replace``).

The buffer pool holds at most two buffer sets, so the common case
(one write of ~1 s per epoch of ~10 s) never allocates: fresh
per-epoch snapshots would leave ~1 GiB of tensor garbage in reference
cycles that the generational GC reclaims far too slowly (measured:
rank-0 RSS ratchets from 6.5 to 18.7 GiB over 50 epochs).  If both
buffer sets are busy (a write still in flight and one queued), the
checkpoint is saved synchronously instead of stalling.

A not-yet-started write to a path is replaced when a newer checkpoint
for the same path is submitted, so ``last.ckpt`` always holds the
newest fully-written epoch; distinct paths (periodic checkpoints) are
never dropped.

If a background write fails, the error is recorded and subsequent
saves fall back to synchronous writes so training keeps producing
checkpoints; the error is re-raised by ``flush()``.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch

_MAX_POOL_SETS = 2


def cpu_snapshot(obj: Any) -> Any:
    """Recursively copy tensors to freshly allocated host memory."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        return t.cpu() if t.device.type != "cpu" else t.clone()
    if isinstance(obj, dict):
        return {k: cpu_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_snapshot(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_snapshot(v) for v in obj)
    return obj


def _empty_like_pinned(t: torch.Tensor) -> torch.Tensor:
    try:
        return torch.empty_like(t, device="cpu", pin_memory=True)
    except RuntimeError:
        return torch.empty_like(t, device="cpu")


class AsyncCheckpointWriter:
    """Single background thread writing checkpoints atomically."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        # Jobs (snapshot dict, path, buffer-set index), FIFO; a submit
        # for a path replaces a queued job for the same path.
        self._jobs: list[tuple[Any, str, int]] = []
        self._writing = False
        self._writing_path: str | None = None
        self._error: BaseException | None = None
        self._failed = False
        # Buffer sets: list of {key: host tensor}; free set = indices
        # owned by nobody (grows to at most _MAX_POOL_SETS).
        self._pool: list[dict[str, torch.Tensor]] = []
        self._free: set[int] = set()
        self._pinned = False
        self._thread = threading.Thread(
            target=self._run, name="ckpt-writer", daemon=True
        )
        self._thread.start()

    # -- producer side (trainer thread) ------------------------------

    def submit(self, ckpt: Any, path: str) -> None:
        """Snapshot ``ckpt`` to host buffers and queue the write."""
        if self._failed:
            # Background writing failed earlier: keep checkpoints
            # flowing with a synchronous save.
            _atomic_save(cpu_snapshot(ckpt), path)
            return
        with self._cv:
            if self._error is not None:
                err = self._error
                raise RuntimeError(f"checkpoint writer failed earlier: {err!r}")
            busy = False
            if not self._free:
                if len(self._pool) < _MAX_POOL_SETS:
                    idx = len(self._pool)
                    self._pool.append({})
                else:
                    # A write is in flight and one is queued (or the
                    # writer is mid-pickup): saves are ~10x shorter than
                    # epochs, so this is rare — write synchronously
                    # rather than stall.
                    busy = True
            else:
                idx = self._free.pop()
        if busy:
            # A stale queued or in-flight write for this path must not
            # clobber the newer synchronous write: wait for it to drain
            # and drop any queued job for the same path.
            with self._cv:
                self._cv.wait_for(
                    lambda: self._writing_path != path
                    and all(p != path for _, p, _ in self._jobs),
                    timeout=600.0,
                )
                for j_idx in range(len(self._jobs) - 1, -1, -1):
                    _c, p, i = self._jobs[j_idx]
                    if p == path:
                        self._free.add(i)
                        self._jobs.pop(j_idx)
            _atomic_save(cpu_snapshot(ckpt), path)
            return
        try:
            snapshot = self._snapshot(ckpt, idx)
        except BaseException:
            self._release(idx)
            raise
        if torch.cuda.is_initialized():
            # non_blocking D2H copies are stream-ordered; make sure the
            # buffers are complete before the writer thread reads them.
            torch.cuda.current_stream().synchronize()
        with self._cv:
            for j_idx, (c, p, i) in enumerate(self._jobs):
                if p == path:
                    self._free.add(i)
                    self._jobs.pop(j_idx)
                    break
            self._jobs.append((snapshot, path, idx))
            self._cv.notify_all()

    def _snapshot(self, obj: Any, idx: int, prefix: str = "") -> Any:
        """Mirror ``obj`` into buffer set ``idx``."""
        buffers = self._pool[idx]
        if isinstance(obj, torch.Tensor):
            t = obj.detach()
            buf = buffers.get(prefix)
            if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
                buf = _empty_like_pinned(t)
                buffers[prefix] = buf
                if buf.is_pinned():
                    self._pinned = True
            buf.copy_(t, non_blocking=self._pinned)
            return buf
        if isinstance(obj, dict):
            return {
                k: self._snapshot(v, idx, f"{prefix}.{k}")
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [
                self._snapshot(v, idx, f"{prefix}[{i}]")
                for i, v in enumerate(obj)
            ]
        if isinstance(obj, tuple):
            return tuple(
                self._snapshot(v, idx, f"{prefix}[{i}]")
                for i, v in enumerate(obj)
            )
        return obj

    def _release(self, idx: int) -> None:
        with self._cv:
            self._free.add(idx)
            self._cv.notify_all()

    def flush(self, timeout: float = 600.0) -> None:
        """Block until every submitted checkpoint is on disk."""
        with self._cv:
            done = self._cv.wait_for(
                lambda: (
                    not self._jobs
                    and not self._writing
                    and self._free == set(range(len(self._pool)))
                ),
                timeout,
            )
        if not done:
            raise RuntimeError(f"checkpoint writer flush timed out ({timeout}s)")
        if self._error is not None:
            err = self._error
            raise RuntimeError(f"checkpoint writer failed: {err!r}")

    # -- consumer side (writer thread) --------------------------------

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                ckpt, path, idx = self._jobs.pop(0)
                self._writing = True
                self._writing_path = path
            try:
                _atomic_save(ckpt, path)
            except BaseException as e:  # noqa: BLE001
                with self._cv:
                    self._error = e
                    self._failed = True
                    self._jobs = []
                    self._writing = False
                    self._writing_path = None
                    self._free = set(range(len(self._pool)))
                    self._cv.notify_all()
                print(
                    f"[AsyncCheckpointWriter] background write to {path} "
                    f"failed ({e!r}); falling back to synchronous saves",
                    flush=True,
                )
            else:
                with self._cv:
                    self._writing = False
                    self._writing_path = None
                    self._free.add(idx)
                    self._cv.notify_all()


def _atomic_save(ckpt: Any, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
