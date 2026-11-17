"""Shared plumbing for CuPy kernels: a compile cache and launching on torch's current stream."""

from __future__ import annotations

import functools
import threading

import torch

try:
    import cupy as cp

    HAS_CUPY = True
except (ImportError, OSError, RuntimeError):  # pragma: no cover - CuPy missing
    cp = None
    HAS_CUPY = False

# Dynamic shared memory a block may use without the kernel opting in to more.
DEFAULT_SHARED_BYTES = 48 * 1024

_kernels: dict = {}
_kernel_lock = threading.Lock()


@functools.cache
def _shared_memory_limit(device_index: int) -> int:
    return cp.cuda.Device(device_index).attributes["MaxSharedMemoryPerBlockOptin"]


def shared_memory_limit(device: torch.device) -> int:
    """The most dynamic shared memory (bytes) a block can opt in to on CUDA ``device``."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _shared_memory_limit(index)


def compile_kernel(key, src: str, name: str, dynamic_shared: int = 0, options=()):
    """The compiled RawKernel ``name`` of ``src`` for the current CuPy device, cached under ``key``
    (unique across callers).

    ``dynamic_shared`` is the largest ``shared_mem`` (bytes) the kernel will be launched with.
    Above the 48 KB default the kernel opts in to that much; the caller keeps it within
    shared_memory_limit(), since a larger request fails with CUDA_ERROR_INVALID_VALUE.
    """
    key = (cp.cuda.device.get_device_id(), key)
    with _kernel_lock:
        k = _kernels.get(key)
        if k is None:
            k = cp.RawKernel(src, name, options=tuple(options))
            k.compile()
            if dynamic_shared > DEFAULT_SHARED_BYTES:
                k.max_dynamic_shared_size_bytes = dynamic_shared
            _kernels[key] = k
        return k


class cupy_stream:
    """Run CuPy kernel launches on the current torch stream of ``device``."""

    def __init__(self, device: torch.device):
        self.dev = cp.cuda.Device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.stream = cp.cuda.ExternalStream(
            torch.cuda.current_stream(device).cuda_stream
        )

    def __enter__(self):
        self.dev.__enter__()
        self.stream.__enter__()
        return self

    def __exit__(self, *exc):
        self.stream.__exit__(*exc)
        self.dev.__exit__(*exc)
