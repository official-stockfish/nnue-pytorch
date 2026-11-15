"""Shared plumbing for the CuPy kernels of the tiled FT backward and the fused fake quantization."""

from __future__ import annotations

import threading

import torch

try:
    import cupy as cp

    HAS_CUPY = True
except (ImportError, OSError, RuntimeError):  # pragma: no cover - CuPy missing
    cp = None
    HAS_CUPY = False

_kernels: dict = {}
_kernel_lock = threading.Lock()


def compile_kernel(key, src: str, name: str, dynamic_shared: int = 0, options=()):
    """The compiled RawKernel ``name`` of ``src``, cached under ``key`` (unique across callers)."""
    with _kernel_lock:
        k = _kernels.get(key)
        if k is None:
            k = cp.RawKernel(src, name, options=tuple(options))
            k.compile()
            if dynamic_shared > 48 * 1024:
                # Raise the kernel's dynamic shared memory ceiling to the device's opt-in maximum:
                # a launch only allocates what it asks for, and a launch above the ceiling fails
                # with CUDA_ERROR_INVALID_VALUE.
                optin = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).shared_memory_per_block_optin
                k.max_dynamic_shared_size_bytes = max(dynamic_shared, optin)
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
