"""Copy a native batch's arrays to the target device."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from _ctypes import CArgObject

    from ._native import SparseBatch


unpack_sparse_indices: Callable[[torch.Tensor, int, int, int, int], torch.Tensor] | None

try:
    from ._cuda_transport import unpack_sparse_indices
except ImportError:
    unpack_sparse_indices = None


def _is_cpu(device: str | torch.device) -> bool:
    return torch.device(device).type == "cpu"


def copy_to_device(
    t: torch.Tensor, device: str | torch.device, use_pinned_memory: bool = False
) -> torch.Tensor:
    """Copy ``t`` off native batch memory (freed with the batch) onto ``device``; asynchronously
    to CUDA with pinned memory."""
    if use_pinned_memory and torch.cuda.is_available():
        out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        # memmove releases the GIL and dispatches no PyTorch CPU workers.
        ctypes.memmove(out.data_ptr(), t.data_ptr(), t.numel() * t.element_size())
        return out if _is_cpu(device) else out.to(device, non_blocking=True)
    out = t.clone()
    return out if _is_cpu(device) else out.to(device)


def compact_transport_available(
    device: str | torch.device, use_pinned_memory: bool
) -> bool:
    """Whether batches for this device can cross as packed IDs and be unpacked there."""
    return (
        use_pinned_memory
        and torch.device(device).type == "cuda"
        and torch.version.hip is None
        and unpack_sparse_indices is not None
    )


def move_indices(
    batch: SparseBatch,
    device: str | torch.device,
    use_pinned_memory: bool,
    *,
    unpack: Callable[[CArgObject, int], None],
) -> torch.Tensor:
    """White rows, black rows (padded with -1) and layer-stack indices, as one int32 block on
    ``device``. With the compact transport only the packed IDs cross and a kernel unpacks them;
    otherwise ``unpack`` does it on the host."""
    size, max_active = batch.size, batch.max_active_features
    num_white = batch.num_active_white_features
    num_black = batch.num_active_black_features
    if compact_transport_available(device, use_pinned_memory):
        assert unpack_sparse_indices is not None
        packed = torch.from_numpy(
            np.ctypeslib.as_array(
                batch.packed, shape=(3 * size + 2 + num_white + num_black,)
            )
        )
        return unpack_sparse_indices(
            copy_to_device(packed, device, use_pinned_memory=True),
            size,
            max_active,
            num_white,
            num_black,
        )
    pin = use_pinned_memory and torch.cuda.is_available()
    # torch.full spreads the -1 fill and the page faults of the fresh buffer over its threads.
    dense = torch.full(
        (size * (2 * max_active + 1),), -1, dtype=torch.int32, pin_memory=pin
    )
    unpack(ctypes.byref(batch), dense.data_ptr())
    if _is_cpu(device):
        return dense
    return dense.to(device, non_blocking=pin)
