"""Copy batch data from the CPU to the target device."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


@dataclass(frozen=True)
class CompactTransportPolicy:
    min_dense_bytes: int = 32 * 1024 * 1024
    min_compression_ratio: float = 2
    enabled: bool = True

    def should_pack(self, dense_elements: int, packed_elements: int) -> bool:
        return (
            self.enabled
            and dense_elements * 4 >= self.min_dense_bytes
            and packed_elements * self.min_compression_ratio < dense_elements
        )


def _pin_and_move(
    t: torch.Tensor,
    device: str | torch.device,
    use_pinned_memory: bool = False,
    dtype: torch.dtype | None = None,
    *,
    copy_mode: Literal["memcpy", "torch"] = "memcpy",
) -> torch.Tensor:
    if dtype is None:
        dtype = t.dtype

    # Must copy off SparseBatch-backed memory before it is freed
    if torch.cuda.is_available() and use_pinned_memory:
        if copy_mode not in ("memcpy", "torch"):
            raise ValueError(f"copy_mode must be memcpy or torch, got {copy_mode!r}")
        out = torch.empty(
            t.shape, dtype=dtype, layout=t.layout, device="cpu", pin_memory=True
        )
        if (
            copy_mode == "memcpy"
            and t.device.type == "cpu"
            and t.is_contiguous()
            and t.dtype == dtype
            and not t.is_conj()
            and not t.is_neg()
        ):
            # Release the GIL without dispatching additional PyTorch CPU workers.
            ctypes.memmove(out.data_ptr(), t.data_ptr(), t.numel() * t.element_size())
        else:
            out.copy_(t)
        if device == "cpu" or (
            isinstance(device, torch.device) and device.type == "cpu"
        ):
            return out
        return out.to(device=device, non_blocking=True)

    out = torch.empty(t.shape, dtype=dtype, layout=t.layout, device="cpu")
    out.copy_(t)
    if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
        return out
    return out.to(device=device)


def unpack_sparse_indices_host(
    packed: torch.Tensor,
    batch_size: int,
    max_active: int,
    num_white: int,
    num_black: int,
) -> torch.Tensor:
    """CPU counterpart of unpack_sparse_indices: the dense int32 block (white
    rows, black rows padded with -1, then layer-stack indices) from the packed
    layout. Returns a new tensor, independent of ``packed``."""
    out = torch.full((batch_size * (2 * max_active + 1),), -1, dtype=torch.int32)
    ids_start = 2 * (batch_size + 1)
    rows = torch.arange(batch_size)
    for perspective, (start, count) in enumerate(
        ((ids_start, num_white), (ids_start + num_white, num_black))
    ):
        offsets = packed[
            perspective * (batch_size + 1) : (perspective + 1) * (batch_size + 1)
        ].long()
        per_row = offsets[1:] - offsets[:-1]
        row = torch.repeat_interleave(rows, per_row)
        col = torch.arange(count) - torch.repeat_interleave(offsets[:-1], per_row)
        out[perspective * batch_size * max_active + row * max_active + col] = packed[
            start : start + count
        ]
    stacks_start = ids_start + num_white + num_black
    out[2 * batch_size * max_active :] = packed[
        stacks_start : stacks_start + batch_size
    ]
    return out


def compact_transport_available(
    device: str | torch.device, use_pinned_memory: bool
) -> bool:
    """Whether batches for this device can take the packed transport at all
    (the per-batch size check of CompactTransportPolicy comes on top)."""
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
    native_owned: bool,
    pack: Callable[[CArgObject, int, int], bool],
    policy: CompactTransportPolicy,
) -> torch.Tensor:
    """Copy feature IDs, skipping unused slots when that saves enough space.

    Packed int32 ABI: white/black offsets, white/black IDs, layer-stack
    indices. Each offsets array contains size + 1 elements.
    Native provenance is required because packing reads private C++ row counts.
    A batch the loader worker prepacked (white is NULL) always takes the packed
    path; packing it is a plain copy of the layout the worker built.
    """
    num_positions = batch.size
    max_features_per_position = batch.max_active_features
    dense_ids_per_perspective = num_positions * max_features_per_position
    layer_stack_elements = num_positions
    dense_elements = (
        dense_ids_per_perspective  # White IDs, including padding.
        + dense_ids_per_perspective  # Black IDs, including padding.
        + layer_stack_elements
    )

    # Each perspective needs a start offset per row and a final end offset.
    offsets_per_perspective = num_positions + 1
    packed_elements = (
        offsets_per_perspective  # White offsets.
        + offsets_per_perspective  # Black offsets.
        + batch.num_active_white_features
        + batch.num_active_black_features
        + layer_stack_elements
    )
    # The loader worker built only the packed layout: there are no dense rows.
    # pack() then just copies that layout (GIL released).
    prepacked = native_owned and not batch.white
    if (
        native_owned
        and compact_transport_available(device, use_pinned_memory)
        and (prepacked or policy.should_pack(dense_elements, packed_elements))
    ):
        assert unpack_sparse_indices is not None
        packed = torch.empty(packed_elements, dtype=torch.int32, pin_memory=True)
        if not pack(ctypes.byref(batch), packed.data_ptr(), packed_elements):
            raise RuntimeError("Native sparse batch packing failed")
        return unpack_sparse_indices(
            packed.to(device=device, non_blocking=True),
            num_positions,
            max_features_per_position,
            batch.num_active_white_features,
            batch.num_active_black_features,
        )
    if prepacked:  # No dense rows to fall back on.
        packed = torch.empty(packed_elements, dtype=torch.int32)
        if not pack(ctypes.byref(batch), packed.data_ptr(), packed_elements):
            raise RuntimeError("Native sparse batch packing failed")
        dense = unpack_sparse_indices_host(
            packed,
            num_positions,
            max_features_per_position,
            batch.num_active_white_features,
            batch.num_active_black_features,
        )
        return _pin_and_move(dense, device, use_pinned_memory)

    dense = torch.from_numpy(
        np.ctypeslib.as_array(batch.white, shape=(dense_elements,))
    )
    return _pin_and_move(dense, device, use_pinned_memory)
