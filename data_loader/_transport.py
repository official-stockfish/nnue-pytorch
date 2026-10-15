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

    Packed int32 ABI: white/black offsets, white/black IDs, piece counts,
    queen buckets. Each offsets array contains size + 1 elements.
    Native provenance is required because packing reads private C++ row counts.
    """
    num_positions = batch.size
    max_features_per_position = batch.max_active_features
    dense_ids_per_perspective = num_positions * max_features_per_position
    piece_count_elements = num_positions
    queen_bucket_elements = num_positions
    dense_elements = (
        dense_ids_per_perspective  # White IDs, including padding.
        + dense_ids_per_perspective  # Black IDs, including padding.
        + piece_count_elements
        + queen_bucket_elements
    )

    # Each perspective needs a start offset per row and a final end offset.
    offsets_per_perspective = num_positions + 1
    packed_elements = (
        offsets_per_perspective  # White offsets.
        + offsets_per_perspective  # Black offsets.
        + batch.num_active_white_features
        + batch.num_active_black_features
        + piece_count_elements
        + queen_bucket_elements
    )
    if (
        native_owned
        and use_pinned_memory
        and torch.device(device).type == "cuda"
        and torch.version.hip is None
        and unpack_sparse_indices is not None
        and policy.should_pack(dense_elements, packed_elements)
    ):
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

    dense = torch.from_numpy(
        np.ctypeslib.as_array(batch.white, shape=(dense_elements,))
    )
    return _pin_and_move(dense, device, use_pinned_memory)
