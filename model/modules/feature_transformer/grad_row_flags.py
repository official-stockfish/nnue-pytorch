"""Row flags for the FT weight gradient of the tiled backward.

The tiled FT backward allocates the weight gradient with ``torch.zeros`` and adds only to the rows of
the batch's active features; it records those rows in a byte mask and registers (gradient, mask) here.
The fused merge + fake-quantization backward (fused_weight_quant), which receives that gradient next,
looks it up and sums the virtual weight's buckets over the flagged rows only.

A lookup succeeds for a tensor that aliases the registered gradient: same data pointer, shape and
version counter, with the registered storage still alive. (Under torch.compile the gradient reaches the
next backward as a regenerated alias, a new TensorImpl over the same memory, so object identity does
not work.) The registration holds only a weak reference to the storage, so it never keeps a gradient
alive, and once the storage is freed a new tensor at the same address cannot match it.
"""

from __future__ import annotations

import torch
from torch.multiprocessing.reductions import StorageWeakRef

_entry: tuple | None = None  # (storage weak ref, data_ptr, shape, version, flags)


def register(grad: torch.Tensor, flags: torch.Tensor) -> None:
    """Record that the rows of 2D ``grad`` with ``flags[r] == 0`` are exactly +0 (replaces the
    previous registration)."""
    global _entry
    _entry = None
    if (
        grad.dim() == 2
        and grad.dtype == torch.float32
        and grad.is_contiguous()
        and flags.numel() == grad.shape[0]
    ):
        _entry = (
            StorageWeakRef(grad.untyped_storage()),
            grad.data_ptr(),
            tuple(grad.shape),
            grad._version,
            flags,
        )


def lookup(g: torch.Tensor) -> torch.Tensor | None:
    """The row flags of ``g`` if it aliases the registered gradient (unchanged since), else None."""
    entry = _entry
    if entry is None or g.dtype != torch.float32 or not g.is_contiguous():
        return None
    storage, ptr, shape, version, flags = entry
    if (
        storage.expired()
        or g.data_ptr() != ptr
        or tuple(g.shape) != shape
        or g._version != version
    ):
        return None
    return flags
