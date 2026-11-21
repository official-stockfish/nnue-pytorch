"""Tiled segmented-reduction backward for the fused double feature transformer.

Replaces the per-position atomic scatter with tiles of sorted entries: one block per (tile, column
slice) holds the tile's gradient rows in shared memory, sums each row segment in registers and
issues one coalesced atomicAdd per (row, column) per tile. Same math as the plain kernel; fp32 sums
in a different order, like the atomics.

A batch flows through three modules:

1. tiled_ft_prep.py: the loader's prefetch thread orders the batch's lists into tiles and sorts
   every tile's entries (register_prep), keyed by the batch's index tensor.
2. FusedDoubleFtFunction.forward claims that preparation (claim_prep) and keeps it for its backward.
3. backward() below turns the output gradient into the gradients of the two accumulators and
   scatters them into the weight gradient, tile by tile.

The kernels are in tiled_ft_kernel.py.
"""

from __future__ import annotations

import numpy as np
import torch

from ...cupy_utils import HAS_CUPY, cupy_stream, shared_memory_limit
from . import tiled_ft_kernel as kernel
from .tiled_ft_prep import STATS, TiledPrep, prepare, warn_once


def enabled(l1_size: int, num_inputs: int, device: torch.device | None = None) -> bool:
    """Whether the tiled backward handles this layer, on ``device`` when one is given; otherwise
    the fused FT uses its direct scatter.

    It needs CUDA (the kernels assume 32-lane warps and CUB, which ROCm's compiler rejects), a
    width that the column slices divide, rows that fit the ROW_BITS of an entry key, and a GPU
    whose blocks can hold a tile's gradient slice in shared memory."""
    if not (HAS_CUPY and torch.version.hip is None):
        return False
    if (
        l1_size < kernel.COLUMNS_PER_BLOCK
        or l1_size % kernel.COLUMNS_PER_BLOCK
        or num_inputs > 1 << kernel.ROW_BITS
    ):
        return False
    return device is None or (
        device.type == "cuda"
        and shared_memory_limit(device) >= kernel.SCATTER_SHARED_BYTES
    )


@torch.compiler.disable(recursive=True)
def backward(
    us,
    them,
    white_indices,
    black_indices,
    grad_l0,
    clamped_out,
    num_inputs: int,
    max_ft_activation: float,
    prep: TiledPrep | None = None,
):
    """(grad_weight, grad_bias) of the fused double FT, whose white and black lists share weight
    and bias, from the gradient of its output and the clamped values its forward saved. ``prep``
    is the batch's preparation; without one it is built here."""
    dev = grad_l0.device
    batch_size, l1_size = grad_l0.shape
    if prep is None:
        STATS["fallback"] += 1
        warn_once(
            "no_prefetch",
            "tiled FT backward: no preparation was prefetched for this batch, so it is built "
            "in the backward (one host sync per step). Training batches get it from the "
            "loader's prefetch hook (NNUE.prefetch_hook).",
        )
        prep = prepare(white_indices, black_indices)
    if prep.n_lists != 2 * batch_size:
        raise RuntimeError("tiled backward: preparation does not match the batch")

    grad_white = torch.empty(batch_size, l1_size, dtype=torch.float32, device=dev)
    grad_black = torch.empty_like(grad_white)
    grad_weight = torch.zeros(num_inputs, l1_size, dtype=torch.float32, device=dev)
    grad_bias = torch.zeros(l1_size, dtype=torch.float32, device=dev)
    with cupy_stream(dev):
        kernel.prologue_kernel(l1_size)(
            grid=((batch_size * (l1_size // 2) + 255) // 256,),
            block=(256,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                grad_l0.data_ptr(),
                clamped_out.data_ptr(),
                np.float32(max_ft_activation),
                grad_white.data_ptr(),
                grad_black.data_ptr(),
                np.int32(batch_size),
            ),
        )
        kernel.scatter_kernel(l1_size)(
            grid=(prep.n_tiles, l1_size // kernel.COLUMNS_PER_BLOCK),
            block=(32 * kernel.WARPS_PER_BLOCK,),
            shared_mem=kernel.SCATTER_SHARED_BYTES,
            args=(
                grad_white.data_ptr(),
                grad_black.data_ptr(),
                prep.order.data_ptr(),
                prep.tile_start.data_ptr(),
                prep.entries.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(prep.n_lists),
                np.int32(batch_size),
            ),
        )
    return grad_weight, grad_bias
