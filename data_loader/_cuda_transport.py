"""Put feature IDs back into padded rows on the GPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_kernel(
    packed: tl.tensor,
    output: tl.tensor,
    batch: tl.tensor,
    active: tl.constexpr,
    nw: tl.tensor,
    nb: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    perspective = tl.program_id(1)
    offsets = perspective * (batch + 1)
    start = tl.load(packed + offsets + row)
    end = tl.load(packed + offsets + row + 1)
    ids = 2 * (batch + 1) + tl.where(perspective == 0, 0, nw)
    k = tl.arange(0, BLOCK)
    values = tl.load(
        packed + ids + start + k, (k < active) & (k < end - start), other=-1
    )
    destination = (perspective * batch + row).to(tl.int64) * active
    tl.store(output + destination + k, values, k < active)
    # Each perspective also copies one of the per-position metadata arrays.
    metadata = 2 * (batch + 1) + nw + nb + perspective * batch + row
    value = tl.load(packed + metadata)
    tl.store(
        output + 2 * batch.to(tl.int64) * active + perspective * batch + row, value
    )


@torch.compiler.disable
def unpack_sparse_indices(
    packed: torch.Tensor,
    batch_size: int,
    max_active: int,
    num_white: int,
    num_black: int,
) -> torch.Tensor:
    """Restore padded rows and copy batch metadata on the current CUDA stream."""
    if packed.dtype != torch.int32 or packed.ndim != 1 or not packed.is_contiguous():
        raise ValueError("packed must be a contiguous one-dimensional int32 tensor")
    if min(batch_size, max_active, num_white, num_black) < 0:
        raise ValueError("transport dimensions must be nonnegative")
    if packed.numel() != 4 * batch_size + 2 + num_white + num_black:
        raise ValueError("packed size does not match transport dimensions")
    output = torch.empty(
        batch_size * (2 * max_active + 2), dtype=torch.int32, device=packed.device
    )
    if batch_size:
        with torch.cuda.device(packed.device):
            _unpack_kernel[(batch_size, 2)](
                packed,
                output,
                batch_size,
                max_active,
                num_white,
                num_black,
                BLOCK=triton.next_power_of_2(max(1, max_active)),
            )
    return output
