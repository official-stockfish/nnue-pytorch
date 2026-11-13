"""Row flags for a gradient whose unflagged rows are exactly +0.

The tiled FT backward allocates the FT weight gradient with ``torch.zeros`` and adds only to the rows of
the batch's active features; it records those rows in a byte mask and registers (gradient, mask) here.
The merged-weight backward registers each parameter's row slice of it, and the fused RangerLite update
looks a parameter's ``.grad`` up: for an unflagged row it uses +0 instead of reading the gradient (the
same value), and skips elements whose update is the identity.

A lookup succeeds for a tensor that aliases a registered one: same data pointer, shape and version
counter. (Under torch.compile the gradient reaches the parameter as a regenerated alias, a new
TensorImpl over the same memory, so neither object identity nor a weak reference works, and a strong
reference to the tensor would make autograd copy it.) Each entry holds the gradient's storage, so the
memory cannot be freed and reused by another tensor while registered; any in-place change through the
looked-up tensor bumps its version and makes the optimizer read the gradient again. Registrations are
dropped by the next backward's ``reset`` and by ``clear()`` at the end of the optimizer step.
"""
from __future__ import annotations

import torch

_entries: list = []  # (storage, data_ptr, shape, version, flags)


def register(grad: torch.Tensor, flags: torch.Tensor, reset: bool = False) -> None:
    """Record that the rows of 2D ``grad`` with ``flags[r] == 0`` are exactly +0. ``reset`` drops all
    earlier registrations (one per backward)."""
    global _entries
    if reset:
        _entries = []
    if grad.dim() == 2 and grad.dtype == torch.float32 and grad.is_contiguous() and flags.numel() == grad.shape[0]:
        _entries.append((grad.untyped_storage(), grad.data_ptr(), tuple(grad.shape), grad._version, flags))


def clear() -> None:
    global _entries
    _entries = []


def lookup(g: torch.Tensor):
    """The row flags of ``g`` if it aliases a registered gradient (unchanged since), else None."""
    if g.dtype != torch.float32 or not g.is_contiguous():
        return None
    for _storage, ptr, shape, version, flags in _entries:
        if g.data_ptr() == ptr and tuple(g.shape) == shape:
            return flags if g._version == version else None
    return None


_SSQ_SRC = r"""
// Sum of squares of the flagged rows of a (rows, cols) fp32 matrix; one block walks whole rows.
extern "C" __global__ void flagged_ssq(const float* __restrict__ g, const unsigned char* __restrict__ flags,
                                       long long rows, int cols, double* __restrict__ out) {
    float s = 0.0f;
    for (long long r = blockIdx.x; r < rows; r += gridDim.x) {
        if (!flags[r]) continue;   // the row is exactly +0
        const float* gr = g + r * cols;
        if ((cols & 3) == 0) {
            for (int c = threadIdx.x * 4; c < cols; c += blockDim.x * 4) {
                const float4 v = *(const float4*)(gr + c);
                s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            for (int c = threadIdx.x; c < cols; c += blockDim.x) s += gr[c] * gr[c];
        }
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
        if (threadIdx.x == 0) atomicAdd(out, (double)s);
    }
}
"""
_ssq_kernel = None


def total_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """L2 norm of all ``grads`` (like ``torch.nn.utils.get_total_norm(grads, 2.0)``), reading only the
    flagged rows of registered gradients. Same value up to fp32 summation order."""
    global _ssq_kernel
    flagged, rest = [], []
    for g in grads:
        f = lookup(g) if g.is_cuda else None
        (flagged if f is not None else rest).append((g, f))
    if not flagged:
        return torch.nn.utils.get_total_norm(grads, 2.0)
    import cupy as cp
    import numpy as np

    from model.modules.feature_transformer import tiled_ft_backward as _tb
    if _ssq_kernel is None:
        _ssq_kernel = cp.RawKernel(_SSQ_SRC, "flagged_ssq")
    dev = flagged[0][0].device
    acc = torch.zeros((), dtype=torch.float64, device=dev)
    with _tb._cupy_stream(dev):
        for g, f in flagged:
            rows, cols = g.shape
            _ssq_kernel((min(rows, 82 * 8),), (256,), (g.data_ptr(), f.data_ptr(), np.int64(rows), np.int32(cols), acc.data_ptr()))
    norms = [acc.to(torch.float32).sqrt()]
    if rest:
        norms += list(torch._foreach_norm([g for g, _ in rest], 2.0))
    return torch.linalg.vector_norm(torch.stack([n.to(dev) for n in norms]), 2.0)
