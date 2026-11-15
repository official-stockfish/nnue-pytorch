"""Tiled backward of the fused double feature transformer.

The tiled scatter (tiled_ft_backward) works on the gradients of the two
raw accumulators (white, black), each (B, L1). This module computes
them from grad_l0 and the forward's clamped values (the prologue: the
per-position part of the fused function's backward) before the scatter.
"""

from __future__ import annotations

import numpy as np
import torch

from . import tiled_ft_backward as _tb

USE_TILED_BACKWARD = _tb._HAS_CUPY and _tb.TILED_BACKWARD_ENABLED

_PROLOGUE_SRC = r"""
extern "C" __global__ void tiled_ft_prologue(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ clamped_out, const float max_act,
    float* __restrict__ grad_w, float* __restrict__ grad_b, const int batch_size)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)batch_size * $H) return;
    const int b = (int)(t / $H), i = (int)(t % $H);
    const float u = us[b], th = them[b];
    const size_t r = (size_t)b * $L1, c = (size_t)b * 4 * $H;
    const float cw0 = clamped_out[c + 0 * $H + i], cw1 = clamped_out[c + 1 * $H + i];
    const float cb0 = clamped_out[c + 2 * $H + i], cb1 = clamped_out[c + 3 * $H + i];
    const float g = grad_l0[r + i], gh = grad_l0[r + $H + i];
    const float dw0 = (cw0 == 0.0f || cw0 == max_act) ? 0.0f : g * cw1;
    const float dw1 = (cw1 == 0.0f || cw1 == max_act) ? 0.0f : g * cw0;
    const float db0 = (cb0 == 0.0f || cb0 == max_act) ? 0.0f : gh * cb1;
    const float db1 = (cb1 == 0.0f || cb1 == max_act) ? 0.0f : gh * cb0;
    grad_w[r + i] = u * dw0 + th * db0;
    grad_w[r + $H + i] = u * dw1 + th * db1;
    grad_b[r + i] = th * dw0 + u * db0;
    grad_b[r + $H + i] = th * dw1 + u * db1;
}
"""


def _kernel(name: str, src: str, l1_size: int):
    h = l1_size // 2
    return _tb._get_kernel(
        (name, l1_size), src.replace("$H", str(h)).replace("$L1", str(l1_size)), name
    )


def _launch(kernel, batch_size: int, l1_size: int, args):
    n = batch_size * (l1_size // 2)
    kernel(grid=((n + 255) // 256,), block=(256,), args=args)


def lookup_backward_prep(white_indices):
    return _tb.lookup_prep(white_indices) if USE_TILED_BACKWARD else None


@torch.compiler.disable(recursive=True)
def backward(
    us,
    them,
    white_indices,
    black_indices,
    grad_l0,
    clamped_out,
    num_inputs,
    max_ft_activation,
    l1_size,
    prep=None,
):
    """(grad_weight, grad_bias) of the fused double FT, or None when the tiled backward does not apply."""
    if not USE_TILED_BACKWARD or not _tb.applicable(l1_size):
        return None
    batch_size = white_indices.shape[0]
    grad_w = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
    grad_b = torch.empty_like(grad_w)
    with _tb._cupy_stream(us.device):
        _launch(
            _kernel("tiled_ft_prologue", _PROLOGUE_SRC, l1_size),
            batch_size,
            l1_size,
            (
                us.data_ptr(),
                them.data_ptr(),
                grad_l0.data_ptr(),
                clamped_out.data_ptr(),
                np.float32(max_ft_activation),
                grad_w.data_ptr(),
                grad_b.data_ptr(),
                np.int32(batch_size),
            ),
        )
    return _tb.tiled_backward(
        white_indices, black_indices, grad_w, grad_b, num_inputs, l1_size, prep=prep
    )
