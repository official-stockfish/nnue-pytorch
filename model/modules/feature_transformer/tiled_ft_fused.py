"""Tiled forward / backward of the fused double feature transformer.

The tiled kernels work on the two raw accumulators (white, black),
each (B, L1) = sum of the active weight rows + bias. This module adds
the per-position parts of the fused function around them: the epilogue
(perspective mix, clamp, pairwise product) after the forward, and the
prologue (accumulator gradients) before the backward's scatter.
"""

from __future__ import annotations

import numpy as np
import torch

from . import tiled_ft_backward as _tb
from . import tiled_ft_forward as _tf

USE_TILED_FORWARD = _tf._HAS_CUPY and _tf.TILED_FORWARD_ENABLED
USE_TILED_BACKWARD = _tb._HAS_CUPY and _tb.TILED_BACKWARD_ENABLED

_EPILOGUE_SRC = r"""
__device__ __forceinline__ void epi_elem(const int b, const int i,
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ acc_w, const float* __restrict__ acc_b, const float max_act,
    float* __restrict__ l0_out, float* __restrict__ clamped_out,
    const int act_q, const float act_s, const float act_eps, const float act_inv, const float act_c)
{
    const float u = us[b], th = them[b];
    const size_t r = (size_t)b * $L1;
    const float w0 = acc_w[r + i], w1 = acc_w[r + i + $H], b0 = acc_b[r + i], b1 = acc_b[r + i + $H];
    float l0_w0 = u * w0 + th * b0;
    float l0_w1 = u * w1 + th * b1;
    float l0_b0 = u * b0 + th * w0;
    float l0_b1 = u * b1 + th * w1;
    if (l0_w0 < 0.0f) l0_w0 = 0.0f; else if (l0_w0 > max_act) l0_w0 = max_act;
    if (l0_w1 < 0.0f) l0_w1 = 0.0f; else if (l0_w1 > max_act) l0_w1 = max_act;
    if (l0_b0 < 0.0f) l0_b0 = 0.0f; else if (l0_b0 > max_act) l0_b0 = max_act;
    if (l0_b1 < 0.0f) l0_b1 = 0.0f; else if (l0_b1 > max_act) l0_b1 = max_act;
    float o0 = l0_w0 * l0_w1, o1 = l0_b0 * l0_b1;
    if (act_q) {   // fake_quantize_ft_act * correction, same roundings as fused_weight_quant.act_quant
        o0 = __fmul_rn(__fmul_rn(floorf(__fadd_rn(__fmul_rn(o0, act_s), act_eps)), act_inv), act_c);
        o1 = __fmul_rn(__fmul_rn(floorf(__fadd_rn(__fmul_rn(o1, act_s), act_eps)), act_inv), act_c);
    }
    l0_out[r + i] = o0;
    l0_out[r + $H + i] = o1;
    if (clamped_out == nullptr) return;
    const size_t c = (size_t)b * 4 * $H;
    clamped_out[c + 0 * $H + i] = l0_w0;
    clamped_out[c + 1 * $H + i] = l0_w1;
    clamped_out[c + 2 * $H + i] = l0_b0;
    clamped_out[c + 3 * $H + i] = l0_b1;
}

extern "C" __global__ void tiled_ft_epilogue(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ acc_w, const float* __restrict__ acc_b, const float max_act,
    float* __restrict__ l0_out, float* __restrict__ clamped_out, const int batch_size,
    const int act_q, const float act_s, const float act_eps, const float act_inv, const float act_c)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)batch_size * $H) return;
    epi_elem((int)(t / $H), (int)(t % $H), us, them, acc_w, acc_b, max_act, l0_out, clamped_out, act_q, act_s, act_eps,
             act_inv, act_c);
}

// After the gather's fused epilogue: one warp per position recomputes l0_ with the reference epilogue
// where the gather's shortcut may differ (marked positions, us / them not (1, 0) or (0, 1)).
extern "C" __global__ void tiled_ft_fixup(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ acc_w, const float* __restrict__ acc_b, const float max_act,
    float* __restrict__ l0_out, const unsigned char* __restrict__ mark, const int batch_size,
    const int act_q, const float act_s, const float act_eps, const float act_inv, const float act_c)
{
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= batch_size) return;
    const int b = (int)w;
    const float u = us[b], th = them[b];
    if (!mark[b] && ((u == 1.0f && th == 0.0f) || (u == 0.0f && th == 1.0f))) return;
    for (int i = threadIdx.x & 31; i < $H; i += 32)
        epi_elem(b, i, us, them, acc_w, acc_b, max_act, l0_out, nullptr, act_q, act_s, act_eps, act_inv, act_c);
}
"""

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


_PROLOGUE_ACC_SRC = r"""
__device__ __forceinline__ void pro_elem(const int b, const int i,
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ acc_w, const float* __restrict__ acc_b,
    const float max_act, float* __restrict__ grad_w, float* __restrict__ grad_b)
{
    const float u = us[b], th = them[b];
    const size_t r = (size_t)b * $L1;
    // Recompute the clamped values exactly as tiled_ft_epilogue does (same expressions, same roundings).
    const float w0 = acc_w[r + i], w1 = acc_w[r + i + $H], b0 = acc_b[r + i], b1 = acc_b[r + i + $H];
    float cw0 = u * w0 + th * b0;
    float cw1 = u * w1 + th * b1;
    float cb0 = u * b0 + th * w0;
    float cb1 = u * b1 + th * w1;
    if (cw0 < 0.0f) cw0 = 0.0f; else if (cw0 > max_act) cw0 = max_act;
    if (cw1 < 0.0f) cw1 = 0.0f; else if (cw1 > max_act) cw1 = max_act;
    if (cb0 < 0.0f) cb0 = 0.0f; else if (cb0 > max_act) cb0 = max_act;
    if (cb1 < 0.0f) cb1 = 0.0f; else if (cb1 > max_act) cb1 = max_act;
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

extern "C" __global__ void tiled_ft_prologue_acc(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ acc_w, const float* __restrict__ acc_b,
    const float max_act, float* __restrict__ grad_w, float* __restrict__ grad_b, const int batch_size)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)batch_size * $H) return;
    pro_elem((int)(t / $H), (int)(t % $H), us, them, grad_l0, acc_w, acc_b, max_act, grad_w, grad_b);
}

// Fused prologue (the scatter computes the accumulator gradients): per list in tile order, its id, the
// half of grad_l0 it reads (white: 0 when us == 1) and whether its position needs the reference rows
// (us / them not (1, 0) or (0, 1)); one warp per such position writes them with the reference formulas.
extern "C" __global__ void tiled_ft_pro_codes(const int* __restrict__ order, const float* __restrict__ us,
    const float* __restrict__ them, unsigned* __restrict__ code, const int n_lists, const int batch_size)
{
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= n_lists) return;
    const int lid = order[li];
    const bool white = lid < batch_size;
    const int b = white ? lid : lid - batch_size;
    const float u = us[b], th = them[b];
    const bool binary = (u == 1.0f && th == 0.0f) || (u == 0.0f && th == 1.0f);
    code[li] = (unsigned)lid | ((white == (u == 1.0f)) ? 0u : (1u << 30)) | (binary ? 0u : (1u << 31));
}

extern "C" __global__ void tiled_ft_pro_ref(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ acc_w, const float* __restrict__ acc_b,
    const float max_act, float* __restrict__ grad_w, float* __restrict__ grad_b, const int batch_size)
{
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= batch_size) return;
    const int b = (int)w;
    const float u = us[b], th = them[b];
    if ((u == 1.0f && th == 0.0f) || (u == 0.0f && th == 1.0f)) return;
    for (int i = threadIdx.x & 31; i < $H; i += 32)
        pro_elem(b, i, us, them, grad_l0, acc_w, acc_b, max_act, grad_w, grad_b);
}
"""


def _kernel(name: str, src: str, l1_size: int):
    h = l1_size // 2
    return _tb._get_kernel((name, l1_size), src.replace("$H", str(h)).replace("$L1", str(l1_size)), name)


def _launch(kernel, batch_size: int, l1_size: int, args):
    n = batch_size * (l1_size // 2)
    kernel(grid=((n + 255) // 256,), block=(256,), args=args)


import os
PRO_FUSED = True   # the scatter computes the accumulator gradients
ACT_FUSED = True   # the epilogue applies act_quant (FusedDoubleFtFunction checks this)


@torch.compiler.disable(recursive=True)
def forward(us, them, white_indices, black_indices, weight, bias, max_ft_activation, l1_size, keep_acc=False,
            act_quant=None):
    """(l0_, clamped_out) like the fused forward kernel, or None when the tiled forward does not apply.
    With ``keep_acc`` (only when the tiled backward will run) it returns (l0_, (acc_w, acc_b)) instead:
    the backward recomputes the clamped values from the raw accumulators, which saves writing
    ``clamped_out`` (as large as both accumulators together)."""
    if not USE_TILED_FORWARD or not _tf.applicable(l1_size, weight.shape[1]):
        return None
    prep = _tf.lookup_prep(white_indices)
    if prep is None:
        _tf.STATS["fallback"] += 1
        return None
    batch_size = white_indices.shape[0]
    aq = ((np.int32(1), np.float32(act_quant[0]), np.float32(act_quant[1]), np.float32(1.0) / np.float32(act_quant[0]),
           np.float32(act_quant[2])) if act_quant is not None else
          (np.int32(0), np.float32(0), np.float32(0), np.float32(0), np.float32(0)))
    if keep_acc and isinstance(prep, _tf._tp.PullPrep) and _tf._tp.epi_applicable(l1_size) and us.is_contiguous() \
            and them.is_contiguous() and us.dtype == torch.float32 and them.dtype == torch.float32 \
            and us.numel() == batch_size and them.numel() == batch_size:
        # the gather writes l0_ itself; the fixup recomputes the positions its shortcut does not cover
        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        mark = torch.zeros(batch_size, dtype=torch.uint8, device=us.device)
        acc_w, acc_b = _tf.tiled_forward(white_indices, black_indices, weight, bias, l1_size, prep,
                                         epi=(us, them, l0_, mark, np.float32(max_ft_activation), *aq))
        with _tb._cupy_stream(us.device):
            _kernel("tiled_ft_fixup", _EPILOGUE_SRC, l1_size)(grid=((batch_size * 32 + 255) // 256,), block=(256,), args=(
                us.data_ptr(), them.data_ptr(), acc_w.data_ptr(), acc_b.data_ptr(), np.float32(max_ft_activation),
                l0_.data_ptr(), mark.data_ptr(), np.int32(batch_size), *aq))
        return l0_, (acc_w, acc_b)
    acc_w, acc_b = _tf.tiled_forward(white_indices, black_indices, weight, bias, l1_size, prep)
    l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
    clamped_out = None if keep_acc else torch.empty(
        batch_size, 4, l1_size // 2, dtype=torch.float32, device=us.device)
    with _tb._cupy_stream(us.device):
        _launch(_kernel("tiled_ft_epilogue", _EPILOGUE_SRC, l1_size), batch_size, l1_size, (
            us.data_ptr(), them.data_ptr(), acc_w.data_ptr(), acc_b.data_ptr(), np.float32(max_ft_activation),
            l0_.data_ptr(), 0 if keep_acc else clamped_out.data_ptr(), np.int32(batch_size),
            *((np.int32(1), np.float32(act_quant[0]), np.float32(act_quant[1]), np.float32(1.0) / np.float32(act_quant[0]),
               np.float32(act_quant[2])) if act_quant is not None else
              (np.int32(0), np.float32(0), np.float32(0), np.float32(0), np.float32(0)))))
    return (l0_, (acc_w, acc_b)) if keep_acc else (l0_, clamped_out)


def backward_will_run(l1_size: int) -> bool:
    return USE_TILED_BACKWARD and _tb.applicable(l1_size)


def lookup_backward_prep(white_indices):
    return _tb.lookup_prep(white_indices) if USE_TILED_BACKWARD else None


@torch.compiler.disable(recursive=True)
def backward(us, them, white_indices, black_indices, grad_l0, clamped_out, num_inputs, max_ft_activation,
             l1_size, prep=None, acc=None):
    """(grad_weight, grad_bias) of the fused double FT, or None when the tiled backward does not apply.
    ``acc`` = (acc_w, acc_b) from ``forward(keep_acc=True)`` replaces ``clamped_out``."""
    if not USE_TILED_BACKWARD or not _tb.applicable(l1_size):
        return None
    batch_size = white_indices.shape[0]
    grad_w = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
    grad_b = torch.empty_like(grad_w)
    if acc is not None and PRO_FUSED and prep is not None and grad_l0.is_contiguous() and us.is_contiguous() \
            and them.is_contiguous() and us.dtype == torch.float32 and them.dtype == torch.float32 \
            and us.numel() == batch_size and them.numel() == batch_size and acc[0].shape[1] == l1_size \
            and _tb.COLUMNS_PER_BLOCK == 64 and l1_size % 64 == 0 and 2 * batch_size < (1 << 30):
        # fused prologue: the scatter computes the accumulator gradients; grad_w / grad_b only receive the
        # reference rows of positions with non-binary us / them
        n_lists = 2 * batch_size
        code = torch.empty(n_lists, dtype=torch.int32, device=us.device)
        with _tb._cupy_stream(us.device):
            _kernel("tiled_ft_pro_codes", _PROLOGUE_ACC_SRC, l1_size)(grid=((n_lists + 255) // 256,), block=(256,), args=(
                prep.order.data_ptr(), us.data_ptr(), them.data_ptr(), code.data_ptr(), np.int32(n_lists), np.int32(batch_size)))
            _kernel("tiled_ft_pro_ref", _PROLOGUE_ACC_SRC, l1_size)(grid=((batch_size * 32 + 255) // 256,), block=(256,), args=(
                us.data_ptr(), them.data_ptr(), grad_l0.data_ptr(), acc[0].data_ptr(), acc[1].data_ptr(),
                np.float32(max_ft_activation), grad_w.data_ptr(), grad_b.data_ptr(), np.int32(batch_size)))
        return _tb.tiled_backward(white_indices, black_indices, grad_w, grad_b, num_inputs, l1_size, prep=prep,
                                  pro=(grad_l0, acc[0], acc[1], code, max_ft_activation))
    with _tb._cupy_stream(us.device):
        if acc is not None:
            _launch(_kernel("tiled_ft_prologue_acc", _PROLOGUE_ACC_SRC, l1_size), batch_size, l1_size, (
                us.data_ptr(), them.data_ptr(), grad_l0.data_ptr(), acc[0].data_ptr(), acc[1].data_ptr(),
                np.float32(max_ft_activation), grad_w.data_ptr(), grad_b.data_ptr(), np.int32(batch_size)))
        else:
            _launch(_kernel("tiled_ft_prologue", _PROLOGUE_SRC, l1_size), batch_size, l1_size, (
                us.data_ptr(), them.data_ptr(), grad_l0.data_ptr(), clamped_out.data_ptr(), np.float32(max_ft_activation),
                grad_w.data_ptr(), grad_b.data_ptr(), np.int32(batch_size)))
    return _tb.tiled_backward(white_indices, black_indices, grad_w, grad_b, num_inputs, l1_size, prep=prep)
