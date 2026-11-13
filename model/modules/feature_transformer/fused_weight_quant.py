"""One-pass merge + fake quantization of the feature-transformer weight.

``ComposedFeatureTransformer.merged_weight_and_bias(fake_quantize_weights=True)`` builds the effective
weight as ``cat([w, w_k + virtual.repeat(NB, 1), ...])`` and fake-quantizes it with the STE
``hard + (x - x.detach())``. For the default model that is a ~570 MB tensor, and the chain of
repeat / add / cat / mul / round / div / sub / add makes about eight full passes over it every step,
plus their backward. Here one kernel writes ``round((w + v) * s) * (1 / s)`` per feature segment
straight into the merged tensor, and the backward hands each weight its slice of the merged gradient
(the STE gradient is the identity) and reduces the virtual weight's slice over the buckets.

The result is bit-identical to the reference chain: PyTorch's CUDA ``mul``/``div`` by a Python scalar
compute in fp32 with ``float(s)`` and ``float(1) / float(s)``, ``round`` is round-half-to-even
(``rint``), ``x - x.detach()`` is exactly zero, and the virtual gradient is the same
``view(NB, nv, L).sum(0)`` that ``repeat``'s backward performs.
"""

from __future__ import annotations

import numpy as np
import torch

from model.optimizers import grad_row_flags as _grf

try:
    import cupy as cp

    from . import tiled_ft_backward as _tb

    _HAS_CUPY = True
except Exception:  # noqa: BLE001 - optional
    _HAS_CUPY = False

_SRC = r"""
extern "C" __global__ void merge_quant(float* __restrict__ out, const float* __restrict__ w,
                                       const float* __restrict__ v, long long n, long long vn,
                                       float s, float inv) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float x = w[i];
        if (v != nullptr) x = x + v[i % vn];
        out[i] = rintf(x * s) * inv;
    }
}

// Same per-element math, 16 B per access. Thread j owns float4 j of the virtual weight (or of the
// whole segment when there is none) and walks the nb buckets, so each virtual element is read once.
extern "C" __global__ void merge_quant4(float4* __restrict__ out, const float4* __restrict__ w,
                                        const float4* __restrict__ v, int vn4, int nb,
                                        float s, float inv) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < vn4; j += blockDim.x * gridDim.x) {
        const float4 a = v != nullptr ? v[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int b = 0; b < nb; ++b) {
            const long long i = (long long)b * vn4 + j;
            float4 x = w[i];
            if (v != nullptr) { x.x = x.x + a.x; x.y = x.y + a.y; x.z = x.z + a.z; x.w = x.w + a.w; }
            out[i] = make_float4(rintf(x.x * s) * inv, rintf(x.y * s) * inv, rintf(x.z * s) * inv, rintf(x.w * s) * inv);
        }
    }
}
"""
_kernel = None
_kernel4 = None
ENABLED = True


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_SRC, "merge_quant")
        _kernel.compile()
    return _kernel


def _get_kernel4():
    global _kernel4
    if _kernel4 is None:
        _kernel4 = cp.RawKernel(_SRC, "merge_quant4")
        _kernel4.compile()
    return _kernel4


def _aligned16(*ts):
    return all(t is None or t.data_ptr() % 16 == 0 for t in ts)


def segments_of(features):
    """[(weight, virtual_weight or None, num_buckets)] for the feature set, or None when some feature
    merges its weight in another way (then the caller uses the reference path)."""
    segs = []
    for f in features:
        w = f.weight
        v = getattr(f, "virtual_weight", None)
        if v is not None:
            nb = getattr(f, "NUM_BUCKETS", None)
            if (
                nb is None
                or w.shape[0] != v.shape[0] * nb
                or w.shape[1:] != v.shape[1:]
            ):
                return None
            segs.append((w, v, nb))
        else:
            if f.merged_weight() is not w:
                return None
            segs.append((w, None, 1))
    return segs


def usable(segs) -> bool:
    return (
        _HAS_CUPY
        and segs is not None
        and all(
            w.is_cuda
            and w.dtype == torch.float32
            and w.is_contiguous()
            and (
                v is None
                or (v.is_cuda and v.dtype == torch.float32 and v.is_contiguous())
            )
            for w, v, _ in segs
        )
    )


class _MergeFakeQuant(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, scale, layout, *params):
        s = np.float32(scale)
        inv = np.float32(1.0) / s
        rows = sum(params[i].shape[0] for i, (has_v, _) in _iter_layout(layout))
        cols = params[0].shape[1]
        out = torch.empty(rows, cols, device=params[0].device, dtype=torch.float32)
        k = _get_kernel()
        r0 = 0
        with _tb._cupy_stream(out.device):
            for i, (has_v, _nb) in _iter_layout(layout):
                w = params[i]
                n = w.numel()
                if n:
                    v = params[i + 1] if has_v else None
                    o = out[r0:]
                    vn = n // _nb if v is not None else n
                    if (
                        n % 4 == 0
                        and vn % 4 == 0
                        and vn // 4 < 2**31
                        and _aligned16(o, w, v)
                    ):
                        vn4 = vn // 4
                        blocks = min((vn4 + 255) // 256, 82 * 16)
                        _get_kernel4()(
                            (blocks,),
                            (256,),
                            (
                                o.data_ptr(),
                                w.data_ptr(),
                                0 if v is None else v.data_ptr(),
                                np.int32(vn4),
                                np.int32(_nb if v is not None else 1),
                                s,
                                inv,
                            ),
                        )
                    else:
                        blocks = min((n + 255) // 256, 65535 * 8)
                        k(
                            (blocks,),
                            (256,),
                            (
                                o.data_ptr(),
                                w.data_ptr(),
                                0 if v is None else v.data_ptr(),
                                np.int64(n),
                                np.int64(1 if v is None else v.numel()),
                                s,
                                inv,
                            ),
                        )
                r0 += w.shape[0]
        ctx.layout = layout
        ctx.shapes = [p.shape for p in params]
        return out

    @staticmethod
    def backward(ctx, grad):
        flags = _grf.lookup(
            grad
        )  # rows of the tiled FT gradient that are not exactly +0
        grad = grad.contiguous()
        grads = [None] * len(ctx.shapes)
        r0 = 0
        for i, (has_v, nb) in _iter_layout(ctx.layout):
            nr = ctx.shapes[i][0]
            g = grad[r0 : r0 + nr]
            grads[i] = g
            if flags is not None:
                _grf.register(g, flags[r0 : r0 + nr])
            if has_v:
                vs = ctx.shapes[i + 1]
                fl = None if flags is None else flags[r0 : r0 + nr]
                grads[i + 1] = (
                    _bucket_sum(g, fl, nb, vs)
                    if fl is not None
                    else g.view(nb, vs[0], *vs[1:]).sum(0)
                )
            r0 += nr
        return (None, None, *grads)


# Virtual-weight gradient: the sum over the buckets, in the order of PyTorch's reduction (4 accumulators,
# bucket b into accumulator b % 4, then ((a0 + a1) + a2) + a3), reading only the flagged rows: an unflagged
# row is exactly +0, and adding +0 changes no sum (no partial sum is -0: the gradient has no -0 entries).
_BSUM_SRC = r"""
extern "C" __global__ void bucket_sum(const float4* __restrict__ g, const unsigned char* __restrict__ flags,
                                      float4* __restrict__ out, const int nb, const int rows, const int cols4)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)rows * cols4) return;
    const int r = (int)(t / cols4), c = (int)(t - (long long)r * cols4);
    float4 a[4];
    bool has[4] = {false, false, false, false};
    for (int b = 0; b < nb; ++b) {
        if (!flags[(long long)b * rows + r]) continue;
        const float4 v = g[((long long)b * rows + r) * cols4 + c];
        const int k = b & 3;
        if (has[k]) { a[k].x += v.x; a[k].y += v.y; a[k].z += v.z; a[k].w += v.w; } else { a[k] = v; has[k] = true; }
    }
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool any = false;
    for (int k = 0; k < 4; ++k) {
        if (!has[k]) continue;
        if (any) { s.x += a[k].x; s.y += a[k].y; s.z += a[k].z; s.w += a[k].w; } else { s = a[k]; any = true; }
    }
    out[t] = s;
}
"""
_bsum_kernel = None
BUCKET_SUM = True


@torch.compiler.disable
def _bucket_sum(g, flags, nb, vs):
    """g.view(nb, *vs).sum(0) for a contiguous fp32 g whose unflagged rows are +0 (bit-identical)."""
    global _bsum_kernel
    cols = int(np.prod(vs[1:])) if len(vs) > 1 else 1
    rows = vs[0]
    if not (
        BUCKET_SUM
        and _HAS_CUPY
        and g.is_cuda
        and g.dtype == torch.float32
        and g.is_contiguous()
        and cols % 4 == 0
        and g.data_ptr() % 16 == 0
        and flags.dtype == torch.uint8
        and flags.numel() == nb * rows
    ):
        return g.view(nb, *vs).sum(0)
    if _bsum_kernel is None:
        _bsum_kernel = cp.RawKernel(_BSUM_SRC, "bucket_sum")
    out = torch.empty(*vs, dtype=torch.float32, device=g.device)
    n = rows * (cols // 4)
    with _tb._cupy_stream(g.device):
        _bsum_kernel(
            ((n + 255) // 256,),
            (256,),
            (
                g.data_ptr(),
                flags.data_ptr(),
                out.data_ptr(),
                np.int32(nb),
                np.int32(rows),
                np.int32(cols // 4),
            ),
        )
    return out


def _iter_layout(layout):
    """layout: tuple of (has_virtual, num_buckets) per feature; params are w[, v] per feature in order."""
    i = 0
    for has_v, nb in layout:
        yield i, (has_v, nb)
        i += 2 if has_v else 1


def merge_and_fake_quantize(segs, scale):
    layout = tuple((v is not None, nb) for _, v, nb in segs)
    params = [p for w, v, _ in segs for p in ((w, v) if v is not None else (w,))]
    return _MergeFakeQuant.apply(scale, layout, *params)


# ---- FT activation: fake_quantize_ft_act(l0_) * l0_correction_factor in one pass --------------------
# Reference: hard = x.mul(s).add(eps).floor().div(s); out = (hard + (x - x.detach())) * c. PyTorch's CUDA
# scalar mul/add/div run in fp32 with float(s), float(eps) and float(1) / float(s) (a scalar divisor is
# applied as a reciprocal multiply), so the kernel uses the same roundings with no FMA contraction.
# The STE gradient is grad * c (identity when c == 1).
_ACT_SRC = r"""
extern "C" __global__ void act_quant(float* __restrict__ out, const float* __restrict__ x, long long n,
                                     float s, float eps, float inv, float c) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float h = __fmul_rn(floorf(__fadd_rn(__fmul_rn(x[i], s), eps)), inv);
        out[i] = __fmul_rn(h, c);
    }
}
"""
_act_kernel = None


class _ActFakeQuant(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, x, scale, eps, corr):
        global _act_kernel
        if _act_kernel is None:
            _act_kernel = cp.RawKernel(_ACT_SRC, "act_quant")
            _act_kernel.compile()
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        s = np.float32(scale)
        if n:
            with _tb._cupy_stream(x.device):
                _act_kernel(
                    (min((n + 255) // 256, 65535 * 8),),
                    (256,),
                    (
                        out.data_ptr(),
                        x.data_ptr(),
                        np.int64(n),
                        s,
                        np.float32(eps),
                        np.float32(1.0) / s,
                        np.float32(corr),
                    ),
                )
        ctx.corr = corr
        return out

    @staticmethod
    def backward(ctx, grad):
        return (grad if ctx.corr == 1.0 else grad * ctx.corr), None, None, None


def act_usable(x: torch.Tensor) -> bool:
    return ENABLED and _HAS_CUPY and x.is_cuda and x.dtype == torch.float32


def fake_quantize_act_and_scale(x, scale, eps, corr):
    return _ActFakeQuant.apply(x, scale, eps, corr)
