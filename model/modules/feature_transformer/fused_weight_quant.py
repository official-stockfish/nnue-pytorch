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

from ...cupy_utils import HAS_CUPY, compile_kernel, cupy_stream

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


def _merge_quant_kernel():
    return compile_kernel(("merge_quant",), _SRC, "merge_quant")


def _merge_quant4_kernel():
    return compile_kernel(("merge_quant4",), _SRC, "merge_quant4")


def _aligned16(*ts):
    return all(t is None or t.data_ptr() % 16 == 0 for t in ts)


def segments_of(features):
    """[(weight, virtual_weight or None, num_buckets)] for the feature set, from the layout each feature
    declares (InputFeature.virtual_layout), or None when some feature merges its weight another way
    (then the caller uses merged_weight())."""
    segs = []
    for f in features:
        w = f.weight
        layout = f.virtual_layout()
        if layout is not None:
            v, nb = layout
            if w.shape[0] != v.shape[0] * nb or w.shape[1:] != v.shape[1:]:
                return None
            segs.append((w, v, nb))
        elif f.merged_weight() is w:
            segs.append((w, None, 1))
        else:
            return None
    return segs


def usable(segs) -> bool:
    return (
        HAS_CUPY
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
        r0 = 0
        with cupy_stream(out.device):
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
                        _merge_quant4_kernel()(
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
                        _merge_quant_kernel()(
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
        grad = grad.contiguous()
        grads = [None] * len(ctx.shapes)
        r0 = 0
        for i, (has_v, nb) in _iter_layout(ctx.layout):
            nr = ctx.shapes[i][0]
            g = grad[r0 : r0 + nr]
            grads[i] = g
            if has_v:
                # the virtual weight is repeated over the buckets: sum its gradient over them
                grads[i + 1] = g.view(nb, *ctx.shapes[i + 1]).sum(0)
            r0 += nr
        return (None, None, *grads)


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
