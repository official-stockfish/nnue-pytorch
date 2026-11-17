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
// out = rint((w + v) * s) * inv, 16 B per access. Thread j owns float4 j of the virtual weight (or of
// the whole segment when there is none) and walks the nb buckets, so each virtual element is read
// once.
extern "C" __global__ void merge_quant(float4* __restrict__ out, const float4* __restrict__ w,
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


# Grid of the kernel's grid-stride loop: enough blocks to occupy the GPU; the loop covers the rest.
_MAX_BLOCKS = 82 * 16


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


def _float4_addressable(t: torch.Tensor) -> bool:
    """A contiguous fp32 CUDA matrix that the kernel can read as 16-byte-aligned float4 rows."""
    return (
        t.is_cuda
        and t.dtype == torch.float32
        and t.is_contiguous()
        and t.dim() == 2
        and t.shape[1] % 4 == 0
        and t.data_ptr() % 16 == 0
        and t.numel() // 4 < 2**31
    )


def usable(segs) -> bool:
    """Whether the kernel can merge these segments; otherwise the caller uses the reference chain."""
    return (
        HAS_CUPY
        and segs is not None
        and all(
            _float4_addressable(w) and (v is None or _float4_addressable(v))
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
            kernel = compile_kernel(("merge_quant",), _SRC, "merge_quant")
            for i, (has_v, nb) in _iter_layout(layout):
                w = params[i]
                if w.numel():
                    v = params[i + 1] if has_v else None
                    # float4s of the virtual weight, or of the whole weight when there is none
                    n4 = (w if v is None else v).numel() // 4
                    kernel(
                        (min((n4 + 255) // 256, _MAX_BLOCKS),),
                        (256,),
                        (
                            out[r0:].data_ptr(),
                            w.data_ptr(),
                            0 if v is None else v.data_ptr(),
                            np.int32(n4),
                            np.int32(1 if v is None else nb),
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
