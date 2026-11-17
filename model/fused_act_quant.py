"""One-pass activation fake quantization with a scale: the CuPy form of
``QuantizationManager.fake_quantize_ft_act(x) * l0_correction_factor``.

The reference computes ``hard = x.mul(s).add(eps).floor().div(s)`` and
``out = (hard + (x - x.detach())) * c``: seven passes over the activations, plus their backward.
PyTorch's CUDA scalar mul / add / div run in fp32 with ``float(s)``, ``float(eps)`` and
``float(1) / float(s)`` (a scalar divisor is applied as a reciprocal multiply), so the kernel uses
the same roundings with no FMA contraction and its output is bit-identical. The STE gradient is
``grad * c``.
"""

from __future__ import annotations

import numpy as np
import torch

from .cupy_utils import HAS_CUPY, compile_kernel, cupy_stream

_SRC = r"""
extern "C" __global__ void act_quant(float* __restrict__ out, const float* __restrict__ x, long long n,
                                     float s, float eps, float inv, float c) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float h = __fmul_rn(floorf(__fadd_rn(__fmul_rn(x[i], s), eps)), inv);
        out[i] = __fmul_rn(h, c);
    }
}
"""


class _ActFakeQuant(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, x, scale, eps, corr):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        s = np.float32(scale)
        if n:
            with cupy_stream(x.device):
                compile_kernel(("act_quant",), _SRC, "act_quant")(
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


def usable(x: torch.Tensor) -> bool:
    return HAS_CUPY and x.is_cuda and x.dtype == torch.float32


def fake_quantize_act_and_scale(x, scale, eps, corr):
    """``floor(x * scale + eps) / scale * corr`` with a straight-through gradient."""
    return _ActFakeQuant.apply(x, scale, eps, corr)
