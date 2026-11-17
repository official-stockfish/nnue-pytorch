"""Fused merge + fake quantization of the FT weight against the QuantizationManager reference it
replaces (bit-identical, gradients included)."""

import numpy as np
import pytest
import torch

from model.cupy_utils import HAS_CUPY, cupy_stream
from model.modules import ComposedFeatureTransformer, get_feature_cls
from model.modules.feature_transformer import fused_weight_quant as FWQ
from model.quantize import QuantizationConfig, QuantizationManager

needs_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_CUPY), reason="needs CUDA + CuPy"
)


def _ft(device):
    torch.manual_seed(0)
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats+PP_3Wide+K32H3^"),
        64,
        QuantizationManager(QuantizationConfig()),
    ).to(device)
    with torch.no_grad():
        for f in ft.features:
            if f.virtual_layout() is not None:
                f.virtual_layout()[0].normal_(std=0.01)
    return ft


def test_segments_follow_declared_layouts():
    ft = _ft("cpu")
    segs = FWQ.segments_of(ft.features)
    assert [
        (w is f.weight, v is None, nb) for (w, v, nb), f in zip(segs, ft.features)
    ] == [
        (True, True, 1),
        (True, True, 1),
        (True, False, 96),
    ]

    # merged_weight() is built from the same declaration
    threats, k32h3 = ft.features[0], ft.features[-1]
    assert threats.merged_weight() is threats.weight
    assert torch.equal(
        k32h3.merged_weight(),
        k32h3.weight + k32h3.virtual_weight.repeat(k32h3.NUM_BUCKETS, 1),
    )

    class OtherMerge(torch.nn.Module):
        weight = torch.zeros(2, 2)

        def virtual_layout(self):
            return None

        def merged_weight(self):
            return self.weight * 2

    assert FWQ.segments_of([OtherMerge()]) is None


@needs_cuda
def test_fused_weight_path_matches_quantization_manager():
    ft = _ft("cuda")
    qm = ft.quantization
    params = list(ft.parameters())
    merged, bias = ft.merged_weight_and_bias(fake_quantize_weights=True)
    ref_merged = qm.fake_quantize_weights(
        torch.cat([f.merged_weight() for f in ft.features]), "ft_weight"
    )
    ref_bias = qm.fake_quantize_weights(ft.bias, "ft_bias")
    assert torch.equal(merged, ref_merged) and torch.equal(bias, ref_bias)
    g = torch.randn_like(merged)
    got = torch.autograd.grad(merged, params, g, allow_unused=True)
    want = torch.autograd.grad(ref_merged, params, g, allow_unused=True)
    for a, b in zip(got, want):
        assert (a is None and b is None) or torch.equal(a, b)


@needs_cuda
@pytest.mark.parametrize("cols", [1032, 520, 6])
def test_merge_quant_vectorized_bit_identical(cols):
    """The vectorized merge (merge_quant4) == the scalar merge_quant kernel == the torch chain."""

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(cols)
    s = 254.0
    nb, nv = 32, 37
    w0 = torch.randn(301, cols, generator=gen, device=dev) * 0.3
    w1 = torch.randn(nb * nv, cols, generator=gen, device=dev) * 0.3
    v = torch.randn(nv, cols, generator=gen, device=dev) * 0.3
    # exact rounding ties
    w0[:7] = (
        torch.randint(-300, 300, (7, cols), generator=gen, device=dev).float() + 0.5
    ) / s
    w1[:5] = (
        torch.randint(-300, 300, (5, cols), generator=gen, device=dev).float() + 0.5
    ) / s
    v[:2] = 0.0
    out = FWQ.merge_and_fake_quantize([(w0, None, 1), (w1, v, nb)], s)
    x = torch.cat([w0, w1 + v.repeat(nb, 1)])
    ref = torch.round(x * s) * (np.float32(1.0) / np.float32(s)).item()
    assert torch.equal(out, ref)
    # the scalar kernel on the same segments
    sc = torch.empty_like(out)
    with cupy_stream(dev):
        for r0, (ww, vv) in ((0, (w0, None)), (301, (w1, v))):
            FWQ._merge_quant_kernel()(
                ((ww.numel() + 255) // 256,),
                (256,),
                (
                    sc[r0:].data_ptr(),
                    ww.data_ptr(),
                    0 if vv is None else vv.data_ptr(),
                    np.int64(ww.numel()),
                    np.int64(1 if vv is None else vv.numel()),
                    np.float32(s),
                    np.float32(1.0) / np.float32(s),
                ),
            )
    assert torch.equal(out, sc)
