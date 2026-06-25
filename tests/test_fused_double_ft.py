import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.functions import _HAS_CUPY_KERNELS
from model.modules import (
    DoubleFeatureTransformer,
    get_use_fused_double_ft,
    set_use_fused_double_ft,
)


class DummyComposedFeatures:
    def __init__(self, weight, bias, l1_size):
        self.weight = weight
        self.bias = bias
        self.l1_size = l1_size

        # Simple quantization mock
        class DummyQuantization:
            max_ft_activation = 127.0
            l0_correction_factor = 1.0

            def clip_ft_act(self, x):
                return torch.clamp(x, 0.0, 127.0)

            def fake_quantize_ft_act(self, x):
                return x

        self.quantization = DummyQuantization()

    def merged_weight_and_bias(self, fake_quantize_weights=False):
        b = self.bias[: self.l1_size]
        pb = torch.zeros_like(self.bias[self.l1_size :], dtype=b.dtype)
        bias = torch.cat([b, pb], dim=0)
        return self.weight, bias


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
def test_fused_double_ft():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    batch_size = 4
    max_active = 32
    num_inputs = 100
    l1 = 32
    num_psqt_buckets = 8

    output_size = l1 + num_psqt_buckets

    us = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    them = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")

    # ensure non-negative indices and some padding (-1)
    white_indices = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    white_indices[:, -2:] = -1
    black_indices = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    black_indices[:, -2:] = -1

    psqt_indices = torch.randint(
        0, num_psqt_buckets, (batch_size,), dtype=torch.int64, device="cuda"
    )

    weight = torch.randn(
        num_inputs,
        output_size,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    bias = torch.randn(
        output_size, dtype=torch.float32, device="cuda", requires_grad=True
    )

    dummy_features = DummyComposedFeatures(weight, bias, l1)
    double_ft = DoubleFeatureTransformer(dummy_features)

    orig_fused = get_use_fused_double_ft()
    try:
        # 1) Fused kernel
        set_use_fused_double_ft(True)
        l0_fused, wpsqt_fused, bpsqt_fused = double_ft(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts=False,
            fake_quantize_weights=False,
        )

        loss_fused = l0_fused.sum() + wpsqt_fused.sum() + bpsqt_fused.sum()
        loss_fused.backward()

        grad_weight_fused = weight.grad.clone()
        grad_bias_fused = bias.grad.clone()

        # 2) Fallback
        weight.grad.zero_()
        bias.grad.zero_()

        set_use_fused_double_ft(False)
        l0_fallback, wpsqt, bpsqt = double_ft(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts=False,
            fake_quantize_weights=False,
        )

        loss_fallback = l0_fallback.sum() + wpsqt.sum() + bpsqt.sum()
        loss_fallback.backward()

        # Compare
        torch.testing.assert_close(l0_fused, l0_fallback, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(wpsqt_fused, wpsqt, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(bpsqt_fused, bpsqt, atol=1e-5, rtol=1e-4)

        torch.testing.assert_close(
            grad_weight_fused, weight.grad, atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(grad_bias_fused, bias.grad, atol=1e-4, rtol=1e-3)
    finally:
        set_use_fused_double_ft(orig_fused)
