"""TF32 dense-forward parity under fake quantization, and the reason for the guard.

With act and weight fake-quantization on, every dense GEMM input is k/128
(k <= 127) and every weight k/128 or k/64: both are exact in TF32's 11-bit
significand, and products are exact in the fp32 accumulator, so enabling TF32
cannot change the forward result. Without quantization the inputs leave that
grid and TF32 rounds them, which is why train.py only enables TF32 for
quantized runs.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.config import LayerStacksConfig
from model.modules.layer_stacks import LayerStacks
from model.quantize import QuantizationConfig, QuantizationManager


def _tf32_supported():
    # TF32 matmuls need an NVIDIA GPU with compute capability >= 8 (Ampere).
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 8
    )


_TF32_REASON = "requires an NVIDIA GPU with compute capability >= 8 (Ampere)"


def _make_layer_stacks(count):
    torch.manual_seed(1234)
    layer = LayerStacks(count, LayerStacksConfig(L1=1024, L2=32, L3=32), QuantizationManager(QuantizationConfig())).cuda()
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    return layer


def _quantized_input(batch):
    # k/128 with k in [0, 127], as after FT-output fake quantization + clip.
    return (torch.randint(0, 128, (batch, 1024), device="cuda") / 128.0).float()


@pytest.mark.skipif(not _tf32_supported(), reason=_TF32_REASON)
@pytest.mark.parametrize("count", [8, 64])
@pytest.mark.parametrize("batch", [17, 32768])
def test_dense_forward_bit_identical_under_quantization(count, batch):
    layer = _make_layer_stacks(count)
    x = _quantized_input(batch)
    ls_indices = ((torch.randint(1, 33, (batch,), device="cuda", dtype=torch.int64) - 1) // 4)

    torch.backends.cuda.matmul.allow_tf32 = False
    expected = layer(x, ls_indices, True, True)
    torch.backends.cuda.matmul.allow_tf32 = True
    actual = layer(x, ls_indices, True, True)
    torch.backends.cuda.matmul.allow_tf32 = False

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _tf32_supported(), reason=_TF32_REASON)
def test_quantized_gemm_matches_fp64_reference():
    # Raw L1-shaped GEMM with grid inputs: k/128 acts, k/128 weights, k/16384
    # bias, as after fake quantization. Products are exact in both precisions,
    # so TF32 vs fp64 differs only by fp32 accumulation order.
    x = torch.randint(0, 128, (4096, 1024), device="cuda").float() / 128.0
    w = torch.randint(-127, 128, (256, 1024), device="cuda").float() / 128.0
    b = torch.randint(-127, 128, (256,), device="cuda").float() / 16384.0

    torch.backends.cuda.matmul.allow_tf32 = True
    actual = torch.nn.functional.linear(x, w, b)
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = torch.nn.functional.linear(x.double(), w.double(), b.double())

    torch.testing.assert_close(actual, expected.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _tf32_supported(), reason=_TF32_REASON)
def test_tf32_changes_forward_without_quantization():
    layer = _make_layer_stacks(8)
    batch = 4096
    x = torch.randn(batch, 1024, device="cuda")
    ls_indices = ((torch.randint(1, 33, (batch,), device="cuda", dtype=torch.int64) - 1) // 4)

    torch.backends.cuda.matmul.allow_tf32 = False
    expected = layer(x, ls_indices, False, False)
    torch.backends.cuda.matmul.allow_tf32 = True
    actual = layer(x, ls_indices, False, False)
    torch.backends.cuda.matmul.allow_tf32 = False

    # Pins why train.py gates TF32 on fake quantization.
    assert not torch.equal(actual, expected)
