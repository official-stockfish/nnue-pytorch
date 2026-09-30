import os
import sys
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules import stacked_linear
from model.modules.stacked_linear import FactorizedStackedLinear, grouped_l1
from model.quantize import QuantizationConfig, QuantizationManager

GPU_AVAILABLE = torch.cuda.is_available()
CUDA_AVAILABLE = GPU_AVAILABLE and torch.version.hip is None
OPTIMIZED_AVAILABLE = (
    CUDA_AVAILABLE
    and grouped_l1 is not None
    and torch.cuda.get_device_capability()[0] >= 8
)


def _reference(layer, x, indices, quantize=False):
    weight = layer.linear.weight.reshape(layer.count, layer.out_features, layer.in_features)
    weight = weight + layer.factorized_linear.weight
    bias = layer.linear.bias.reshape(layer.count, layer.out_features) + layer.factorized_linear.bias
    if quantize:
        weight = layer.quantization.fake_quantize_weights(weight, "ls_l1_weight")
        bias = layer.quantization.fake_quantize_weights(bias, "ls_l1_bias")
    indices = indices.flatten().long()
    return torch.bmm(weight[indices], x.unsqueeze(-1)).squeeze(-1) + bias[indices]


@pytest.mark.skipif(not OPTIMIZED_AVAILABLE, reason="NVIDIA SM80+, CuPy and Triton required")
@pytest.mark.parametrize("width", [1024, 1152, 1280])
@pytest.mark.parametrize("batch", [1, 17, 257])
@pytest.mark.parametrize("concentrated", [False, True])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("outputs", [16, 32, 48, 64, 128])
def test_grouped_forward_and_all_gradients(width, batch, concentrated, quantize, outputs, monkeypatch):
    _check_forward_and_all_gradients(width, batch, concentrated, quantize, outputs, 8, monkeypatch)


def _check_forward_and_all_gradients(width, batch, concentrated, quantize, outputs, count, monkeypatch):
    torch.manual_seed(123)
    layer = FactorizedStackedLinear(width, outputs, count, QuantizationManager(QuantizationConfig()), "ls_l1").cuda()
    # Distinct bucket weights catch incorrect routing/strides, unlike the
    # identical initial weights used by StackedLinear.
    with torch.no_grad():
        layer.linear.weight.normal_(std=0.02)
        layer.linear.bias.normal_(std=0.02)
        layer.factorized_linear.weight.normal_(std=0.02)
        layer.factorized_linear.bias.normal_(std=0.02)
    # Strided all-zero, dense and 75%-zero inputs; uneven/empty buckets and partial tiles.
    storage = torch.randn(batch, width, 2, device="cuda")
    zero_fraction = {1: 1.0, 17: 0.0, 257: 0.75}[batch]
    storage[..., 0].masked_fill_(torch.rand(batch, width, device="cuda") < zero_fraction, 0)
    x = storage[..., 0].detach().requires_grad_()
    indices = torch.randint(0, count, (batch, 1), device="cuda", dtype=torch.int32)
    indices[-1] = count - 1
    if concentrated:
        indices.fill_(count - 1)
    upstream = torch.randn(batch, outputs, device="cuda")
    parameters = tuple(layer.parameters())
    optimized = Mock(wraps=grouped_l1)
    monkeypatch.setattr(stacked_linear, "grouped_l1", optimized)
    monkeypatch.setenv("NNUE_GROUPED_L1", "1")

    # Both the CuPy router and Triton matmuls must respect the caller's stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reference = _reference(layer, x, indices, quantize)
        expected = torch.autograd.grad(reference, (x, *parameters), upstream)
        actual = layer(x, indices, quantize)
        gradients = torch.autograd.grad(actual, (x, *parameters), upstream)
    torch.cuda.current_stream().wait_stream(stream)
    optimized.assert_called_once()

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    for grad, ref_grad in zip(gradients, expected):
        torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)
    # A zero activation does not permit dropping its straight-through gradient.
    if zero_fraction:
        assert torch.count_nonzero(gradients[0][x == 0]) > 0


@pytest.mark.skipif(not OPTIMIZED_AVAILABLE, reason="NVIDIA SM80+, CuPy and Triton required")
@pytest.mark.parametrize("outputs", [1, 17, 31, 33, 63, 65, 96, 127])
def test_grouped_partial_output_tiles(outputs, monkeypatch):
    test_grouped_forward_and_all_gradients(1024, 17, False, False, outputs, monkeypatch)


@pytest.mark.skipif(not OPTIMIZED_AVAILABLE, reason="NVIDIA SM80+, CuPy and Triton required")
@pytest.mark.parametrize("count", [1, 3, 4, 16, 32, 256])
@pytest.mark.parametrize("concentrated", [False, True])
@pytest.mark.parametrize("quantize", [False, True])
def test_grouped_variable_stack_count(count, concentrated, quantize, monkeypatch):
    _check_forward_and_all_gradients(1024, 257, concentrated, quantize, 32, count, monkeypatch)


@pytest.mark.skipif(not OPTIMIZED_AVAILABLE, reason="NVIDIA SM80+, CuPy and Triton required")
@pytest.mark.parametrize("count,expected", [(1, False), (8, False), (16, False), (31, False), (32, True), (64, True), (256, True)])
def test_grouped_dispatch_gate(count, expected, monkeypatch):
    monkeypatch.delenv("NNUE_GROUPED_L1", raising=False)
    torch.manual_seed(123)
    layer = FactorizedStackedLinear(1024, 32, count, QuantizationManager(QuantizationConfig()), "ls_l1").cuda()
    with torch.no_grad():
        layer.linear.weight.normal_(std=0.02)
        layer.linear.bias.normal_(std=0.02)
        layer.factorized_linear.weight.normal_(std=0.02)
        layer.factorized_linear.bias.normal_(std=0.02)
    x = torch.randn(257, 1024, device="cuda", requires_grad=True)
    indices = torch.randint(0, count, (257, 1), device="cuda", dtype=torch.int32)
    optimized = Mock(wraps=grouped_l1)
    monkeypatch.setattr(stacked_linear, "grouped_l1", optimized)

    actual = layer(x, indices, False)
    if expected:
        optimized.assert_called_once()
    else:
        # The dense path must produce identical results when the gate closes
        # (cuBLAS uses TF32 here; the bmm reference does not).
        optimized.assert_not_called()
        torch.testing.assert_close(actual, _reference(layer, x, indices), atol=2e-3, rtol=2e-4)


@pytest.mark.skipif(not OPTIMIZED_AVAILABLE, reason="NVIDIA SM80+, CuPy and Triton required")
def test_grouped_dispatch_env_override(monkeypatch):
    layer = FactorizedStackedLinear(1024, 32, 64, QuantizationManager(QuantizationConfig()), "ls_l1").cuda()
    x = torch.randn(17, 1024, device="cuda")
    indices = torch.randint(0, 64, (17, 1), device="cuda", dtype=torch.int32)
    optimized = Mock(wraps=grouped_l1)
    monkeypatch.setattr(stacked_linear, "grouped_l1", optimized)

    monkeypatch.setenv("NNUE_GROUPED_L1", "0")
    layer(x, indices, False)
    optimized.assert_not_called()

    monkeypatch.setenv("NNUE_GROUPED_L1", "1")
    layer(x, indices, False)
    optimized.assert_called_once()


@pytest.mark.skipif(not GPU_AVAILABLE or grouped_l1 is None, reason="GPU, CuPy and Triton required")
def test_grouped_router_switches_stack_count():
    from model.modules.grouped_linear import _route_rows

    # Revisit a specialization after using another count; several CUDA blocks
    # must agree on the offsets. Bucket zero is deliberately left empty.
    for count in (8, 3, 256, 1, 8, 3):
        indices = (torch.arange(1025, device="cuda") % max(1, count - 1) + (count > 1)).long()
        rows, counts = _route_rows(indices, count)
        assert rows.shape == (count, len(indices))
        torch.testing.assert_close(counts.long(), torch.bincount(indices, minlength=count))
        for bucket, population in enumerate(counts.tolist()):
            expected = torch.where(indices == bucket)[0]
            actual = rows[bucket, :population].long().sort().values
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA or ROCm GPU required")),
])
@pytest.mark.parametrize("case", ["dependencies", "width", "outputs", "buckets", "dtype", "empty", "autocast"])
def test_grouped_fallback(device, case, monkeypatch):
    width = 264 if case == "width" else 1024
    outputs = 129 if case == "outputs" else 32
    buckets = 257 if case == "buckets" else 8
    dtype = torch.float64 if case == "dtype" else torch.float32
    batch = 0 if case == "empty" else 3
    layer = FactorizedStackedLinear(width, outputs, buckets, QuantizationManager(QuantizationConfig()), "ls_l1")
    layer = layer.to(device=device, dtype=dtype)
    x = torch.randn(batch, width, device=device, dtype=dtype, requires_grad=True)
    indices = torch.arange(batch, device=device).reshape(-1, 1) % buckets
    optimized = Mock(side_effect=AssertionError("Unsupported inputs must use the PyTorch fallback"))
    monkeypatch.setattr(stacked_linear, "grouped_l1", None if case == "dependencies" else optimized)

    with torch.autocast(device_type=device, enabled=case == "autocast"):
        actual = layer(x, indices)
        # Use the original all-buckets operation for matching autocast semantics.
        if case == "autocast":
            weight = layer.linear.weight + layer.factorized_linear.weight.repeat(buckets, 1)
            bias = layer.linear.bias + layer.factorized_linear.bias.repeat(buckets)
            expected = layer.select_output(torch.nn.functional.linear(x, weight, bias), indices)
        else:
            expected = _reference(layer, x, indices)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=2e-4)
    parameters = (x, *layer.parameters())
    gradients = torch.autograd.grad(actual.sum(), parameters)
    reference_gradients = torch.autograd.grad(expected.sum(), parameters)
    for grad, ref_grad in zip(gradients, reference_gradients):
        torch.testing.assert_close(grad, ref_grad, atol=1e-3, rtol=1e-3)
    optimized.assert_not_called()


@pytest.mark.parametrize("count", [1, 3, 8, 16])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not GPU_AVAILABLE or torch.version.hip is None, reason="ROCm GPU required"
    )),
])
def test_grouped_fallback_compiles(count, device):
    layer = FactorizedStackedLinear(1024, 32, count, QuantizationManager(QuantizationConfig()), "ls_l1").to(device)
    with torch.no_grad():
        layer.linear.weight.normal_(std=0.02)
        layer.linear.bias.normal_(std=0.02)
    x = torch.randn(3, 1024, device=device, requires_grad=True)
    indices = torch.tensor([[0], [count // 2], [count - 1]], device=device)
    compiled = torch.compile(layer, backend="aot_eager")
    actual = compiled(x, indices, True)
    expected = _reference(layer, x, indices, True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    parameters = (x, *layer.parameters())
    gradients = torch.autograd.grad(actual.sum(), parameters)
    reference_gradients = torch.autograd.grad(expected.sum(), parameters)
    for grad, ref_grad in zip(gradients, reference_gradients):
        torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)
