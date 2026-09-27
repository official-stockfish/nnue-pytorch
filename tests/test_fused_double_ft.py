import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)
from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
@pytest.mark.parametrize("l1", [32, 64, 96, 264, 1024, 1152, 1280, 2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 9])
def test_fused_double_ft(l1, batch_size):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    max_active = 96
    num_inputs = 100
    output_size = l1

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
    # Empty rows, unequal perspective lengths and a partial backward tile.
    for row in range(batch_size):
        white_indices[row, (row * 17) % max_active :] = -1
        black_indices[row, (row * 23 + 1) % max_active :] = -1

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

    # 1) Fused kernel
    l0_fused = double_feature_transform(
        us,
        them,
        white_indices,
        black_indices,
        weight,
        bias,
        127.0,  # max_ft_activation
        l1,  # l1_size
        "fused",
    )

    # Distinct upstream gradients catch column-routing errors in tiled kernels.
    grad_output = torch.randn_like(l0_fused)
    loss_fused = (l0_fused * grad_output).sum()
    loss_fused.backward()

    grad_weight_fused = weight.grad.clone()
    grad_bias_fused = bias.grad.clone()

    # 2) Fallback
    weight.grad.zero_()
    bias.grad.zero_()

    l0_fallback = double_feature_transform(
        us,
        them,
        white_indices,
        black_indices,
        weight,
        bias,
        127.0,  # max_ft_activation
        l1,  # l1_size
        "torch",
    )

    loss_fallback = (l0_fallback * grad_output).sum()
    loss_fallback.backward()

    # Compare
    torch.testing.assert_close(l0_fused, l0_fallback, atol=1e-3, rtol=1e-4)

    torch.testing.assert_close(grad_weight_fused, weight.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(grad_bias_fused, bias.grad, atol=1e-4, rtol=1e-3)
