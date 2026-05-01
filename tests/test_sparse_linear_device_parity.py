import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.functions import _torch_sparse_linear


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sparse_linear_cpu_mps_parity():
    torch.manual_seed(42)

    batch_size = 8
    input_size = 60720
    max_active_features = 128
    output_size = 1032

    feature_indices = (
        torch.rand(batch_size, max_active_features) * input_size
    ).to(dtype=torch.int32)

    # Exercise padding behavior. Negative indices should be ignored
    feature_indices[:, -4:] = -1

    feature_values = torch.rand(batch_size, max_active_features, dtype=torch.float32)
    weight = torch.randn(input_size, output_size, dtype=torch.float32)
    bias = torch.randn(output_size, dtype=torch.float32)

    cpu_out = _torch_sparse_linear(feature_indices, feature_values, weight, bias)

    mps_out = _torch_sparse_linear(
        feature_indices.to("mps"),
        feature_values.to("mps"),
        weight.to("mps"),
        bias.to("mps"),
    ).cpu()

    # Small tolerance for floating point differences
    assert torch.allclose(cpu_out, mps_out, atol=1e-5, rtol=1e-5)
