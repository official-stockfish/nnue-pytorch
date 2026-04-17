import os
import sys
import time

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules import DoubleFeatureTransformer
from model.modules.feature_transformer.functions import (
    SparseLinearFunction,
)


def SparseLinearFunctionEmulate(
    input_indices: torch.Tensor,
    input_values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    batch_size = input_indices.shape[0]
    num_inputs = weight.shape[0]
    max_active_indices = input_indices.shape[1]
    inputs = torch.zeros(
        batch_size, num_inputs, dtype=torch.float32, device=weight.device
    )
    for i in range(batch_size):
        for j in range(max_active_indices):
            feature = input_indices[i, j]
            value = input_values[i, j]
            if feature < 0:
                continue
            inputs[i, feature] += value

    return torch.mm(inputs, weight) + bias


def _run_test(device: torch.device):
    BATCH_SIZE = 16
    INPUT_SIZE = 10
    MAX_ACTIVE_FEATURES = 32
    STRIDE = 128
    MAX_ERROR = 1e-4

    torch.manual_seed(0)
    weight0 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
    bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
    torch.manual_seed(0)
    weight1 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
    bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
    indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(
        dtype=torch.int32
    )
    indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(
        dtype=torch.int32
    )
    values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
    values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

    output00 = SparseLinearFunctionEmulate(
        indices0.clone(), values0.clone(), weight0, bias0
    )
    output01 = SparseLinearFunctionEmulate(
        indices1.clone(), values1.clone(), weight0, bias0
    )
    output10 = SparseLinearFunction.apply(
        indices0.clone().to(device),
        values0.clone().to(device),
        weight1.to(device),
        bias1.to(device),
    )
    output11 = SparseLinearFunction.apply(
        indices1.clone().to(device),
        values1.clone().to(device),
        weight1.to(device),
        bias1.to(device),
    )

    assert torch.max(torch.abs(output00.cpu() - output10.cpu())) < MAX_ERROR
    assert torch.max(torch.abs(output01.cpu() - output11.cpu())) < MAX_ERROR
    (output00 - output01).sum().backward()
    (output10 - output11).sum().backward()
    assert torch.max(torch.abs(weight0.grad.cpu() - weight1.grad.cpu())) < MAX_ERROR
    assert torch.max(torch.abs(bias0.grad.cpu() - bias1.grad.cpu())) < MAX_ERROR
    print(f"Test passed on {device}.")


def _run_padding_test(device: torch.device):
    """Negative entries in feature_indices must be treated as padding."""
    BATCH_SIZE = 8
    INPUT_SIZE = 10
    N_ACTIVE = 5
    N_PADDING = 3
    STRIDE = 64
    MAX_ERROR = 1e-4

    torch.manual_seed(42)
    weight = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, device=device)
    bias = torch.rand(STRIDE, dtype=torch.float32, device=device)

    active_indices = (torch.rand(BATCH_SIZE, N_ACTIVE) * INPUT_SIZE).to(
        dtype=torch.int32, device=device
    )
    active_values = torch.rand(
        BATCH_SIZE, N_ACTIVE, dtype=torch.float32, device=device
    )
    padding_indices = torch.full(
        (BATCH_SIZE, N_PADDING), -1, dtype=torch.int32, device=device
    )
    # Non-zero values in padding slots must not affect the output.
    padding_values = torch.rand(
        BATCH_SIZE, N_PADDING, dtype=torch.float32, device=device
    )

    out_unpadded = SparseLinearFunction.apply(
        active_indices, active_values, weight, bias
    )
    indices_padded = torch.cat([active_indices, padding_indices], dim=1)
    values_padded = torch.cat([active_values, padding_values], dim=1)
    out_padded = SparseLinearFunction.apply(
        indices_padded, values_padded, weight, bias
    )

    assert torch.max(torch.abs(out_unpadded - out_padded)) < MAX_ERROR
    print(f"Padding test passed on {device}.")


def test_cpu():
    _run_test(torch.device("cpu"))
    _run_padding_test(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    _run_test(torch.device("cuda"))
    _run_padding_test(torch.device("cuda"))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_mps():
    _run_test(torch.device("mps"))
    _run_padding_test(torch.device("mps"))


def bench():
    INPUT_SIZE = 40960
    BATCH_SIZE = 8192
    ITERS = 64
    STRIDE = 264
    MAX_ACTIVE_FEATURES = 64

    def get_fake_indices():
        return torch.cat(
            [
                torch.sort(
                    (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4)) * INPUT_SIZE,
                    dim=1,
                )[0].to(dtype=torch.int32),
                torch.full(
                    (BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        ).cuda()

    layer = DoubleFeatureTransformer(INPUT_SIZE, STRIDE).cuda()
    indices0 = get_fake_indices()
    values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()
    indices1 = get_fake_indices()
    values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()

    start = time.time()

    for _ in range(ITERS):
        output0, output1 = layer(indices0, values0, indices1, values1)
        output0 = torch.clamp(output0, 0.0, 1.0)
        output1 = torch.clamp(output1, 0.0, 1.0)

        g = ((output0 - output1) ** 2).mean()
        g.backward()

        torch.cuda.synchronize()

    end = time.time()

    print("{} pos/s".format((ITERS * BATCH_SIZE) / (end - start)))


if __name__ == "__main__":
    _run_test(torch.device("cpu"))
    _run_padding_test(torch.device("cpu"))
    if torch.cuda.is_available():
        _run_test(torch.device("cuda"))
        _run_padding_test(torch.device("cuda"))
    if torch.backends.mps.is_available():
        _run_test(torch.device("mps"))
        _run_padding_test(torch.device("mps"))
    if torch.cuda.is_available():
        bench()
