"""Aggregation parity with unique row indices, cross-row overlap and streams."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required",
)
@pytest.mark.parametrize("width", [1024, 1152, 1280])
@pytest.mark.parametrize("batch,active", [(1, 33), (17, 288), (1025, 33)])
@pytest.mark.parametrize("kind", ["empty", "unique", "overlap"])
@pytest.mark.parametrize("separate_stream", [False, True])
def test_aggregated_ft(width, batch, active, kind, separate_stream):
    if torch.version.hip is not None or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 specialization")
    from model.modules.feature_transformer.aggregated_ft_kernel import (
        aggregated_ft_backward,
    )

    torch.manual_seed(718)
    device = torch.device("cuda")
    stream = torch.cuda.Stream() if separate_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Delay input production so a launch on the default stream races it.
        if separate_stream:
            torch.cuda._sleep(2_000_000)
        us, them = torch.randn(batch, 1, 2, device=device).unbind(-1)
        us, them = us.contiguous(), them.contiguous()
        clamped = torch.rand(batch, 4, width // 2, device=device)
        clamped[clamped < 0.2] = 0
        clamped[clamped > 0.8] = 1
        grad = torch.randn(batch, width, device=device)
        white = torch.full((batch, active), -1, dtype=torch.int32, device=device)
        black = torch.full_like(white, -1)
        if kind != "empty":
            for row in range(batch):
                n = min(31, row % 32)
                if kind == "overlap":
                    # Repeat IDs across positions/perspectives, never within a row.
                    white[row, :n] = torch.arange(n, device=device)
                    black[row, : n // 2] = torch.arange(n // 2, device=device)
                else:
                    white[row, :n] = torch.randperm(64, device=device)[:n]
                    black[row, : n // 2] = torch.randperm(64, device=device)[: n // 2]
            # A full row exercises compaction capacity and no padding.
            white[-1] = torch.arange(active, device=device)

        w0, w1, b0, b1 = clamped.unbind(1)
        d0, d1 = grad.chunk(2, dim=1)
        dw0 = d0 * w1 * ((w0 > 0) & (w0 < 1))
        dw1 = d0 * w0 * ((w1 > 0) & (w1 < 1))
        db0 = d1 * b1 * ((b0 > 0) & (b0 < 1))
        db1 = d1 * b0 * ((b1 > 0) & (b1 < 1))
        gw0, gw1 = us * dw0 + them * db0, us * dw1 + them * db1
        gb0, gb1 = them * dw0 + us * db0, them * dw1 + us * db1
        expected_weight = torch.zeros(512, width, device=device)
        for indices, value in [
            (white, torch.cat((gw0, gw1), dim=1)),
            (black, torch.cat((gb0, gb1), dim=1)),
        ]:
            rows, cols = (indices >= 0).nonzero(as_tuple=True)
            expected_weight.index_add_(0, indices[rows, cols].long(), value[rows])
        expected_bias = torch.cat((gw0 + gb0, gw1 + gb1), dim=1).sum(0)
        actual_weight = torch.zeros_like(expected_weight)
        actual_bias = torch.zeros_like(expected_bias)
        aggregated_ft_backward(
            us, them, white, black, grad, clamped, actual_weight, actual_bias, 1.0
        )
        torch.testing.assert_close(actual_weight, expected_weight, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(actual_bias, expected_bias, atol=3e-4, rtol=3e-4)
    stream.synchronize()
