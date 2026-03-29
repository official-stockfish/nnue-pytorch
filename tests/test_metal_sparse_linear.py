"""Tests for the custom Metal sparse linear kernel.

Verifies forward and backward correctness against a pure-Python reference,
and benchmarks throughput vs the PyTorch embedding_bag fallback.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.functions import _torch_sparse_linear


def _reference_sparse_linear(indices, values, weight, bias):
    """Naive reference: materialise full dense input then matmul."""
    batch_size = indices.shape[0]
    num_inputs = weight.shape[0]
    max_active = indices.shape[1]
    dense = torch.zeros(batch_size, num_inputs, dtype=torch.float32, device=weight.device)
    for b in range(batch_size):
        for k in range(max_active):
            idx = indices[b, k].item()
            if idx == -1:
                break
            dense[b, idx] += values[b, k].item()
    return dense @ weight + bias


def test_forward():
    """Metal forward matches the naive reference within tolerance."""
    from model.modules.feature_transformer.metal import (
        MetalSparseLinearFunction,
        is_available,
    )
    assert is_available(), "Metal extension not available"

    BATCH = 32
    NUM_INPUTS = 64
    MAX_ACTIVE = 16
    OUTPUT_SIZE = 128
    MAX_ERR = 1e-4

    torch.manual_seed(42)
    weight_cpu = torch.rand(NUM_INPUTS, OUTPUT_SIZE, dtype=torch.float32)
    bias_cpu = torch.rand(OUTPUT_SIZE, dtype=torch.float32)
    indices_cpu = (torch.rand(BATCH, MAX_ACTIVE) * NUM_INPUTS).to(torch.int32)
    values_cpu = torch.rand(BATCH, MAX_ACTIVE, dtype=torch.float32)

    ref = _reference_sparse_linear(indices_cpu, values_cpu, weight_cpu, bias_cpu)

    weight_mps = weight_cpu.clone().to("mps")
    bias_mps = bias_cpu.clone().to("mps")
    indices_mps = indices_cpu.clone().to("mps")
    values_mps = values_cpu.clone().to("mps")

    metal_out = MetalSparseLinearFunction.apply(
        indices_mps, values_mps, weight_mps, bias_mps
    )
    err = torch.max(torch.abs(ref - metal_out.cpu())).item()
    assert err < MAX_ERR, f"Forward max error {err:.6e} exceeds {MAX_ERR}"
    print(f"  forward  OK  (max err = {err:.2e})")


def test_forward_with_sentinels():
    """Forward works when some slots contain the -1 sentinel."""
    from model.modules.feature_transformer.metal import (
        MetalSparseLinearFunction,
        is_available,
    )
    assert is_available()

    BATCH = 16
    NUM_INPUTS = 32
    MAX_ACTIVE = 32
    OUTPUT_SIZE = 64

    torch.manual_seed(7)
    weight_cpu = torch.rand(NUM_INPUTS, OUTPUT_SIZE, dtype=torch.float32)
    bias_cpu = torch.rand(OUTPUT_SIZE, dtype=torch.float32)

    indices_cpu = torch.cat([
        (torch.rand(BATCH, MAX_ACTIVE * 3 // 4) * NUM_INPUTS).to(torch.int32),
        torch.full((BATCH, MAX_ACTIVE // 4), -1, dtype=torch.int32),
    ], dim=1)
    values_cpu = torch.rand(BATCH, MAX_ACTIVE, dtype=torch.float32)

    ref = _reference_sparse_linear(indices_cpu, values_cpu, weight_cpu, bias_cpu)
    metal_out = MetalSparseLinearFunction.apply(
        indices_cpu.to("mps"), values_cpu.to("mps"),
        weight_cpu.to("mps"), bias_cpu.to("mps"),
    )
    err = torch.max(torch.abs(ref - metal_out.cpu())).item()
    assert err < 1e-4, f"Sentinel forward err {err:.6e}"
    print(f"  sentinel OK  (max err = {err:.2e})")


def test_backward():
    """Metal backward produces correct weight and bias gradients."""
    from model.modules.feature_transformer.metal import (
        MetalSparseLinearFunction,
        is_available,
    )
    assert is_available()

    BATCH = 16
    NUM_INPUTS = 32
    MAX_ACTIVE = 16
    OUTPUT_SIZE = 64
    MAX_ERR = 1e-3

    torch.manual_seed(99)
    weight_ref = torch.rand(NUM_INPUTS, OUTPUT_SIZE, dtype=torch.float32, requires_grad=True)
    bias_ref = torch.rand(OUTPUT_SIZE, dtype=torch.float32, requires_grad=True)
    weight_mps = weight_ref.detach().clone().to("mps").requires_grad_(True)
    bias_mps = bias_ref.detach().clone().to("mps").requires_grad_(True)

    indices = (torch.rand(BATCH, MAX_ACTIVE) * NUM_INPUTS).to(torch.int32)
    values = torch.rand(BATCH, MAX_ACTIVE, dtype=torch.float32)

    out_ref = _torch_sparse_linear(indices, values, weight_ref, bias_ref)
    out_mps = MetalSparseLinearFunction.apply(
        indices.to("mps"), values.to("mps"), weight_mps, bias_mps,
    )

    loss_ref = out_ref.sum()
    loss_mps = out_mps.sum()
    loss_ref.backward()
    loss_mps.backward()

    w_err = torch.max(torch.abs(weight_ref.grad - weight_mps.grad.cpu())).item()
    b_err = torch.max(torch.abs(bias_ref.grad - bias_mps.grad.cpu())).item()
    assert w_err < MAX_ERR, f"weight_grad max err {w_err:.6e}"
    assert b_err < MAX_ERR, f"bias_grad max err {b_err:.6e}"
    print(f"  backward OK  (w_err = {w_err:.2e}, b_err = {b_err:.2e})")


def test_matches_embedding_bag():
    """Metal forward matches the PyTorch embedding_bag fallback path."""
    from model.modules.feature_transformer.metal import (
        MetalSparseLinearFunction,
        is_available,
    )
    assert is_available()

    BATCH = 64
    NUM_INPUTS = 256
    MAX_ACTIVE = 32
    OUTPUT_SIZE = 128

    torch.manual_seed(0)
    weight = torch.rand(NUM_INPUTS, OUTPUT_SIZE, dtype=torch.float32)
    bias = torch.rand(OUTPUT_SIZE, dtype=torch.float32)
    indices = (torch.rand(BATCH, MAX_ACTIVE) * NUM_INPUTS).to(torch.int32)
    values = torch.rand(BATCH, MAX_ACTIVE, dtype=torch.float32)

    eb_out = _torch_sparse_linear(indices, values, weight, bias)
    metal_out = MetalSparseLinearFunction.apply(
        indices.to("mps"), values.to("mps"),
        weight.to("mps"), bias.to("mps"),
    )
    err = torch.max(torch.abs(eb_out - metal_out.cpu())).item()
    assert err < 1e-4, f"embedding_bag vs Metal err {err:.6e}"
    print(f"  vs emb_bag OK (max err = {err:.2e})")


def bench():
    """Throughput comparison: Metal kernel vs embedding_bag on MPS."""
    from model.modules.feature_transformer.metal import (
        MetalSparseLinearFunction,
        is_available,
    )
    if not is_available():
        print("  Metal extension not available, skipping bench")
        return

    BATCH = 8192
    NUM_INPUTS = 24576
    MAX_ACTIVE = 32
    OUTPUT_SIZE = 1032
    WARMUP = 4
    ITERS = 16

    torch.manual_seed(0)
    weight = torch.rand(NUM_INPUTS, OUTPUT_SIZE, dtype=torch.float32, device="mps")
    bias = torch.rand(OUTPUT_SIZE, dtype=torch.float32, device="mps")
    indices = torch.cat([
        (torch.rand(BATCH, MAX_ACTIVE * 3 // 4) * NUM_INPUTS).to(torch.int32),
        torch.full((BATCH, MAX_ACTIVE // 4), -1, dtype=torch.int32),
    ], dim=1).to("mps")
    values = torch.rand(BATCH, MAX_ACTIVE, dtype=torch.float32, device="mps")

    for label, fn in [
        ("embedding_bag", lambda: _torch_sparse_linear(indices, values, weight, bias)),
        ("Metal kernel",  lambda: MetalSparseLinearFunction.apply(indices, values, weight, bias)),
    ]:
        for _ in range(WARMUP):
            fn()
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  {label:15s}: {ITERS / elapsed:8.1f} fwd/s  "
              f"({elapsed / ITERS * 1000:6.2f} ms/fwd)")


if __name__ == "__main__":
    print("Metal sparse linear tests")
    print("-" * 40)

    test_forward()
    test_forward_with_sentinels()
    test_backward()
    test_matches_embedding_bag()

    print()
    print("Benchmark (forward only)")
    print("-" * 40)
    bench()

    print()
    print("All tests passed.")
