import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.modules import (
    ComposedFeatures,
    DoubleFeatureTransformer,
    get_use_fused_double_ft,
    set_use_fused_double_ft,
)
from model.modules.feature_transformer.functions import SparseLinearFunction
from model.modules.features import get_feature_cls
from model.quantize import QuantizationManager


def run_bench():
    # Benchmark settings
    INPUT_SIZE = 40960
    BATCH_SIZE = 8192
    WARMUP_ITERS = 10
    ITERS = 64
    STRIDE = 264
    MAX_ACTIVE_FEATURES = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Helper to generate fake indices
    def get_fake_indices():
        return torch.cat(
            [
                torch.sort(
                    (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4))
                    * INPUT_SIZE,
                    dim=1,
                )[0].to(dtype=torch.int32),
                torch.full(
                    (BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        ).to(device)

    indices0 = get_fake_indices()
    indices1 = get_fake_indices()
    us = torch.randn(BATCH_SIZE, 1, dtype=torch.float32, device=device)
    them = torch.randn(BATCH_SIZE, 1, dtype=torch.float32, device=device)
    piece_count = torch.randint(1, 32, (BATCH_SIZE,), dtype=torch.int32, device=device)
    psqt_indices = (piece_count - 1) // 4

    # 1) Benchmark: Direct SparseLinearFunction
    print("Benchmarking direct SparseLinearFunction...")
    weight = torch.randn(
        INPUT_SIZE, STRIDE, dtype=torch.float32, device=device, requires_grad=True
    )
    bias = torch.randn(
        STRIDE, dtype=torch.float32, device=device, requires_grad=True
    )

    # Warmup
    for _ in range(WARMUP_ITERS):
        output0 = SparseLinearFunction.apply(indices0, weight, bias)
        output1 = SparseLinearFunction.apply(indices1, weight, bias)
        g = ((output0 - output1) ** 2).mean()
        g.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(ITERS):
        output0 = SparseLinearFunction.apply(indices0, weight, bias)
        output1 = SparseLinearFunction.apply(indices1, weight, bias)
        output0 = torch.clamp(output0, 0.0, 1.0)
        output1 = torch.clamp(output1, 0.0, 1.0)

        g = ((output0 - output1) ** 2).mean()
        g.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    print(
        "Direct SparseLinearFunction: {} pos/s".format(
            (ITERS * BATCH_SIZE) / (end - start)
        )
    )

    # 2) Benchmark: DoubleFeatureTransformer (Fused vs Fallback)
    config = ModelConfig()
    quantization = QuantizationManager(config.quantize_config)
    feature_classes = get_feature_cls("HalfKAv2_hm^")

    # Instantiate ComposedFeatures and wrap with DoubleFeatureTransformer
    features = ComposedFeatures(
        feature_classes, STRIDE, num_psqt_buckets=8, quantization=quantization
    )
    double_ft = DoubleFeatureTransformer(features).to(device)

    # PyTorch Compile option if supported/requested
    if hasattr(torch, "compile"):
        try:
            print("Compiling DoubleFeatureTransformer...")
            compiled_double_ft = torch.compile(double_ft)
        except Exception as e:
            print(f"Compilation failed or not supported: {e}")
            compiled_double_ft = double_ft
    else:
        compiled_double_ft = double_ft

    for use_fused in [False, True]:
        set_use_fused_double_ft(use_fused)
        mode_str = "Fused" if use_fused else "Fallback"
        print(f"Benchmarking DoubleFeatureTransformer ({mode_str})...")

        # Reset gradients
        for p in compiled_double_ft.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Warmup
        for _ in range(WARMUP_ITERS):
            l0_, wpsqt, bpsqt = compiled_double_ft(
                us,
                them,
                indices0,
                indices1,
                psqt_indices,
                fake_quantize_acts=True,
                fake_quantize_weights=True,
            )
            g = l0_.sum() + wpsqt.sum() + bpsqt.sum()
            g.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(ITERS):
            # Zero grad in loop like standard training step
            for p in compiled_double_ft.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            l0_, wpsqt, bpsqt = compiled_double_ft(
                us,
                them,
                indices0,
                indices1,
                psqt_indices,
                fake_quantize_acts=True,
                fake_quantize_weights=True,
            )
            g = l0_.sum() + wpsqt.sum() + bpsqt.sum()
            g.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"DoubleFeatureTransformer ({mode_str}): { (ITERS * BATCH_SIZE) / (end - start) } pos/s"
        )


if __name__ == "__main__":
    run_bench()
