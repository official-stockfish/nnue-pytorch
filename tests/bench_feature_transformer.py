import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.modules import (
    ComposedFeatures,
    DoubleFeatureTransformer,
    set_double_ft_impl,
)
from model.modules.feature_transformer.functions import SparseLinearFunction
from model.modules.features import get_feature_cls
from model.quantize import QuantizationManager


def run_bench():
    # Base Benchmark settings
    BATCH_SIZE = 8192
    WARMUP_ITERS = 10
    ITERS = 64
    STRIDE = 264

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Instantiate Features FIRST to determine true input bounds
    config = ModelConfig()
    quantization = QuantizationManager(config.quantize_config)
    feature_classes = get_feature_cls("HalfKAv2_hm^")

    features = ComposedFeatures(
        feature_classes, STRIDE, num_psqt_buckets=8, quantization=quantization
    )
    double_ft = DoubleFeatureTransformer(features).to(device)

    if hasattr(torch, "compile"):
        try:
            print("Compiling DoubleFeatureTransformer...")
            compiled_double_ft = torch.compile(double_ft)
        except Exception as e:
            print(f"Compilation failed or not supported: {e}")
            compiled_double_ft = double_ft
    else:
        compiled_double_ft = double_ft

    # Extract the exact hardware-allocated feature bounds
    ACTUAL_INPUT_SIZE = features.NUM_INPUTS
    ACTUAL_MAX_ACTIVE = features.MAX_ACTIVE_FEATURES
        
    print(f"Detected internal feature size limit: {ACTUAL_INPUT_SIZE}")
    print(f"Detected max active features: {ACTUAL_MAX_ACTIVE}")

    # 2) Generate indices strictly bounded to ACTUAL_INPUT_SIZE
    def get_fake_indices():
        random_indices = torch.randint(
            0, ACTUAL_INPUT_SIZE, 
            (BATCH_SIZE, ACTUAL_MAX_ACTIVE * 3 // 4), 
            dtype=torch.int32, 
            device=device
        )
        sorted_indices, _ = torch.sort(random_indices, dim=1)
        padding = torch.full(
            (BATCH_SIZE, ACTUAL_MAX_ACTIVE // 4), 
            -1, 
            dtype=torch.int32, 
            device=device
        )
        return torch.cat([sorted_indices, padding], dim=1)

    indices0 = get_fake_indices()
    indices1 = get_fake_indices()
    
    # 'us' and 'them' indicator floats
    us = torch.randint(0, 2, (BATCH_SIZE, 1), dtype=torch.float32, device=device)
    them = 1.0 - us
    
    piece_count = torch.randint(1, 32, (BATCH_SIZE,), dtype=torch.int64, device=device)
    psqt_indices = (piece_count - 1) // 4

    print("Benchmarking SparseLinearFunction...")
    weight = torch.randn(
        ACTUAL_INPUT_SIZE, STRIDE, dtype=torch.float32, device=device, requires_grad=True
    )
    bias = torch.randn(
        STRIDE, dtype=torch.float32, device=device, requires_grad=True
    )

    for _ in range(WARMUP_ITERS):
        output0 = SparseLinearFunction.apply(indices0, weight, bias)
        output0 = torch.clamp(output0, 0.0, 1.0)
        g = output0.mean()
        g.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(ITERS):
        output0 = SparseLinearFunction.apply(indices0, weight, bias)
        output0 = torch.clamp(output0, 0.0, 1.0)
        g = output0.mean()
        g.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    print(
        "Direct SparseLinearFunction: {} pos/s".format(
            (ITERS * BATCH_SIZE) / (end - start)
        )
    )

    for mode in ["torch", "sparse", "fused"]:
        set_double_ft_impl(mode)
        mode_str = mode.capitalize()
        print(f"Benchmarking DoubleFeatureTransformer ({mode_str})...")

        try:
            for p in compiled_double_ft.parameters():
                if p.grad is not None:
                    p.grad.zero_()

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
        except RuntimeError as e:
            print(f"Error during {mode_str}: {e}")
            continue

        start = time.time()
        for _ in range(ITERS):
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
