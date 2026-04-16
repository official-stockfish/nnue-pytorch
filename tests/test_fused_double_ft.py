import os
import sys
import torch

# Standard path alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.fused_functions import (
    fused_double_ft_op,
    set_use_fused_double_ft,
)

def test_fused_consistency():
    print(">>> Running Mathematical Consistency Test...")

    BATCH_SIZE = 16
    INPUT_SIZE = 256
    MAX_ACTIVE_FEATURES = 32
    L1 = 128
    EXTRA = 8
    FT_MAX_VAL = 0.8
    MAX_ERROR = 1e-5

    device = torch.device('cuda')

    # Force pure FP32 to ensure bit-level consistency between custom CUDA and cuBLAS
    torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(42)
    weight_base = torch.randn(INPUT_SIZE, L1 + EXTRA, device=device) * 0.1
    bias_base = torch.randn(L1 + EXTRA, device=device) * 0.1

    # We use clone().requires_grad_() to get separate grad buffers for the same weights
    weight0 = weight_base.clone().requires_grad_(True)
    bias0 = bias_base.clone().requires_grad_(True)

    weight1 = weight_base.clone().requires_grad_(True)
    bias1 = bias_base.clone().requires_grad_(True)

    # Using unique indices per row to avoid any summation-order divergence
    w_indices = torch.argsort(torch.rand(BATCH_SIZE, INPUT_SIZE, device=device), dim=1)[:, :MAX_ACTIVE_FEATURES].to(torch.int32)
    b_indices = torch.argsort(torch.rand(BATCH_SIZE, INPUT_SIZE, device=device), dim=1)[:, :MAX_ACTIVE_FEATURES].to(torch.int32)
    w_values = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, device=device)
    b_values = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, device=device)

    us = torch.rand(BATCH_SIZE, device=device)
    them = torch.rand(BATCH_SIZE, device=device)

    # Run Unfused Mode
    set_use_fused_double_ft(False)
    out_l0_unfused, wpsqt_unfused, bpsqt_unfused = fused_double_ft_op(
        w_indices, w_values, b_indices, b_values,
        weight0, bias0, us, them, FT_MAX_VAL, L1, EXTRA
    )

    # Create a grad signal that tests the CReLU mask and perspective mixing
    loss_unfused = (out_l0_unfused.pow(2).sum() + wpsqt_unfused.sum() + bpsqt_unfused.sum())
    loss_unfused.backward()

    # Run Fused Mode
    set_use_fused_double_ft(True)
    out_l0_fused, wpsqt_fused, bpsqt_fused = fused_double_ft_op(
        w_indices, w_values, b_indices, b_values,
        weight1, bias1, us, them, FT_MAX_VAL, L1, EXTRA
    )

    loss_fused = (out_l0_fused.pow(2).sum() + wpsqt_fused.sum() + bpsqt_fused.sum())
    loss_fused.backward()

    # 4. Consistency Verification
    results = [
        ("Forward L1", torch.max(torch.abs(out_l0_unfused - out_l0_fused))),
        ("Forward WPSQT", torch.max(torch.abs(wpsqt_unfused - wpsqt_fused))),
        ("Forward BPSQT", torch.max(torch.abs(bpsqt_unfused - bpsqt_fused))),
        ("Weight Grad", torch.max(torch.abs(weight0.grad - weight1.grad))),
        ("Bias Grad", torch.max(torch.abs(bias0.grad - bias1.grad))),
    ]

    print(f"{'Check':<15} | {'Max Difference':<20} | {'Status'}")
    print("-" * 50)

    all_passed = True
    for name, diff in results:
        status = "PASSED" if diff < MAX_ERROR else "FAILED"
        if status == "FAILED": all_passed = False
        print(f"{name:<15} | {diff.item():<20.2e} | {status}")

    if all_passed:
        print("\nConsistency verified: Fused and Unfused modes are mathematically identical.")
    else:
        print("\nConsistency Check FAILED. Verify custom kernel rematerialization logic.")

    return all_passed

def bench_fused_double_ft():
    print("\n>>> Running Performance Benchmark for Training Step...")

    # Realistic training dimensions
    INPUT_SIZE = 40960
    BATCH_SIZE = 16384
    ITERS = 100
    L1 = 512
    EXTRA = 32
    MAX_ACTIVE = 64
    FT_MAX_VAL = 1.0
    device = torch.device('cuda')

    # Data gen (simulating sparse input with padding)
    def get_indices():
        return torch.cat([
            torch.sort(torch.rand(BATCH_SIZE, 48, device=device), dim=1)[0].mul(INPUT_SIZE).to(torch.int32),
            torch.full((BATCH_SIZE, 16), -1, dtype=torch.int32, device=device)
        ], dim=1)

    w_idx = get_indices()
    b_idx = get_indices()
    w_val = torch.rand(BATCH_SIZE, MAX_ACTIVE, device=device)
    b_val = torch.rand(BATCH_SIZE, MAX_ACTIVE, device=device)
    us = torch.rand(BATCH_SIZE, device=device)
    them = torch.rand(BATCH_SIZE, device=device)

    weight = torch.rand(INPUT_SIZE, L1 + EXTRA, device=device, requires_grad=True)
    bias = torch.rand(L1 + EXTRA, device=device, requires_grad=True)

    scores = {}

    for mode in [False, True]:
        set_use_fused_double_ft(mode)
        mode_str = "FUSED" if mode else "UNFUSED"

        # Warmup (triggering JIT/Cupy compilation)
        for _ in range(10):
            o1, o2, o3 = fused_double_ft_op(w_idx, w_val, b_idx, b_val, weight, bias, us, them, FT_MAX_VAL, L1, EXTRA)
            (o1.sum() + o2.sum() + o3.sum()).backward()
            weight.grad.zero_(); bias.grad.zero_()

        # Benchmark using CUDA Events for precision
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(ITERS):
            o1, o2, o3 = fused_double_ft_op(w_idx, w_val, b_idx, b_val, weight, bias, us, them, FT_MAX_VAL, L1, EXTRA)
            loss = (o1.sum() + o2.sum() + o3.sum())
            loss.backward()
            weight.grad.zero_(); bias.grad.zero_()
        end_event.record()

        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)

        pos_per_sec = (ITERS * BATCH_SIZE) / (time_ms / 1000.0)
        scores[mode_str] = pos_per_sec
        print(f"{mode_str:<8}: {pos_per_sec:,.0f} pos/s")

    speedup = scores["FUSED"] / scores["UNFUSED"]
    print(f"\nOptimization Result: {speedup:.2f}x Speedup")


if __name__ == "__main__":
    if test_fused_consistency():
        bench_fused_double_ft()
