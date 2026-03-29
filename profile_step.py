"""Profile each phase of a training step on MPS.

Usage:
    python profile_step.py test79-2022-05-may-12tb7p.min-v2.binpack \
        --features='Full_Threats+HalfKAv2_hm^' --batch-size=16384 \
        --warmup=10 --steps=50
"""
import argparse
import os
import time
from collections import defaultdict

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import data_loader
import model as M


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--features", default="Full_Threats+HalfKAv2_hm^")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--l1", type=int, default=1024)
    parser.add_argument("--l2", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps")

    cfg = M.ModelConfig(L1=args.l1, L2=args.l2)
    nnue = M.NNUEModel(args.features, cfg, M.QuantizationConfig())
    nnue.to(device)
    nnue.train()

    print(f"Model params: {sum(p.numel() for p in nnue.parameters()):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Features: {args.features}")
    print(f"NUM_INPUTS: {nnue.input.NUM_INPUTS}")
    print(f"MAX_ACTIVE: {nnue.input.MAX_ACTIVE_FEATURES}")
    print()

    ds = data_loader.SparseBatchDataset(
        nnue.input_feature_name,
        [args.dataset],
        args.batch_size,
        num_workers=1,
    )
    data_iter = iter(ds)

    optimizer = torch.optim.AdamW(nnue.parameters(), lr=1e-3)

    timings = defaultdict(list)

    total_steps = args.warmup + args.steps
    for step in range(total_steps):
        recording = step >= args.warmup

        # --- Phase 1: Data loading (CPU) ---
        t0 = time.monotonic()
        batch = next(data_iter)
        t_data_cpu = time.monotonic() - t0

        # --- Phase 2: CPU -> MPS transfer ---
        t0 = time.monotonic()
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        sync()
        t_transfer = time.monotonic() - t0

        us, them, w_idx, w_val, b_idx, b_val, outcome, score, psqt_idx, ls_idx = batch

        # --- Phase 3: Weight clipping ---
        sync()
        t0 = time.monotonic()
        nnue.clip_weights()
        sync()
        t_clip = time.monotonic() - t0

        # --- Phase 4: Model forward ---
        sync()
        t0 = time.monotonic()
        x = nnue(us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx)
        sync()
        t_forward = time.monotonic() - t0

        # --- Phase 5: Loss computation ---
        sync()
        t0 = time.monotonic()
        scorenet = x * nnue.quantization.nnue2score
        q = (scorenet - 270) / 340
        qm = (-scorenet - 270) / 340
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())
        s = (score - 270) / 380
        sm = (-score - 270) / 380
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())
        pt = pf * 1.0 + outcome * 0.0
        loss = torch.pow(torch.abs(pt - qf), 2.5)
        weights = 1 + 9.24 * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), 0.7)
        loss = (loss * weights).sum() / weights.sum()
        sync()
        t_loss = time.monotonic() - t0

        # --- Phase 6: Backward ---
        sync()
        t0 = time.monotonic()
        optimizer.zero_grad()
        loss.backward()
        sync()
        t_backward = time.monotonic() - t0

        # --- Phase 7: Optimizer step ---
        sync()
        t0 = time.monotonic()
        optimizer.step()
        sync()
        t_optim = time.monotonic() - t0

        # --- Phase 8: Total step time ---
        t_total = t_data_cpu + t_transfer + t_clip + t_forward + t_loss + t_backward + t_optim

        if recording:
            timings["data_cpu"].append(t_data_cpu)
            timings["transfer"].append(t_transfer)
            timings["clip"].append(t_clip)
            timings["forward"].append(t_forward)
            timings["loss"].append(t_loss)
            timings["backward"].append(t_backward)
            timings["optimizer"].append(t_optim)
            timings["total"].append(t_total)

        if step < 3 or (recording and (step - args.warmup) % 10 == 0):
            print(
                f"step {step:>3d}: "
                f"data={t_data_cpu*1e3:6.1f}ms  "
                f"xfer={t_transfer*1e3:5.1f}ms  "
                f"clip={t_clip*1e3:5.1f}ms  "
                f"fwd={t_forward*1e3:6.1f}ms  "
                f"loss={t_loss*1e3:5.1f}ms  "
                f"bwd={t_backward*1e3:6.1f}ms  "
                f"opt={t_optim*1e3:5.1f}ms  "
                f"total={t_total*1e3:6.1f}ms  "
                f"loss_val={loss.item():.6f}",
                flush=True,
            )

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY (median of {} steps)".format(args.steps))
    print("=" * 80)

    total_median = 0.0
    for phase in ["data_cpu", "transfer", "clip", "forward", "loss", "backward", "optimizer"]:
        vals = sorted(timings[phase])
        med = vals[len(vals) // 2]
        mn = vals[0]
        mx = vals[-1]
        avg = sum(vals) / len(vals)
        total_median += med
        pct = med / (sorted(timings["total"])[len(timings["total"]) // 2]) * 100
        print(
            f"  {phase:<12s}: median={med*1e3:7.2f}ms  "
            f"avg={avg*1e3:7.2f}ms  "
            f"min={mn*1e3:7.2f}ms  "
            f"max={mx*1e3:7.2f}ms  "
            f"({pct:5.1f}%)"
        )

    total_vals = sorted(timings["total"])
    total_med = total_vals[len(total_vals) // 2]
    print(f"\n  {'TOTAL':<12s}: median={total_med*1e3:7.2f}ms")
    print(f"  Throughput : {args.batch_size / total_med:,.0f} pos/s")
    print()


if __name__ == "__main__":
    main()
