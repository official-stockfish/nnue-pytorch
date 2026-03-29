"""Training speed benchmark for MPS.

Measures wall-clock time per training step and per-phase (forward, loss,
backward, optimizer) using the full model and real data.  Uses the actual
optimizer wrapper from the codebase so results reflect each commit's
configuration.

Usage:
    python training_speedtest.py test79-2022-05-may-12tb7p.min-v2.binpack
"""
import gc
import json
import os
import sys
import time
from collections import defaultdict

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import data_loader
import model as M


BATCH_SIZE = 65536
NUM_WORKERS = 8
WARMUP = 5
STEPS = 30
L1, L2 = 1024, 31
FEATURES = "Full_Threats+HalfKAv2_hm^"
NUM_BATCHES_PER_EPOCH = 100
MAX_EPOCH = 800


def sync():
    torch.mps.synchronize()


def compute_loss_pytorch(scorenet, score, outcome):
    in_offset, in_scaling = 270.0, 340.0
    out_offset, out_scaling = 270.0, 380.0
    pow_exp = 2.5
    w1, w2 = 3.3553547771220007, 0.7006821612968052

    q = (scorenet - in_offset) / in_scaling
    qm = (-scorenet - in_offset) / in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    s = (score - out_offset) / out_scaling
    sm = (-score - out_offset) / out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

    pt = pf * 1.0 + outcome * 0.0
    loss = torch.pow(torch.abs(pt - qf), pow_exp)
    weights = 1 + (2.0**w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), w2)
    return (loss * weights).sum() / weights.sum()


def compute_loss_metal(scorenet, score, outcome):
    from model.modules.feature_transformer.metal import metal_fused_loss

    return metal_fused_loss(
        scorenet, score, outcome,
        270.0, 340.0, 270.0, 380.0,
        1.0, 2.5, 0.0, 3.3553547771220007, 0.7006821612968052,
    )


def make_optimizer(nnue_model):
    """Use the actual codebase wrapper so each commit's config is reflected."""
    from model.optimizers.ranger21_wrapper import Ranger21Wrapper, Ranger21Config

    train_params = [
        {"params": list(nnue_model.input.parameters()), "lr": 1.0},
        {"params": list(nnue_model.layer_stacks.parameters()), "lr": 1.0},
    ]

    wrapper = Ranger21Wrapper(
        Ranger21Config(gamma=0.995),
        max_epoch=MAX_EPOCH,
        num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
    )

    try:
        optimizers, _ = wrapper.configure_optimizers(train_params)
        return optimizers[0]
    except Exception as e:
        print(f"  WARNING: Wrapper optimizer failed ({e}), retrying without momentum_type...",
              flush=True)
        import ranger21 as r21
        opt = r21.Ranger21(
            train_params, lr=1.0, betas=(0.9, 0.999), eps=1e-7,
            using_gc=False, using_normgc=False, weight_decay=0.0,
            num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
            num_epochs=MAX_EPOCH, warmdown_active=False,
            use_warmup=False, use_adaptive_gradient_clipping=False,
            softplus=False, pnm_momentum_factor=0.0,
            lookahead_active=False, normloss_active=False,
            logging_active=False,
        )
        return opt


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "test79-2022-05-may-12tb7p.min-v2.binpack"
    json_out = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(dataset):
        print(f"Error: dataset '{dataset}' not found", file=sys.stderr)
        sys.exit(1)

    device = torch.device("mps")

    cfg = M.ModelConfig(L1=L1, L2=L2)
    nnue = M.NNUEModel(FEATURES, cfg, M.QuantizationConfig())
    nnue.to(device)
    nnue.train()
    nnue2score = nnue.quantization.nnue2score

    has_metal_loss = False
    try:
        from model.modules.feature_transformer.metal import is_available, metal_fused_loss
        has_metal_loss = is_available()
    except (ImportError, ModuleNotFoundError):
        pass

    optimizer = make_optimizer(nnue)
    opt_name = type(optimizer).__name__
    loss_fn_name = "Metal" if has_metal_loss else "PyTorch"

    print("=" * 70, flush=True)
    print("MPS Training Speed Benchmark", flush=True)
    print("=" * 70, flush=True)
    print(f"  Batch size:  {BATCH_SIZE:,}", flush=True)
    print(f"  Warmup:      {WARMUP} steps", flush=True)
    print(f"  Measured:    {STEPS} steps", flush=True)
    print(f"  Optimizer:   {opt_name}", flush=True)
    print(f"  Loss:        {loss_fn_name}", flush=True)
    print(f"  Model:       L1={L1}, L2={L2}", flush=True)
    print(f"  Features:    {FEATURES}", flush=True)
    print(f"  Params:      {sum(p.numel() for p in nnue.parameters()):,}", flush=True)
    print(flush=True)

    ds = data_loader.SparseBatchDataset(
        nnue.input_feature_name, [dataset], BATCH_SIZE,
        num_workers=NUM_WORKERS, device="mps",
    )
    data_iter = iter(ds)

    num_batches = WARMUP + STEPS
    print(f"Loading {num_batches} batches...", flush=True)
    batches = []
    load_times = []
    for i in range(num_batches):
        sync()
        t0 = time.monotonic()
        batches.append(next(data_iter))
        sync()
        load_times.append(time.monotonic() - t0)
    load_med = sorted(load_times)[len(load_times) // 2]
    print(f"Loaded {len(batches)} batches (median load: {load_med*1e3:.1f}ms).\n",
          flush=True)

    gc.collect()
    gc.disable()

    timings = defaultdict(list)

    for i in range(WARMUP + STEPS):
        recording = i >= WARMUP
        batch = batches[i]
        us, them, w_idx, w_val, b_idx, b_val, outcome, score, psqt_idx, ls_idx = batch

        sync()
        t_start = time.monotonic()

        t0 = time.monotonic()
        if i % 10 == 0:
            nnue.clip_weights()
        sync()
        t_clip = time.monotonic() - t0

        t0 = time.monotonic()
        scorenet = nnue(us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx) * nnue2score
        sync()
        t_fwd = time.monotonic() - t0

        t0 = time.monotonic()
        if has_metal_loss:
            loss = compute_loss_metal(scorenet, score, outcome)
        else:
            loss = compute_loss_pytorch(scorenet, score, outcome)
        sync()
        t_loss = time.monotonic() - t0

        t0 = time.monotonic()
        optimizer.zero_grad()
        loss.backward()
        sync()
        t_bwd = time.monotonic() - t0

        t0 = time.monotonic()
        optimizer.step()
        sync()
        t_opt = time.monotonic() - t0

        t_total = time.monotonic() - t_start

        if recording:
            timings["clip"].append(t_clip)
            timings["forward"].append(t_fwd)
            timings["loss"].append(t_loss)
            timings["backward"].append(t_bwd)
            timings["optimizer"].append(t_opt)
            timings["total"].append(t_total)

        if i < 3 or (recording and (i - WARMUP) % 10 == 0):
            print(
                f"  step {i:>3d}: "
                f"clip={t_clip*1e3:5.1f}ms  "
                f"fwd={t_fwd*1e3:6.1f}ms  "
                f"loss={t_loss*1e3:5.1f}ms  "
                f"bwd={t_bwd*1e3:6.1f}ms  "
                f"opt={t_opt*1e3:5.1f}ms  "
                f"total={t_total*1e3:6.1f}ms",
                flush=True,
            )

    gc.enable()

    results = {}
    print(flush=True)
    print("-" * 70, flush=True)
    print(f"  PHASE BREAKDOWN  (median of {STEPS} steps)", flush=True)
    print("-" * 70, flush=True)

    total_vals = sorted(timings["total"])
    total_med = total_vals[len(total_vals) // 2]

    for phase in ["clip", "forward", "loss", "backward", "optimizer"]:
        vals = sorted(timings[phase])
        med = vals[len(vals) // 2]
        avg = sum(vals) / len(vals)
        mn, mx = vals[0], vals[-1]
        pct = med / total_med * 100 if total_med > 0 else 0
        results[phase] = {"median": round(med * 1e3, 2), "avg": round(avg * 1e3, 2),
                          "min": round(mn * 1e3, 2), "max": round(mx * 1e3, 2)}
        print(
            f"  {phase:<12s}: median={med*1e3:7.2f}ms  "
            f"avg={avg*1e3:7.2f}ms  "
            f"min={mn*1e3:7.2f}ms  max={mx*1e3:7.2f}ms  "
            f"({pct:5.1f}%)",
            flush=True,
        )

    avg_total = sum(timings["total"]) / len(timings["total"])
    results["total"] = {"median": round(total_med * 1e3, 2), "avg": round(avg_total * 1e3, 2)}
    results["throughput"] = round(BATCH_SIZE / total_med)
    results["data_load"] = {"median": round(load_med * 1e3, 2)}
    results["optimizer_name"] = opt_name
    results["loss_name"] = loss_fn_name

    print(flush=True)
    print(f"  {'TOTAL':<12s}: median={total_med*1e3:7.2f}ms  avg={avg_total*1e3:7.2f}ms",
          flush=True)
    print(f"  Median throughput:  {BATCH_SIZE / total_med:>10,.0f} pos/s", flush=True)
    print("-" * 70, flush=True)
    print(flush=True)

    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  JSON results: {json_out}", flush=True)


if __name__ == "__main__":
    main()
