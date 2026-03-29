"""End-to-end equivalence test: Metal kernel path vs embedding_bag fallback.

Runs the same inputs through both code paths and verifies that forward
outputs *and* backward gradients match within floating-point tolerance.

Tests at three levels:
  1. Raw sparse_linear call (single perspective)
  2. DoubleFeatureTransformer (both perspectives, shared weight)
  3. Full NNUEModel forward (feature transformer + dense stacks + PSQT)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.functions import _torch_sparse_linear
from model.modules.feature_transformer.metal import (
    MetalSparseLinearFunction,
    is_available,
    metal_l0_mixing,
    metal_fused_double_forward_l0,
    metal_fused_composed_double_forward_l0,
    metal_sqr_crelu,
)

DEVICE = "mps"
FWD_TOL = 1e-4
GRAD_TOL = 1e-3


def _make_inputs(batch, num_inputs, max_active, *, sentinel_frac=0.25, seed=42):
    torch.manual_seed(seed)
    active = int(max_active * (1 - sentinel_frac))
    idx = torch.cat(
        [
            (torch.rand(batch, active) * num_inputs).to(torch.int32),
            torch.full((batch, max_active - active), -1, dtype=torch.int32),
        ],
        dim=1,
    )
    val = torch.rand(batch, max_active, dtype=torch.float32)
    return idx, val


# ---------------------------------------------------------------------------
# Level 1: raw sparse_linear
# ---------------------------------------------------------------------------
def test_sparse_linear_forward():
    assert is_available(), "Metal extension not available"

    for batch, n_in, n_act, n_out, label in [
        (16, 32, 8, 64, "small"),
        (64, 256, 32, 128, "medium"),
        (512, 4096, 32, 1032, "training-shape"),
    ]:
        idx, val = _make_inputs(batch, n_in, n_act)
        w = torch.rand(n_in, n_out)
        b = torch.rand(n_out)

        ref = _torch_sparse_linear(idx, val, w, b)
        metal = MetalSparseLinearFunction.apply(
            idx.to(DEVICE), val.to(DEVICE), w.to(DEVICE), b.to(DEVICE)
        ).cpu()

        err = (ref - metal).abs().max().item()
        assert err < FWD_TOL, f"[{label}] fwd err {err:.2e} >= {FWD_TOL}"
        print(f"  sparse_linear fwd [{label:14s}]  err={err:.2e}  PASS")


def test_sparse_linear_backward():
    assert is_available()

    for batch, n_in, n_act, n_out, label in [
        (16, 32, 8, 64, "small"),
        (64, 256, 32, 128, "medium"),
        (512, 4096, 32, 1032, "training-shape"),
    ]:
        idx, val = _make_inputs(batch, n_in, n_act)
        w_ref = torch.rand(n_in, n_out, requires_grad=True)
        b_ref = torch.rand(n_out, requires_grad=True)
        w_met = w_ref.detach().clone().to(DEVICE).requires_grad_(True)
        b_met = b_ref.detach().clone().to(DEVICE).requires_grad_(True)

        go = torch.rand(batch, n_out)
        _torch_sparse_linear(idx, val, w_ref, b_ref).backward(go)
        MetalSparseLinearFunction.apply(
            idx.to(DEVICE), val.to(DEVICE), w_met, b_met
        ).backward(go.to(DEVICE))

        w_err = (w_ref.grad - w_met.grad.cpu()).abs().max().item()
        b_err = (b_ref.grad - b_met.grad.cpu()).abs().max().item()
        assert w_err < GRAD_TOL, f"[{label}] w_grad err {w_err:.2e}"
        assert b_err < GRAD_TOL, f"[{label}] b_grad err {b_err:.2e}"
        print(
            f"  sparse_linear bwd [{label:14s}]  w_err={w_err:.2e}  "
            f"b_err={b_err:.2e}  PASS"
        )


# ---------------------------------------------------------------------------
# Level 2: DoubleFeatureTransformer (both perspectives, shared weight)
# ---------------------------------------------------------------------------
def test_double_feature_transformer():
    assert is_available()

    from model.modules.feature_transformer.module import DoubleFeatureTransformer

    BATCH, NUM_IN, MAX_ACT, OUT = 64, 256, 32, 128

    torch.manual_seed(7)
    ft = DoubleFeatureTransformer(NUM_IN, OUT)
    w_ref = ft.weight.detach().clone().requires_grad_(True)
    b_ref = ft.bias.detach().clone().requires_grad_(True)
    w_met = ft.weight.detach().clone().to(DEVICE).requires_grad_(True)
    b_met = ft.bias.detach().clone().to(DEVICE).requires_grad_(True)

    idx0, val0 = _make_inputs(BATCH, NUM_IN, MAX_ACT, seed=10)
    idx1, val1 = _make_inputs(BATCH, NUM_IN, MAX_ACT, seed=20)

    ref0 = _torch_sparse_linear(idx0, val0, w_ref, b_ref)
    ref1 = _torch_sparse_linear(idx1, val1, w_ref, b_ref)
    met0 = MetalSparseLinearFunction.apply(
        idx0.to(DEVICE), val0.to(DEVICE), w_met, b_met
    )
    met1 = MetalSparseLinearFunction.apply(
        idx1.to(DEVICE), val1.to(DEVICE), w_met, b_met
    )

    fwd_err0 = (ref0 - met0.cpu()).abs().max().item()
    fwd_err1 = (ref1 - met1.cpu()).abs().max().item()
    assert fwd_err0 < FWD_TOL and fwd_err1 < FWD_TOL

    loss_ref = (ref0 - ref1).pow(2).mean()
    loss_met = (met0 - met1).pow(2).mean()
    loss_ref.backward()
    loss_met.backward()

    w_err = (w_ref.grad - w_met.grad.cpu()).abs().max().item()
    b_err = (b_ref.grad - b_met.grad.cpu()).abs().max().item()
    assert w_err < GRAD_TOL, f"DFT w_grad err {w_err:.2e}"
    assert b_err < GRAD_TOL, f"DFT b_grad err {b_err:.2e}"
    print(
        f"  DoubleFeatureTransformer     fwd_err={max(fwd_err0, fwd_err1):.2e}  "
        f"w_grad_err={w_err:.2e}  b_grad_err={b_err:.2e}  PASS"
    )


# ---------------------------------------------------------------------------
# Level 2b: Fused L0 mixing kernel
# ---------------------------------------------------------------------------
def test_fused_l0_mixing():
    assert is_available()

    from model.modules.feature_transformer.metal import metal_l0_mixing

    for B, L1, PSQT, label in [
        (16, 64, 4, "small"),
        (64, 128, 8, "medium"),
        (512, 1024, 8, "training-shape"),
    ]:
        OUT = L1 + PSQT
        torch.manual_seed(77)
        wp = torch.rand(B, OUT, device=DEVICE)
        bp = torch.rand(B, OUT, device=DEVICE)
        us = torch.randint(0, 2, (B, 1), device=DEVICE, dtype=torch.float32)
        them = 1 - us

        # Reference: PyTorch ops
        wo, wpsqt_ref = wp.split(L1, dim=1)
        bo, bpsqt_ref = bp.split(L1, dim=1)
        l0 = (us * torch.cat([wo, bo], dim=1)) + (them * torch.cat([bo, wo], dim=1))
        l0 = torch.clamp(l0, 0.0, 1.0)
        s = l0.split(L1 // 2, dim=1)
        l0_ref = torch.cat([s[0] * s[1], s[2] * s[3]], dim=1) * (127 / 128)

        # Metal kernel
        l0_met, wpsqt_met, bpsqt_met = metal_l0_mixing(wp, bp, us, them, L1, PSQT)

        fwd_err = (l0_ref - l0_met).abs().max().item()
        assert fwd_err < FWD_TOL, f"[{label}] l0 fwd err {fwd_err:.2e}"
        assert (wpsqt_ref - wpsqt_met).abs().max().item() == 0
        assert (bpsqt_ref - bpsqt_met).abs().max().item() == 0

        # Backward
        wp_ref = wp.detach().clone().requires_grad_(True)
        bp_ref = bp.detach().clone().requires_grad_(True)
        wp_met = wp.detach().clone().requires_grad_(True)
        bp_met = bp.detach().clone().requires_grad_(True)

        wo, wpsqt = wp_ref.split(L1, dim=1)
        bo, bpsqt = bp_ref.split(L1, dim=1)
        l0 = (us * torch.cat([wo, bo], dim=1)) + (them * torch.cat([bo, wo], dim=1))
        l0 = torch.clamp(l0, 0.0, 1.0)
        s = l0.split(L1 // 2, dim=1)
        l0_r = torch.cat([s[0] * s[1], s[2] * s[3]], dim=1) * (127 / 128)
        (l0_r.sum() + wpsqt.sum() + bpsqt.sum()).backward()

        l0_m, wp_m, bp_m = metal_l0_mixing(wp_met, bp_met, us, them, L1, PSQT)
        (l0_m.sum() + wp_m.sum() + bp_m.sum()).backward()

        gwp_err = (wp_ref.grad - wp_met.grad).abs().max().item()
        gbp_err = (bp_ref.grad - bp_met.grad).abs().max().item()
        assert gwp_err < GRAD_TOL, f"[{label}] gwp err {gwp_err:.2e}"
        assert gbp_err < GRAD_TOL, f"[{label}] gbp err {gbp_err:.2e}"

        print(
            f"  L0 mixing [{label:14s}]  fwd={fwd_err:.2e}  "
            f"gwp={gwp_err:.2e}  gbp={gbp_err:.2e}  PASS"
        )


# ---------------------------------------------------------------------------
# Level 2c: Fused double sparse_linear + L0 mixing (single autograd node)
# ---------------------------------------------------------------------------
def test_fused_double_forward_l0():
    assert is_available()

    for B, N_IN, MAX_ACT, L1, PSQT, label in [
        (16, 64, 8, 32, 4, "small"),
        (64, 256, 32, 128, 8, "medium"),
        (512, 4096, 32, 1024, 8, "training-shape"),
    ]:
        OUT = L1 + PSQT
        torch.manual_seed(55)
        w_idx, w_val = _make_inputs(B, N_IN, MAX_ACT, seed=10)
        b_idx, b_val = _make_inputs(B, N_IN, MAX_ACT, seed=20)
        us = torch.randint(0, 2, (B, 1), dtype=torch.float32)
        them = 1 - us

        # Reference: embedding_bag + PyTorch L0 ops
        w_ref = torch.rand(N_IN, OUT, requires_grad=True)
        b_ref = torch.rand(OUT, requires_grad=True)
        wp_ref = _torch_sparse_linear(w_idx, w_val, w_ref, b_ref)
        bp_ref = _torch_sparse_linear(b_idx, b_val, w_ref, b_ref)
        wo, wpsqt_r = wp_ref.split(L1, dim=1)
        bo, bpsqt_r = bp_ref.split(L1, dim=1)
        l0 = (us * torch.cat([wo, bo], dim=1)) + (them * torch.cat([bo, wo], dim=1))
        l0 = torch.clamp(l0, 0.0, 1.0)
        s = l0.split(L1 // 2, dim=1)
        l0_ref = torch.cat([s[0] * s[1], s[2] * s[3]], dim=1) * (127 / 128)
        (l0_ref.sum() + wpsqt_r.sum() + bpsqt_r.sum()).backward()

        # Metal fused path
        w_met = w_ref.detach().clone().to(DEVICE).requires_grad_(True)
        b_met = b_ref.detach().clone().to(DEVICE).requires_grad_(True)
        l0_m, wp_m, bp_m = metal_fused_double_forward_l0(
            w_idx.to(DEVICE), w_val.to(DEVICE),
            b_idx.to(DEVICE), b_val.to(DEVICE),
            w_met, b_met,
            us.to(DEVICE), them.to(DEVICE), L1, PSQT,
        )
        (l0_m.sum() + wp_m.sum() + bp_m.sum()).backward()

        fwd_err = (l0_ref - l0_m.cpu()).abs().max().item()
        wg_err = (w_ref.grad - w_met.grad.cpu()).abs().max().item()
        bg_err = (b_ref.grad - b_met.grad.cpu()).abs().max().item()
        assert fwd_err < FWD_TOL, f"[{label}] fwd err {fwd_err:.2e}"
        assert wg_err < GRAD_TOL, f"[{label}] w_grad err {wg_err:.2e}"
        assert bg_err < GRAD_TOL, f"[{label}] b_grad err {bg_err:.2e}"
        print(
            f"  fused double+L0 [{label:14s}]  fwd={fwd_err:.2e}  "
            f"wg={wg_err:.2e}  bg={bg_err:.2e}  PASS"
        )


# ---------------------------------------------------------------------------
# Level 2d: composed fused double sparse_linear + L0 mixing (virtual_weight)
# ---------------------------------------------------------------------------
def test_fused_composed_double_forward_l0():
    assert is_available()

    for B, N_A, N_B, VW_P, MAX_ACT, L1, PSQT, label in [
        (16, 32, 16, 8, 8, 24, 4, "small"),
        (64, 128, 64, 16, 32, 96, 8, "medium"),
        (256, 2048, 1024, 64, 32, 512, 8, "training-like"),
    ]:
        N_IN = N_A + N_B
        OUT = L1 + PSQT
        NUM_BUCKETS = N_B // VW_P
        torch.manual_seed(77)

        w_idx, w_val = _make_inputs(B, N_IN, MAX_ACT, seed=10)
        b_idx, b_val = _make_inputs(B, N_IN, MAX_ACT, seed=20)
        us = torch.randint(0, 2, (B, 1), dtype=torch.float32)
        them = 1 - us

        weight_a = torch.rand(N_A, OUT, requires_grad=True)
        weight_b = torch.rand(N_B, OUT, requires_grad=True)
        virtual_w = torch.rand(VW_P, OUT, requires_grad=True)
        bias_ref = torch.rand(OUT, requires_grad=True)

        # Reference: merge weights, then use the non-composed fused path
        merged_b = (
            weight_b.detach().view(NUM_BUCKETS, VW_P, OUT)
            + virtual_w.detach().unsqueeze(0)
        ).view(N_B, OUT)
        merged = torch.cat([weight_a.detach(), merged_b], dim=0).requires_grad_(True)
        bias_pt = bias_ref.detach().clone().requires_grad_(True)

        wp_r = _torch_sparse_linear(w_idx, w_val, merged, bias_pt)
        bp_r = _torch_sparse_linear(b_idx, b_val, merged, bias_pt)
        wo, wpsqt_r = wp_r.split(L1, dim=1)
        bo, bpsqt_r = bp_r.split(L1, dim=1)
        l0 = (us * torch.cat([wo, bo], dim=1)) + (them * torch.cat([bo, wo], dim=1))
        l0 = torch.clamp(l0, 0.0, 1.0)
        s = l0.split(L1 // 2, dim=1)
        l0_ref = torch.cat([s[0] * s[1], s[2] * s[3]], dim=1) * (127 / 128)
        (l0_ref.sum() + wpsqt_r.sum() + bpsqt_r.sum()).backward()

        # Metal composed path
        wa_m = weight_a.detach().clone().to(DEVICE).requires_grad_(True)
        wb_m = weight_b.detach().clone().to(DEVICE).requires_grad_(True)
        vw_m = virtual_w.detach().clone().to(DEVICE).requires_grad_(True)
        bi_m = bias_ref.detach().clone().to(DEVICE).requires_grad_(True)
        l0_m, wp_m, bp_m = metal_fused_composed_double_forward_l0(
            w_idx.to(DEVICE), w_val.to(DEVICE),
            b_idx.to(DEVICE), b_val.to(DEVICE),
            wa_m, wb_m, vw_m, bi_m,
            us.to(DEVICE), them.to(DEVICE), L1, PSQT, N_A, VW_P,
        )
        (l0_m.sum() + wp_m.sum() + bp_m.sum()).backward()

        fwd_err = (l0_ref - l0_m.cpu()).abs().max().item()

        # Reconstruct component grads from the merged weight grad
        ref_wa_g = merged.grad[:N_A]
        ref_wb_g = merged.grad[N_A:]
        ref_vw_g = ref_wb_g.view(NUM_BUCKETS, VW_P, OUT).sum(0)

        wa_err = (ref_wa_g - wa_m.grad.cpu()).abs().max().item()
        wb_err = (ref_wb_g - wb_m.grad.cpu()).abs().max().item()
        vw_err = (ref_vw_g - vw_m.grad.cpu()).abs().max().item()
        bg_err = (bias_pt.grad - bi_m.grad.cpu()).abs().max().item()

        assert fwd_err < FWD_TOL, f"[{label}] fwd err {fwd_err:.2e}"
        assert wa_err < GRAD_TOL, f"[{label}] wa_grad err {wa_err:.2e}"
        assert wb_err < GRAD_TOL, f"[{label}] wb_grad err {wb_err:.2e}"
        assert vw_err < GRAD_TOL, f"[{label}] vw_grad err {vw_err:.2e}"
        assert bg_err < GRAD_TOL, f"[{label}] bias_grad err {bg_err:.2e}"
        print(
            f"  composed fused [{label:14s}]  fwd={fwd_err:.2e}  "
            f"wa={wa_err:.2e}  wb={wb_err:.2e}  vw={vw_err:.2e}  "
            f"bg={bg_err:.2e}  PASS"
        )


# ---------------------------------------------------------------------------
# Level 2e: fused squared-clamp-relu activation
# ---------------------------------------------------------------------------
def test_fused_sqr_crelu():
    assert is_available()

    for B, L2, label in [
        (16, 7, "small"),
        (64, 15, "medium"),
        (512, 31, "training-shape"),
    ]:
        torch.manual_seed(99)
        l1c = torch.randn(B, L2 + 1, device=DEVICE)

        # Reference: PyTorch ops
        l1x_v, l1x_out_v = l1c.split(L2, dim=1)
        l1x_ref = torch.clamp(
            torch.cat([torch.pow(l1x_v, 2.0) * (255 / 256), l1x_v], dim=1),
            0.0, 1.0,
        )
        l1x_out_ref = l1x_out_v.squeeze(1)

        # Metal kernel
        l1x_met, l1x_out_met = metal_sqr_crelu(l1c, L2)

        fwd_err = (l1x_ref - l1x_met).abs().max().item()
        out_err = (l1x_out_ref - l1x_out_met).abs().max().item()
        assert fwd_err < FWD_TOL, f"[{label}] l1x fwd err {fwd_err:.2e}"
        assert out_err < FWD_TOL, f"[{label}] l1x_out err {out_err:.2e}"

        # Backward
        l1c_pt = l1c.clone().detach().requires_grad_(True)
        l1x_v2, l1x_out_v2 = l1c_pt.split(L2, dim=1)
        l1x_pt = torch.clamp(
            torch.cat([torch.pow(l1x_v2, 2.0) * (255 / 256), l1x_v2], dim=1),
            0.0, 1.0,
        )
        l1x_out_pt = l1x_out_v2.squeeze(1)

        grad_l1x = torch.randn_like(l1x_pt)
        grad_out = torch.randn(B, device=DEVICE)
        (l1x_pt * grad_l1x + l1x_out_pt.unsqueeze(1) * grad_out.unsqueeze(1)).sum().backward()
        ref_grad = l1c_pt.grad.clone()

        l1c_met = l1c.clone().detach().requires_grad_(True)
        l1x_m, l1x_out_m = metal_sqr_crelu(l1c_met, L2)
        (l1x_m * grad_l1x + l1x_out_m.unsqueeze(1) * grad_out.unsqueeze(1)).sum().backward()
        met_grad = l1c_met.grad.clone()

        grad_err = (ref_grad - met_grad).abs().max().item()
        assert grad_err < GRAD_TOL, f"[{label}] grad err {grad_err:.2e}"
        print(
            f"  sqr_crelu [{label:14s}]  fwd={fwd_err:.2e}  "
            f"out={out_err:.2e}  grad={grad_err:.2e}  PASS"
        )


# ---------------------------------------------------------------------------
# Level 3: full NNUEModel forward+backward
# ---------------------------------------------------------------------------
def test_full_model():
    """Run identical inputs through the model twice — once forced through
    embedding_bag, once through Metal — and compare outputs + gradients."""
    assert is_available()

    from model.modules.feature_transformer import functions as fn_module

    BATCH, NUM_IN, MAX_ACT = 64, 256, 32
    L1, PSQT, LS_BUCKETS = 128, 8, 4
    OUT = L1 + PSQT

    torch.manual_seed(123)
    idx_w, val_w = _make_inputs(BATCH, NUM_IN, MAX_ACT, seed=30)
    idx_b, val_b = _make_inputs(BATCH, NUM_IN, MAX_ACT, seed=40)
    us = torch.randint(0, 2, (BATCH, 1), dtype=torch.float32)
    them = 1 - us
    psqt_idx = torch.randint(0, PSQT, (BATCH,))
    ls_idx = torch.randint(0, LS_BUCKETS, (BATCH,))

    def run_with_backend(use_metal: bool):
        """Build a fresh model, run forward+backward, return outputs+grads."""
        torch.manual_seed(999)
        weight = torch.rand(NUM_IN, OUT)
        bias = torch.rand(OUT)

        from model.modules.config import LayerStacksConfig
        from model.modules.layer_stacks import LayerStacks

        ls_config = LayerStacksConfig(L1=L1, L2=15, L3=16)

        torch.manual_seed(999)
        stacks = LayerStacks(LS_BUCKETS, ls_config)

        if use_metal:
            dev = DEVICE
            weight = weight.to(dev).requires_grad_(True)
            bias = bias.to(dev).requires_grad_(True)
            stacks = stacks.to(dev)
            _us = us.to(dev)
            _them = them.to(dev)
            _psqt = psqt_idx.to(dev)
            _ls = ls_idx.to(dev)
            _idx_w = idx_w.to(dev)
            _val_w = val_w.to(dev)
            _idx_b = idx_b.to(dev)
            _val_b = val_b.to(dev)
        else:
            weight = weight.requires_grad_(True)
            bias = bias.requires_grad_(True)
            _us, _them = us, them
            _psqt, _ls = psqt_idx, ls_idx
            _idx_w, _val_w = idx_w, val_w
            _idx_b, _val_b = idx_b, val_b

        if use_metal:
            l0, wpsqt, bpsqt = metal_fused_double_forward_l0(
                _idx_w, _val_w, _idx_b, _val_b,
                weight, bias, _us, _them, L1, PSQT,
            )
        else:
            wp = _torch_sparse_linear(_idx_w, _val_w, weight, bias)
            bp = _torch_sparse_linear(_idx_b, _val_b, weight, bias)
            w_out, wpsqt = wp.split(L1, dim=1)
            b_out, bpsqt = bp.split(L1, dim=1)
            l0 = (_us * torch.cat([w_out, b_out], dim=1)) + (
                _them * torch.cat([b_out, w_out], dim=1)
            )
            l0 = torch.clamp(l0, 0.0, 1.0)
            l0_s = torch.split(l0, L1 // 2, dim=1)
            l0 = torch.cat([l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]], dim=1) * (
                127 / 128
            )

        x = stacks(l0, _ls)
        psqt_unsq = _psqt.unsqueeze(1)
        x = x + (wpsqt.gather(1, psqt_unsq) - bpsqt.gather(1, psqt_unsq)) * (
            _us - 0.5
        )

        loss = x.sum()
        loss.backward()

        def to_cpu(t):
            return t.cpu() if t.is_cuda or t.device.type == "mps" else t

        dense_grads = {}
        for name, p in stacks.named_parameters():
            if p.grad is not None:
                dense_grads[name] = to_cpu(p.grad.clone())

        return (
            to_cpu(x.detach()),
            to_cpu(weight.grad.clone()),
            to_cpu(bias.grad.clone()),
            dense_grads,
        )

    out_ref, wg_ref, bg_ref, dg_ref = run_with_backend(use_metal=False)
    out_met, wg_met, bg_met, dg_met = run_with_backend(use_metal=True)

    fwd_err = (out_ref - out_met).abs().max().item()
    wg_err = (wg_ref - wg_met).abs().max().item()
    bg_err = (bg_ref - bg_met).abs().max().item()

    print(f"  Full model fwd              err={fwd_err:.2e}  ", end="")
    assert fwd_err < 1e-3, f"model output err {fwd_err:.2e}"
    print("PASS")

    print(f"  Full model weight_grad      err={wg_err:.2e}  ", end="")
    assert wg_err < GRAD_TOL, f"weight_grad err {wg_err:.2e}"
    print("PASS")

    print(f"  Full model bias_grad        err={bg_err:.2e}  ", end="")
    assert bg_err < GRAD_TOL, f"bias_grad err {bg_err:.2e}"
    print("PASS")

    for name in sorted(dg_ref.keys()):
        if name not in dg_met:
            continue
        err = (dg_ref[name] - dg_met[name]).abs().max().item()
        status = "PASS" if err < GRAD_TOL else "FAIL"
        print(f"  Full model grad {name:30s}  err={err:.2e}  {status}")
        assert err < GRAD_TOL, f"dense grad {name} err {err:.2e}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("Metal vs embedding_bag equivalence tests")
    print("=" * 68)
    print()

    print("Level 1: raw sparse_linear")
    print("-" * 40)
    test_sparse_linear_forward()
    test_sparse_linear_backward()

    print()
    print("Level 2: DoubleFeatureTransformer")
    print("-" * 40)
    test_double_feature_transformer()

    print()
    print("Level 2b: fused L0 mixing kernel")
    print("-" * 40)
    test_fused_l0_mixing()

    print()
    print("Level 2c: fused double sparse_linear + L0 mixing")
    print("-" * 40)
    test_fused_double_forward_l0()

    print()
    print("Level 2d: composed fused double forward + L0 mixing")
    print("-" * 40)
    test_fused_composed_double_forward_l0()

    print()
    print("Level 2e: fused squared-clamp-relu activation")
    print("-" * 40)
    test_fused_sqr_crelu()

    print()
    print("Level 3: full NNUEModel forward + backward")
    print("-" * 40)
    test_full_model()

    print()
    print("All equivalence tests passed.")
