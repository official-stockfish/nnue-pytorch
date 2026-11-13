"""Tiled forward/backward of the fused double feature transformer vs an fp64 reference and vs the
upstream fused kernels."""

import numpy as np
import pytest
import torch

from model.modules.feature_transformer import fused_ft_functions as FF
from model.modules.feature_transformer import tiled_ft_backward as TB
from model.modules.feature_transformer import tiled_ft_forward as TF
from model.modules.feature_transformer import tiled_ft_fused as TFF

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and FF._HAS_CUPY_KERNELS and TB._HAS_CUPY),
    reason="needs CUDA + CuPy",
)

MAX_ACT = 1.0


def _indices(B, maxa, num_inputs, gen, dev):
    # left-packed lists of distinct rows with -1 padding
    n = torch.randint(maxa // 2, maxa + 1, (B,), generator=gen, device=dev)
    rows = torch.sort(torch.rand(B, num_inputs, generator=gen, device=dev).argsort(1)[:, :maxa], dim=1)[0]
    slot = torch.arange(maxa, device=dev)[None, :]
    return torch.where(slot < n[:, None], rows, torch.full_like(rows, -1)).to(torch.int32).contiguous()


def _run(tiled, us, them, w, b, W, bias, l1, g, prefetch):
    old = TFF.USE_TILED_FORWARD, TFF.USE_TILED_BACKWARD
    old_enabled = TF.TILED_FORWARD_ENABLED
    TFF.USE_TILED_FORWARD = TFF.USE_TILED_BACKWARD = tiled
    TF.TILED_FORWARD_ENABLED = tiled or prefetch  # registration gate (default off on GH200)
    try:
        if tiled and prefetch:
            bp = TB.register_prep(w, b)
            TF.register_prep(w, b, order_perm=bp.order, from_backward=bp)
        Wp = W.clone().requires_grad_()
        bp_ = bias.clone().requires_grad_()
        out = FF.FusedDoubleFtFunction.apply(us, them, w, b, Wp, bp_, MAX_ACT, l1)
        out.backward(g)
    finally:
        TFF.USE_TILED_FORWARD, TFF.USE_TILED_BACKWARD = old
        TF.TILED_FORWARD_ENABLED = old_enabled
    return out, Wp.grad, bp_.grad


def _clamped_fp32(us, them, w, b, W, bias, l1):
    """Clamp decisions of the fp32 forward (the fp64 reference takes its linear region from them)."""
    B, maxa = w.shape
    l0 = torch.empty(B, l1, device=W.device)
    c = torch.empty(B, 4, l1 // 2, device=W.device)
    FF.make_fused_double_ft_forward_kernel(maxa, l1)(grid=(B,), args=(
        us.data_ptr(), them.data_ptr(), w.data_ptr(), b.data_ptr(), W.data_ptr(), bias.data_ptr(),
        np.float32(MAX_ACT), l0.data_ptr(), c.data_ptr(), np.int32(l1)))
    return c


def _reference(us, them, w, b, W, bias, l1, g, lin):
    W64, b64 = W.double(), bias.double()
    h = l1 // 2

    def acc(idx):
        m = idx >= 0
        return torch.nn.functional.embedding_bag(idx.clamp(min=0).long(), W64, mode="sum",
                                                 per_sample_weights=m.double()) + b64

    aw, ab = acc(w), acc(b)
    u, t = us.double(), them.double()
    x = torch.cat([u * aw + t * ab, u * ab + t * aw], 1)          # [w0 w1 b0 b1]
    c = x.clamp(0.0, MAX_ACT)
    c0, c1, c2, c3 = c.split(h, 1)
    l0 = torch.cat([c0 * c1, c2 * c3], 1)
    g64 = g.double()
    gl, gh = g64[:, :h], g64[:, h:]
    lin = lin.reshape(-1, 4, h)
    d = [lin[:, 0] * gl * c1, lin[:, 1] * gl * c0, lin[:, 2] * gh * c3, lin[:, 3] * gh * c2]
    gaw = torch.cat([u * d[0] + t * d[2], u * d[1] + t * d[3]], 1)
    gab = torch.cat([t * d[0] + u * d[2], t * d[1] + u * d[3]], 1)
    gW = torch.zeros_like(W64)
    for idx, ga in ((w, gaw), (b, gab)):
        m = idx >= 0
        gW.index_add_(0, idx.clamp(min=0).long()[m], ga.repeat_interleave(idx.shape[1], 0)[m.reshape(-1)])
    return l0, gW, gaw.sum(0) + gab.sum(0)


def rel(x, y):
    return ((x.double() - y.double()).abs().max() / y.double().abs().max()).item()


@pytest.mark.parametrize("B,l1", [(1000, 1024), (4096, 512)])
@pytest.mark.parametrize("prefetch", [True, False])
def test_tiled_matches_reference(B, l1, prefetch):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    num_inputs, maxa = 3000, 64
    w = _indices(B, maxa, num_inputs, gen, dev)
    b = _indices(B, maxa, num_inputs, gen, dev)
    W = torch.randn(num_inputs, l1, generator=gen, device=dev) * 0.05
    bias = torch.rand(l1, generator=gen, device=dev)
    us = (torch.rand(B, 1, generator=gen, device=dev) < 0.5).float()
    them = 1.0 - us
    g = torch.randn(B, l1, generator=gen, device=dev)

    c32 = _clamped_fp32(us, them, w, b, W, bias, l1)
    lin = ((c32 != 0) & (c32 != MAX_ACT)).double()
    ref_l0, ref_gW, ref_gb = _reference(us, them, w, b, W, bias, l1, g, lin)

    fwd_fb, hits = TF.STATS["fallback"], TF.STATS["prep_hit"]
    t_out, t_gW, t_gb = _run(True, us, them, w, b, W, bias, l1, g, prefetch)
    if prefetch:
        assert TF.STATS["prep_hit"] == hits + 1
    else:
        assert TF.STATS["fallback"] == fwd_fb + 1
    u_out, u_gW, u_gb = _run(False, us, them, w, b, W, bias, l1, g, prefetch)

    # The tiled and upstream paths are bit-identical to each other (rel == 0
    # measured); both carry the same ~1.2e-6 fp32 accumulation noise against
    # the exact fp64 reference at these widths, so the bound is the fp32
    # noise scale (2e-6), not the fp64 comparison's ideal.
    assert rel(t_out, ref_l0) < 2e-6 and rel(u_out, ref_l0) < 2e-6
    assert rel(t_gW, ref_gW) < 1e-6 and rel(t_gb, ref_gb) < 1e-6
    assert rel(t_out, u_out) < 2e-6 and rel(t_gW, u_gW) < 2e-6 and rel(t_gb, u_gb) < 2e-6
    print(f"B={B} l1={l1} prefetch={prefetch}: out tiled {rel(t_out, ref_l0):.1e} upstream {rel(u_out, ref_l0):.1e}; "
          f"gW tiled {rel(t_gW, ref_gW):.1e} upstream {rel(u_gW, ref_gW):.1e}; "
          f"gb tiled {rel(t_gb, ref_gb):.1e} upstream {rel(u_gb, ref_gb):.1e}")


@pytest.mark.parametrize("B", [1000, 4096])
def test_pull_forward_bit_identical(B):
    """The pull gather sums bias + rows in ascending row order, exactly like the 16-list gather."""
    from model.modules.feature_transformer import tiled_ft_pull as TP
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(B)
    maxa, num_inputs, l1 = 64, 3000, 512
    w = _indices(B, maxa, num_inputs, gen, dev)
    b = _indices(B, maxa, num_inputs, gen, dev)
    W = torch.randn(num_inputs, l1 + 8, generator=gen, device=dev)
    bias = torch.randn(l1 + 8, generator=gen, device=dev)
    bp = TB.prepare(w, b)
    pp = TP.prepare(w, b, bp)
    assert pp is not None
    fp = TF.prepare(w, b, T=16, order_perm=bp.order, from_backward=bp)
    ref = TF.tiled_forward(w, b, W, bias, l1, fp)
    out = TF.tiled_forward(w, b, W, bias, l1, pp)
    assert torch.equal(ref[0], out[0]) and torch.equal(ref[1], out[1])


def test_block_sort_matches_global_sort():
    """256-list tiles too large for one shared-memory sort: sub-tile sorts + rank merge == global sort."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    w = _indices(4096, 128, 20000, gen, dev)
    b = _indices(4096, 128, 20000, gen, dev)
    p1 = TB.prepare(w, b, sort="block")
    p2 = TB.prepare(w, b, sort="global")
    assert p1.max_tile > TB.SORT_CAP_MAX and p1.sub_sorted is not None
    assert torch.equal(p1.e_key, p2.e_key) and torch.equal(p1.tile_start, p2.tile_start)


@pytest.mark.parametrize("corr", [1.0, 0.75])
def test_fused_act_quant_bit_identical(corr):
    """act_quant fused into the tiled epilogue == FT output followed by fake_quantize_act_and_scale."""
    from model.modules.feature_transformer import fused_weight_quant as FWQ
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3)
    B, maxa, num_inputs, l1 = 2048, 64, 3000, 512
    w = _indices(B, maxa, num_inputs, gen, dev)
    b = _indices(B, maxa, num_inputs, gen, dev)
    W = torch.randn(num_inputs, l1 + 8, generator=gen, device=dev) * 0.05
    bias = torch.rand(l1 + 8, generator=gen, device=dev) * 0.5
    us = (torch.rand(B, 1, generator=gen, device=dev) < 0.5).float()
    them = 1.0 - us
    aq = (128.0, 1e-5, corr)
    outs = []
    for fused in (False, True):
        bp = TB.register_prep(w, b)
        TF.register_prep(w, b, order_perm=bp.order, from_backward=bp)
        if fused:
            out = FF.FusedDoubleFtFunction.apply(us, them, w, b, W, bias, MAX_ACT, l1, aq)
        else:
            out = FWQ.fake_quantize_act_and_scale(FF.FusedDoubleFtFunction.apply(us, them, w, b, W, bias, MAX_ACT, l1), *aq)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("cols", [1032, 520, 6])
def test_merge_quant_vectorized_bit_identical(cols):
    """The vectorized merge (merge_quant4) == the scalar merge_quant kernel == the torch chain."""
    from model.modules.feature_transformer import fused_weight_quant as FWQ
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(cols)
    s = 254.0
    nb, nv = 32, 37
    w0 = torch.randn(301, cols, generator=gen, device=dev) * 0.3
    w1 = torch.randn(nb * nv, cols, generator=gen, device=dev) * 0.3
    v = torch.randn(nv, cols, generator=gen, device=dev) * 0.3
    # exact rounding ties
    w0[:7] = (torch.randint(-300, 300, (7, cols), generator=gen, device=dev).float() + 0.5) / s
    w1[:5] = (torch.randint(-300, 300, (5, cols), generator=gen, device=dev).float() + 0.5) / s
    v[:2] = 0.0
    out = FWQ.merge_and_fake_quantize([(w0, None, 1), (w1, v, nb)], s)
    x = torch.cat([w0, w1 + v.repeat(nb, 1)])
    ref = torch.round(x * s) * (np.float32(1.0) / np.float32(s)).item()
    assert torch.equal(out, ref)
    # the scalar kernel on the same segments
    k = FWQ._get_kernel()
    sc = torch.empty_like(out)
    with TB._cupy_stream(dev):
        for r0, (ww, vv) in ((0, (w0, None)), (301, (w1, v))):
            k(((ww.numel() + 255) // 256,), (256,), (sc[r0:].data_ptr(), ww.data_ptr(), 0 if vv is None else vv.data_ptr(),
                                                    np.int64(ww.numel()), np.int64(1 if vv is None else vv.numel()),
                                                    np.float32(s), np.float32(1.0) / np.float32(s)))
    assert torch.equal(out, sc)


def _fused_inputs(B, seed, L1=1024, N=4096, maxa=64):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = _indices(B, maxa, N, gen, dev); b = _indices(B, maxa, N, gen, dev)
    us = (torch.rand(B, 1, generator=gen, device=dev) < 0.5).float(); them = 1.0 - us
    us[3] = 0.5; them[3] = 0.5; us[5] = 1.0; them[5] = 1.0                 # non-binary positions
    W = (torch.randn(N, L1, generator=gen, device=dev) * 0.02 * 64).round() / 64
    W[W == 0] = 0.0
    W[torch.rand(N, L1, generator=gen, device=dev) < 0.3] = -0.0            # -0 weights: -0 accumulators
    bias = (torch.randn(L1, generator=gen, device=dev) * 0.3 * 64).round() / 64
    bias[:64] = -0.0
    return us, them, w, b, W, bias


@pytest.mark.parametrize("aq", [None, (127.0, 0.25, 1.0), (127.0, 0.0, 0.5)])
def test_fused_epilogue_bit_identical(aq):
    from model.modules.feature_transformer import tiled_ft_pull as TP
    # The tiled forward is off by default on GH200; this test exercises its
    # epilogue, so enable it for the duration of of the call.
    import model.modules.feature_transformer.tiled_ft_fused as _TFF

    _fwd_default = TF.TILED_FORWARD_ENABLED
    _fused_default = _TFF.USE_TILED_FORWARD
    TF.TILED_FORWARD_ENABLED = True       # enables register_prep in the forward module
    _TFF.USE_TILED_FORWARD = True         # enables the fused forward call
    try:
        us, them, w, b, W, bias = _fused_inputs(2048, 3)
        outs = []
        old = TP.EPI_FUSED
        try:
            for fused in (False, True):
                TP.EPI_FUSED = fused
                bp = TB.register_prep(w, b)
                TF.register_prep(w, b, order_perm=bp.order, from_backward=bp)
                outs.append(
                    TFF.forward(us, them, w, b, W, bias, MAX_ACT, 1024, keep_acc=True, act_quant=aq))
        finally:
            TP.EPI_FUSED = old
        (l0a, (aw, ab)), (l0b, (bw, bb)) = outs
        assert torch.equal(l0a.view(torch.int32), l0b.view(torch.int32))
        assert torch.equal(aw, bw) and torch.equal(ab, bb)
    finally:
        TF.TILED_FORWARD_ENABLED = _fwd_default
        _TFF.USE_TILED_FORWARD = _fused_default


def test_fused_prologue_single_tile_bit_identical():
    # one tile (2B <= 256 lists): every (row, column) gets one atomic, so the result is deterministic
    us, them, w, b, W, bias = _fused_inputs(128, 4)
    gen = torch.Generator(device="cuda").manual_seed(5)
    aw = torch.randn(128, 1024, generator=gen, device="cuda") * 0.6 + 0.3
    ab = torch.randn(128, 1024, generator=gen, device="cuda") * 0.6 + 0.3
    aw[aw.abs() < 0.05] = -0.0
    gl0 = torch.randn(128, 1024, generator=gen, device="cuda") * 1e-3
    prep = TB.prepare(w, b)
    old = TFF.PRO_FUSED
    try:
        res = []
        for pro in (False, True):
            TFF.PRO_FUSED = pro
            res.append(TFF.backward(us, them, w, b, gl0, None, W.shape[0], MAX_ACT, 1024, prep=prep, acc=(aw, ab)))
    finally:
        TFF.PRO_FUSED = old
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])


@pytest.mark.parametrize("p", [0.02, 0.6, 1.0])
def test_bucket_sum_matches_torch(p):
    from model.modules.feature_transformer import fused_weight_quant as FWQ
    gen = torch.Generator(device="cuda").manual_seed(6)
    nb, vs, cols = 32, 768, 1024
    fl = (torch.rand(nb * vs, generator=gen, device="cuda") < p).to(torch.uint8)
    g = torch.randn(nb * vs, cols, generator=gen, device="cuda") * fl[:, None].float()
    g[g == 0] = 0.0
    assert torch.equal(FWQ._bucket_sum(g, fl, nb, (vs, cols)), g.view(nb, vs, cols).sum(0))
