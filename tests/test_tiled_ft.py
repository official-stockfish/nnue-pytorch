"""Tiled backward of the fused double feature transformer vs an fp64 reference and vs the
upstream fused kernels."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from model.cupy_utils import cupy_stream, shared_memory_limit
from model.modules.feature_transformer import fused_ft_functions as FF
from model.modules.feature_transformer import tiled_ft_backward as TB
from model.modules.feature_transformer import tiled_ft_kernel as TK
from model.modules.feature_transformer import tiled_ft_prep as TP

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and FF._HAS_CUPY_KERNELS
        and TB.enabled(1024, 1 << 16, torch.device("cuda"))
    ),
    reason="needs CUDA + CuPy and a GPU that the tiled backward supports",
)

MAX_ACT = 1.0


def _indices(B, maxa, num_inputs, gen, dev, active=None):
    # left-packed lists of distinct rows with -1 padding; ``active`` rows per list, else random
    if active is None:
        n = torch.randint(maxa // 2, maxa + 1, (B,), generator=gen, device=dev)
    else:
        n = torch.full((B,), active, device=dev)
    rows = torch.sort(
        torch.rand(B, num_inputs, generator=gen, device=dev).argsort(1)[:, :maxa], dim=1
    )[0]
    slot = torch.arange(maxa, device=dev)[None, :]
    return (
        torch.where(slot < n[:, None], rows, torch.full_like(rows, -1))
        .to(torch.int32)
        .contiguous()
    )


def _run(tiled, us, them, w, b, W, bias, l1, g, prefetch):
    """Forward and backward of the fused FT, with the tiled backward or the direct scatter."""
    with pytest.MonkeyPatch.context() as patch:
        if not tiled:
            patch.setattr(TB, "enabled", lambda *args: False)
        elif prefetch:
            TP.register_prep(w, b)
        Wp = W.clone().requires_grad_()
        bp_ = bias.clone().requires_grad_()
        out = FF.FusedDoubleFtFunction.apply(us, them, w, b, Wp, bp_, MAX_ACT, l1)
        out.backward(g)
    return out, Wp.grad, bp_.grad


def _clamped_fp32(us, them, w, b, W, bias, l1):
    """Clamp decisions of the fp32 forward (the fp64 reference takes its linear region from them)."""
    B, maxa = w.shape
    l0 = torch.empty(B, l1, device=W.device)
    c = torch.empty(B, 4, l1 // 2, device=W.device)
    FF.make_fused_double_ft_forward_kernel(maxa, l1)(
        grid=(B,),
        args=(
            us.data_ptr(),
            them.data_ptr(),
            w.data_ptr(),
            b.data_ptr(),
            W.data_ptr(),
            bias.data_ptr(),
            np.float32(MAX_ACT),
            l0.data_ptr(),
            c.data_ptr(),
            np.int32(l1),
        ),
    )
    return c


def _reference(us, them, w, b, W, bias, l1, g, lin):
    W64, b64 = W.double(), bias.double()
    h = l1 // 2

    def acc(idx):
        m = idx >= 0
        return (
            torch.nn.functional.embedding_bag(
                idx.clamp(min=0).long(), W64, mode="sum", per_sample_weights=m.double()
            )
            + b64
        )

    aw, ab = acc(w), acc(b)
    u, t = us.double(), them.double()
    x = torch.cat([u * aw + t * ab, u * ab + t * aw], 1)  # [w0 w1 b0 b1]
    c = x.clamp(0.0, MAX_ACT)
    c0, c1, c2, c3 = c.split(h, 1)
    l0 = torch.cat([c0 * c1, c2 * c3], 1)
    g64 = g.double()
    gl, gh = g64[:, :h], g64[:, h:]
    lin = lin.reshape(-1, 4, h)
    d = [
        lin[:, 0] * gl * c1,
        lin[:, 1] * gl * c0,
        lin[:, 2] * gh * c3,
        lin[:, 3] * gh * c2,
    ]
    gaw = torch.cat([u * d[0] + t * d[2], u * d[1] + t * d[3]], 1)
    gab = torch.cat([t * d[0] + u * d[2], t * d[1] + u * d[3]], 1)
    gW = torch.zeros_like(W64)
    for idx, ga in ((w, gaw), (b, gab)):
        m = idx >= 0
        gW.index_add_(
            0,
            idx.clamp(min=0).long()[m],
            ga.repeat_interleave(idx.shape[1], 0)[m.reshape(-1)],
        )
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

    fallbacks, hits = TP.STATS["fallback"], TP.STATS["prep_hit"]
    t_out, t_gW, t_gb = _run(True, us, them, w, b, W, bias, l1, g, prefetch)
    if prefetch:
        assert TP.STATS["prep_hit"] == hits + 1
    else:
        assert TP.STATS["fallback"] == fallbacks + 1
    u_out, u_gW, u_gb = _run(False, us, them, w, b, W, bias, l1, g, prefetch)

    # The tiled and upstream paths are bit-identical to each other (rel == 0
    # measured); both carry the same ~1.2e-6 fp32 accumulation noise against
    # the exact fp64 reference at these widths, so the bound is the fp32
    # noise scale (2e-6), not the fp64 comparison's ideal.
    assert rel(t_out, ref_l0) < 2e-6 and rel(u_out, ref_l0) < 2e-6
    assert rel(t_gW, ref_gW) < 1e-6 and rel(t_gb, ref_gb) < 1e-6
    assert (
        rel(t_out, u_out) < 2e-6 and rel(t_gW, u_gW) < 2e-6 and rel(t_gb, u_gb) < 2e-6
    )
    print(
        f"B={B} l1={l1} prefetch={prefetch}: out tiled {rel(t_out, ref_l0):.1e} upstream {rel(u_out, ref_l0):.1e}; "
        f"gW tiled {rel(t_gW, ref_gW):.1e} upstream {rel(u_gW, ref_gW):.1e}; "
        f"gb tiled {rel(t_gb, ref_gb):.1e} upstream {rel(u_gb, ref_gb):.1e}"
    )


# Lists of a fixed length pick the sort path. A sub-tile of 64 lists with 96 rows each fills the
# default sort (256 threads x 24 keys) exactly, and 97 rows need 32 keys per thread. 99 rows make a
# tile of 101,376 B, the shared memory of an sm_86 / sm_89 block. 300 rows are more than the sort's
# largest size holds, on any GPU; a tile that the GPU's shared memory cannot hold falls back too.
@pytest.mark.parametrize(
    "active,keys_per_thread", [(48, 24), (96, 24), (97, 32), (99, 32), (300, None)]
)
def test_tile_sort_matches_global_sort(active, keys_per_thread, monkeypatch):
    """Sub-tile sorts + rank merge == one global sort, for the sort sizes and the fallback."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    w = _indices(1024, 300, 20000, gen, dev, active)
    b = _indices(1024, 300, 20000, gen, dev, active)
    sort_kernel = Mock(wraps=TK.subtile_sort_kernel)
    monkeypatch.setattr(TK, "subtile_sort_kernel", sort_kernel)
    fallbacks = TP.STATS["global_sort_fallback"]
    if TK.TILE_LISTS * active * 4 > shared_memory_limit(dev):
        keys_per_thread = None

    tiled = TP.prepare(w, b)
    if keys_per_thread is None:
        sort_kernel.assert_not_called()
        assert TP.STATS["global_sort_fallback"] == fallbacks + 1
    else:
        sort_kernel.assert_called_once_with(keys_per_thread)
        assert TP.STATS["global_sort_fallback"] == fallbacks

    reference = TP.prepare(w, b, sort="global")
    assert torch.equal(tiled.entries, reference.entries)
    assert torch.equal(tiled.tile_start, reference.tile_start)
    assert torch.equal(tiled.order, reference.order)

    # The entries are exactly the batch's (list, row) pairs.
    tile = torch.repeat_interleave(
        torch.arange(tiled.n_tiles, device=dev),
        (tiled.tile_start[1:] - tiled.tile_start[:-1]).long(),
    )
    local = tiled.entries & ((1 << TK.LOCAL_BITS) - 1)
    lists = tiled.order[tile * TK.TILE_LISTS + local].long()
    pairs = (lists << TK.ROW_BITS) | (tiled.entries >> TK.LOCAL_BITS)
    rows = torch.cat([w, b]).long()
    list_ids = torch.arange(rows.shape[0], device=dev)[:, None].expand_as(rows)
    expected = ((list_ids << TK.ROW_BITS) | rows)[rows >= 0]
    assert torch.equal(torch.sort(pairs)[0], torch.sort(expected)[0])


@pytest.mark.parametrize(
    "keys_per_thread",
    range(TK.SORT_MIN_KEYS_PER_THREAD, TK.SORT_MAX_KEYS_PER_THREAD + 1, 8),
)
def test_subtile_sort_sizes(keys_per_thread):
    """Every size of the sub-tile sort kernel, on full and on partly filled sub-tiles."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(keys_per_thread)
    capacity = TK.SORT_THREADS * keys_per_thread
    sizes = torch.tensor([capacity, 0, 1, capacity - 1, capacity // 3, 2], device=dev)
    sub_start = torch.zeros(sizes.numel() + 1, dtype=torch.int32, device=dev)
    sub_start[1:] = torch.cumsum(sizes, 0)
    # unique keys per sub-tile: distinct (row, list in sub-tile) pairs, with the sub-tile's high
    # list-in-tile bits (here those of the third sub-tile of a tile)
    keys = torch.empty(int(sub_start[-1]), dtype=torch.int32, device=dev)
    for i, n in enumerate(sizes.tolist()):
        pair = torch.randperm(TK.SUBTILE_LISTS << 14, generator=gen, device=dev)[:n]
        row, local = pair >> 6, (pair & 63) | 128
        keys[sub_start[i] : sub_start[i + 1]] = (row << TK.LOCAL_BITS) | local
    expected = torch.cat(
        [
            torch.sort(keys[sub_start[i] : sub_start[i + 1]])[0]
            for i in range(sizes.numel())
        ]
    )
    with cupy_stream(dev):
        TK.subtile_sort_kernel(keys_per_thread)(
            grid=(sizes.numel(),),
            block=(TK.SORT_THREADS,),
            args=(sub_start.data_ptr(), keys.data_ptr()),
        )
    assert torch.equal(keys, expected)


def test_small_shared_memory_keeps_the_plain_backward(monkeypatch):
    """A GPU whose blocks cannot hold a tile's gradient slice (sm_75: 64 KB) is left to the other
    backward kernels instead of failing in the scatter launch."""
    dev = torch.device("cuda")
    monkeypatch.setattr(TB, "shared_memory_limit", lambda device: 64 * 1024)
    assert not TB.enabled(1024, 1 << 16, dev)
    assert TB.enabled(1024, 1 << 16)  # the layer itself is supported

    gen = torch.Generator(device=dev).manual_seed(1)
    B, l1, num_inputs, maxa = 256, 128, 500, 32
    w = _indices(B, maxa, num_inputs, gen, dev)
    b = _indices(B, maxa, num_inputs, gen, dev)
    W = torch.randn(num_inputs, l1, generator=gen, device=dev) * 0.05
    bias = torch.rand(l1, generator=gen, device=dev)
    us = (torch.rand(B, 1, generator=gen, device=dev) < 0.5).float()
    g = torch.randn(B, l1, generator=gen, device=dev)
    fallbacks = TP.STATS["fallback"]
    _, gW, gb = _run(True, us, 1.0 - us, w, b, W, bias, l1, g, prefetch=False)
    assert TP.STATS["fallback"] == fallbacks  # the tiled backward did not run
    _, ref_gW, ref_gb = _run(False, us, 1.0 - us, w, b, W, bias, l1, g, prefetch=False)
    assert rel(gW, ref_gW) < 2e-6 and rel(gb, ref_gb) < 2e-6
