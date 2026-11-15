"""Tiled segmented-reduction backward for the double feature transformer.

Replaces the per-position atomic scatter with tiles of sorted entries:
one block per (tile, column slice) holds the tile's gradient rows in
shared memory, sums each row segment in registers and issues one
coalesced atomicAdd per (row, column) per tile. The preparation (the
sort and the tile keys) is computed by the loader's prefetch worker on
its own CUDA stream, keyed by the batch's index tensor. Same math as
the plain kernel; fp32 sums in a different order, like the atomics.
"""

from __future__ import annotations

import collections
import os
import threading
import warnings
import weakref
from dataclasses import dataclass

import numpy as np
import torch

from model.optimizers import grad_row_flags as _grf

from .cupy_utils import HAS_CUPY, compile_kernel, cp, cupy_stream

# NNUE_TILED_BACKWARD=0 disables the tiled backward (the fused FT then uses its plain scatter).
TILED_BACKWARD_ENABLED = os.environ.get("NNUE_TILED_BACKWARD", "1") != "0"
TILE_SIZE = 256
COLUMNS_PER_BLOCK = 64
WARPS_PER_BLOCK = 32

ROW_BITS = 20  # row index bits in the tile keys (feature sets up to 1M rows)
LOCAL_BITS = 8  # T <= 256
MAX_T = 256
MAX_SHARED_BYTES = (
    99 * 1024
)  # dynamic shared memory per block we allow (GA102: 101376 B)
SORT_CAP_MAX = (
    16384  # max entries per tile for the in-block sort (64 KB of shared memory)
)

STATS = (
    collections.Counter()
)  # "prep_registered", "prep_hit", "fallback", "global_sort_fallback"


def _render(src: str, **kw) -> str:
    kw = dict(
        kw,
        LIST_ROW="((li) < batch_size ? white + (int64_t)(li) * MAXA : black + (int64_t)((li) - batch_size) * MAXA)",
    )
    for k, v in kw.items():
        src = src.replace("@" + k + "@", str(v))
    return src


def _cub_options() -> tuple:
    inc = os.path.join(
        os.path.dirname(cp.__file__), "_core", "include", "cupy", "_cccl"
    )
    return (
        "--std=c++17",
        "-I" + os.path.join(inc, "cub"),
        "-I" + os.path.join(inc, "libcudacxx", "include"),
    )


_HDR = "typedef int int32_t; typedef long long int64_t; typedef unsigned int uint32_t; typedef unsigned char uint8_t;\n"

# ----------------------------------------------------------------------------- kernels

_ORDER_KEY_SRC = (
    _HDR
    + r"""
// One warp per list: lanes read consecutive slots (coalesced) up to the first empty slot, then the
// warp reduces the per-lane minima / maximum / count (min, max and + are order-independent).
extern "C" __global__ void order_key(const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    const uint32_t* __restrict__ hash_a, const uint32_t* __restrict__ hash_b, int64_t* __restrict__ keys,
    int32_t* __restrict__ counts, const int32_t n_lists, const int32_t batch_size, const int32_t bucket_first_row, const int32_t bucket_rows)
{
    const int32_t MAXA = @MAXA@, NKEYS = @NKEYS@, BITS = @BITS@;
    const int32_t li = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (li >= n_lists) return;
    const int32_t* row = @LIST_ROW@;
    uint32_t ha[NKEYS], hb[NKEYS];
    for (int j = 0; j < NKEYS; ++j) { ha[j] = hash_a[j]; hb[j] = hash_b[j]; }
    uint32_t mn[NKEYS]; int32_t mx = -1; int32_t n = 0;
    for (int j = 0; j < NKEYS; ++j) mn[j] = (1u << BITS) - 1;
    for (int k0 = 0; k0 < MAXA; k0 += 32) {
        const int32_t k = k0 + lane;
        const int32_t r = k < MAXA ? row[k] : -1;
        const unsigned valid = __ballot_sync(0xFFFFFFFFu, r >= 0);
        const unsigned lead = ~valid ? (((~valid) & (0u - (~valid))) - 1u) : 0xFFFFFFFFu;   // lanes before the first empty slot
        if ((lead >> lane) & 1u) {
            ++n;
            mx = max(mx, r);
            for (int j = 0; j < NKEYS; ++j) {
                const uint32_t h = (((uint32_t)r * ha[j] + hb[j]) >> 7) & ((1u << BITS) - 1);
                mn[j] = min(mn[j], h);
            }
        }
        if (lead != 0xFFFFFFFFu) break;
    }
    for (int o = 16; o > 0; o >>= 1) {
        n += __shfl_xor_sync(0xFFFFFFFFu, n, o);
        mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
        for (int j = 0; j < NKEYS; ++j) mn[j] = min(mn[j], __shfl_xor_sync(0xFFFFFFFFu, mn[j], o));
    }
    if (lane != 0) return;
    int64_t key = 0;
    if (bucket_rows > 0) { int32_t kb = (mx - bucket_first_row) / bucket_rows; kb = kb < 0 ? 0 : (kb > 127 ? 127 : kb); key = kb; }
    for (int j = 0; j < NKEYS; ++j) key = (key << BITS) | (int64_t)mn[j];
    keys[li] = key;
    counts[li] = n;
}
"""
)


def _order_key_kernel(max_active: int, n_keys: int = 4, key_bits: int = 14):
    src = _render(_ORDER_KEY_SRC, MAXA=max_active, NKEYS=n_keys, BITS=key_bits)
    return compile_kernel(("order_key", max_active, n_keys, key_bits), src, "order_key")


_TILE_KEY_SRC = (
    _HDR
    + r"""
// 32-bit keys (row << LOCAL_BITS | local list index), written per list at its offset; the lists of
// a tile are consecutive in tile order, so the keys of a tile are contiguous (unsorted).
// One warp per list: lanes read 32 consecutive slots and write 32 consecutive keys (coalesced).
extern "C" __global__ void tile_key(const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    const int32_t* __restrict__ order, const int32_t* __restrict__ offsets, uint32_t* __restrict__ keys,
    const int32_t n_lists, const int32_t batch_size)
{
    const int32_t MAXA = @MAXA@, T = @T@, LOCAL_BITS = @LOCAL_BITS@;
    const int32_t p = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // position in tile order
    const int lane = threadIdx.x & 31;
    if (p >= n_lists) return;
    const int32_t li = order[p];
    const int32_t* row = @LIST_ROW@;
    const uint32_t local = (uint32_t)(p % T);
    const int32_t o = offsets[p];
    for (int k = lane; k < MAXA; k += 32) {
        const int32_t r = row[k];
        const unsigned active = __ballot_sync(0xFFFFFFFFu, r >= 0);
        if (r >= 0) keys[o + k] = ((uint32_t)r << LOCAL_BITS) | local;   // lists are left-packed: slot k of a valid entry = its rank
        if (active != 0xFFFFFFFFu) break;
    }
}
"""
)


def _tile_key_kernel(max_active: int, T: int):
    src = _render(_TILE_KEY_SRC, MAXA=max_active, T=T, LOCAL_BITS=LOCAL_BITS)
    return compile_kernel(("tile_key", max_active, T), src, "tile_key")


_TILE_SORT_SRC = (
    _HDR
    + r"""
// In-place bitonic sort of every tile's keys (one block per tile, shared memory, capacity CAP).
extern "C" __global__ void tile_sort(const int32_t* __restrict__ tile_start, uint32_t* __restrict__ keys)
{
    extern __shared__ uint32_t sh[];
    const int32_t tile = blockIdx.x;
    const int32_t e0 = tile_start[tile], n = tile_start[tile + 1] - e0;
    if (n <= 1) return;
    int32_t N2 = 1; while (N2 < n) N2 <<= 1;               // n <= CAP = @CAP@ is guaranteed by the host
    for (int i = threadIdx.x; i < N2; i += blockDim.x) sh[i] = (i < n) ? keys[e0 + i] : 0xFFFFFFFFu;
    __syncthreads();
    for (int k = 2; k <= N2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int q = threadIdx.x; q < (N2 >> 1); q += blockDim.x) {   // one compare per pair (i, i + j)
                const int i = ((q & ~(j - 1)) << 1) | (q & (j - 1)), ixj = i + j;
                const uint32_t a = sh[i], b = sh[ixj];
                const bool up = (i & k) == 0;
                if ((a > b) == up) { sh[i] = b; sh[ixj] = a; }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) keys[e0 + i] = sh[i];
}
"""
)


_MERGE_SRC = (
    _HDR
    + r"""
// One block per sub-tile s of tile t = s / NSUB: entry i of the sorted run s goes to
// tile_start[t] + i + (number of smaller keys in the other runs of the tile).
extern "C" __global__ void merge_runs(const int32_t* __restrict__ sub_start, const int32_t* __restrict__ tile_start,
    const uint32_t* __restrict__ keys, uint32_t* __restrict__ out, const int32_t n_sub)
{
    const int NSUB = @NSUB@;
    const int32_t s = blockIdx.x, t = s / NSUB, s0 = t * NSUB;
    const int32_t a = sub_start[s], n = sub_start[s + 1] - a;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t k = keys[a + i];
        int32_t rank = i;
        for (int r = s0; r < s0 + NSUB && r < n_sub; ++r) {
            if (r == s) continue;
            int32_t lo = sub_start[r], hi = sub_start[r + 1];
            while (lo < hi) { const int32_t mid = (lo + hi) >> 1; if (keys[mid] < k) lo = mid + 1; else hi = mid; }
            rank += lo - sub_start[r];
        }
        out[tile_start[t] + rank] = k;
    }
}
"""
)


_TILE_RSORT_SRC = r"""
#include <cub/block/block_radix_sort.cuh>
// Every tile's keys sorted in place by a block-wide radix sort (CUB, BS threads x IPT keys in
// registers; tiles of n <= BS * IPT keys). Keys are unique, so the result equals any other sort.
// Padding keys (all ones) sort after the real keys even where the low NBITS bits tie: the sort is
// stable and the padding comes last in the blocked input.
extern "C" __global__ void __launch_bounds__(@BS@) tile_rsort(const int* __restrict__ tile_start, unsigned* __restrict__ keys)
{
    typedef cub::BlockRadixSort<unsigned, @BS@, @IPT@, cub::NullType, @RB@> Sort;
    __shared__ typename Sort::TempStorage tmp;
    const int e0 = tile_start[blockIdx.x], n = tile_start[blockIdx.x + 1] - e0;
    if (n <= 1) return;
    unsigned v[@IPT@];
#pragma unroll
    for (int i = 0; i < @IPT@; ++i) {   // blocked: padding last
        const int j = threadIdx.x * @IPT@ + i;
        const unsigned k = j < n ? keys[e0 + j] : 0xFFFFFFFFu;
        v[i] = ((k >> @LB@) << @SB@) | (k & ((1u << @SB@) - 1u));   // drop the local-index bits that are constant in the tile
    }
    Sort(tmp).SortBlockedToStriped(v, 0, @NBITS@);
    const unsigned hi = keys[e0] & (((1u << @LB@) - 1u) ^ ((1u << @SB@) - 1u));
#pragma unroll
    for (int i = 0; i < @IPT@; ++i) {
        const int j = i * @BS@ + threadIdx.x;
        if (j < n) keys[e0 + j] = ((v[i] >> @SB@) << @LB@) | hi | (v[i] & ((1u << @SB@) - 1u));
    }
}
"""
RSORT_BS = 256
RSORT_IPT = 24
RSORT_RB = 4


def _tile_rsort_kernel(sub_bits: int):
    """Sort of tiles of 2**sub_bits lists whose local list indices (LOCAL_BITS) share their high bits."""
    src = _render(
        _TILE_RSORT_SRC,
        BS=RSORT_BS,
        IPT=RSORT_IPT,
        RB=RSORT_RB,
        NBITS=ROW_BITS + sub_bits,
        LB=LOCAL_BITS,
        SB=sub_bits,
    )
    return compile_kernel(
        ("tile_rsort", RSORT_BS, RSORT_IPT, RSORT_RB, sub_bits),
        src,
        "tile_rsort",
        options=_cub_options(),
    )


_MERGE_SH_SRC = (
    _HDR
    + r"""
// One block per tile: the tile's NSUB sorted runs are loaded into shared memory, and entry i of run s
// goes to tile_start[t] + (i - start of s) + (number of smaller keys in the other runs).
extern "C" __global__ void __launch_bounds__(1024) merge_runs_sh(const int32_t* __restrict__ sub_start,
    const int32_t* __restrict__ tile_start, const uint32_t* __restrict__ keys, uint32_t* __restrict__ out, const int32_t n_sub)
{
    extern __shared__ uint32_t sk[];
    const int NSUB = @NSUB@;
    const int32_t t = blockIdx.x, e0 = tile_start[t], n = tile_start[t + 1] - e0;
    int32_t rs[NSUB + 1];
#pragma unroll
    for (int r = 0; r <= NSUB; ++r) rs[r] = sub_start[min(t * NSUB + r, n_sub)] - e0;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) sk[i] = keys[e0 + i];
    __syncthreads();
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t k = sk[i];
        int32_t rank = 0;
#pragma unroll
        for (int r = 0; r < NSUB; ++r) {
            if (i >= rs[r] && i < rs[r + 1]) { rank += i - rs[r]; continue; }
            int32_t lo = rs[r], hi = rs[r + 1];
            while (lo < hi) { const int32_t mid = (lo + hi) >> 1; if (sk[mid] < k) lo = mid + 1; else hi = mid; }
            rank += lo - rs[r];
        }
        out[e0 + rank] = k;
    }
}
"""
)


def _merge_sh_kernel(nsub: int, cap: int):
    return compile_kernel(
        ("merge_runs_sh", nsub, cap),
        _render(_MERGE_SH_SRC, NSUB=nsub),
        "merge_runs_sh",
        dynamic_shared=cap * 4,
    )


def _merge_kernel(nsub: int):
    return compile_kernel(
        ("merge_runs", nsub), _render(_MERGE_SRC, NSUB=nsub), "merge_runs"
    )


def _tile_sort_kernel(cap: int):
    src = _render(_TILE_SORT_SRC, CAP=cap)
    return compile_kernel(("tile_sort", cap), src, "tile_sort", dynamic_shared=cap * 4)


_SCATTER_SRC = (
    _HDR
    + r"""
#define GV @GV@
// One block per (tile, column slice). NWARPS warps; lane owns columns col0 + lane + 32*c, c < CPL.
extern "C" __global__ void __launch_bounds__(@NWARPS@ * 32) tiled_scatter(
    const float* __restrict__ Gw, const float* __restrict__ Gb, const int32_t* __restrict__ order,
    const int32_t* __restrict__ tile_start, const uint32_t* __restrict__ e_key,
    float* __restrict__ grad_weight, float* __restrict__ grad_bias, const int32_t n_lists, const int32_t batch_size,
    uint8_t* __restrict__ row_touched)
{
    const int32_t T = @T@, SLICE = @SLICE@, OUT = @OUT@, CPL = @CPL@, LOCAL_BITS = @LOCAL_BITS@;
    extern __shared__ float Gsh[];                          // T * SLICE
    const int32_t tile = blockIdx.x, slice = blockIdx.y;
    const int32_t col0 = slice * SLICE;

    // 1. the tile's gradient rows for this column slice; every thread issues all its loads (PF list ids,
    // then PF values) before its shared-memory stores
    const int NT = @NWARPS@ * 32, PF = (T * SLICE + NT - 1) / NT;
    int32_t lid[PF];
#pragma unroll
    for (int k = 0; k < PF; ++k) {
        const int idx = threadIdx.x + k * NT;
        const int32_t li = tile * T + idx / SLICE;
        lid[k] = (idx < T * SLICE && li < n_lists) ? __ldg(&order[li]) : -1;
    }
    float v[PF];
#pragma unroll
    for (int k = 0; k < PF; ++k) {
        const int c = (threadIdx.x + k * NT) % SLICE;
        v[k] = lid[k] < 0 ? 0.0f : lid[k] < batch_size ? __ldg(&Gw[(int64_t)lid[k] * OUT + col0 + c])
                                                       : __ldg(&Gb[(int64_t)(lid[k] - batch_size) * OUT + col0 + c]);
    }
#pragma unroll
    for (int k = 0; k < PF; ++k) {
        const int idx = threadIdx.x + k * NT, l = idx / SLICE, c = idx - l * SLICE;
        if (idx < T * SLICE) Gsh[l * SLICE + (c & 31) * CPL + (c >> 5)] = v[k];   // column c at lane (c % 32), slot c / 32
    }
    __syncthreads();

    // 2. bias gradient: column sums of the tile
    if (threadIdx.x < SLICE) {
        float s = 0.0f;
        const int pc = (threadIdx.x & 31) * CPL + (threadIdx.x >> 5);
        for (int l = 0; l < T; ++l) s += Gsh[l * SLICE + pc];
        if (s != 0.0f) atomicAdd(&grad_bias[col0 + threadIdx.x], s);
    }

    // 3. row segments: warp w walks part w of the tile's sorted entry stream
    const int32_t e0 = tile_start[tile], e1 = tile_start[tile + 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t* ksh = (uint32_t*)(Gsh + T * SLICE) + warp * 32;
    const int32_t per = (e1 - e0 + @NWARPS@ - 1) / @NWARPS@;
    const int32_t s = e0 + warp * per;
    const int32_t e = min(s + per, e1);
    float acc[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) acc[c] = 0.0f;
    uint32_t cur = 0xFFFFFFFFu;
    // Same entry order, sums and flushes as a one-entry-at-a-time walk; the shared-memory loads of U
    // entries are issued before their adds, and the next 32 keys are fetched one batch ahead.
    const int U = 8;
    uint32_t nextkey = (s + lane < e) ? __ldg(&e_key[s + lane]) : 0u;
    for (int32_t base = s; base < e; base += 32) {
        const int32_t nk = min(32, e - base);
        const uint32_t mykey = nextkey;
        nextkey = (base + 32 + lane < e) ? __ldg(&e_key[base + 32 + lane]) : 0u;
        __syncwarp();
        ksh[lane] = mykey;                                  // the 32 keys, read back as broadcasts
        __syncwarp();
        int j = 0;
        for (; j + U <= nk; j += U) {
            uint32_t key[U];
            GV gv[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                key[u] = ksh[j + u];
                const int l = (int)(key[u] & ((1u << LOCAL_BITS) - 1));
                gv[u] = *(const GV*)(Gsh + l * SLICE + lane * CPL);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const uint32_t r = key[u] >> LOCAL_BITS;
                if (r != cur) {
                    if (cur != 0xFFFFFFFFu) {
                        if (lane == 0 && slice == 0) row_touched[cur] = 1;
                        float* gw = grad_weight + (int64_t)cur * OUT + col0 + lane;
#pragma unroll
                        for (int c = 0; c < CPL; ++c) { atomicAdd(gw + 32 * c, acc[c]); acc[c] = 0.0f; }
                    }
                    cur = r;
                }
                const float* g = (const float*)&gv[u];
#pragma unroll
                for (int c = 0; c < CPL; ++c) acc[c] += g[c];
            }
        }
        for (; j < nk; ++j) {
            const uint32_t key = ksh[j];
            const uint32_t r = key >> LOCAL_BITS;
            const int l = (int)(key & ((1u << LOCAL_BITS) - 1));
            if (r != cur) {
                if (cur != 0xFFFFFFFFu) {
                    if (lane == 0 && slice == 0) row_touched[cur] = 1;
                    float* gw = grad_weight + (int64_t)cur * OUT + col0 + lane;
#pragma unroll
                    for (int c = 0; c < CPL; ++c) { atomicAdd(gw + 32 * c, acc[c]); acc[c] = 0.0f; }
                }
                cur = r;
            }
            const GV gv = *(const GV*)(Gsh + l * SLICE + lane * CPL);   // this lane's CPL columns in one load
            const float* g = (const float*)&gv;
#pragma unroll
            for (int c = 0; c < CPL; ++c) acc[c] += g[c];
        }
    }
    if (cur != 0xFFFFFFFFu) {
        if (lane == 0 && slice == 0) row_touched[cur] = 1;
        float* gw = grad_weight + (int64_t)cur * OUT + col0 + lane;
#pragma unroll
        for (int c = 0; c < CPL; ++c) atomicAdd(gw + 32 * c, acc[c]);
    }
}
"""
)


def _scatter_kernel(
    T: int,
    slice_cols: int,
    l1_size: int,
    out_size: int,
    nwarps: int,
):
    if (
        slice_cols not in (32, 64, 128)
        or slice_cols > 32 * nwarps
        or l1_size % slice_cols
    ):
        raise ValueError(
            f"slice must be a multiple of 32, at most {32 * nwarps}, and divide l1_size; got {slice_cols}"
        )
    if nwarps not in (4, 8, 16, 32):
        raise ValueError(f"nwarps must be 4, 8, 16 or 32, got {nwarps}")
    shm = T * slice_cols * 4 + nwarps * 128
    if shm > MAX_SHARED_BYTES:
        raise ValueError(
            f"T={T} x slice={slice_cols} needs {shm} B of shared memory > {MAX_SHARED_BYTES}"
        )
    src = _render(
        _SCATTER_SRC,
        T=T,
        SLICE=slice_cols,
        OUT=out_size,
        CPL=slice_cols // 32,
        LOCAL_BITS=LOCAL_BITS,
        NWARPS=nwarps,
        GV={1: "float", 2: "float2", 4: "float4"}[slice_cols // 32],
    )
    key = ("scatter", T, slice_cols, out_size, nwarps)
    return compile_kernel(key, src, "tiled_scatter", dynamic_shared=shm)


_PSQT_SRC = (
    _HDR
    + r"""
// One thread per (list, slot): the PSQT columns of the row's weight gradient (zero gradients skipped).
extern "C" __global__ void psqt_scatter(
    const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    const float* __restrict__ Gw, const float* __restrict__ Gb, float* __restrict__ grad_weight,
    const int32_t n_lists, const int32_t batch_size, uint8_t* __restrict__ row_touched)
{
    const int32_t MAXA = @MAXA@, L1 = @L1@, OUT = @OUT@;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t li = (int32_t)(gid / MAXA); const int32_t k = (int32_t)(gid % MAXA);
    if (li >= n_lists) return;
    const int32_t r = (@LIST_ROW@)[k];
    if (r < 0) return;
    const float* g = (li < batch_size ? Gw + (int64_t)li * OUT : Gb + (int64_t)(li - batch_size) * OUT) + L1;
    float* gw = grad_weight + (int64_t)r * OUT + L1;
    for (int p = 0; p < OUT - L1; ++p) { const float v = g[p]; if (v != 0.0f) { row_touched[r] = 1; atomicAdd(gw + p, v); } }
}
"""
)


def _psqt_scatter_kernel(max_active: int, l1_size: int, out_size: int):
    src = _render(_PSQT_SRC, MAXA=max_active, L1=l1_size, OUT=out_size)
    return compile_kernel(
        ("psqt_scatter", max_active, l1_size, out_size), src, "psqt_scatter"
    )


# ----------------------------------------------------------------------------- preparation


@dataclass
class TiledPrep:
    """Index-only preparation of one batch: list order and per-tile row-sorted entries."""

    T: int
    n_lists: int
    order: (
        torch.Tensor
    )  # (n_lists,) int32: list ids in tile order (white lists 0..B-1, black B..2B-1)
    tile_start: torch.Tensor  # (n_tiles + 1,) int32
    e_key: torch.Tensor  # (E,) uint32 (stored as int32): row << LOCAL_BITS | local, sorted within each tile
    n_tiles: int
    max_tile: int  # largest number of entries in one tile

    def record_stream(self, stream):
        for t in (self.order, self.tile_start, self.e_key):
            t.record_stream(stream)


@torch.compiler.disable
def list_order(
    white: torch.Tensor,
    black: torch.Tensor,
    buckets: tuple[int, int] | None = None,
    seed: int = 0,
):
    """Permutation of the perspective lists (white 0..B-1, black B..2B-1; int32) that forms the tiles,
    and the number of active rows per list (int32, list order).

    Lists are sorted by a min-hash of their rows, so lists that share rows tend to share a tile.
    ``buckets`` = (first_row, rows_per_bucket) groups them by bucket first: the highest row of every
    list falls in a bucket-major feature that starts at first_row (the last feature of the set; see
    ComposedFeatureTransformer.prefetch_preparer). Any order gives the same sums up to fp32 rounding."""
    B, maxa = white.shape
    n_lists = 2 * B
    dev = white.device
    first_row, rows_per_bucket = buckets if buckets is not None else (0, 0)
    g = torch.Generator(device=dev).manual_seed(seed)
    a = (torch.randint(1, 2**31 - 1, (4,), generator=g, device=dev) | 1).to(torch.int32)
    b = torch.randint(0, 2**31 - 1, (4,), generator=g, device=dev).to(torch.int32)
    keys = torch.empty(n_lists, dtype=torch.int64, device=dev)
    counts = torch.empty(n_lists, dtype=torch.int32, device=dev)
    with cupy_stream(dev):
        _order_key_kernel(maxa)(
            grid=((n_lists * 32 + 255) // 256,),
            block=(256,),
            args=(
                white.data_ptr(),
                black.data_ptr(),
                a.data_ptr(),
                b.data_ptr(),
                keys.data_ptr(),
                counts.data_ptr(),
                np.int32(n_lists),
                np.int32(B),
                np.int32(first_row),
                np.int32(rows_per_bucket),
            ),
        )
    o = torch.sort(keys)[1].to(torch.int32)
    return o, counts


@torch.compiler.disable
def prepare(
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    T: int = TILE_SIZE,
    buckets: tuple[int, int] | None = None,
    sort: str = "block",
) -> TiledPrep:
    """Build the tile structure for a batch (both perspectives; only the indices are read).
    ``buckets``: see list_order. ``sort="global"`` forces the fallback used when a tile's entries
    do not fit the shared-memory sorts (for tests)."""
    assert white_indices.dtype == torch.int32 and white_indices.is_cuda
    assert 1 <= T <= MAX_T
    dev = white_indices.device
    white_indices = white_indices.contiguous()
    black_indices = black_indices.contiguous()
    B, maxa = white_indices.shape
    n_lists = 2 * B
    n_tiles = (n_lists + T - 1) // T
    o, counts = list_order(white_indices, black_indices, buckets)
    counts = counts[o.long()]
    offsets = torch.zeros(n_lists + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(counts, 0)
    tile_start = torch.cat([offsets[0:n_lists:T], offsets[n_lists:]]).contiguous()
    tile_n = tile_start[1:] - tile_start[:-1]
    SUB = (
        64 if T % 64 == 0 and T > 64 else T
    )  # sub-tiles: sorted in shared memory, then merged
    sub_start = (
        tile_start
        if SUB == T
        else torch.cat([offsets[0:n_lists:SUB], offsets[n_lists:]]).contiguous()
    )
    sub_n = sub_start[1:] - sub_start[:-1]
    E, max_tile, max_sub = (
        int(x) for x in torch.stack([offsets[-1], tile_n.max(), sub_n.max()]).tolist()
    )  # one host sync per batch
    n_entries = E
    max_sub = max(max_sub, 1)
    E, max_tile = max(E, 1), max(max_tile, 1)
    keys = torch.empty(E, dtype=torch.int32, device=dev)
    with cupy_stream(dev):
        _tile_key_kernel(maxa, T)(
            grid=((n_lists * 32 + 255) // 256,),
            block=(256,),
            args=(
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                o.data_ptr(),
                offsets.data_ptr(),
                keys.data_ptr(),
                np.int32(n_lists),
                np.int32(B),
            ),
        )
    cap = 1024
    while (
        cap < max_tile
    ):  # shared-memory capacity from the actual largest tile (one bitonic sort per tile)
        cap <<= 1
    cap_sub = 1024
    while cap_sub < max_sub:
        cap_sub <<= 1
    if sort == "block" and cap <= SORT_CAP_MAX:
        with cupy_stream(dev):
            _tile_sort_kernel(cap)(
                grid=(n_tiles,),
                block=(1024,),
                shared_mem=cap * 4,
                args=(tile_start.data_ptr(), keys.data_ptr()),
            )
    elif sort == "block" and SUB < T and cap_sub <= SORT_CAP_MAX:
        # T = 256 tiles (~23k entries) exceed shared memory: sort the 64-list sub-tiles in shared memory,
        # then merge every tile's T / SUB sorted runs by rank (keys are unique across runs: the local list
        # index differs). Same keys as the global sort.
        merged = torch.empty_like(keys)
        with cupy_stream(dev):
            if (
                max_sub <= RSORT_BS * RSORT_IPT
                and SUB & (SUB - 1) == 0
                and T % SUB == 0
            ):
                # a sub-tile's local list indices (p % T, 64 consecutive p) differ only in their low log2(SUB) bits
                _tile_rsort_kernel(SUB.bit_length() - 1)(
                    grid=(n_lists // SUB + (n_lists % SUB > 0),),
                    block=(RSORT_BS,),
                    args=(sub_start.data_ptr(), keys.data_ptr()),
                )
            else:
                _tile_sort_kernel(cap_sub)(
                    grid=(n_lists // SUB + (n_lists % SUB > 0),),
                    block=(512,),
                    shared_mem=cap_sub * 4,  # BS
                    args=(sub_start.data_ptr(), keys.data_ptr()),
                )
        with cupy_stream(dev):
            if max_tile * 4 <= MAX_SHARED_BYTES:
                mcap = 4096
                while mcap < max_tile:
                    mcap += 4096
                _merge_sh_kernel(T // SUB, mcap)(
                    grid=(n_tiles,),
                    block=(1024,),
                    shared_mem=max_tile * 4,
                    args=(
                        sub_start.data_ptr(),
                        tile_start.data_ptr(),
                        keys.data_ptr(),
                        merged.data_ptr(),
                        np.int32(sub_n.shape[0]),
                    ),
                )
            else:
                _merge_kernel(T // SUB)(
                    grid=(int(sub_n.shape[0]),),
                    block=(256,),
                    args=(
                        sub_start.data_ptr(),
                        tile_start.data_ptr(),
                        keys.data_ptr(),
                        merged.data_ptr(),
                        np.int32(sub_n.shape[0]),
                    ),
                )
        keys = merged
    else:
        if sort == "block":
            STATS["global_sort_fallback"] += (
                1  # a tile exceeded the shared-memory capacity
            )
        tile_of = torch.repeat_interleave(
            torch.arange(n_tiles, device=dev, dtype=torch.int64),
            tile_n.long(),
            output_size=n_entries,
        )  # size known: no host sync
        full = (tile_of << 32) | (keys.to(torch.int64) & 0xFFFFFFFF)
        keys = (torch.sort(full)[0] & 0xFFFFFFFF).to(torch.int32)
    return TiledPrep(T, n_lists, o, tile_start, keys, n_tiles, max_tile)


# side table: data_ptr(white_indices) -> (weakref to the registered tensor, shape, prep)
_PREP_CACHE_MAX = 16
_prep_cache: collections.OrderedDict[int, tuple] = collections.OrderedDict()
_prep_lock = threading.Lock()


def register_prep(
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    buckets: tuple[int, int] | None = None,
) -> TiledPrep | None:
    """Compute and store the preparation for a batch, keyed by its white index tensor, for
    FusedDoubleFtFunction to pick up (lookup_prep). Called on the loader's prefetch thread, on its
    CUDA stream. Returns the preparation, or None when the tiled backward is off."""
    if not (TILED_BACKWARD_ENABLED and HAS_CUPY and white_indices.is_cuda):
        return None
    prep = prepare(white_indices, black_indices, buckets=buckets)
    with _prep_lock:
        _prep_cache[white_indices.data_ptr()] = (
            weakref.ref(white_indices),
            tuple(white_indices.shape),
            prep,
        )
        _prep_cache.move_to_end(white_indices.data_ptr())
        while len(_prep_cache) > _PREP_CACHE_MAX:
            _prep_cache.popitem(last=False)
    STATS["prep_registered"] += 1
    return prep


def lookup_prep(white_indices: torch.Tensor) -> TiledPrep | None:
    """Pop the preparation registered for this index block, if it is still valid."""
    with _prep_lock:
        entry = _prep_cache.pop(white_indices.data_ptr(), None)
    if entry is None:
        return None
    ref, shape, prep = entry
    src = ref()
    if (
        src is None
        or src.data_ptr() != white_indices.data_ptr()
        or tuple(white_indices.shape) != shape
    ):
        return None
    # produced on the prefetch stream; the consumer stream already waited on the batch's ready event
    prep.record_stream(torch.cuda.current_stream(white_indices.device))
    STATS["prep_hit"] += 1
    return prep


# ----------------------------------------------------------------------------- backward


def applicable(
    l1_size: int, num_inputs: int, slice_cols: int = COLUMNS_PER_BLOCK
) -> bool:
    """Whether the tiled backward can run for this layer (else the plain kernel is used): CUDA (the
    kernels assume 32-lane warps and CUB), a width the column slices divide, and rows that fit the
    ROW_BITS of the tile keys."""
    return (
        HAS_CUPY
        and torch.version.hip is None
        and l1_size % slice_cols == 0
        and l1_size >= slice_cols
        and num_inputs <= 1 << ROW_BITS
    )


_warned_fallback = False


@torch.compiler.disable
def tiled_backward(
    white_indices,
    black_indices,
    grad_w,
    grad_b,
    num_inputs,
    l1_size,
    prep: TiledPrep | None = None,
    slice_cols: int = COLUMNS_PER_BLOCK,
    nwarps: int = WARPS_PER_BLOCK,
):
    """Weight/bias gradient of the double sparse linear layer (white and black lists share weight
    and bias); ``grad_w`` / ``grad_b`` (B, OUT) are the output gradients. Columns l1_size.. (the
    PSQT columns) use plain atomics. Returns (grad_weight, grad_bias)."""
    global _warned_fallback
    dev = grad_w.device
    B, maxa = white_indices.shape
    OUT = grad_w.shape[1]
    if prep is None:
        STATS["fallback"] += 1
        if not _warned_fallback:
            _warned_fallback = True
            warnings.warn(
                "tiled FT backward: no preparation was prefetched for this batch, so it is built "
                "in the backward (one host sync per step). Training batches get it from the "
                "loader's prefetch hook (NNUE.prefetch_hook).",
                stacklevel=2,
            )
        # the fallback tile size must fit the caller's column slice into shared memory
        T_fb = min(TILE_SIZE, MAX_T, (MAX_SHARED_BYTES // (slice_cols * 4)) // 32 * 32)
        prep = prepare(white_indices, black_indices, T=T_fb)
    n_lists = 2 * B
    if prep.n_lists != n_lists:
        raise RuntimeError("tiled backward: preparation does not match the batch")
    white_indices = white_indices.contiguous()
    black_indices = black_indices.contiguous()
    grad_w = grad_w.contiguous()
    grad_b = grad_b.contiguous()

    grad_weight = torch.zeros(num_inputs, OUT, dtype=torch.float32, device=dev)
    grad_bias = torch.zeros(OUT, dtype=torch.float32, device=dev)
    # row_touched[r] != 0 for every row the kernels below add to; the other rows of grad_weight stay
    # exactly +0 (registered for the optimizer, see grad_row_flags)
    row_touched = torch.zeros(num_inputs, dtype=torch.uint8, device=dev)
    if OUT > l1_size:
        grad_bias[l1_size:] = grad_w[:, l1_size:].sum(0) + grad_b[:, l1_size:].sum(0)
    with cupy_stream(dev):
        _scatter_kernel(prep.T, slice_cols, l1_size, OUT, nwarps)(
            grid=(prep.n_tiles, l1_size // slice_cols),
            block=(32 * nwarps,),
            shared_mem=prep.T * slice_cols * 4 + nwarps * 128,
            args=(
                grad_w.data_ptr(),
                grad_b.data_ptr(),
                prep.order.data_ptr(),
                prep.tile_start.data_ptr(),
                prep.e_key.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(n_lists),
                np.int32(B),
                row_touched.data_ptr(),
            ),
        )
        if OUT > l1_size:
            _psqt_scatter_kernel(maxa, l1_size, OUT)(
                grid=((n_lists * maxa + 255) // 256,),
                block=(256,),
                args=(
                    white_indices.data_ptr(),
                    black_indices.data_ptr(),
                    grad_w.data_ptr(),
                    grad_b.data_ptr(),
                    grad_weight.data_ptr(),
                    np.int32(n_lists),
                    np.int32(B),
                    row_touched.data_ptr(),
                ),
            )
    _grf.register(grad_weight, row_touched, reset=True)
    return grad_weight, grad_bias
