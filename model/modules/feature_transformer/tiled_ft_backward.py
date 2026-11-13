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
import weakref
from dataclasses import dataclass

import numpy as np
import torch

try:
    import cupy as cp

    _HAS_CUPY = True
except (ImportError, OSError, RuntimeError):  # pragma: no cover - CuPy missing
    cp = None
    _HAS_CUPY = False

TILED_BACKWARD_ENABLED = True
TILE_SIZE = 256
COLUMNS_PER_BLOCK = 64
LIST_ORDER = "kb_minhash"
TILE_SORT = "block"
BLOCK_GRID = "tile"
WARPS_PER_BLOCK = 32
HALFKA_FIRST_ROW = 64368
PREFETCH_PREP = True

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

_kernels: dict = {}
_kernel_lock = threading.Lock()


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


def _get_kernel(key, src, name, dynamic_shared: int = 0, options: tuple = ()):
    with _kernel_lock:
        k = _kernels.get(key)
        if k is None:
            k = cp.RawKernel(src, name, options=options)
            k.compile()
            if dynamic_shared > 48 * 1024:
                # raise the kernel's dynamic shared memory ceiling to the device's
                # opt-in maximum: the launch only allocates what it demands, so the
                # highest valid ceiling is free and removes the sensitivity of the
                # cap bookkeeping to the environment (a launch above the ceiling
                # fails with CUDA_ERROR_INVALID_VALUE, as observed on another host).
                optin = torch.cuda.get_device_properties(
                    0
                ).shared_memory_per_block_optin
                k.max_dynamic_shared_size_bytes = max(dynamic_shared, optin)
            _kernels[key] = k
        return k


class _cupy_stream:
    """Run CuPy kernel launches on the current torch stream of the current device."""

    def __init__(self, device: torch.device):
        self.dev = cp.cuda.Device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.stream = cp.cuda.ExternalStream(
            torch.cuda.current_stream(device).cuda_stream
        )

    def __enter__(self):
        self.dev.__enter__()
        self.stream.__enter__()
        return self

    def __exit__(self, *exc):
        self.stream.__exit__(*exc)
        self.dev.__exit__(*exc)


_HDR = "typedef int int32_t; typedef long long int64_t; typedef unsigned int uint32_t; typedef unsigned char uint8_t;\n"

# ----------------------------------------------------------------------------- kernels

_ORDER_KEY_SRC = (
    _HDR
    + r"""
// One warp per list: lanes read consecutive slots (coalesced) up to the first empty slot, then the
// warp reduces the per-lane minima / maximum / count (min, max and + are order-independent).
extern "C" __global__ void order_key(const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    const uint32_t* __restrict__ hash_a, const uint32_t* __restrict__ hash_b, int64_t* __restrict__ keys,
    int32_t* __restrict__ counts, const int32_t n_lists, const int32_t batch_size, const int32_t hk_off, const int32_t kb_first)
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
    if (kb_first) { int32_t kb = (mx - hk_off) / 768; kb = kb < 0 ? 0 : (kb > 127 ? 127 : kb); key = kb; }
    for (int j = 0; j < NKEYS; ++j) key = (key << BITS) | (int64_t)mn[j];
    keys[li] = key;
    counts[li] = n;
}
"""
)


def _order_key_kernel(max_active: int, n_keys: int = 4, key_bits: int = 14):
    src = _render(_ORDER_KEY_SRC, MAXA=max_active, NKEYS=n_keys, BITS=key_bits)
    return _get_kernel(("order_key", max_active, n_keys, key_bits), src, "order_key")


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
    return _get_kernel(("tile_key", max_active, T), src, "tile_key")


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
MERGE_SHARED = True
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
    return _get_kernel(
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
    return _get_kernel(
        ("merge_runs_sh", nsub, cap),
        _render(_MERGE_SH_SRC, NSUB=nsub),
        "merge_runs_sh",
        dynamic_shared=cap * 4,
    )


def _merge_kernel(nsub: int):
    return _get_kernel(
        ("merge_runs", nsub), _render(_MERGE_SRC, NSUB=nsub), "merge_runs"
    )


def _tile_sort_kernel(cap: int):
    src = _render(_TILE_SORT_SRC, CAP=cap)
    return _get_kernel(("tile_sort", cap), src, "tile_sort", dynamic_shared=cap * 4)


_SCATTER_SRC = (
    _HDR
    + r"""
#define GV @GV@
// One block per (tile, column slice). NWARPS warps; lane owns columns col0 + lane + 32*c, c < CPL.
extern "C" __global__ void __launch_bounds__(@NWARPS@ * 32) tiled_scatter(
    const float* __restrict__ Gw, const float* __restrict__ Gb, const int32_t* __restrict__ order, const int32_t* __restrict__ tile_map,
    const int32_t* __restrict__ tile_start, const uint32_t* __restrict__ e_key,
    float* __restrict__ grad_weight, float* __restrict__ grad_bias, const int32_t n_lists, const int32_t batch_size,
    uint8_t* __restrict__ row_touched,
    const float* __restrict__ grad_l0, const float* __restrict__ acc_w, const float* __restrict__ acc_b,
    const uint32_t* __restrict__ code, const float max_act)
{
    const int32_t T = @T@, SLICE = @SLICE@, OUT = @OUT@, CPL = @CPL@, LOCAL_BITS = @LOCAL_BITS@;
    // PRO: the block's SLICE columns are SLICE / 2 columns of the first half of L1 and the same columns of
    // the second half, and the tile load computes the accumulator gradients itself (fused prologue)
    const int32_t PRO = @PRO@, H = @H@, HS = SLICE / 2;
    extern __shared__ float Gsh[];                          // T * SLICE
    const int32_t tile = tile_map[@SLICE_FASTEST@ ? blockIdx.y : blockIdx.x];
    const int32_t slice = @SLICE_FASTEST@ ? blockIdx.x : blockIdx.y;
    const int32_t col0 = slice * SLICE;
    const int32_t colh = slice * HS;
#define SCOL(x) (PRO ? ((x) < HS ? colh + (x) : H + colh + (x) - HS) : col0 + (x))

    // 1. the tile's gradient rows for this column slice; every thread issues all its loads (PF list ids,
    // then PF values) before its shared-memory stores
    if (PRO) {
        // per (list, column pair): code = list id | half << 30 | reference << 31 (tiled_ft_fused.pro_codes).
        // With us / them = (1, 0) or (0, 1) the list's gradient at columns (i, i + H) depends only on its own
        // accumulators and on grad_l0 in its half; other positions use the reference rows in Gw / Gb.
        const int NT = @NWARPS@ * 32, NP = T * HS, PF = (NP + NT - 1) / NT;
        uint32_t cd[PF];
#pragma unroll
        for (int k = 0; k < PF; ++k) {
            const int idx = threadIdx.x + k * NT;
            const int32_t li = tile * T + idx / HS;
            cd[k] = (idx < NP && li < n_lists) ? __ldg(&code[li]) : 0xFFFFFFFFu;
        }
        float a0[PF], a1[PF], g[PF];
#pragma unroll
        for (int k = 0; k < PF; ++k) {
            const int x = (threadIdx.x + k * NT) % HS;
            a0[k] = a1[k] = g[k] = 0.0f;
            if (cd[k] != 0xFFFFFFFFu) {
                const int32_t lid = (int32_t)(cd[k] & 0x3FFFFFFFu);
                const bool white = lid < batch_size, ref = (cd[k] >> 31) != 0u;
                const int64_t b = white ? lid : lid - batch_size;
                const float* src = ref ? (white ? Gw : Gb) : (white ? acc_w : acc_b);
                a0[k] = __ldg(&src[b * OUT + colh + x]);
                a1[k] = __ldg(&src[b * OUT + H + colh + x]);
                if (!ref) g[k] = __ldg(&grad_l0[b * OUT + ((cd[k] >> 30) & 1u) * H + colh + x]);
            }
        }
#pragma unroll
        for (int k = 0; k < PF; ++k) {
            const int idx = threadIdx.x + k * NT, l = idx / HS, x = idx - l * HS;
            float v0 = a0[k], v1 = a1[k];
            if (cd[k] != 0xFFFFFFFFu && (cd[k] >> 31) == 0u) {
                float c0 = a0[k], c1 = a1[k];
                if (c0 < 0.0f) c0 = 0.0f; else if (c0 > max_act) c0 = max_act;
                if (c1 < 0.0f) c1 = 0.0f; else if (c1 > max_act) c1 = max_act;
                v0 = (c0 == 0.0f || c0 == max_act) ? 0.0f : g[k] * c1;
                v1 = (c1 == 0.0f || c1 == max_act) ? 0.0f : g[k] * c0;
            }
            if (idx < NP) {
                Gsh[l * SLICE + (x & 31) * CPL + (x >> 5)] = v0;
                Gsh[l * SLICE + ((x + HS) & 31) * CPL + ((x + HS) >> 5)] = v1;
            }
        }
    } else {
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
    }
    __syncthreads();

    // 2. bias gradient: column sums of the tile
    if (threadIdx.x < SLICE) {
        float s = 0.0f;
        const int pc = (threadIdx.x & 31) * CPL + (threadIdx.x >> 5);
        for (int l = 0; l < T; ++l) s += Gsh[l * SLICE + pc];
        if (s != 0.0f) atomicAdd(&grad_bias[SCOL(threadIdx.x)], s);
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
                        float* gw = grad_weight + (int64_t)cur * OUT + SCOL(lane);
#pragma unroll
                        for (int c = 0; c < CPL; ++c) { atomicAdd(gw + (PRO ? c * (H - HS) : 0) + 32 * c, acc[c]); acc[c] = 0.0f; }
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
                    float* gw = grad_weight + (int64_t)cur * OUT + SCOL(lane);
#pragma unroll
                    for (int c = 0; c < CPL; ++c) { atomicAdd(gw + (PRO ? c * (H - HS) : 0) + 32 * c, acc[c]); acc[c] = 0.0f; }
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
        float* gw = grad_weight + (int64_t)cur * OUT + SCOL(lane);
#pragma unroll
        for (int c = 0; c < CPL; ++c) atomicAdd(gw + (PRO ? c * (H - HS) : 0) + 32 * c, acc[c]);
    }
}
"""
)


from model.optimizers import grad_row_flags as _grf


def _scatter_kernel(
    T: int,
    slice_cols: int,
    l1_size: int,
    out_size: int,
    slice_fastest: bool,
    nwarps: int,
    pro: bool = False,
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
        SLICE_FASTEST=1 if slice_fastest else 0,
        NWARPS=nwarps,
        GV={1: "float", 2: "float2", 4: "float4"}[slice_cols // 32],
        PRO=1 if pro else 0,
        H=l1_size // 2,
    )
    key = ("scatter", T, slice_cols, l1_size, out_size, slice_fastest, nwarps, pro)
    return _get_kernel(key, src, "tiled_scatter", dynamic_shared=shm)


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
    return _get_kernel(
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
    tile_map: (
        torch.Tensor
    )  # (n_tiles,) int32: block index -> tile (identity unless grid == "shuffle")

    def record_stream(self, stream):
        for t in (self.order, self.tile_start, self.e_key, self.tile_map):
            t.record_stream(stream)


def _mark(timers, name, dev):
    if timers is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream(dev))
        timers.setdefault("_events", []).append((name, e))


def finish_timers(timers, dev=None):
    """Turn the events recorded by prepare()/tiled_backward() into ms per stage (synchronises)."""
    ev = timers.pop("_events", [])
    if not ev:
        return timers
    torch.cuda.synchronize(dev)
    from itertools import pairwise

    for (n0, e0), (n1, e1) in pairwise(ev):
        if n1 == "start":
            continue
        timers[n1] = timers.get(n1, 0.0) + e0.elapsed_time(e1)
    return timers


def list_counts(white: torch.Tensor, black: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [(white >= 0).sum(1, dtype=torch.int32), (black >= 0).sum(1, dtype=torch.int32)]
    )


@torch.compiler.disable
def list_order(
    white: torch.Tensor,
    black: torch.Tensor,
    order: str = LIST_ORDER,
    hk_off: int = HALFKA_FIRST_ROW,
    seed: int = 0,
    with_counts: bool = False,
):
    """Permutation of the perspective lists (white 0..B-1, black B..2B-1; int32) used to form the tiles.
    With ``with_counts`` also returns the number of active rows per list (int32, list order)."""
    B, maxa = white.shape
    n_lists = 2 * B
    dev = white.device
    if order == "identity":
        o = torch.arange(n_lists, device=dev, dtype=torch.int32)
        return (o, list_counts(white, black)) if with_counts else o
    if order not in ("kb_minhash", "minhash"):
        raise ValueError(f"unknown list order {order!r}")
    g = torch.Generator(device=dev).manual_seed(seed)
    a = (torch.randint(1, 2**31 - 1, (4,), generator=g, device=dev) | 1).to(torch.int32)
    b = torch.randint(0, 2**31 - 1, (4,), generator=g, device=dev).to(torch.int32)
    keys = torch.empty(n_lists, dtype=torch.int64, device=dev)
    counts = torch.empty(n_lists, dtype=torch.int32, device=dev)
    with _cupy_stream(dev):
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
                np.int32(hk_off),
                np.int32(1 if order == "kb_minhash" else 0),
            ),
        )
    o = torch.sort(keys)[1].to(torch.int32)
    return (o, counts) if with_counts else o


def _sort_cap(T: int) -> int:
    cap = 1
    while cap < T * 128:
        cap <<= 1
    return cap


@torch.compiler.disable
def prepare(
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    T: int = TILE_SIZE,
    order: str = LIST_ORDER,
    hk_off: int = HALFKA_FIRST_ROW,
    sort: str = TILE_SORT,
    grid: str = BLOCK_GRID,
    timers: dict | None = None,
    order_perm: torch.Tensor | None = None,
    list_counts_cache: torch.Tensor | None = None,
) -> TiledPrep:
    """Build the tile structure for a batch (both perspectives; only the indices are read).
    ``order_perm``: an already computed list permutation (int32, 2B) to reuse instead of ``list_order``
    (the tiled forward builds its smaller tiles from the backward's order)."""
    assert white_indices.dtype == torch.int32 and white_indices.is_cuda
    assert 1 <= T <= MAX_T
    dev = white_indices.device
    _mark(timers, "start", dev)
    white_indices = white_indices.contiguous()
    black_indices = black_indices.contiguous()
    B, maxa = white_indices.shape
    n_lists = 2 * B
    n_tiles = (n_lists + T - 1) // T
    if order_perm is None:
        o, counts = list_order(
            white_indices, black_indices, order, hk_off, with_counts=True
        )
    else:
        o, counts = (
            order_perm,
            (
                list_counts(white_indices, black_indices)
                if list_counts_cache is None
                else list_counts_cache
            ),
        )
    _mark(timers, "order", dev)
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
    with _cupy_stream(dev):
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
    _mark(timers, "keys", dev)
    cap = 1024
    while (
        cap < max_tile
    ):  # shared-memory capacity from the actual largest tile (one bitonic sort per tile)
        cap <<= 1
    cap_sub = 1024
    while cap_sub < max_sub:
        cap_sub <<= 1
    sub_sorted = None
    if sort == "block" and cap <= SORT_CAP_MAX:
        with _cupy_stream(dev):
            _tile_sort_kernel(cap)(
                grid=(n_tiles,),
                block=(1024,),
                shared_mem=cap * 4,
                args=(tile_start.data_ptr(), keys.data_ptr()),
            )
    elif sort == "block" and SUB < T and cap_sub <= SORT_CAP_MAX:
        # T = 256 tiles (~23k entries) exceed shared memory: sort the 64-list sub-tiles in shared memory,
        # then merge every tile's T / SUB sorted runs by rank (keys are unique across runs: the local list
        # index differs). Same keys as the global sort; the sorted sub-tiles serve the pull forward.
        merged = torch.empty_like(keys)
        with _cupy_stream(dev):
            if (
                RSORT_BS > 0
                and max_sub <= RSORT_BS * RSORT_IPT
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
        _mark(timers, "subsort", dev)
        with _cupy_stream(dev):
            if max_tile * 4 <= MAX_SHARED_BYTES and MERGE_SHARED:
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
        sub_sorted = (SUB, sub_start, keys, max_sub)
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
    _mark(timers, "sort", dev)
    if grid == "shuffle":
        tile_map = torch.randperm(
            n_tiles, device=dev, generator=torch.Generator(device=dev).manual_seed(1)
        ).to(torch.int32)
    else:
        tile_map = torch.arange(n_tiles, device=dev, dtype=torch.int32)
    prep = TiledPrep(T, n_lists, o, tile_start, keys, n_tiles, tile_map)
    prep.counts = counts  # active rows per list, in list order (int32); reusable by the tiled forward
    prep.max_tile = max_tile  # largest number of entries in one tile (the tiled forward sizes its staging with it)
    prep.sub_sorted = (
        sub_sorted  # (SUB, sub_start, row-sorted sub-tile keys, max entries) or None
    )
    return prep


# side table: data_ptr(white_indices) -> (weakref to the registered tensor, shape, prep)
_PREP_CACHE_MAX = 16
_prep_cache: collections.OrderedDict[int, tuple] = collections.OrderedDict()
_prep_lock = threading.Lock()


def register_prep(
    white_indices: torch.Tensor, black_indices: torch.Tensor, **kw
) -> TiledPrep | None:
    """Compute and store the preparation for a batch (called by the loader's prefetch worker on
    its own CUDA stream). Never raises into the loader: failures only mean a fallback later.
    Returns the preparation (the tiled forward reuses its list order) or None when disabled."""
    if not (
        TILED_BACKWARD_ENABLED and PREFETCH_PREP and _HAS_CUPY and white_indices.is_cuda
    ):
        return None
    prep = prepare(white_indices, black_indices, **kw)
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


def applicable(l1_size: int, slice_cols: int = COLUMNS_PER_BLOCK) -> bool:
    """Whether the tiled backward can run for this layer width (else the plain kernel is used)."""
    return _HAS_CUPY and l1_size % slice_cols == 0 and l1_size >= slice_cols


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
    grid: str = BLOCK_GRID,
    nwarps: int = WARPS_PER_BLOCK,
    timers: dict | None = None,
    pro=None,
):
    """Weight/bias gradient of the double sparse linear layer (white and black lists share weight
    and bias); ``grad_w`` / ``grad_b`` (B, OUT) are the output gradients. Columns l1_size.. (the
    PSQT columns) use plain atomics. Returns (grad_weight, grad_bias).
    ``pro`` = (grad_l0, acc_w, acc_b, code, max_act): the scatter computes the output gradients from
    grad_l0 and the accumulators itself (tiled_ft_fused); grad_w / grad_b then only hold the reference
    rows of the positions flagged in ``code``."""
    dev = grad_w.device
    B, maxa = white_indices.shape
    OUT = grad_w.shape[1]
    _mark(timers, "start", dev)
    if prep is None:
        STATS["fallback"] += 1
        # the fallback tile size must fit the caller's column slice into shared memory
        T_fb = min(TILE_SIZE, MAX_T, (MAX_SHARED_BYTES // (slice_cols * 4)) // 32 * 32)
        prep = prepare(white_indices, black_indices, T=T_fb, grid=grid)
        _mark(timers, "prep_fallback", dev)
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
    _mark(timers, "alloc", dev)
    with _cupy_stream(dev):
        n_slices = l1_size // slice_cols
        slice_fastest = grid == "slice"
        launch_grid = (
            (n_slices, prep.n_tiles) if slice_fastest else (prep.n_tiles, n_slices)
        )
        if pro is not None and (OUT != l1_size or slice_cols != 64):
            raise ValueError(
                "tiled backward: the fused prologue needs OUT == l1_size and 64-column slices"
            )
        extra = (
            (
                pro[0].data_ptr(),
                pro[1].data_ptr(),
                pro[2].data_ptr(),
                pro[3].data_ptr(),
                np.float32(pro[4]),
            )
            if pro is not None
            else (0, 0, 0, 0, np.float32(0))
        )
        _scatter_kernel(
            prep.T, slice_cols, l1_size, OUT, slice_fastest, nwarps, pro is not None
        )(
            grid=launch_grid,
            block=(32 * nwarps,),
            shared_mem=prep.T * slice_cols * 4 + nwarps * 128,
            args=(
                grad_w.data_ptr(),
                grad_b.data_ptr(),
                prep.order.data_ptr(),
                prep.tile_map.data_ptr(),
                prep.tile_start.data_ptr(),
                prep.e_key.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(n_lists),
                np.int32(B),
                row_touched.data_ptr(),
                *extra,
            ),
        )
        _mark(timers, "scatter", dev)
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
        _mark(timers, "psqt", dev)
    _grf.register(grad_weight, row_touched, reset=True)
    return grad_weight, grad_bias
