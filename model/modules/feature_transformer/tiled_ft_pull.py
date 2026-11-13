"""Pull-style tiled forward of the double feature transformer.

Turns the gather around: the tile's distinct rows (ascending) are
staged into shared memory chunk by chunk (cp.async, double-buffered),
every warp owns LPW lists and keeps their accumulators in registers.
Bit-identical to the 16-list tiled gather (the sum of every output is
bias + rows in ascending order). The tile's row items are read from
shared memory once per warp, not once per list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import tiled_ft_backward as _tb

import os
TL = 64  # lists per tile (divides the backward's tile)
NS = 2   # cp.async pipeline stages
MINB = 3   # min resident blocks per SM (register cap)
R = 32            # staged rows per chunk (entries store the row inside the chunk as uint8)
LPW = 8  # lists per warp
NW = TL // LPW    # warps per block
VEC = 4  # columns per lane
SLICE = 32 * VEC  # columns per block
EPI_FUSED = True   # the gather also writes l0_ (tiled_ft_fused)
BUILD_KERNEL = True   # 0: torch index glue + pull_items / pull_lists
ITEMS_KERNEL = True   # 0: the per-tile list kernel builds the items
MAX_CHUNKS = 16384 // TL   # shared counters of the list kernel (max distinct rows per tile = R * MAX_CHUNKS)

_HDR = "typedef int int32_t; typedef long long int64_t; typedef unsigned int uint32_t; typedef unsigned char uint8_t; typedef unsigned short uint16_t;\n"

_LISTS_SRC = _HDR + r"""
// One block per tile. Entries arrive row-sorted (the backward's keys); seg_incl[e] - 1 - drow_start[t]
// is the entry's rank among the tile's distinct rows. Counting sort by (chunk, list), then every
// (chunk, list) bucket is sorted by row (deterministic; buckets hold ~1-2 entries).
extern "C" __global__ void pull_lists(
    const uint32_t* __restrict__ e_key, const int32_t* __restrict__ seg_incl, const int32_t* __restrict__ tile_start,
    const int32_t* __restrict__ drow_start, const int32_t* __restrict__ chunk_start,
    uint8_t* __restrict__ cnt_g, int32_t* __restrict__ wstart_g, uint8_t* __restrict__ ent_g,
    uint8_t* __restrict__ wn_g, uint16_t* __restrict__ items_g)
{
    const int32_t TL = @TL@, R = @R@, LPW = @LPW@, NW = @NW@, MAXC = @MAXC@, LOCAL_BITS = @LOCAL_BITS@;
    __shared__ __align__(16) uint16_t sc[MAXC * TL];
    __shared__ int32_t tsum[TL];
    const int32_t t = blockIdx.x, l = threadIdx.x;
    const int32_t e0 = tile_start[t], e1 = tile_start[t + 1];
    const int32_t dr = drow_start[t] + 1;
    const int32_t c0 = chunk_start[t], nch = chunk_start[t + 1] - c0;
    for (int i = l; i < nch * TL; i += TL) sc[i] = 0;
    __syncthreads();
    for (int32_t e = e0 + l; e < e1; e += TL) {
        const int32_t r = seg_incl[e] - dr;
        const int32_t ll = (int32_t)(e_key[e] & ((1u << LOCAL_BITS) - 1)) % TL;
        atomicAdd((unsigned int*)&sc[((r / R) * TL + ll) & ~1], ((r / R) * TL + ll) & 1 ? 65536u : 1u);
    }
    __syncthreads();
    // exclusive prefix over the nch * TL counters (chunk-major, then list): thread l scans [l*nch, (l+1)*nch)
    int32_t s = 0;
    for (int i = l * nch; i < (l + 1) * nch; ++i) s += sc[i];
    tsum[l] = s;
    __syncthreads();
    if (l == 0) { int32_t a = 0; for (int i = 0; i < TL; ++i) { const int32_t v = tsum[i]; tsum[i] = a; a += v; } }
    __syncthreads();
    {
        int32_t a = tsum[l];
        for (int i = l * nch; i < (l + 1) * nch; ++i) {
            const int32_t v = sc[i];
            const int32_t c = i / TL, ll = i - c * TL;
            cnt_g[(int64_t)(c0 + c) * TL + ll] = (uint8_t)v;
            if (ll % LPW == 0) wstart_g[(int64_t)(c0 + c) * NW + ll / LPW] = e0 + a;
            sc[i] = (uint16_t)a;
            a += v;
        }
    }
    __syncthreads();
    for (int32_t e = e0 + l; e < e1; e += TL) {
        const int32_t r = seg_incl[e] - dr;
        const int32_t ll = (int32_t)(e_key[e] & ((1u << LOCAL_BITS) - 1)) % TL;
        const int32_t c = r / R, i = c * TL + ll;
        const unsigned old = atomicAdd((unsigned int*)&sc[i & ~1], i & 1 ? 65536u : 1u);
        const int32_t pos = (i & 1) ? (int32_t)(old >> 16) : (int32_t)(old & 0xFFFFu);
        ent_g[e0 + pos] = (uint8_t)(r - c * R);
    }
    __syncthreads();
    // sort list l's bucket of every chunk (start = cursor - count)
    for (int c = 0; c < nch; ++c) {
        const int32_t n = cnt_g[(int64_t)(c0 + c) * TL + l];
        if (n < 2) continue;
        uint8_t* bk = ent_g + e0 + sc[c * TL + l] - n;
        for (int i = 1; i < n; ++i) {
            const uint8_t v = bk[i]; int j = i;
            while (j > 0 && bk[j - 1] > v) { bk[j] = bk[j - 1]; --j; }
            bk[j] = v;
        }
    }
    __syncthreads();
    // per (chunk, warp): the rows its LPW lists use, ascending, each with the mask of the lists that use it
    // (row | mask << 8), written from the warp's first entry on (items <= entries); wn = number of items
    {
        for (int p = l; p < nch * NW; p += TL) {
            const int c = p / NW, w = p - c * NW;
            const int64_t g = (int64_t)(c0 + c);
            uint32_t m[LPW];
            uint32_t u = 0u;
#pragma unroll
            for (int j = 0; j < LPW; ++j) {
                const int32_t ll = w * LPW + j, n = cnt_g[g * TL + ll];
                const uint8_t* bk = ent_g + e0 + sc[c * TL + ll] - n;
                uint32_t mm = 0u;
                for (int i = 0; i < n; ++i) mm |= 1u << bk[i];
                m[j] = mm; u |= mm;
            }
            const int32_t ws = wstart_g[g * NW + w];
            int32_t k = 0;
            while (u) {
                const int32_t r = __ffs(u) - 1;
                u &= u - 1;
                uint32_t mk = 0u;
#pragma unroll
                for (int j = 0; j < LPW; ++j) mk |= ((m[j] >> r) & 1u) << j;
                items_g[ws + k++] = (uint16_t)(r | (mk << 8));
            }
            wn_g[g * NW + w] = (uint8_t)k;
        }
    }
}
"""

_ITEMS_SRC = _HDR + r"""
// One warp per (tile, chunk) (global chunk index = warp; warps past the last chunk exit). Entries
// arrive row-sorted (the backward's keys), so a chunk's entries (distinct-row ranks c*R .. c*R+R-1)
// are one contiguous range, found by binary search on seg_incl. The warp ORs every entry's row bit
// into the mask of its list and counts the entries of every warp group (shared memory), then lane w
// writes group w's first entry (wstart), its items (row | mask of its lists using the row << 8,
// ascending rows) and their number (wn): the same values as the per-tile counting sort did.
extern "C" __global__ void __launch_bounds__(256) pull_items(
    const uint32_t* __restrict__ e_key, const int32_t* __restrict__ seg_incl, const int32_t* __restrict__ tile_start,
    const int32_t* __restrict__ drow_start, const int32_t* __restrict__ chunk_start, const int32_t n_tiles,
    int32_t* __restrict__ wstart_g, uint8_t* __restrict__ wn_g, uint16_t* __restrict__ items_g)
{
    const int32_t TL = @TL@, R = @R@, LPW = @LPW@, NW = @NW@, LOCAL_BITS = @LOCAL_BITS@;
    __shared__ uint32_t sm[8][TL];
    __shared__ int32_t sn[8][NW];
    const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    const int32_t gc = blockIdx.x * 8 + wib;
    if (gc >= chunk_start[n_tiles]) return;
    int32_t lo = 0, hi = n_tiles;                 // last t with chunk_start[t] <= gc
    while (hi - lo > 1) { const int32_t mid = (lo + hi) >> 1; if (chunk_start[mid] <= gc) lo = mid; else hi = mid; }
    const int32_t t = lo, c = gc - chunk_start[t];
    const int32_t e0 = tile_start[t], e1 = tile_start[t + 1];
    const int32_t d0 = drow_start[t], dn = drow_start[t + 1] - d0;
    for (int i = lane; i < TL; i += 32) sm[wib][i] = 0u;
    if (lane < NW) sn[wib][lane] = 0;
    // first entry with distinct-row rank >= c * R (rank = seg_incl - 1 - d0), and the same for (c + 1) * R
    int32_t bnd[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int32_t target = d0 + (c + q) * R;         // first e with seg_incl[e] > target
        if (q == 1 && (c + 1) * R >= dn) { bnd[q] = e1; continue; }
        int32_t a = e0, b = e1;
        while (a < b) { const int32_t m = (a + b) >> 1; if (seg_incl[m] > target) b = m; else a = m + 1; }
        bnd[q] = a;
    }
    const int32_t cs = bnd[0], ce = bnd[1];
    __syncwarp();
    for (int32_t e = cs + lane; e < ce; e += 32) {
        const int32_t rr = seg_incl[e] - 1 - d0 - c * R;
        const int32_t ll = (int32_t)(e_key[e] & ((1u << LOCAL_BITS) - 1)) % TL;
        atomicOr(&sm[wib][ll], 1u << rr);
        atomicAdd(&sn[wib][ll / LPW], 1);
    }
    __syncwarp();
    if (lane < NW) {
        int32_t ws = cs;
        for (int w = 0; w < lane; ++w) ws += sn[wib][w];
        uint32_t m[LPW]; uint32_t u = 0u;
#pragma unroll
        for (int j = 0; j < LPW; ++j) { m[j] = sm[wib][lane * LPW + j]; u |= m[j]; }
        const int64_t g = (int64_t)gc;
        wstart_g[g * NW + lane] = ws;
        int32_t k = 0;
        while (u) {
            const int32_t r = __ffs(u) - 1;
            u &= u - 1;
            uint32_t mk = 0u;
#pragma unroll
            for (int j = 0; j < LPW; ++j) mk |= ((m[j] >> r) & 1u) << j;
            items_g[ws + k++] = (uint16_t)(r | (mk << 8));
        }
        wn_g[g * NW + lane] = (uint8_t)k;
    }
}
"""

_BUILD_SRC = _HDR + r"""
// Fused pull preparation from the row-sorted tile keys (no per-entry global intermediates).
// sub_distinct: number of distinct rows of every tile.
extern "C" __global__ void __launch_bounds__(256) sub_distinct(const uint32_t* __restrict__ e_key,
    const int32_t* __restrict__ tile_start, int32_t* __restrict__ nd)
{
    const int32_t LOCAL_BITS = @LOCAL_BITS@;
    __shared__ int32_t s;
    const int32_t t = blockIdx.x, e0 = tile_start[t], e1 = tile_start[t + 1];
    if (threadIdx.x == 0) s = 0;
    __syncthreads();
    int32_t c = 0;
    for (int32_t e = e0 + threadIdx.x; e < e1; e += blockDim.x)
        c += (e == e0) || ((e_key[e] >> LOCAL_BITS) != (e_key[e - 1] >> LOCAL_BITS));
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xFFFFFFFFu, c, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&s, c);
    __syncthreads();
    if (threadIdx.x == 0) nd[t] = s;
}

// One block: drow_start = exclusive prefix of nd, chunk_start = exclusive prefix of ceil(nd / R) (n + 1 values each).
extern "C" __global__ void __launch_bounds__(1024) sub_offsets(const int32_t* __restrict__ nd, const int32_t n,
    int32_t* __restrict__ drow_start, int32_t* __restrict__ chunk_start)
{
    const int32_t R = @R@;
    __shared__ int32_t wa[32], wb[32];
    __shared__ int32_t carry_a, carry_b;
    if (threadIdx.x == 0) { carry_a = 0; carry_b = 0; drow_start[0] = 0; chunk_start[0] = 0; }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int32_t base = 0; base < n; base += blockDim.x) {
        const int32_t i = base + threadIdx.x;
        int32_t a = i < n ? nd[i] : 0;
        int32_t b = (a + R - 1) / R;
        for (int o = 1; o < 32; o <<= 1) {
            const int32_t xa = __shfl_up_sync(0xFFFFFFFFu, a, o), xb = __shfl_up_sync(0xFFFFFFFFu, b, o);
            if (lane >= o) { a += xa; b += xb; }
        }
        if (lane == 31) { wa[w] = a; wb[w] = b; }
        __syncthreads();
        if (w == 0) {
            int32_t ya = wa[lane], yb = wb[lane];
            for (int o = 1; o < 32; o <<= 1) {
                const int32_t xa = __shfl_up_sync(0xFFFFFFFFu, ya, o), xb = __shfl_up_sync(0xFFFFFFFFu, yb, o);
                if (lane >= o) { ya += xa; yb += xb; }
            }
            wa[lane] = ya; wb[lane] = yb;
        }
        __syncthreads();
        const int32_t pa = carry_a + a + (w ? wa[w - 1] : 0), pb = carry_b + b + (w ? wb[w - 1] : 0);
        if (i < n) { drow_start[i + 1] = pa; chunk_start[i + 1] = pb; }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) { carry_a = pa; carry_b = pb; }
        __syncthreads();
    }
}

// One block (8 warps) per tile: the distinct-row rank of every entry (block scan, shared memory),
// the tile's distinct rows, then one warp per chunk builds wstart / wn / items as pull_items does.
extern "C" __global__ void __launch_bounds__(256) pull_build(
    const uint32_t* __restrict__ e_key, const int32_t* __restrict__ tile_start,
    const int32_t* __restrict__ drow_start, const int32_t* __restrict__ chunk_start,
    int32_t* __restrict__ tile_rows, int32_t* __restrict__ wstart_g, uint8_t* __restrict__ wn_g, uint16_t* __restrict__ items_g)
{
    const int32_t TL = @TL@, R = @R@, LPW = @LPW@, NW = @NW@, LOCAL_BITS = @LOCAL_BITS@;
    extern __shared__ uint16_t rk[];          // distinct-row rank of every entry of the tile
    __shared__ uint32_t sm[8][TL];
    __shared__ int32_t sn[8][NW];
    __shared__ int32_t wsum[8];
    const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    const int32_t t = blockIdx.x, e0 = tile_start[t], n = tile_start[t + 1] - e0;
    const int32_t d0 = drow_start[t], c0 = chunk_start[t], nch = chunk_start[t + 1] - c0;
    // contiguous segment per thread: count, block exclusive scan, then ranks and distinct rows
    const int32_t ch = (n + 255) / 256, i0 = min(threadIdx.x * ch, n), i1 = min(i0 + ch, n);
    int32_t cnt = 0;
    uint32_t prev = i0 > 0 ? (e_key[e0 + i0 - 1] >> LOCAL_BITS) : 0xFFFFFFFFu;
    for (int32_t i = i0; i < i1; ++i) { const uint32_t r = e_key[e0 + i] >> LOCAL_BITS; cnt += (i == 0) || (r != prev); prev = r; }
    int32_t x = cnt;
    for (int o = 1; o < 32; o <<= 1) { const int32_t y = __shfl_up_sync(0xFFFFFFFFu, x, o); if (lane >= o) x += y; }
    if (lane == 31) wsum[wib] = x;
    __syncthreads();
    int32_t before = x - cnt;
    for (int w = 0; w < wib; ++w) before += wsum[w];
    prev = i0 > 0 ? (e_key[e0 + i0 - 1] >> LOCAL_BITS) : 0xFFFFFFFFu;
    int32_t rank = before - 1;
    for (int32_t i = i0; i < i1; ++i) {
        const uint32_t r = e_key[e0 + i] >> LOCAL_BITS;
        if ((i == 0) || (r != prev)) { ++rank; tile_rows[d0 + rank] = (int32_t)r; }
        rk[i] = (uint16_t)rank;
        prev = r;
    }
    __syncthreads();
    for (int32_t c = wib; c < nch; c += 8) {
        for (int i = lane; i < TL; i += 32) sm[wib][i] = 0u;
        if (lane < NW) sn[wib][lane] = 0;
        int32_t bnd[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {             // first entry with rank >= (c + q) * R
            const int32_t target = (c + q) * R;
            int32_t a = 0, b = n;
            if (q == 1 && c + 1 == nch) a = n;
            while (a < b) { const int32_t m = (a + b) >> 1; if ((int32_t)rk[m] >= target) b = m; else a = m + 1; }
            bnd[q] = a;
        }
        __syncwarp();
        for (int32_t i = bnd[0] + lane; i < bnd[1]; i += 32) {
            const int32_t rr = (int32_t)rk[i] - c * R;
            const int32_t ll = (int32_t)(e_key[e0 + i] & ((1u << LOCAL_BITS) - 1)) % TL;
            atomicOr(&sm[wib][ll], 1u << rr);
            atomicAdd(&sn[wib][ll / LPW], 1);
        }
        __syncwarp();
        if (lane < NW) {
            int32_t ws = e0 + bnd[0];
            for (int w = 0; w < lane; ++w) ws += sn[wib][w];
            uint32_t m[LPW]; uint32_t u = 0u;
#pragma unroll
            for (int j = 0; j < LPW; ++j) { m[j] = sm[wib][lane * LPW + j]; u |= m[j]; }
            const int64_t g = (int64_t)(c0 + c);
            wstart_g[g * NW + lane] = ws;
            int32_t k = 0;
            while (u) {
                const int32_t r = __ffs(u) - 1;
                u &= u - 1;
                uint32_t mk = 0u;
#pragma unroll
                for (int j = 0; j < LPW; ++j) mk |= ((m[j] >> r) & 1u) << j;
                items_g[ws + k++] = (uint16_t)(r | (mk << 8));
            }
            wn_g[g * NW + lane] = (uint8_t)k;
        }
        __syncwarp();
    }
}
"""

_PULL_SRC = _HDR + r"""
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

#define VT @VT@
#define UNROLL @UNROLL@
__device__ __forceinline__ float2 vadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float4 vadd(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
extern "C" __global__ void __launch_bounds__(@NW@ * 32, @MINB@) pull_gather(
    const float* __restrict__ weight, const float* __restrict__ bias, const int32_t* __restrict__ order,
    const int32_t* __restrict__ tile_rows, const int32_t* __restrict__ drow_start, const int32_t* __restrict__ chunk_start,
    const uint8_t* __restrict__ wn_g, const int32_t* __restrict__ wstart_g, const uint16_t* __restrict__ items_g,
    float* __restrict__ out_w, float* __restrict__ out_b, const int32_t n_lists, const int32_t batch_size,
    const float* __restrict__ us, const float* __restrict__ them, float* __restrict__ l0_out, uint8_t* __restrict__ mark,
    const float max_act, const int act_q, const float act_s, const float act_eps, const float act_inv, const float act_c)
{
    const int32_t TL = @TL@, R = @R@, LPW = @LPW@, NW = @NW@, SLICE = @SLICE@, OUT = @OUT@, VEC = SLICE / 32;
    // EPI: the block's columns are SLICE / 2 columns of the first half of L1 and the same columns of the
    // second half (lane l and lane l ^ 16 hold a pairwise-product pair); else SLICE consecutive columns
    const int32_t EPI = @EPI@, H = @H@, L1 = 2 * @H@;
    const int32_t NS = @NS@;
    extern __shared__ __align__(16) float wsh_dyn[];   // NS stages of R * SLICE floats
    const int32_t t = blockIdx.x;
    const int32_t col0 = blockIdx.y * SLICE;
    const int32_t colh = blockIdx.y * (SLICE / 2);
#define COLMAP(x) (EPI ? ((x) < SLICE / 2 ? colh + (x) : H + colh + (x) - SLICE / 2) : col0 + (x))
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int32_t d0 = drow_start[t], nd = drow_start[t + 1] - d0;
    const int32_t c0 = chunk_start[t], nch = chunk_start[t + 1] - c0;
    const VT bv = *(const VT*)(bias + COLMAP(VEC * lane));
    VT acc[LPW];
#pragma unroll
    for (int j = 0; j < LPW; ++j) acc[j] = bv;
    // staging: R rows x SLICE floats = R * SLICE / 4 pieces of 16 B
    const int NT = NW * 32, PPR = SLICE / 4, PPT = R * SLICE / 4 / NT;   // 16-byte pieces per thread per chunk
    int32_t rowreg[PPT];
    auto load_rows = [&](int c) {
#pragma unroll
        for (int i = 0; i < PPT; ++i) {
            const int32_t ri = c * R + (tid + i * NT) / PPR;
            rowreg[i] = (c < nch && ri < nd) ? __ldg(&tile_rows[d0 + ri]) : -1;
        }
    };
    auto stage = [&](int buf) {               // stages the chunk whose rows are in rowreg
#pragma unroll
        for (int i = 0; i < PPT; ++i) {
            const int p = tid + i * NT, rr = p / PPR, q = p - rr * PPR;
            const int32_t row = rowreg[i];
            cp_async16(wsh_dyn + buf * (R * SLICE) + rr * SLICE + q * 4,
                       weight + (int64_t)(row < 0 ? 0 : row) * OUT + COLMAP(q * 4), row < 0 ? 0 : 16);
        }
        cp_async_commit();
    };
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) { load_rows(s); stage(s); }
    load_rows(NS - 1);
    const int32_t clast = c0 + (nch > 0 ? nch - 1 : 0);
    // items of (chunk c, this warp): lane k holds item k (a chunk has at most R = 32 distinct rows)
    int32_t ws = __ldg(&wstart_g[(int64_t)c0 * NW + warp]);
    int32_t ws_n = __ldg(&wstart_g[(int64_t)min(c0 + 1, clast) * NW + warp]);
    int32_t n = (int32_t)__ldg(&wn_g[(int64_t)c0 * NW + warp]);
    int32_t item = (int32_t)__ldg(&items_g[ws + lane]);
    for (int32_t c = 0; c < nch; ++c) {
        const int buf = c % NS;
        cp_async_wait<NS - 2>();
        __syncthreads();                       // chunk c staged; every warp is done with chunk c - 1
        stage((c + NS - 1) % NS);              // chunk c + NS - 1 (empty pieces past the tile)
        load_rows(c + NS);
        // metadata of chunk c + 1 (and the warp start of c + 2) while chunk c is processed
        const int32_t gn = min(c0 + c + 1, clast);
        const int32_t n_n = (int32_t)__ldg(&wn_g[(int64_t)gn * NW + warp]);
        const int32_t item_n = (int32_t)__ldg(&items_g[ws_n + lane]);
        const int32_t ws_nn = __ldg(&wstart_g[(int64_t)min(c0 + c + 2, clast) * NW + warp]);
        // every staged row the warp uses is read from shared memory once and added to each of its lists
        // that uses it; every list adds its rows in ascending order
        const float* wb = wsh_dyn + buf * (R * SLICE) + VEC * lane;
        int32_t k = 0;
#if UNROLL >= 4
        for (; k + 3 < n; k += 4) {
            int32_t it[4]; VT v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) it[q] = __shfl_sync(0xFFFFFFFFu, item, k + q);
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = *(const VT*)(wb + (it[q] & 255) * SLICE);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
                for (int j = 0; j < LPW; ++j) if ((it[q] >> (8 + j)) & 1) acc[j] = vadd(acc[j], v[q]);
            }
        }
#endif
#if UNROLL >= 2
        for (; k + 1 < n; k += 2) {
            const int32_t it0 = __shfl_sync(0xFFFFFFFFu, item, k), it1 = __shfl_sync(0xFFFFFFFFu, item, k + 1);
            const VT v0 = *(const VT*)(wb + (it0 & 255) * SLICE);
            const VT v1 = *(const VT*)(wb + (it1 & 255) * SLICE);
#pragma unroll
            for (int j = 0; j < LPW; ++j) if ((it0 >> (8 + j)) & 1) acc[j] = vadd(acc[j], v0);
#pragma unroll
            for (int j = 0; j < LPW; ++j) if ((it1 >> (8 + j)) & 1) acc[j] = vadd(acc[j], v1);
        }
#endif
        for (; k < n; ++k) {
            const int32_t it = __shfl_sync(0xFFFFFFFFu, item, k);
            const VT v = *(const VT*)(wb + (it & 255) * SLICE);
#pragma unroll
            for (int j = 0; j < LPW; ++j) if ((it >> (8 + j)) & 1) acc[j] = vadd(acc[j], v);
        }
        n = n_n; item = item_n; ws = ws_n; ws_n = ws_nn;
    }
#pragma unroll
    for (int j = 0; j < LPW; ++j) {
        const int32_t li = t * TL + warp * LPW + j;
        if (li < n_lists) {
            const int32_t lid = __ldg(&order[li]);
            float* dst = lid < batch_size ? out_w + (int64_t)lid * OUT : out_b + (int64_t)(lid - batch_size) * OUT;
            *(VT*)(dst + COLMAP(VEC * lane)) = acc[j];
        }
    }
    if (!EPI) return;
    // fused epilogue: with us / them = (1, 0) or (0, 1) the perspective mix selects this list's own
    // accumulators, so the list writes its half of l0_ (white: half 0 when us == 1, else half 1) with
    // the epilogue's clamp / product / quantization. A list whose values include -0 or a non-finite
    // value (where u * w + th * b could differ from w) marks its position; tiled_ft_fixup recomputes
    // marked and non-binary positions with the reference epilogue.
#pragma unroll
    for (int j = 0; j < LPW; ++j) {
        const int32_t li = t * TL + warp * LPW + j;
        if (li >= n_lists) continue;                       // uniform over the warp
        const int32_t lid = __ldg(&order[li]);
        const float a[4] = {acc[j].x, acc[j].y, acc[j].z, acc[j].w};
        float p[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) p[q] = __shfl_xor_sync(0xFFFFFFFFu, a[q], 16);
        bool bad = false;
#pragma unroll
        for (int q = 0; q < 4; ++q) bad |= (a[q] == 0.0f && signbit(a[q])) || !isfinite(a[q]);
        const bool white = lid < batch_size;
        const int32_t b = white ? lid : lid - batch_size;
        if (__any_sync(0xFFFFFFFFu, bad) && lane == 0) mark[b] = 1;
        if (lane < 16) {
            const float u = __ldg(&us[b]);
            const int half = (white == (u == 1.0f)) ? 0 : 1;
            float o[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float x0 = a[q], x1 = p[q];
                if (x0 < 0.0f) x0 = 0.0f; else if (x0 > max_act) x0 = max_act;
                if (x1 < 0.0f) x1 = 0.0f; else if (x1 > max_act) x1 = max_act;
                float v = x0 * x1;
                if (act_q) v = __fmul_rn(__fmul_rn(floorf(__fadd_rn(__fmul_rn(v, act_s), act_eps)), act_inv), act_c);
                o[q] = v;
            }
            *(float4*)(l0_out + (int64_t)b * L1 + half * H + colh + 4 * lane) = make_float4(o[0], o[1], o[2], o[3]);
        }
    }
}
"""


def _lists_kernel():
    src = _tb._render(_LISTS_SRC, TL=TL, R=R, LPW=LPW, NW=NW, MAXC=MAX_CHUNKS, LOCAL_BITS=_tb.LOCAL_BITS)
    return _tb._get_kernel(("pull_lists", TL, R, LPW), src, "pull_lists")


def _items_kernel():
    src = _tb._render(_ITEMS_SRC, TL=TL, R=R, LPW=LPW, NW=NW, LOCAL_BITS=_tb.LOCAL_BITS)
    return _tb._get_kernel(("pull_items", TL, R, LPW), src, "pull_items")


def _build_kernels(cap: int):
    src = _tb._render(_BUILD_SRC, TL=TL, R=R, LPW=LPW, NW=NW, LOCAL_BITS=_tb.LOCAL_BITS)
    key = ("pull_build", TL, R, LPW)
    return (_tb._get_kernel(key + ("d",), src, "sub_distinct"), _tb._get_kernel(key + ("o",), src, "sub_offsets"),
            _tb._get_kernel(key + ("b", cap), src, "pull_build", dynamic_shared=cap * 2))


def _pull_kernel(out_size: int, epi_h: int = 0):
    src = _tb._render(_PULL_SRC, TL=TL, R=R, LPW=LPW, NW=NW, SLICE=SLICE, OUT=out_size, NS=NS, VT={2: "float2", 4: "float4"}[VEC], MINB=MINB, UNROLL=int(4),
                      EPI=1 if epi_h else 0, H=max(epi_h, 1))
    shm = NS * R * SLICE * 4
    assert (R * SLICE // 4) % (NW * 32) == 0, "R * SLICE / 4 must be a multiple of the block size"
    assert R <= 32 and LPW <= 8, "items: one bit per staged row in a 32-bit mask, list masks in 8 bits"
    return _tb._get_kernel(("pull_gather", TL, R, LPW, SLICE, out_size, NS, VEC, MINB, 4, epi_h), src, "pull_gather", dynamic_shared=max(shm, 1))


@dataclass
class PullPrep:
    n_lists: int
    n_tiles: int
    order: torch.Tensor
    tile_rows: torch.Tensor
    drow_start: torch.Tensor
    chunk_start: torch.Tensor
    cnt: torch.Tensor
    wstart: torch.Tensor
    ent: torch.Tensor
    wn: torch.Tensor
    items: torch.Tensor

    def record_stream(self, stream):
        for t in (self.order, self.tile_rows, self.drow_start, self.chunk_start, self.cnt, self.wstart, self.ent,
                  self.wn, self.items):
            if t is not None:
                t.record_stream(stream)


@torch.compiler.disable
def prepare(white_indices: torch.Tensor, black_indices: torch.Tensor, bp) -> PullPrep | None:
    """From the backward's preparation ``bp`` (tiles of TL lists, row-sorted keys). None when a tile
    has more than R * MAX_CHUNKS distinct rows (the caller then uses the 16-list gather)."""
    dev = white_indices.device
    ss = getattr(bp, "sub_sorted", None)
    if bp.T != TL and ss is not None and ss[0] == TL:
        # the backward's row-sorted 64-list sub-tiles (local list index still relative to the 256-list tile)
        sub = _tb.TiledPrep(TL, bp.n_lists, bp.order, ss[1], ss[2], int(ss[1].shape[0]) - 1, bp.tile_map)
        sub.max_tile = ss[3]
        bp = sub
    elif bp.T != TL:
        if bp.T % TL:
            return None
        from . import tiled_ft_forward as _tf
        bp = _tf._derive_from_backward(bp, TL, dev, None)
    if getattr(bp, "max_tile", 65536) >= 65536:   # 16-bit prefix counters of the list kernel
        return None
    B, maxa = white_indices.shape
    E = int(bp.e_key.shape[0])
    if bp.max_tile > R * MAX_CHUNKS:      # distinct rows of a tile <= its entries (max_tile: no extra host sync)
        return None
    if BUILD_KERNEL and NW <= 8 and R <= 32 and bp.max_tile * 2 <= _tb.MAX_SHARED_BYTES - 16384:
        # fused: distinct rows per tile -> offsets (one block) -> ranks, distinct rows and items per tile
        n_chunks = E // R + bp.n_tiles                                    # upper bound of sum(ceil(nd / R))
        cap = 4096
        while cap < bp.max_tile:
            cap += 4096
        k_d, k_o, k_b = _build_kernels(cap)
        nd = torch.empty(bp.n_tiles, dtype=torch.int32, device=dev)
        drow_start = torch.empty(bp.n_tiles + 1, dtype=torch.int32, device=dev)
        chunk_start = torch.empty(bp.n_tiles + 1, dtype=torch.int32, device=dev)
        tile_rows = torch.empty(max(E, 1), dtype=torch.int32, device=dev)
        wstart = torch.empty(max(n_chunks, 1) * NW, dtype=torch.int32, device=dev)
        wn = torch.empty(max(n_chunks, 1) * NW, dtype=torch.uint8, device=dev)
        items = torch.zeros(E + 64, dtype=torch.int16, device=dev)        # padded: the gather reads 32 items past a warp start
        with _tb._cupy_stream(dev):
            k_d(grid=(bp.n_tiles,), block=(256,), args=(bp.e_key.data_ptr(), bp.tile_start.data_ptr(), nd.data_ptr()))
            k_o(grid=(1,), block=(1024,), args=(nd.data_ptr(), np.int32(bp.n_tiles), drow_start.data_ptr(), chunk_start.data_ptr()))
            k_b(grid=(bp.n_tiles,), block=(256,), shared_mem=bp.max_tile * 2, args=(
                bp.e_key.data_ptr(), bp.tile_start.data_ptr(), drow_start.data_ptr(), chunk_start.data_ptr(),
                tile_rows.data_ptr(), wstart.data_ptr(), wn.data_ptr(), items.data_ptr()))
        return PullPrep(bp.n_lists, bp.n_tiles, bp.order, tile_rows, drow_start, chunk_start, None, wstart, None, wn, items)
    # no host syncs below: sizes use upper bounds, masks are replaced by scatters
    rows = bp.e_key >> _tb.LOCAL_BITS
    starts = torch.zeros(E + 1, dtype=torch.bool, device=dev)
    starts[1:E] = rows[1:] != rows[:-1]
    starts[bp.tile_start[:-1].long()] = True                       # index E (empty last tiles) lands in the pad slot
    starts = starts[:E]
    seg_before = torch.cumsum(starts, 0, dtype=torch.int32)          # inclusive
    tile_rows = torch.empty(max(E, 1), dtype=torch.int32, device=dev)
    tile_rows[(seg_before - 1).long()] = rows                          # every entry of a segment writes the same row
    drow_start = torch.zeros(bp.n_tiles + 1, dtype=torch.int32, device=dev)
    ends = bp.tile_start[1:].long()
    drow_start[1:] = torch.where(ends > 0, seg_before[(ends - 1).clamp(min=0)], torch.zeros_like(ends, dtype=torch.int32))
    nd = drow_start[1:] - drow_start[:-1]
    nch = (nd + R - 1) // R
    chunk_start = torch.zeros(bp.n_tiles + 1, dtype=torch.int32, device=dev)
    chunk_start[1:] = torch.cumsum(nch, 0)
    n_chunks = E // R + bp.n_tiles                                    # upper bound of sum(ceil(nd / R))
    use_items = ITEMS_KERNEL and NW <= 8 and R <= 32
    cnt = None if use_items else torch.empty(max(n_chunks, 1) * TL, dtype=torch.uint8, device=dev)
    wstart = torch.empty(max(n_chunks, 1) * NW, dtype=torch.int32, device=dev)
    ent = None if use_items else torch.zeros(E + 64, dtype=torch.uint8, device=dev)
    wn = torch.empty(max(n_chunks, 1) * NW, dtype=torch.uint8, device=dev)
    items = torch.zeros(E + 64, dtype=torch.int16, device=dev)        # padded: the gather reads 32 items past a warp start
    with _tb._cupy_stream(dev):
        if use_items:      # cnt / ent are only intermediates of the list kernel; the gather reads wstart / wn / items
            _items_kernel()(grid=((max(n_chunks, 1) + 7) // 8,), block=(256,), args=(
                bp.e_key.data_ptr(), seg_before.data_ptr(), bp.tile_start.data_ptr(), drow_start.data_ptr(),
                chunk_start.data_ptr(), np.int32(bp.n_tiles), wstart.data_ptr(), wn.data_ptr(), items.data_ptr()))
        else:
            _lists_kernel()(grid=(bp.n_tiles,), block=(TL,), args=(
                bp.e_key.data_ptr(), seg_before.data_ptr(), bp.tile_start.data_ptr(), drow_start.data_ptr(),
                chunk_start.data_ptr(), cnt.data_ptr(), wstart.data_ptr(), ent.data_ptr(), wn.data_ptr(), items.data_ptr()))
    return PullPrep(bp.n_lists, bp.n_tiles, bp.order, tile_rows, drow_start, chunk_start, cnt, wstart, ent, wn, items)


def epi_applicable(l1_size: int) -> bool:
    return EPI_FUSED and VEC == 4 and l1_size % 2 == 0 and (l1_size // 2) % (SLICE // 2) == 0


@torch.compiler.disable
def gather(weight, bias, l1_size, prep: PullPrep, out_w, out_b, epi=None):
    """Columns 0..l1_size of the two outputs (B, OUT) in original batch order. ``epi`` = (us, them, l0_out,
    mark, max_act, act_q, act_s, act_eps, act_inv, act_c): also write l0_ (fused epilogue, see the kernel)."""
    OUT = bias.shape[0]
    B = out_w.shape[0]
    if epi is None:
        extra = (0, 0, 0, 0, np.float32(0), np.int32(0), np.float32(0), np.float32(0), np.float32(0), np.float32(0))
    else:
        us, them, l0_out, mark = epi[:4]
        extra = (us.data_ptr(), them.data_ptr(), l0_out.data_ptr(), mark.data_ptr(), *epi[4:])
    with _tb._cupy_stream(weight.device):
        _pull_kernel(OUT, l1_size // 2 if epi is not None else 0)(grid=(prep.n_tiles, l1_size // SLICE), block=(NW * 32,), shared_mem=NS * R * SLICE * 4, args=(
            weight.data_ptr(), bias.data_ptr(), prep.order.data_ptr(), prep.tile_rows.data_ptr(), prep.drow_start.data_ptr(),
            prep.chunk_start.data_ptr(), prep.wn.data_ptr(), prep.wstart.data_ptr(), prep.items.data_ptr(),
            out_w.data_ptr(), out_b.data_ptr(), np.int32(prep.n_lists), np.int32(B), *extra))
