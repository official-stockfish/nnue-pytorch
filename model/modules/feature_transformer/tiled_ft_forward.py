"""Tiled forward for the double feature transformer.

Replaces the per-position gather with a tiled gather: the perspective
lists are ordered by king bucket and min-hash so that consecutive lists
share rows, then grouped into small tiles; one block per (tile, column
slice) keeps the tile's accumulator rows in shared memory and reads
every distinct row of the tile once. Same math as the plain forward
(fp32 sums in a different order: bias first, rows sorted), outputs in
the original batch order. Noise-level on GH200 (the plain gather
matches), so it is off by default; +2% on consumer GPUs.
"""
from __future__ import annotations

import collections
import os
import threading
import weakref
from dataclasses import dataclass

import numpy as np
import torch

from . import tiled_ft_backward as _tb
from . import tiled_ft_pull as _tp

try:
    import cupy as cp

    _HAS_CUPY = True
except (ImportError, OSError, RuntimeError):  # pragma: no cover - CuPy missing
    cp = None
    _HAS_CUPY = False

TILED_FORWARD_ENABLED = False
TILE_SIZE = 16
COLUMNS_PER_BLOCK = 512
VEC = 4
PREFETCH_DEPTH = 8
PREFETCH_PREP = True
PULL_ENABLED = True

LOCAL_BITS = _tb.LOCAL_BITS
MAX_SHARED_BYTES = _tb.MAX_SHARED_BYTES

STATS = collections.Counter()  # "prep_registered", "prep_hit", "fallback", "register_failed"

_HDR = "typedef int int32_t; typedef long long int64_t; typedef unsigned int uint32_t; typedef unsigned char uint8_t; typedef unsigned short uint16_t;\n"

# ----------------------------------------------------------------------------- kernels

_SEGMENTS_SRC = _HDR + r"""
// One block per tile. Entry e of the tile starts a row segment if e == 0 or its row differs from the
// previous entry's row (keys are sorted within the tile). Writes the segment starts (relative to the
// tile's first entry) at seg_start[tile_start[tile] + rank] and the segment count of the tile.
extern "C" __global__ void tile_segments(const int32_t* __restrict__ tile_start, const uint32_t* __restrict__ e_key,
    uint16_t* __restrict__ seg_start, int32_t* __restrict__ tile_nseg)
{
    const int NT = @NT@, NW = NT / 32, LOCAL_BITS = @LOCAL_BITS@;
    __shared__ int warp_sums[NW];
    __shared__ int carry;
    const int32_t tile = blockIdx.x;
    const int32_t e0 = tile_start[tile], n = tile_start[tile + 1] - e0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) carry = 0;
    __syncthreads();
    for (int32_t base = 0; base < n; base += NT) {
        const int32_t e = base + tid;
        int flag = 0;
        if (e < n) {
            const uint32_t row = e_key[e0 + e] >> LOCAL_BITS;
            flag = (e == 0 || row != (e_key[e0 + e - 1] >> LOCAL_BITS)) ? 1 : 0;
        }
        int v = flag;
        for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(0xFFFFFFFFu, v, o); if (lane >= o) v += t; }
        if (lane == 31) warp_sums[warp] = v;
        __syncthreads();
        if (warp == 0) {
            int s = (lane < NW) ? warp_sums[lane] : 0;
            for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(0xFFFFFFFFu, s, o); if (lane >= o) s += t; }
            if (lane < NW) warp_sums[lane] = s;
        }
        __syncthreads();
        const int rank = carry + v - flag + (warp > 0 ? warp_sums[warp - 1] : 0);
        if (flag) seg_start[e0 + rank] = (uint16_t)e;
        __syncthreads();
        if (tid == 0) carry += warp_sums[NW - 1];
        __syncthreads();
    }
    if (tid == 0) tile_nseg[tile] = carry;
}
"""

_PARTITION_SRC = _HDR + r"""
// Stable SUB-way partition of a backward tile's row-sorted keys into its SUB sub-tiles of T lists
// (sub-tile = local / T): row order is preserved inside every sub-tile, so the sub-tiles are the
// row-sorted key streams the tiled forward needs, without writing and sorting them again.
extern "C" __global__ void tile_partition(const int32_t* __restrict__ big_start, const uint32_t* __restrict__ big_key,
    const int32_t* __restrict__ sub_start, uint32_t* __restrict__ sub_key)
{
    const int NT = @NT@, NW = NT / 32, SUB = @SUB@, T = @T@, LOCAL_BITS = @LOCAL_BITS@;
    __shared__ int base[SUB];
    __shared__ int wcount[NW][SUB];
    __shared__ int wtotal[SUB];
    const int32_t tile = blockIdx.x;
    const int32_t e0 = big_start[tile], n = big_start[tile + 1] - e0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned lt = (1u << lane) - 1u;
    if (tid < SUB) base[tid] = sub_start[tile * SUB + tid];
    __syncthreads();
    for (int32_t b0 = 0; b0 < n; b0 += NT) {
        const int32_t e = b0 + tid;
        uint32_t key = 0; int s = -1;
        if (e < n) { key = big_key[e0 + e]; s = (int)(key & ((1u << LOCAL_BITS) - 1)) / T; }
        int rank = 0;
        for (int s2 = 0; s2 < SUB; ++s2) {
            const unsigned m = __ballot_sync(0xFFFFFFFFu, s == s2);
            if (s == s2) rank = __popc(m & lt);
            if (lane == 0) wcount[warp][s2] = __popc(m);
        }
        __syncthreads();
        if (tid < SUB) {
            int acc = 0;
            for (int w = 0; w < NW; ++w) { const int c = wcount[w][tid]; wcount[w][tid] = acc; acc += c; }
            wtotal[tid] = acc;
        }
        __syncthreads();
        if (s >= 0) {
            const uint32_t local = key & ((1u << LOCAL_BITS) - 1);
            sub_key[base[s] + wcount[warp][s] + rank] = (key & ~((1u << LOCAL_BITS) - 1)) | (local - (uint32_t)(s * T));
        }
        __syncthreads();
        if (tid < SUB) base[tid] += wtotal[tid];
        __syncthreads();
    }
}
"""

_GATHER_SRC = _HDR + r"""
#define VT @VT@
__device__ __forceinline__ float  vadd(float a, float b) { return a + b; }
__device__ __forceinline__ float2 vadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float4 vadd(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
extern "C" __global__ void tiled_gather(
    const float* __restrict__ weight, const float* __restrict__ bias,
    const int32_t* __restrict__ order, const int32_t* __restrict__ tile_start, const int32_t* __restrict__ tile_nseg,
    const uint32_t* __restrict__ e_key, const uint16_t* __restrict__ seg_start_g,
    float* __restrict__ out_w, float* __restrict__ out_b, const int32_t n_lists, const int32_t batch_size, const int32_t smax)
{
    const int32_t T = @T@, SLICE = @SLICE@, VEC = @VEC@, OUT = @OUT@, P = @P@, LOCAL_BITS = @LOCAL_BITS@;
    const int32_t NT = SLICE / VEC;                       // threads per block
    extern __shared__ float acc_sh_f[];                   // T * SLICE floats | staged segment table (smax entries)
    VT* acc_sh = (VT*)acc_sh_f;
    int32_t* srow_sh = (int32_t*)(acc_sh_f + T * SLICE);  // smax
    uint16_t* sstart_sh = (uint16_t*)(srow_sh + smax);    // smax + 1
    uint16_t* eloc_sh = sstart_sh + smax + 2;             // smax (byte offsets local * SLICE * 4)
    const int32_t tile = blockIdx.x;
    const int32_t tid = threadIdx.x;
    const int32_t col = blockIdx.y * SLICE + tid * VEC;   // first column of this thread
    const VT bcol = *(const VT*)(bias + col);
    for (int l = 0; l < T; ++l) acc_sh[l * NT + tid] = bcol;
    const int32_t e0 = tile_start[tile];
    const int32_t n_ent = tile_start[tile + 1] - e0;
    const int32_t n_seg = tile_nseg[tile];
    for (int j = tid; j < n_seg; j += NT) {
        const uint16_t st = __ldg(&seg_start_g[e0 + j]);
        sstart_sh[j] = st;
        srow_sh[j] = (int32_t)(__ldg(&e_key[e0 + st]) >> LOCAL_BITS);
    }
    if (tid == 0) sstart_sh[n_seg] = (uint16_t)n_ent;
    for (int j = tid; j < n_ent; j += NT)
        eloc_sh[j] = (uint16_t)((int32_t)(__ldg(&e_key[e0 + j]) & ((1u << LOCAL_BITS) - 1)) * (SLICE * 4));
    __syncthreads();
    char* my_acc = (char*)acc_sh + tid * sizeof(VT);      // accumulator of list l for this thread: my_acc + l * SLICE * 4
#define ACC(off) (*(VT*)(my_acc + (off)))
#define LOAD_GROUP(W, ST, g) { \
        _Pragma("unroll") for (int p = 0; p < P; ++p) { \
            const int32_t gg = min((g) + p, n_seg - 1); \
            W[p] = __ldcg((const VT*)(weight + (int64_t)srow_sh[gg] * OUT + col)); \
            ST[p] = (int32_t)sstart_sh[min((g) + p, n_seg)]; } \
        ST[P] = (int32_t)sstart_sh[min((g) + P, n_seg)]; }
#define PROCESS_GROUP(W, ST) { \
        _Pragma("unroll") for (int p = 0; p < P; ++p) { \
            const VT wv = W[p]; const int32_t e1 = ST[p + 1]; int32_t e = ST[p]; \
            for (; e + 1 < e1; e += 2) { \
                const int32_t oa = eloc_sh[e], ob = eloc_sh[e + 1]; \
                const VT aa = ACC(oa); const VT ab = ACC(ob); \
                const VT na = vadd(aa, wv); \
                ACC(oa) = na; ACC(ob) = vadd(oa == ob ? na : ab, wv); } \
            if (e < e1) { const int32_t o = eloc_sh[e]; ACC(o) = vadd(ACC(o), wv); } } }
    // warps are independent: each owns its columns of every accumulator row (no block sync needed)
    if (n_seg > 0) {
        // double-buffered: the loads of group g+P are in flight while group g is applied
        VT wA[P], wB[P]; int32_t stA[P + 1], stB[P + 1];
        int32_t g = 0;
        LOAD_GROUP(wA, stA, 0);
        while (g < n_seg) {
            LOAD_GROUP(wB, stB, g + P);
            PROCESS_GROUP(wA, stA);
            g += P;
            if (g >= n_seg) break;
            LOAD_GROUP(wA, stA, g + P);
            PROCESS_GROUP(wB, stB);
            g += P;
        }
    }
    for (int l = 0; l < T; ++l) {
        const int32_t li = tile * T + l;
        if (li < n_lists) {
            const int32_t lid = __ldg(&order[li]);
            float* dst = lid < batch_size ? out_w + (int64_t)lid * OUT : out_b + (int64_t)(lid - batch_size) * OUT;
            *(VT*)(dst + col) = acc_sh[l * NT + tid];
        }
    }
}
"""

_PSQT_SRC = _HDR + r"""
// One warp per list: the PSQT columns l1.. of the output (OUT - l1 <= 32 sums per lane).
extern "C" __global__ void psqt_gather(
    const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ out_w, float* __restrict__ out_b, const int32_t n_lists, const int32_t batch_size)
{
    const int32_t MAXA = @MAXA@, L1 = @L1@, OUT = @OUT@, NP = @OUT@ - @L1@;
    const int32_t li = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // one warp per list
    const int32_t lane = threadIdx.x & 31;
    if (li >= n_lists) return;
    const int32_t b = li < batch_size ? li : li - batch_size;
    const int32_t* row = (li < batch_size) ? white + (int64_t)li * MAXA : black + (int64_t)b * MAXA;
    float s[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) s[p] = 0.0f;
    for (int k0 = 0; k0 < MAXA; k0 += 32) {               // warp-uniform trip count
        const int32_t k = k0 + lane;
        const int32_t r = k < MAXA ? row[k] : -1;
        if (__all_sync(0xffffffffu, r < 0)) break;           // lists are left-packed: the rest is padding
        if (r >= 0) {
#pragma unroll
            for (int p = 0; p < NP; ++p) s[p] += __ldg(&weight[(int64_t)r * OUT + L1 + p]);
        }
    }
    float* dst = (li < batch_size ? out_w : out_b) + (int64_t)b * OUT + L1;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        float v = s[p];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) dst[p] = v + bias[L1 + p];
    }
}
"""

_SEG_THREADS = 256


def _partition_kernel(sub: int, T: int):
    src = _tb._render(_PARTITION_SRC, NT=_SEG_THREADS, SUB=sub, T=T, LOCAL_BITS=LOCAL_BITS)
    return _tb._get_kernel(("fwd_partition", sub, T), src, "tile_partition")


def _segments_kernel():
    src = _tb._render(_SEGMENTS_SRC, NT=_SEG_THREADS, LOCAL_BITS=LOCAL_BITS)
    return _tb._get_kernel(("fwd_segments", _SEG_THREADS), src, "tile_segments")


def _gather_kernel(T: int, slice_cols: int, vec: int, l1_size: int, out_size: int, P: int):
    if vec not in (1, 2, 4) or slice_cols % (32 * vec) or l1_size % slice_cols:
        raise ValueError(f"vec must be 1, 2 or 4 and slice a multiple of 32 * vec dividing l1_size; got vec={vec}, slice={slice_cols}")
    if T * slice_cols * 4 > 65535:
        raise ValueError(f"T={T} x slice={slice_cols}: staged byte offsets are uint16 (T * slice * 4 <= 65535)")
    if P < 1:
        raise ValueError("P must be >= 1")
    vt = {1: "float", 2: "float2", 4: "float4"}[vec]
    src = _tb._render(_GATHER_SRC, VT=vt, T=T, SLICE=slice_cols, VEC=vec, OUT=out_size, P=P, LOCAL_BITS=LOCAL_BITS)
    key = ("fwd_gather", T, slice_cols, vec, l1_size, out_size, P)
    return _tb._get_kernel(key, src, "tiled_gather", dynamic_shared=MAX_SHARED_BYTES)


def _psqt_kernel(max_active: int, l1_size: int, out_size: int):
    src = _tb._render(_PSQT_SRC, MAXA=max_active, L1=l1_size, OUT=out_size)
    return _tb._get_kernel(("fwd_psqt", max_active, l1_size, out_size), src, "psqt_gather")


def shared_bytes(T: int, slice_cols: int, smax: int) -> int:
    """Dynamic shared memory of one gather block: accumulators + staged segment table."""
    return T * slice_cols * 4 + smax * 4 + (smax + 2) * 2 + smax * 2 + 16


# ----------------------------------------------------------------------------- preparation

@dataclass
class ForwardPrep:
    """Index-only preparation of one batch for the tiled forward."""

    T: int
    n_lists: int
    n_tiles: int
    order: torch.Tensor       # (n_lists,) int32: list ids in tile order (shared with the backward's prep)
    tile_start: torch.Tensor  # (n_tiles + 1,) int32: entry offsets of the tiles
    e_key: torch.Tensor       # (E,) int32: row << LOCAL_BITS | local, sorted within each tile
    seg_start: torch.Tensor   # (E,) uint16 (stored as int16): per tile, its segment starts relative to tile_start[tile]
    tile_nseg: torch.Tensor   # (n_tiles,) int32
    max_tile: int             # largest number of entries in one tile (sizes the shared-memory staging)

    def record_stream(self, stream):
        for t in (self.order, self.tile_start, self.e_key, self.seg_start, self.tile_nseg):
            t.record_stream(stream)


@torch.compiler.disable
def prepare(white_indices: torch.Tensor, black_indices: torch.Tensor, T: int = TILE_SIZE,
            order_perm: torch.Tensor | None = None, timers: dict | None = None,
            from_backward: "_tb.TiledPrep | None" = None, **order_kw) -> ForwardPrep:
    """Tiles of T lists (of ``order_perm``, else of ``tiled_ft_backward.list_order``) with per-tile
    row-sorted keys and the segment table. Only the feature indices are read. With ``from_backward``
    (the backward's preparation of the same batch, tile size a multiple of T) the sub-tiles are derived
    from its row-sorted keys by a stable partition instead of being written and sorted again."""
    dev = white_indices.device
    if from_backward is not None and from_backward.T % T == 0 and hasattr(from_backward, "counts"):
        bp = _derive_from_backward(from_backward, T, dev, timers)
    else:
        bp = _tb.prepare(white_indices, black_indices, T=T, order_perm=order_perm, timers=timers, **order_kw)
    E = int(bp.e_key.shape[0])
    seg_start = torch.empty(max(E, 1), dtype=torch.int16, device=dev)
    tile_nseg = torch.empty(bp.n_tiles, dtype=torch.int32, device=dev)
    with _tb._cupy_stream(dev):
        _segments_kernel()(grid=(bp.n_tiles,), block=(_SEG_THREADS,), args=(
            bp.tile_start.data_ptr(), bp.e_key.data_ptr(), seg_start.data_ptr(), tile_nseg.data_ptr()))
    _tb._mark(timers, "segments", dev)
    return ForwardPrep(T, bp.n_lists, bp.n_tiles, bp.order, bp.tile_start, bp.e_key, seg_start, tile_nseg, bp.max_tile)


def _derive_from_backward(bp, T: int, dev, timers):
    """Sub-tiles of T lists of the backward's tiles (T_b = k * T), keys row-sorted within each sub-tile."""
    sub = bp.T // T
    n_lists = bp.n_lists
    n_tiles = (n_lists + T - 1) // T
    offsets = torch.zeros(n_lists + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(bp.counts, 0)
    tile_start = torch.cat([offsets[0:n_lists:T], offsets[n_lists:]]).contiguous()
    tile_n = tile_start[1:] - tile_start[:-1]
    max_tile = max(int(tile_n.max()), 1)   # one host sync (prefetch thread)
    keys = torch.empty(bp.e_key.shape[0], dtype=torch.int32, device=dev)
    with _tb._cupy_stream(dev):
        _partition_kernel(sub, T)(grid=(bp.n_tiles,), block=(_SEG_THREADS,), args=(
            bp.tile_start.data_ptr(), bp.e_key.data_ptr(), tile_start.data_ptr(), keys.data_ptr()))
    _tb._mark(timers, "partition", dev)
    prep = _tb.TiledPrep(T, n_lists, bp.order, tile_start, keys, n_tiles, bp.tile_map)
    prep.max_tile = max_tile
    prep.counts = bp.counts
    return prep


_PREP_CACHE_MAX = 16
_prep_cache: "collections.OrderedDict[int, tuple]" = collections.OrderedDict()
_prep_lock = threading.Lock()


def register_prep(white_indices: torch.Tensor, black_indices: torch.Tensor, order_perm: torch.Tensor | None = None, **kw) -> None:
    """Compute and store the preparation for a batch (loader prefetch worker, on its CUDA stream);
    ``order_perm`` is the backward's list order when the tiled backward is on (else it is computed)."""
    if not (TILED_FORWARD_ENABLED and PREFETCH_PREP and _HAS_CUPY and white_indices.is_cuda):
        return
    prep = None
    if PULL_ENABLED and kw.get("from_backward") is not None:
        prep = _tp.prepare(white_indices, black_indices, kw["from_backward"])   # None: unsupported tile -> 16-list gather
    if prep is None:
        prep = prepare(white_indices, black_indices, order_perm=order_perm, **kw)
    with _prep_lock:
        _prep_cache[white_indices.data_ptr()] = (weakref.ref(white_indices), tuple(white_indices.shape), prep)
        _prep_cache.move_to_end(white_indices.data_ptr())
        while len(_prep_cache) > _PREP_CACHE_MAX:
            _prep_cache.popitem(last=False)
    STATS["prep_registered"] += 1


def lookup_prep(white_indices: torch.Tensor) -> ForwardPrep | None:
    """Pop the preparation registered for this index block, if it is still valid (same rules as the backward's)."""
    with _prep_lock:
        entry = _prep_cache.pop(white_indices.data_ptr(), None)
    if entry is None:
        return None
    ref, shape, prep = entry
    src = ref()
    if src is None or src.data_ptr() != white_indices.data_ptr() or tuple(white_indices.shape) != shape:
        return None
    prep.record_stream(torch.cuda.current_stream(white_indices.device))
    STATS["prep_hit"] += 1
    return prep


# ----------------------------------------------------------------------------- forward

@torch.compiler.disable
def tiled_forward(white_indices, black_indices, weight, bias, l1_size, prep: ForwardPrep,
                  slice_cols: int = COLUMNS_PER_BLOCK, vec: int = VEC, P: int = PREFETCH_DEPTH,
                  timers: dict | None = None, epi=None):
    """Outputs (white, black), each (B, OUT) = sum of the active weight rows + bias, original batch order.
    Columns 0..l1_size use the tiled gather, the remaining (PSQT) columns a per-list warp reduction."""
    dev = weight.device
    B, maxa = white_indices.shape
    OUT = bias.shape[0]
    n_lists = 2 * B
    if prep.n_lists != n_lists:
        raise RuntimeError("tiled forward: preparation does not match the batch")
    if isinstance(prep, _tp.PullPrep):
        return _pull_forward(white_indices, black_indices, weight, bias, l1_size, prep, timers, epi=epi)
    if epi is not None:
        raise ValueError("tiled forward: the fused epilogue needs the pull gather")
    shm = shared_bytes(prep.T, slice_cols, prep.max_tile)
    if shm > MAX_SHARED_BYTES:
        raise ValueError(f"tiled forward: T={prep.T} x slice={slice_cols} with {prep.max_tile} entries per tile needs {shm} B of shared memory")
    _tb._mark(timers, "start", dev)
    out_w = torch.empty(B, OUT, dtype=torch.float32, device=dev)
    out_b = torch.empty(B, OUT, dtype=torch.float32, device=dev)
    white_indices = white_indices.contiguous(); black_indices = black_indices.contiguous()
    with _tb._cupy_stream(dev):
        _gather_kernel(prep.T, slice_cols, vec, l1_size, OUT, P)(
            grid=(prep.n_tiles, l1_size // slice_cols), block=(slice_cols // vec,), shared_mem=shm, args=(
                weight.data_ptr(), bias.data_ptr(), prep.order.data_ptr(), prep.tile_start.data_ptr(), prep.tile_nseg.data_ptr(),
                prep.e_key.data_ptr(), prep.seg_start.data_ptr(), out_w.data_ptr(), out_b.data_ptr(),
                np.int32(n_lists), np.int32(B), np.int32(prep.max_tile)))
        _tb._mark(timers, "gather", dev)
        if OUT > l1_size:
            _psqt_kernel(maxa, l1_size, OUT)(grid=((n_lists * 32 + 255) // 256,), block=(256,), args=(
                white_indices.data_ptr(), black_indices.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                out_w.data_ptr(), out_b.data_ptr(), np.int32(n_lists), np.int32(B)))
        _tb._mark(timers, "psqt", dev)
    return out_w, out_b


@torch.compiler.disable
def _pull_forward(white_indices, black_indices, weight, bias, l1_size, prep, timers=None, epi=None):
    dev = weight.device
    B, maxa = white_indices.shape
    OUT = bias.shape[0]
    n_lists = 2 * B
    if prep.n_lists != n_lists or l1_size % _tp.SLICE:
        raise RuntimeError("tiled forward: pull preparation does not match the batch / layer width")
    _tb._mark(timers, "start", dev)
    out_w = torch.empty(B, OUT, dtype=torch.float32, device=dev)
    out_b = torch.empty(B, OUT, dtype=torch.float32, device=dev)
    white_indices = white_indices.contiguous(); black_indices = black_indices.contiguous()
    _tp.gather(weight, bias, l1_size, prep, out_w, out_b, epi=epi)
    _tb._mark(timers, "gather", dev)
    if OUT > l1_size:
        with _tb._cupy_stream(dev):
            _psqt_kernel(maxa, l1_size, OUT)(grid=((n_lists * 32 + 255) // 256,), block=(256,), args=(
                white_indices.data_ptr(), black_indices.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                out_w.data_ptr(), out_b.data_ptr(), np.int32(n_lists), np.int32(B)))
        _tb._mark(timers, "psqt", dev)
    return out_w, out_b


def applicable(l1_size: int, out_size: int, slice_cols: int = COLUMNS_PER_BLOCK, vec: int = VEC) -> bool:
    """Whether the tiled forward can run for this layer width (else the plain kernel is used)."""
    return (_HAS_CUPY and l1_size % slice_cols == 0 and l1_size >= slice_cols and out_size % vec == 0
            and 0 <= out_size - l1_size <= 32)
