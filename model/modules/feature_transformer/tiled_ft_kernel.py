"""CUDA kernels of the tiled feature-transformer backward.

tiled_ft_prep.py launches the index-only kernels (order_key, tile_key, subtile_sort, merge_runs)
and tiled_ft_backward.py the gradient kernels (prologue, scatter). Every source below is plain
CUDA C; its constants are the #defines that the ``*_kernel`` function puts in front of it.

Vocabulary: a *list* is the active rows of one perspective of one position (white lists are
0..B-1, black lists B..2B-1). A *tile* is TILE_LISTS lists that are consecutive in the tile order.
An *entry* is one (row, list) pair, stored as the key ``row << LOCAL_BITS | list index in its tile``.
"""

from __future__ import annotations

from ...cupy_utils import compile_kernel

TILE_LISTS = 256  # lists per tile
# A tile is sorted as TILE_LISTS / SUBTILE_LISTS sub-tiles, which are then merged.
SUBTILE_LISTS = 64
COLUMNS_PER_BLOCK = 64  # columns of the layer that one scatter block handles
WARPS_PER_BLOCK = 32  # warps of a scatter block
ROW_BITS = 20  # row bits of an entry key: feature sets up to 2**20 rows
LOCAL_BITS = 8  # list-in-tile bits of an entry key: TILE_LISTS <= 2**LOCAL_BITS

# Dynamic shared memory of a scatter block: the tile's gradient rows for its column slice, and
# 32 entry keys per warp.
SCATTER_SHARED_BYTES = TILE_LISTS * COLUMNS_PER_BLOCK * 4 + WARPS_PER_BLOCK * 32 * 4

# A block of the sub-tile sort holds SORT_THREADS * keys_per_thread keys. The kernel is compiled
# for the keys per thread a batch needs: 24 for the usual sub-tile of the default feature set
# (~5.9k keys), more for denser ones. CUB keeps a block's keys in static shared memory, which is
# limited to 48 KB: 40 keys per thread use 42 KB of it.
SORT_THREADS = 256
SORT_MIN_KEYS_PER_THREAD = 24
SORT_MAX_KEYS_PER_THREAD = 40
SORT_RADIX_BITS = 4

# Odd multipliers and offsets of the list order's min-hashes.
_HASH_A = (2047437035, 27379259, 1342816569, 1409073591)
_HASH_B = (1316723239, 1858169645, 314556816, 334338155)
_HASH_BITS = 14

_TYPES = (
    "typedef int int32_t; typedef long long int64_t; typedef unsigned int uint32_t;\n"
)

# The slots of list li, for a kernel with (white, black, batch_size) arguments.
_LIST_ROW = (
    "#define LIST_ROW(li) ((li) < batch_size ? white + (int64_t)(li) * MAXA"
    " : black + (int64_t)((li) - batch_size) * MAXA)\n"
)


def _defines(**constants) -> str:
    return "".join(f"#define {name} {value}\n" for name, value in constants.items())


def _c_array(values) -> str:
    return "{" + ", ".join(f"{v}u" for v in values) + "}"


_ORDER_KEY_SRC = r"""
// The sort key of every list (it groups lists that share rows into the same tile) and its number of
// active rows. One warp per list: lanes read consecutive slots (coalesced) up to the first empty
// slot, then the warp reduces the per-lane minima / maximum / count (min, max and + are
// order-independent).
extern "C" __global__ void order_key(const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    int64_t* __restrict__ keys, int32_t* __restrict__ counts, const int32_t n_lists, const int32_t batch_size,
    const int32_t bucket_first_row, const int32_t bucket_rows)
{
    const uint32_t ha[NUM_HASHES] = HASH_A, hb[NUM_HASHES] = HASH_B;
    const int32_t li = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (li >= n_lists) return;
    const int32_t* row = LIST_ROW(li);
    uint32_t mn[NUM_HASHES]; int32_t mx = -1; int32_t n = 0;
    for (int j = 0; j < NUM_HASHES; ++j) mn[j] = (1u << HASH_BITS) - 1;
    for (int k0 = 0; k0 < MAXA; k0 += 32) {
        const int32_t k = k0 + lane;
        const int32_t r = k < MAXA ? row[k] : -1;
        const unsigned valid = __ballot_sync(0xFFFFFFFFu, r >= 0);
        const unsigned lead = ~valid ? (((~valid) & (0u - (~valid))) - 1u) : 0xFFFFFFFFu;   // lanes before the first empty slot
        if ((lead >> lane) & 1u) {
            ++n;
            mx = max(mx, r);
            for (int j = 0; j < NUM_HASHES; ++j) {
                const uint32_t h = (((uint32_t)r * ha[j] + hb[j]) >> 7) & ((1u << HASH_BITS) - 1);
                mn[j] = min(mn[j], h);
            }
        }
        if (lead != 0xFFFFFFFFu) break;
    }
    for (int o = 16; o > 0; o >>= 1) {
        n += __shfl_xor_sync(0xFFFFFFFFu, n, o);
        mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
        for (int j = 0; j < NUM_HASHES; ++j) mn[j] = min(mn[j], __shfl_xor_sync(0xFFFFFFFFu, mn[j], o));
    }
    if (lane != 0) return;
    // The bucket of the list's highest row (when the feature set has one), then the min-hashes.
    int64_t key = 0;
    if (bucket_rows > 0) { int32_t kb = (mx - bucket_first_row) / bucket_rows; kb = kb < 0 ? 0 : (kb > 127 ? 127 : kb); key = kb; }
    for (int j = 0; j < NUM_HASHES; ++j) key = (key << HASH_BITS) | (int64_t)mn[j];
    keys[li] = key;
    counts[li] = n;
}
"""


def order_key_kernel(max_active: int):
    src = (
        _TYPES
        + _defines(
            MAXA=max_active,
            NUM_HASHES=len(_HASH_A),
            HASH_BITS=_HASH_BITS,
            HASH_A=_c_array(_HASH_A),
            HASH_B=_c_array(_HASH_B),
        )
        + _LIST_ROW
        + _ORDER_KEY_SRC
    )
    return compile_kernel(("tiled_order_key", max_active), src, "order_key")


_TILE_KEY_SRC = r"""
// The entry keys, written per list at its offset; the lists of a tile are consecutive in tile
// order, so the keys of a tile are contiguous (unsorted).
// One warp per list: lanes read 32 consecutive slots and write 32 consecutive keys (coalesced).
extern "C" __global__ void tile_key(const int32_t* __restrict__ white, const int32_t* __restrict__ black,
    const int32_t* __restrict__ order, const int32_t* __restrict__ offsets, uint32_t* __restrict__ keys,
    const int32_t n_lists, const int32_t batch_size)
{
    const int32_t p = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // position in tile order
    const int lane = threadIdx.x & 31;
    if (p >= n_lists) return;
    const int32_t li = order[p];
    const int32_t* row = LIST_ROW(li);
    const uint32_t local = (uint32_t)(p % TILE);
    const int32_t o = offsets[p];
    for (int k = lane; k < MAXA; k += 32) {
        const int32_t r = row[k];
        const unsigned active = __ballot_sync(0xFFFFFFFFu, r >= 0);
        if (r >= 0) keys[o + k] = ((uint32_t)r << LOCAL_BITS) | local;   // lists are left-packed: slot k of a valid entry = its rank
        if (active != 0xFFFFFFFFu) break;
    }
}
"""


def tile_key_kernel(max_active: int):
    src = (
        _TYPES
        + _defines(MAXA=max_active, TILE=TILE_LISTS, LOCAL_BITS=LOCAL_BITS)
        + _LIST_ROW
        + _TILE_KEY_SRC
    )
    return compile_kernel(("tiled_tile_key", max_active), src, "tile_key")


# The #defines follow the include: they must not rename anything inside CUB.
_SUBTILE_SORT_SRC = r"""
// Every sub-tile's keys sorted in place by a block-wide radix sort (CUB, THREADS threads x
// KEYS_PER_THREAD keys in registers; sub-tiles of n <= THREADS * KEYS_PER_THREAD keys). A sub-tile is
// 2**SUB_BITS lists that are consecutive in tile order, so its keys share the high bits of the
// list-in-tile index and only ROW_BITS + SUB_BITS bits are sorted. Keys are unique, so the result
// equals any other sort. Padding keys (all ones) sort after the real keys even where the sorted bits
// tie: the sort is stable and the padding comes last in the blocked input.
extern "C" __global__ void __launch_bounds__(THREADS) subtile_sort(const int* __restrict__ sub_start, unsigned* __restrict__ keys)
{
    typedef cub::BlockRadixSort<unsigned, THREADS, KEYS_PER_THREAD, cub::NullType, RADIX_BITS> Sort;
    __shared__ typename Sort::TempStorage tmp;
    const int e0 = sub_start[blockIdx.x], n = sub_start[blockIdx.x + 1] - e0;
    if (n <= 1) return;
    unsigned v[KEYS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; ++i) {   // blocked: padding last
        const int j = threadIdx.x * KEYS_PER_THREAD + i;
        const unsigned k = j < n ? keys[e0 + j] : 0xFFFFFFFFu;
        v[i] = ((k >> LOCAL_BITS) << SUB_BITS) | (k & ((1u << SUB_BITS) - 1u));   // drop the list-index bits that are constant in the sub-tile
    }
    Sort(tmp).SortBlockedToStriped(v, 0, ROW_BITS + SUB_BITS);
    const unsigned hi = keys[e0] & (((1u << LOCAL_BITS) - 1u) ^ ((1u << SUB_BITS) - 1u));
#pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        const int j = i * THREADS + threadIdx.x;
        if (j < n) keys[e0 + j] = ((v[i] >> SUB_BITS) << LOCAL_BITS) | hi | (v[i] & ((1u << SUB_BITS) - 1u));
    }
}
"""


def subtile_sort_kernel(keys_per_thread: int):
    """Sort of sub-tiles of up to SORT_THREADS * ``keys_per_thread`` keys. CuPy puts the CUB it
    bundles on the include path."""
    src = (
        "#include <cub/block/block_radix_sort.cuh>\n"
        + _defines(
            THREADS=SORT_THREADS,
            KEYS_PER_THREAD=keys_per_thread,
            RADIX_BITS=SORT_RADIX_BITS,
            ROW_BITS=ROW_BITS,
            LOCAL_BITS=LOCAL_BITS,
            SUB_BITS=SUBTILE_LISTS.bit_length() - 1,
        )
        + _SUBTILE_SORT_SRC
    )
    return compile_kernel(
        ("tiled_subtile_sort", keys_per_thread),
        src,
        "subtile_sort",
        options=("--std=c++17",),
    )


_MERGE_RUNS_SRC = r"""
// One block per tile: the tile's NSUB sorted runs (its sub-tiles) are loaded into shared memory, and
// entry i of run s goes to tile_start[t] + (i - start of s) + (number of smaller keys in the other
// runs).
extern "C" __global__ void __launch_bounds__(1024) merge_runs(const int32_t* __restrict__ sub_start,
    const int32_t* __restrict__ tile_start, const uint32_t* __restrict__ keys, uint32_t* __restrict__ out, const int32_t n_sub)
{
    extern __shared__ uint32_t sk[];
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


def merge_runs_kernel(shared_bytes: int):
    """Merge of every tile's sorted sub-tiles; launches use at most ``shared_bytes``."""
    src = _TYPES + _defines(NSUB=TILE_LISTS // SUBTILE_LISTS) + _MERGE_RUNS_SRC
    return compile_kernel(
        ("tiled_merge_runs", shared_bytes),
        src,
        "merge_runs",
        dynamic_shared=shared_bytes,
    )


_PROLOGUE_SRC = r"""
// The per-position part of the fused double FT's backward: the gradients of the two raw
// accumulators (white, black), each (batch, L1), from the output gradient and the forward's clamped
// values. One thread per (position, column pair).
extern "C" __global__ void tiled_ft_prologue(
    const float* __restrict__ us, const float* __restrict__ them,
    const float* __restrict__ grad_l0, const float* __restrict__ clamped_out, const float max_act,
    float* __restrict__ grad_white, float* __restrict__ grad_black, const int batch_size)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)batch_size * HALF) return;
    const int b = (int)(t / HALF), i = (int)(t % HALF);
    const float u = us[b], th = them[b];
    const size_t r = (size_t)b * L1, c = (size_t)b * 4 * HALF;
    const float cw0 = clamped_out[c + 0 * HALF + i], cw1 = clamped_out[c + 1 * HALF + i];
    const float cb0 = clamped_out[c + 2 * HALF + i], cb1 = clamped_out[c + 3 * HALF + i];
    const float g = grad_l0[r + i], gh = grad_l0[r + HALF + i];
    const float dw0 = (cw0 == 0.0f || cw0 == max_act) ? 0.0f : g * cw1;
    const float dw1 = (cw1 == 0.0f || cw1 == max_act) ? 0.0f : g * cw0;
    const float db0 = (cb0 == 0.0f || cb0 == max_act) ? 0.0f : gh * cb1;
    const float db1 = (cb1 == 0.0f || cb1 == max_act) ? 0.0f : gh * cb0;
    grad_white[r + i] = u * dw0 + th * db0;
    grad_white[r + HALF + i] = u * dw1 + th * db1;
    grad_black[r + i] = th * dw0 + u * db0;
    grad_black[r + HALF + i] = th * dw1 + u * db1;
}
"""


def prologue_kernel(l1_size: int):
    src = _defines(L1=l1_size, HALF=l1_size // 2) + _PROLOGUE_SRC
    return compile_kernel(("tiled_prologue", l1_size), src, "tiled_ft_prologue")


_SCATTER_SRC = r"""
typedef float2 GV;   // a lane's CPL columns of one list, loaded at once
#if CPL != 2
#error "GV must hold CPL floats"
#endif
#define NO_ROW 0xFFFFFFFFu

// A lane's columns of the gradient row of an entry's list, from the tile's rows in shared memory.
static __device__ __forceinline__ GV lane_gradient(const float* Gsh, const uint32_t key, const int lane)
{
    const int l = (int)(key & ((1u << LOCAL_BITS) - 1));
    return *(const GV*)(Gsh + l * SLICE + lane * CPL);
}

// Add the sums of a finished row segment to the lane's columns (col, col + 32, ...) of the row's
// weight gradient, and clear them.
static __device__ __forceinline__ void flush_row(float* __restrict__ grad_weight, const uint32_t row,
                                                 const int32_t col, float* acc)
{
    float* gw = grad_weight + (int64_t)row * L1 + col;
#pragma unroll
    for (int c = 0; c < CPL; ++c) { atomicAdd(gw + 32 * c, acc[c]); acc[c] = 0.0f; }
}

// Move a lane's walk to the row of its next entry: a new row flushes the one before it.
static __device__ __forceinline__ void enter_row(const uint32_t row, uint32_t& cur, float* acc,
                                                 float* __restrict__ grad_weight, const int32_t col)
{
    if (row != cur) {
        if (cur != NO_ROW) flush_row(grad_weight, cur, col, acc);
        cur = row;
    }
}

// Add an entry's gradient to the sums of the current row.
static __device__ __forceinline__ void accumulate(float* acc, const GV gv)
{
    const float* g = (const float*)&gv;
#pragma unroll
    for (int c = 0; c < CPL; ++c) acc[c] += g[c];
}

// One block per (tile, column slice). NWARPS warps; lane owns columns col0 + lane + 32*c, c < CPL.
// Gw / Gb are the gradients of the white / black accumulators.
extern "C" __global__ void __launch_bounds__(NWARPS * 32) tiled_scatter(
    const float* __restrict__ Gw, const float* __restrict__ Gb, const int32_t* __restrict__ order,
    const int32_t* __restrict__ tile_start, const uint32_t* __restrict__ e_key,
    float* __restrict__ grad_weight, float* __restrict__ grad_bias, const int32_t n_lists, const int32_t batch_size)
{
    extern __shared__ float Gsh[];                          // TILE * SLICE
    const int32_t tile = blockIdx.x, slice = blockIdx.y;
    const int32_t col0 = slice * SLICE;

    // 1. the tile's gradient rows for this column slice; every thread issues all its loads (PF list ids,
    // then PF values) before its shared-memory stores
    const int NT = NWARPS * 32, PF = (TILE * SLICE + NT - 1) / NT;
    int32_t lid[PF];
#pragma unroll
    for (int k = 0; k < PF; ++k) {
        const int idx = threadIdx.x + k * NT;
        const int32_t li = tile * TILE + idx / SLICE;
        lid[k] = (idx < TILE * SLICE && li < n_lists) ? __ldg(&order[li]) : -1;
    }
    float v[PF];
#pragma unroll
    for (int k = 0; k < PF; ++k) {
        const int c = (threadIdx.x + k * NT) % SLICE;
        v[k] = lid[k] < 0 ? 0.0f : lid[k] < batch_size ? __ldg(&Gw[(int64_t)lid[k] * L1 + col0 + c])
                                                       : __ldg(&Gb[(int64_t)(lid[k] - batch_size) * L1 + col0 + c]);
    }
#pragma unroll
    for (int k = 0; k < PF; ++k) {
        const int idx = threadIdx.x + k * NT, l = idx / SLICE, c = idx - l * SLICE;
        if (idx < TILE * SLICE) Gsh[l * SLICE + (c & 31) * CPL + (c >> 5)] = v[k];   // column c at lane (c % 32), slot c / 32
    }
    __syncthreads();

    // 2. bias gradient: column sums of the tile
    if (threadIdx.x < SLICE) {
        float s = 0.0f;
        const int pc = (threadIdx.x & 31) * CPL + (threadIdx.x >> 5);
        for (int l = 0; l < TILE; ++l) s += Gsh[l * SLICE + pc];
        if (s != 0.0f) atomicAdd(&grad_bias[col0 + threadIdx.x], s);
    }

    // 3. row segments: warp w walks part w of the tile's sorted entry stream
    const int32_t e0 = tile_start[tile], e1 = tile_start[tile + 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t* ksh = (uint32_t*)(Gsh + TILE * SLICE) + warp * 32;
    const int32_t per = (e1 - e0 + NWARPS - 1) / NWARPS;
    const int32_t s = e0 + warp * per;
    const int32_t e = min(s + per, e1);
    float acc[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) acc[c] = 0.0f;
    uint32_t cur = NO_ROW;
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
                gv[u] = lane_gradient(Gsh, key[u], lane);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                enter_row(key[u] >> LOCAL_BITS, cur, acc, grad_weight, col0 + lane);
                accumulate(acc, gv[u]);
            }
        }
        for (; j < nk; ++j) {
            const uint32_t key = ksh[j];
            enter_row(key >> LOCAL_BITS, cur, acc, grad_weight, col0 + lane);
            accumulate(acc, lane_gradient(Gsh, key, lane));
        }
    }
    if (cur != NO_ROW) flush_row(grad_weight, cur, col0 + lane, acc);
}
"""


def scatter_kernel(l1_size: int):
    src = (
        _TYPES
        + _defines(
            TILE=TILE_LISTS,
            SLICE=COLUMNS_PER_BLOCK,
            CPL=COLUMNS_PER_BLOCK // 32,
            NWARPS=WARPS_PER_BLOCK,
            LOCAL_BITS=LOCAL_BITS,
            L1=l1_size,
        )
        + _SCATTER_SRC
    )
    return compile_kernel(
        ("tiled_scatter", l1_size),
        src,
        "tiled_scatter",
        dynamic_shared=SCATTER_SHARED_BYTES,
    )
