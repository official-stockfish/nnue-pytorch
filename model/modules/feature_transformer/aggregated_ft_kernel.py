"""H100 master-net FT backward with tile-local feature aggregation.

Pack the union of features in eight positions with a position/perspective mask.
Each column block sums those contributions before issuing global atomics.
Feature indices must be unique within each position/perspective, as guaranteed
by the feature extractors. Repeated indices across rows are aggregated exactly
apart from floating-point rounding.
"""

from functools import cache

import cupy as cp
import numpy as np
import torch


@cache
def _kernels(active: int, width: int):
    A = active
    half = width // 2
    threads = next(n for n in range(min(128, half), 0, -1) if half % n == 0)
    tile = 8
    maxids = 2 * tile * A
    h = 1 << (maxids - 1).bit_length()
    code = (
        f"#define T {tile}\n#define A {A}\n#define H {h}\n#define M {maxids}\n"
        f"#define K {width}\n#define HALF {half}\n#define B {threads}\n"
        + r"""
extern "C" __global__ void ft_pack_features(const int *w, const int *b, int *ids, unsigned *masks,
                                            int *counts, int batch_size) {
    extern __shared__ unsigned mem[];
    int *keys = (int *)mem;
    unsigned *bits = mem + H;
    __shared__ int n;
    if (threadIdx.x == 0) {
        n = 0;
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        keys[i] = -1;
        bits[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int row = i / A, k = i % A;
        int pos = blockIdx.x * T + row / 2;
        if (pos >= batch_size)
            continue;
        int id = (row & 1) ? b[pos * A + k] : w[pos * A + k];
        if (id < 0)
            continue;
        unsigned slot = ((unsigned)id * 2654435761u) & (H - 1);
        while (true) {
            int old = atomicCAS(keys + slot, -1, id);
            if (old == -1 || old == id) {
                if (old == -1) {
                    // Only the thread that claims a new hash slot appends it.
                    int offset = atomicAdd(&n, 1);
                    ids[blockIdx.x * M + offset] = slot;
                }
                atomicOr(bits + slot, 1u << row);
                break;
            }
            slot = (slot + 1) & (H - 1);
        }
    }
    __syncthreads();
    // Resolve the occupied-slot list in place after all masks are complete.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int slot = ids[blockIdx.x * M + i];
        ids[blockIdx.x * M + i] = keys[slot];
        masks[blockIdx.x * M + i] = bits[slot];
    }
    __syncthreads();
    if (threadIdx.x == 0)
        counts[blockIdx.x] = n;
}
extern "C" __global__ void ft_aggregate_backward(const float *us, const float *them,
                                                 const float *gl, const float *cl, float maxact,
                                                 const int *ids,
                                                 const unsigned *masks, const int *counts,
                                                 float *gw, float *gb, int batch_size) {
    __shared__ float g[2 * T][(2*B)];
    unsigned tid = threadIdx.x, col = tid + B * blockIdx.y;
    float bias0 = 0, bias1 = 0;
    for (int t = 0; t < T; ++t) {
        int row = blockIdx.x * T + t;
        if (row >= batch_size)
            break;
        float w0 = cl[row * (2*K) + col], w1 = cl[row * (2*K) + HALF + col];
        float b0 = cl[row * (2*K) + K + col], b1 = cl[row * (2*K) + (3*HALF) + col];
        float d0 = gl[row * K + col], d1 = gl[row * K + HALF + col];
        float dw0 = (w0 == 0 || w0 == maxact) ? 0 : d0 * w1;
        float dw1 = (w1 == 0 || w1 == maxact) ? 0 : d0 * w0;
        float db0 = (b0 == 0 || b0 == maxact) ? 0 : d1 * b1;
        float db1 = (b1 == 0 || b1 == maxact) ? 0 : d1 * b0;
        float u = us[row], v = them[row];
        float gw0 = u * dw0 + v * db0, gw1 = u * dw1 + v * db1;
        float gb0 = v * dw0 + u * db0, gb1 = v * dw1 + u * db1;
        g[2 * t][tid] = gw0;
        g[2 * t][tid + B] = gw1;
        g[2 * t + 1][tid] = gb0;
        g[2 * t + 1][tid + B] = gb1;
        bias0 += gw0 + gb0;
        bias1 += gw1 + gb1;
    }
    __syncthreads();
    int n = counts[blockIdx.x];
    for (int i = 0; i < n; ++i) {
        // Form the row offset in pointer width before adding column offsets.
        size_t id = (unsigned)ids[blockIdx.x * M + i];
        unsigned mask = masks[blockIdx.x * M + i];
        // Packed masks are nonempty; avoid an add-to-zero and a loop
        // iteration for the common case of a single contributing row.
        int r = __ffs(mask) - 1;
        float v0 = g[r][tid], v1 = g[r][tid + B];
        while ((mask &= mask - 1)) {
            r = __ffs(mask) - 1;
            v0 += g[r][tid];
            v1 += g[r][tid + B];
        }
        if (v0 != 0)
            atomicAdd(gw + id * K + col, v0);
        if (v1 != 0)
            atomicAdd(gw + id * K + HALF + col, v1);
    }
    if (bias0 != 0)
        atomicAdd(gb + col, bias0);
    if (bias1 != 0)
        atomicAdd(gb + HALF + col, bias1);
}
"""
    )
    pack = cp.RawKernel(code, "ft_pack_features")
    pack.compile()
    pack.max_dynamic_shared_size_bytes = h * 8
    aggregate = cp.RawKernel(code, "ft_aggregate_backward")
    aggregate.compile()
    return pack, aggregate, maxids, h * 8, threads


@torch.compiler.disable
def aggregated_ft_backward(
    us, them, white, black, grad, clamped, grad_weight, grad_bias, maxact
):
    """Accumulate on the current stream.

    Biases always accumulate in FP32.  Nonnegative feature indices must be
    unique within each white/black row.
    """
    batch_size, active = white.shape
    tiles = (batch_size + 7) // 8
    with cp.cuda.Device(us.device.index):
        pack, aggregate, capacity, shared_bytes, threads = _kernels(active, grad.shape[1])
        ids = torch.empty((tiles, capacity), device=us.device, dtype=torch.int32)
        masks = torch.empty_like(ids)
        counts = torch.empty(tiles, device=us.device, dtype=torch.int32)
        pack_args = (
            white.data_ptr(),
            black.data_ptr(),
            ids.data_ptr(),
            masks.data_ptr(),
            counts.data_ptr(),
            np.int32(batch_size),
        )
        backward_args = (
            us.data_ptr(),
            them.data_ptr(),
            grad.data_ptr(),
            clamped.data_ptr(),
            np.float32(maxact),
            ids.data_ptr(),
            masks.data_ptr(),
            counts.data_ptr(),
            grad_weight.data_ptr(),
            grad_bias.data_ptr(),
            np.int32(batch_size),
        )
        with cp.cuda.ExternalStream(torch.cuda.current_stream(us.device).cuda_stream):
            pack((tiles,), (512,), pack_args, shared_mem=shared_bytes)
            aggregate((tiles, grad.shape[1] // (2 * threads)), (threads,), backward_args)
