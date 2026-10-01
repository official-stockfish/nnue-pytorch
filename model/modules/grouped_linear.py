"""Selected-bucket L1 matmul for the H100 master-net workload.

Only the requested K→N layer is evaluated, with NNZ-compacted
forward. A GPU row map allows tiled FP32 backward matmuls without sorting/copying
the large activation tensor or reading bucket sizes on the host. Split reductions
keep weight gradients parallel even when bucket populations differ. Input
gradients remain dense: fake quantization uses STE, so a quantized zero can have
a nonzero derivative.
"""

import os

import cupy as cp
import torch
import triton as tr
import triton.language as tl

_route_kernels = {}
_sparse_forward_kernels = {}


def _sparse_forward(x, weight, bias, indices, count):
    """Compact NNZ in shared memory and coalesce loads over adjacent outputs.

    Four warps split each row's nonzeros. The transpose is included in the
    measured cost; parameter storage and dense STE input gradients stay intact.
    """
    width = x.shape[1]
    outputs = bias.numel() // count
    key = (width, outputs)
    kernel = _sparse_forward_kernels.get(key)
    if kernel is None:
        kernel = cp.RawKernel(
            f"#define K {width}\n#define N {outputs}\n" + r"""
extern "C" __global__ void sparse_l1_forward(
    const float* x, const float* weight, const float* bias,
    const long long* buckets, float* output
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int bucket = buckets[row];
    const int column = (N <= 32 ? 0 : blockIdx.y * 32) + lane;
    __shared__ int nnz, indices[K];
    __shared__ float values[K], partial[4][32];
    if (threadIdx.x == 0) nnz = 0;
    __syncthreads();

    for (int k = threadIdx.x; k < K; k += 128) {
        const float value = x[row * K + k];
        const unsigned mask = __ballot_sync(0xffffffff, value != 0.0f);
        int base = 0;
        if (lane == 0) base = atomicAdd(&nnz, __popc(mask));
        base = __shfl_sync(0xffffffff, base, 0);
        if (value != 0.0f) {
            const int slot = base + __popc(mask & ((1u << lane) - 1));
            indices[slot] = k;
            values[slot] = value;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    if (column < N) {
        for (int i = warp; i < nnz; i += 4) {
            const int k = indices[i];
            acc = fmaf(values[i], weight[(bucket * K + k) * N + column], acc);
        }
    }
    partial[warp][lane] = acc;
    __syncthreads();
    if (warp == 0 && column < N) {
        float total = bias[bucket * N + column];
        #pragma unroll
        for (int i = 0; i < 4; ++i) total += partial[i][lane];
        output[row * N + column] = total;
    }
}
""",
            "sparse_l1_forward",
        )
        _sparse_forward_kernels[key] = kernel
    transposed = weight.reshape(count, outputs, width).transpose(1, 2).contiguous()
    output = torch.empty((len(x), outputs), device=x.device, dtype=x.dtype)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream)
    kernel(
        (len(x), tr.cdiv(outputs, 32)),
        (128,),
        (x.data_ptr(), transposed.data_ptr(), bias.data_ptr(), indices.data_ptr(), output.data_ptr()),
        stream=stream,
    )
    return output


@torch.compiler.disable(recursive=False)
def _route_rows(indices, count):
    kernel = _route_kernels.get(count)
    if kernel is None:
        # One thread initializes/reserves each bucket, so count must be <= 256.
        # Specializing the constant preserves the original eight-bucket code.
        kernel = cp.RawKernel(
            f"#define BUCKETS {count}\n" + r"""
extern "C" __global__ void route(const long long* ids,int* rows,int* counts,int batch){
 __shared__ int used[BUCKETS],base[BUCKETS];
 int t=threadIdx.x,r=blockIdx.x*blockDim.x+t;
 if(t<BUCKETS)used[t]=0;
 __syncthreads();
 int b=0,pos=0;
 if(r<batch){b=ids[r];pos=atomicAdd(&used[b],1);}
 __syncthreads();
 if(t<BUCKETS)base[t]=atomicAdd(&counts[t],used[t]);
 __syncthreads();
 if(r<batch)rows[b*batch+base[b]+pos]=r;
}
""",
            "route",
        )
        _route_kernels[count] = kernel
    rows = torch.empty((count, len(indices)), device=indices.device, dtype=torch.int32)
    counts = torch.zeros(count, device=indices.device, dtype=torch.int32)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream(indices.device).cuda_stream)
    kernel(
        (tr.cdiv(len(indices), 256),),
        (256,),
        (indices.data_ptr(), rows.data_ptr(), counts.data_ptr(), len(indices)),
        stream=stream,
    )
    return rows, counts


@tr.jit
def _bucketed_input_gradient(
    G,
    W,
    R,
    C,
    DX,
    BATCH,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    p = tl.program_id(0)
    pk = tl.program_id(1)
    bucket = tl.program_id(2)
    count = tl.load(C + bucket)
    if p * BM < count:
        m = p * BM + tl.arange(0, BM)
        k = pk * BK + tl.arange(0, BK)
        n = tl.arange(0, BN)
        rows = tl.load(R + bucket * BATCH + m, m < count, 0)
        g = tl.load(
            G + rows[:, None] * N + n[None, :],
            (m[:, None] < count) & (n[None, :] < N),
            0,
        )
        w = tl.load(
            W + (bucket * N + n[:, None]) * K + k[None, :],
            (n[:, None] < N) & (k[None, :] < K),
            0,
        )
        acc = tl.dot(g, w, input_precision="ieee")
        tl.store(
            DX + rows[:, None] * K + k[None, :],
            acc,
            (m[:, None] < count) & (k[None, :] < K),
        )


@tr.jit
def _bucketed_weight_gradient(
    X,
    G,
    R,
    C,
    P,
    BATCH,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    pk = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.program_id(2)
    k = pk * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    mi = tl.arange(0, BM)
    count = tl.load(C + b)
    acc = tl.full((BN, BK), 0, tl.float32)
    for start in range(s * BM, count, BM * SPLIT):
        m = start + mi
        rows = tl.load(R + b * BATCH + m, m < count, 0)
        x = tl.load(
            X + rows[:, None] * K + k[None, :],
            (m[:, None] < count) & (k[None, :] < K),
            0,
        )
        g = tl.load(
            G + rows[None, :] * N + n[:, None],
            (m[None, :] < count) & (n[:, None] < N),
            0,
        )
        acc = tl.dot(g, x, acc, input_precision="ieee")
    tl.store(
        P + ((b * SPLIT + s) * N + n[:, None]) * K + k[None, :],
        acc,
        (n[:, None] < N) & (k[None, :] < K),
    )


@tr.jit
def _bucketed_bias_gradient(
    G,
    R,
    C,
    Q,
    BATCH,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    n = tl.arange(0, BN)
    mi = tl.arange(0, 128)
    count = tl.load(C + b)
    acc = tl.full((BN,), 0, tl.float32)
    for start in range(s * 128, count, 128 * SPLIT):
        m = start + mi
        rows = tl.load(R + b * BATCH + m, m < count, 0)
        g = tl.load(
            G + rows[:, None] * N + n[None, :],
            (m[:, None] < count) & (n[None, :] < N),
            0,
        )
        acc += tl.sum(g, 0)
    tl.store(Q + (b * SPLIT + s) * N + n, acc, n < N)


@tr.jit
def _sum_gradient_partials(
    P,
    Q,
    W,
    B,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLIT)
    p = tl.load(P + b * SPLIT * N * K + s[:, None] * N * K + i[None, :], i[None, :] < N * K, 0)
    tl.store(W + b * N * K + i, tl.sum(p, 0), i < N * K)
    if tl.program_id(1) == 0:
        q = tl.load(Q + b * SPLIT * N + s[:, None] * N + i[None, :], i[None, :] < N, 0)
        tl.store(B + b * N + i, tl.sum(q, 0), i < N)


def _input_gradient(g, w, rows, counts, k, bm=32, bk=64):
    out = torch.empty((len(g), k), device=g.device, dtype=g.dtype)
    _bucketed_input_gradient[(tr.cdiv(len(g), bm), tr.cdiv(k, bk), counts.numel())](
        g,
        w,
        rows,
        counts,
        out,
        len(g),
        k,
        g.shape[1],
        bm,
        bk,
        max(16, tr.next_power_of_2(g.shape[1])),
        num_warps=4,
    )
    return out


def _weight_and_bias_gradient(x, g, rows, counts, split=8, bm=32, bk=32):
    batch, k = x.shape
    n = g.shape[1]
    count = counts.numel()  # Tensor metadata; no GPU-to-CPU synchronization.
    p = torch.empty((count, split, n, k), device=x.device, dtype=x.dtype)
    q = torch.empty((count, split, n), device=x.device, dtype=x.dtype)
    w = torch.empty((count * n, k), device=x.device, dtype=x.dtype)
    b = torch.empty(count * n, device=x.device, dtype=x.dtype)
    _bucketed_weight_gradient[(tr.cdiv(k, bk), count, split)](
        x,
        g,
        rows,
        counts,
        p,
        batch,
        k,
        n,
        split,
        bm,
        bk,
        max(16, tr.next_power_of_2(n)),
        num_warps=4,
    )
    _bucketed_bias_gradient[(count, split)](
        g, rows, counts, q, batch, n, split, max(16, tr.next_power_of_2(n)), num_warps=4
    )
    _sum_gradient_partials[(count, tr.cdiv(n * k, 128))](p, q, w, b, k, n, split, 128, num_warps=4)
    return w, b


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, indices, count):
        x = x.contiguous()
        weight = weight.contiguous()
        indices = indices.flatten().to(torch.int64).contiguous()
        rows, counts = _route_rows(indices, count)
        ctx.save_for_backward(x, weight, rows, counts)
        return _sparse_forward(x, weight, bias.contiguous(), indices, count)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rows, counts = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # Quantized zero activations still need dense STE input gradients.
        grad_input = _input_gradient(grad_output, weight, rows, counts, x.shape[1], 64, 64)
        grad_weight, grad_bias = _weight_and_bias_gradient(x, grad_output, rows, counts, 16, 32, 32)
        return grad_input, grad_weight, grad_bias, None, None


# Measured on GH200 (batch 32768, l2 32, widths 1024/1152, fwd+bwd): the dense
# all-buckets cuBLAS path is fastest through 16 stacks (~0.62 ms/step vs ~0.80
# at count 8); the bucketed kernels win from 32 stacks (~0.94 vs ~1.00) and
# scale far better (count 128: ~1.6 vs ~3.4).
_GROUPED_L1_MIN_COUNT = 32


def grouped_l1_preferred(count: int) -> bool:
    """Whether the bucketed kernels should replace the dense path at this count.

    NNUE_GROUPED_L1=1|0 forces the bucketed or dense path regardless of count.
    """
    override = os.environ.get("NNUE_GROUPED_L1", "")
    if override == "1":
        return True
    if override == "0":
        return False
    return count >= _GROUPED_L1_MIN_COUNT


@torch.compiler.disable
def grouped_l1(x, weight, bias, indices, count=8):
    """FP32 K→N linear, 1 <= N <= 128, 1 <= count <= 256; indices in [0, count).

    Bucket counts stay on the GPU. Parameters and output row order are identical
    to a dense K→(count*N) linear followed by selection. First-order training only,
    as with the custom feature transformer. The caller handles other shapes.
    """
    return _GroupedLinear.apply(x, weight, bias, indices, count)
