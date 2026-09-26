import cupy as cp
import torch

from .sparse_linear_kernel import _kernel_with_threads

# Thread target for the forward kernel: largest divisor of l1_half
# that is <= 128 is used. Fewer threads gives each warp more
# independent accumulator chains; the backward uses one thread per
# column (l1_half) since it is atomicAdd-contended.
_FORWARD_THREADS = 128


def _num_threads(l1_half: int, target: int) -> int:
    """Largest divisor of l1_half that is <= target."""
    for i in range(min(target, l1_half), 0, -1):
        if l1_half % i == 0:
            return i
    return 1

_fused_double_ft_forward_kernel_cache = {}

@torch.compiler.disable(recursive=False)
def make_fused_double_ft_forward_kernel(max_active_indices: int, l1_size: int):
    l1_half = l1_size // 2
    num_threads = _num_threads(l1_half, _FORWARD_THREADS)
    output_thread_slice_size = l1_half // num_threads

    key = (max_active_indices, l1_size, num_threads)
    if key not in _fused_double_ft_forward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;

extern "C" __global__
void fused_double_ft_forward(
    const float* __restrict__ us,
    const float* __restrict__ them,
    const int32_t* __restrict__ white_indices,
    const int32_t* __restrict__ black_indices,
    const int64_t* __restrict__ psqt_indices,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float          max_ft_act,
          float* __restrict__ l0_out,
          float* __restrict__ wpsqt_out,
          float* __restrict__ bpsqt_out,
          float* __restrict__ clamped_out,
    const int32_t        output_size
) {
    const uint32_t block_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    const float us_val = __ldg(&us[block_idx]);
    const float them_val = __ldg(&them[block_idx]);

    const int32_t* const w_idx_row = white_indices + block_idx * """ + str(max_active_indices) + r""";
    const int32_t* const b_idx_row = black_indices + block_idx * """ + str(max_active_indices) + r""";

    const int32_t l1_size = """ + str(l1_size) + r""";
    const int32_t l1_half = """ + str(l1_half) + r""";
    const int32_t n_threads = """ + str(num_threads) + r""";
    const int64_t p_idx = __ldg(&psqt_indices[block_idx]);
    float w_psqt_val = __ldg(&bias[l1_size + p_idx]);
    float b_psqt_val = __ldg(&bias[l1_size + p_idx]);

    float w0[""" + str(output_thread_slice_size) + r"""];
    float w1[""" + str(output_thread_slice_size) + r"""];
    float b0[""" + str(output_thread_slice_size) + r"""];
    float b1[""" + str(output_thread_slice_size) + r"""];

    #pragma unroll
    for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
        uint32_t i = s * n_threads + tid;
        w0[s] = __ldg(&bias[i]);
        w1[s] = __ldg(&bias[i + l1_half]);
        b0[s] = __ldg(&bias[i]);
        b1[s] = __ldg(&bias[i + l1_half]);
    }

    for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
        int w_idx = w_idx_row[k];
        if (w_idx == -1) break;

        #pragma unroll
        for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
            uint32_t i = s * n_threads + tid;
            w0[s] += __ldg(&weight[w_idx * output_size + i]);
            w1[s] += __ldg(&weight[w_idx * output_size + i + l1_half]);
        }
    }

    for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
        int b_idx = b_idx_row[k];
        if (b_idx == -1) break;

        #pragma unroll
        for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
            uint32_t i = s * n_threads + tid;
            b0[s] += __ldg(&weight[b_idx * output_size + i]);
            b1[s] += __ldg(&weight[b_idx * output_size + i + l1_half]);
        }
    }

    #pragma unroll
    for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
        uint32_t i = s * n_threads + tid;

        float l0_w0 = us_val * w0[s] + them_val * b0[s];
        float l0_w1 = us_val * w1[s] + them_val * b1[s];
        float l0_b0 = us_val * b0[s] + them_val * w0[s];
        float l0_b1 = us_val * b1[s] + them_val * w1[s];

        if (l0_w0 < 0.0f) l0_w0 = 0.0f; else if (l0_w0 > max_ft_act) l0_w0 = max_ft_act;
        if (l0_w1 < 0.0f) l0_w1 = 0.0f; else if (l0_w1 > max_ft_act) l0_w1 = max_ft_act;
        if (l0_b0 < 0.0f) l0_b0 = 0.0f; else if (l0_b0 > max_ft_act) l0_b0 = max_ft_act;
        if (l0_b1 < 0.0f) l0_b1 = 0.0f; else if (l0_b1 > max_ft_act) l0_b1 = max_ft_act;

        l0_out[block_idx * l1_size + i] = l0_w0 * l0_w1;
        l0_out[block_idx * l1_size + l1_half + i] = l0_b0 * l0_b1;

        const uint32_t clamp_base = block_idx * 4 * l1_half;
        clamped_out[clamp_base + 0 * l1_half + i] = l0_w0;
        clamped_out[clamp_base + 1 * l1_half + i] = l0_w1;
        clamped_out[clamp_base + 2 * l1_half + i] = l0_b0;
        clamped_out[clamp_base + 3 * l1_half + i] = l0_b1;
    }

    if (threadIdx.x == 0) {
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx == -1) break;
            w_psqt_val += __ldg(&weight[w_idx * output_size + l1_size + p_idx]);
        }
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx == -1) break;
            b_psqt_val += __ldg(&weight[b_idx * output_size + l1_size + p_idx]);
        }
        wpsqt_out[block_idx] = w_psqt_val;
        bpsqt_out[block_idx] = b_psqt_val;
    }
}
""",
            "fused_double_ft_forward",
        )
        kernel.compile()
        _fused_double_ft_forward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _fused_double_ft_forward_kernel_cache[key]

BACKWARD_TILE_SIZE = 4

_fused_double_ft_backward_kernel_cache = {}

@torch.compiler.disable(recursive=False)
def make_fused_double_ft_backward_kernel(max_active_indices: int, l1_size: int, tile_size: int = BACKWARD_TILE_SIZE, num_psqt_buckets: int = 8):
    l1_half = l1_size // 2
    # One thread per column; backward is atomicAdd-contended, not
    # latency-limited, so scalar form with full thread count is best.
    # Cap at 1024 (CUDA max threads per block); use a stride loop
    # when l1_half exceeds it.
    num_threads = _num_threads(l1_half, min(l1_half, 1024))
    output_size = l1_size + num_psqt_buckets

    key = (max_active_indices, l1_size, num_threads, tile_size, num_psqt_buckets)
    if key not in _fused_double_ft_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;

extern "C" __global__
void fused_double_ft_backward(
    const float* __restrict__ us,
    const float* __restrict__ them,
    const int32_t* __restrict__ white_indices,
    const int32_t* __restrict__ black_indices,
    const int64_t* __restrict__ psqt_indices,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float          max_ft_act,
    const float* __restrict__ grad_l0,
    const float* __restrict__ grad_wpsqt,
    const float* __restrict__ grad_bpsqt,
    const float* __restrict__ clamped_out,
          float* __restrict__ grad_weight,
          float* __restrict__ grad_bias,
    const int32_t        batch_size,
    const int32_t        output_size
) {
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    const int32_t l1_size = """ + str(l1_size) + r""";
    const int32_t l1_half = """ + str(l1_half) + r""";
    const int32_t n_threads = """ + str(num_threads) + r""";
    const int32_t tile_size = """ + str(tile_size) + r""";

    __shared__ float shared_grad_bias[""" + str(output_size) + r"""];
    for (int i = tid; i < output_size; i += n_threads) {
        shared_grad_bias[i] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < tile_size; ++t) {
        const uint32_t block_idx = tile_idx * tile_size + t;
        if (block_idx >= batch_size) break;

        const float us_val = __ldg(&us[block_idx]);
        const float them_val = __ldg(&them[block_idx]);

        const int32_t* const w_idx_row = white_indices + block_idx * """ + str(max_active_indices) + r""";
        const int32_t* const b_idx_row = black_indices + block_idx * """ + str(max_active_indices) + r""";

        const int64_t p_idx = __ldg(&psqt_indices[block_idx]);
        const float gw_psqt = __ldg(&grad_wpsqt[block_idx]);
        const float gb_psqt = __ldg(&grad_bpsqt[block_idx]);
        const uint32_t clamp_base = block_idx * 4 * l1_half;

        if (tid == 0) {
            shared_grad_bias[l1_size + p_idx] += gw_psqt + gb_psqt;
        }

        for (int col = tid; col < l1_half; col += n_threads) {
            float clamped_w0 = __ldg(&clamped_out[clamp_base + 0 * l1_half + col]);
            float clamped_w1 = __ldg(&clamped_out[clamp_base + 1 * l1_half + col]);
            float clamped_b0 = __ldg(&clamped_out[clamp_base + 2 * l1_half + col]);
            float clamped_b1 = __ldg(&clamped_out[clamp_base + 3 * l1_half + col]);

            float gl0_i   = __ldg(&grad_l0[block_idx * l1_size + col]);
            float gl0_i_h = __ldg(&grad_l0[block_idx * l1_size + l1_half + col]);

            float dw0 = (clamped_w0 == 0.0f || clamped_w0 == max_ft_act) ? 0.0f : gl0_i   * clamped_w1;
            float dw1 = (clamped_w1 == 0.0f || clamped_w1 == max_ft_act) ? 0.0f : gl0_i   * clamped_w0;
            float db0 = (clamped_b0 == 0.0f || clamped_b0 == max_ft_act) ? 0.0f : gl0_i_h * clamped_b1;
            float db1 = (clamped_b1 == 0.0f || clamped_b1 == max_ft_act) ? 0.0f : gl0_i_h * clamped_b0;

            float g_w0 = us_val * dw0 + them_val * db0;
            float g_w1 = us_val * dw1 + them_val * db1;
            float g_b0 = them_val * dw0 + us_val * db0;
            float g_b1 = them_val * dw1 + us_val * db1;

            for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
                int w_idx = w_idx_row[k];
                if (w_idx == -1) break;
                if (col == 0) {
                    atomicAdd(&grad_weight[w_idx * output_size + l1_size + p_idx], gw_psqt);
                }
                atomicAdd(&grad_weight[w_idx * output_size + col],           g_w0);
                atomicAdd(&grad_weight[w_idx * output_size + col + l1_half], g_w1);
            }

            for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
                int b_idx = b_idx_row[k];
                if (b_idx == -1) break;
                if (col == 0) {
                    atomicAdd(&grad_weight[b_idx * output_size + l1_size + p_idx], gb_psqt);
                }
                atomicAdd(&grad_weight[b_idx * output_size + col],           g_b0);
                atomicAdd(&grad_weight[b_idx * output_size + col + l1_half], g_b1);
            }

            shared_grad_bias[col]           += g_w0 + g_b0;
            shared_grad_bias[col + l1_half] += g_w1 + g_b1;
        }
    }

    __syncthreads();
    for (int i = tid; i < output_size; i += n_threads) {
        const float val = shared_grad_bias[i];
        if (val != 0.0f) {
            atomicAdd(&grad_bias[i], val);
        }
    }
}
""",
            "fused_double_ft_backward",
        )
        kernel.compile()
        _fused_double_ft_backward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _fused_double_ft_backward_kernel_cache[key]
