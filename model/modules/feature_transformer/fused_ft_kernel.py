import cupy as cp
import torch

from .sparse_linear_kernel import (
    _get_num_threads_for_backward,
    _get_num_threads_for_forward,
    _kernel_with_threads,
)

_fused_double_ft_forward_kernel_cache = {}

@torch.compiler.disable(recursive=False)
def make_fused_double_ft_forward_kernel(max_active_indices: int, l1_size: int):
    l1_half = l1_size // 2
    num_threads = _get_num_threads_for_forward(l1_half)
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
    const uint32_t slice_offset = threadIdx.x * """ + str(output_thread_slice_size) + r""";

    const float us_val = __ldg(&us[block_idx]);
    const float them_val = __ldg(&them[block_idx]);

    const int32_t* const w_idx_row = white_indices + block_idx * """ + str(max_active_indices) + r""";
    const int32_t* const b_idx_row = black_indices + block_idx * """ + str(max_active_indices) + r""";

    const int32_t l1_size = """ + str(l1_size) + r""";
    const int32_t l1_half = """ + str(l1_half) + r""";
    const int64_t p_idx = __ldg(&psqt_indices[block_idx]);
    float w_psqt_val = __ldg(&bias[l1_size + p_idx]);
    float b_psqt_val = __ldg(&bias[l1_size + p_idx]);

    #pragma unroll
    for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
        uint32_t i = slice_offset + s;
        float w0 = __ldg(&bias[i]);
        float w1 = __ldg(&bias[i + l1_half]);
        float b0 = __ldg(&bias[i]);
        float b1 = __ldg(&bias[i + l1_half]);

        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx != -1) {
                w0 += __ldg(&weight[w_idx * output_size + i]);
                w1 += __ldg(&weight[w_idx * output_size + i + l1_half]);
            } else break;
        }

        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                b0 += __ldg(&weight[b_idx * output_size + i]);
                b1 += __ldg(&weight[b_idx * output_size + i + l1_half]);
            } else break;
        }

        float l0_w0 = us_val * w0 + them_val * b0;
        float l0_w1 = us_val * w1 + them_val * b1;
        float l0_b0 = us_val * b0 + them_val * w0;
        float l0_b1 = us_val * b1 + them_val * w1;

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
            if (w_idx != -1) {
                w_psqt_val += __ldg(&weight[w_idx * output_size + l1_size + p_idx]);
            } else break;
        }
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                b_psqt_val += __ldg(&weight[b_idx * output_size + l1_size + p_idx]);
            } else break;
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
    num_threads = _get_num_threads_for_backward(l1_half)
    output_thread_slice_size = l1_half // num_threads
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
    const uint32_t slice_offset = threadIdx.x * """ + str(output_thread_slice_size) + r""";

    const int32_t l1_size = """ + str(l1_size) + r""";
    const int32_t l1_half = """ + str(l1_half) + r""";
    const int32_t tile_size = """ + str(tile_size) + r""";

    __shared__ float shared_grad_bias[""" + str(output_size) + r"""];
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        shared_grad_bias[i] = 0.0f;
    }
    __syncthreads();

    float g_w0[ """ + str(output_thread_slice_size) + r""" ];
    float g_w1[ """ + str(output_thread_slice_size) + r""" ];
    float g_b0[ """ + str(output_thread_slice_size) + r""" ];
    float g_b1[ """ + str(output_thread_slice_size) + r""" ];
    float bias_acc0[ """ + str(output_thread_slice_size) + r""" ];
    float bias_acc1[ """ + str(output_thread_slice_size) + r""" ];

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

        if (threadIdx.x == 0) {
            shared_grad_bias[l1_size + p_idx] += gw_psqt + gb_psqt;
        }

        #pragma unroll
        for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
            uint32_t i = slice_offset + s;
            float clamped_w0 = __ldg(&clamped_out[clamp_base + 0 * l1_half + i]);
            float clamped_w1 = __ldg(&clamped_out[clamp_base + 1 * l1_half + i]);
            float clamped_b0 = __ldg(&clamped_out[clamp_base + 2 * l1_half + i]);
            float clamped_b1 = __ldg(&clamped_out[clamp_base + 3 * l1_half + i]);

            float gl0_i    = __ldg(&grad_l0[block_idx * l1_size + i]);
            float gl0_i_h  = __ldg(&grad_l0[block_idx * l1_size + l1_half + i]);

            float dw0 = (clamped_w0 == 0.0f || clamped_w0 == max_ft_act) ? 0.0f : gl0_i   * clamped_w1;
            float dw1 = (clamped_w1 == 0.0f || clamped_w1 == max_ft_act) ? 0.0f : gl0_i   * clamped_w0;
            float db0 = (clamped_b0 == 0.0f || clamped_b0 == max_ft_act) ? 0.0f : gl0_i_h * clamped_b1;
            float db1 = (clamped_b1 == 0.0f || clamped_b1 == max_ft_act) ? 0.0f : gl0_i_h * clamped_b0;

            g_w0[s] = us_val * dw0 + them_val * db0;
            g_w1[s] = us_val * dw1 + them_val * db1;
            g_b0[s] = them_val * dw0 + us_val * db0;
            g_b1[s] = them_val * dw1 + us_val * db1;

            bias_acc0[s] = g_w0[s] + g_b0[s];
            bias_acc1[s] = g_w1[s] + g_b1[s];
        }

        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx == -1) break;
            if (threadIdx.x == 0) {
                atomicAdd(&grad_weight[w_idx * output_size + l1_size + p_idx], gw_psqt);
            }
            #pragma unroll
            for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
                uint32_t i = slice_offset + s;
                atomicAdd(&grad_weight[w_idx * output_size + i],           g_w0[s]);
                atomicAdd(&grad_weight[w_idx * output_size + i + l1_half], g_w1[s]);
            }
        }

        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx == -1) break;
            if (threadIdx.x == 0) {
                atomicAdd(&grad_weight[b_idx * output_size + l1_size + p_idx], gb_psqt);
            }
            #pragma unroll
            for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
                uint32_t i = slice_offset + s;
                atomicAdd(&grad_weight[b_idx * output_size + i],           g_b0[s]);
                atomicAdd(&grad_weight[b_idx * output_size + i + l1_half], g_b1[s]);
            }
        }

        #pragma unroll
        for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
            uint32_t i = slice_offset + s;
            shared_grad_bias[i]           += bias_acc0[s];
            shared_grad_bias[i + l1_half] += bias_acc1[s];
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
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
