import cupy as cp
import torch

from .sparse_linear_kernel import (
    _get_num_threads_for_forward,
    _get_num_threads_for_backward,
    _kernel_with_threads,
)

_fused_double_ft_forward_kernel_cache = dict()

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
    const float*   const us,
    const float*   const them,
    const int32_t* const white_indices,
    const int32_t* const black_indices,
    const int64_t* const psqt_indices,
    const float*   const weight,
    const float*   const bias,
    const float          max_ft_act,
          float*   const l0_out,
          float*   const wpsqt_out,
          float*   const bpsqt_out,
    const int32_t        output_size
) {
    const uint32_t block_idx = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * """ + str(output_thread_slice_size) + r""";
    
    const float us_val = us[block_idx];
    const float them_val = them[block_idx];
    
    const int32_t* const w_idx_row = white_indices + block_idx * """ + str(max_active_indices) + r""";
    const int32_t* const b_idx_row = black_indices + block_idx * """ + str(max_active_indices) + r""";
    
    const int32_t l1_size = """ + str(l1_size) + r""";
    const int32_t l1_half = """ + str(l1_half) + r""";
    
    #pragma unroll
    for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
        uint32_t i = slice_offset + s;
        float w0 = bias[i];
        float w1 = bias[i + l1_half];
        float b0 = bias[i];
        float b1 = bias[i + l1_half];
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx != -1) {
                w0 += weight[w_idx * output_size + i];
                w1 += weight[w_idx * output_size + i + l1_half];
            } else break;
        }
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                b0 += weight[b_idx * output_size + i];
                b1 += weight[b_idx * output_size + i + l1_half];
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
    }
    
    if (threadIdx.x == 0) {
        int64_t p_idx = psqt_indices[block_idx];
        float w_psqt_val = bias[l1_size + p_idx];
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx != -1) {
                w_psqt_val += weight[w_idx * output_size + l1_size + p_idx];
            } else break;
        }
        wpsqt_out[block_idx] = w_psqt_val;

        float b_psqt_val = bias[l1_size + p_idx];
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                b_psqt_val += weight[b_idx * output_size + l1_size + p_idx];
            } else break;
        }
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

_fused_double_ft_backward_kernel_cache = dict()

@torch.compiler.disable(recursive=False)
def make_fused_double_ft_backward_kernel(max_active_indices: int, l1_size: int):
    l1_half = l1_size // 2
    num_threads = _get_num_threads_for_backward(l1_half)
    output_thread_slice_size = l1_half // num_threads
    
    key = (max_active_indices, l1_size, num_threads)
    if key not in _fused_double_ft_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;

extern "C" __global__
void fused_double_ft_backward(
    const float*   const us,
    const float*   const them,
    const int32_t* const white_indices,
    const int32_t* const black_indices,
    const int64_t* const psqt_indices,
    const float*   const weight,
    const float*   const bias,
    const float          max_ft_act,
    const float*   const grad_l0,
    const float*   const grad_wpsqt,
    const float*   const grad_bpsqt,
          float*   const grad_weight,
          float*   const grad_bias,
    const int32_t        output_size
) {
    const uint32_t block_idx = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * """ + str(output_thread_slice_size) + r""";
    
    const float us_val = us[block_idx];
    const float them_val = them[block_idx];
    
    const int32_t* const w_idx_row = white_indices + block_idx * """ + str(max_active_indices) + r""";
    const int32_t* const b_idx_row = black_indices + block_idx * """ + str(max_active_indices) + r""";
    
    const int32_t l1_size = """ + str(l1_size) + r""";
    const int32_t l1_half = """ + str(l1_half) + r""";
    
    #pragma unroll
    for (uint32_t s = 0; s < """ + str(output_thread_slice_size) + r"""; ++s) {
        uint32_t i = slice_offset + s;
        float w0 = bias[i];
        float w1 = bias[i + l1_half];
        float b0 = bias[i];
        float b1 = bias[i + l1_half];
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx != -1) {
                w0 += weight[w_idx * output_size + i];
                w1 += weight[w_idx * output_size + i + l1_half];
            } else break;
        }
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                b0 += weight[b_idx * output_size + i];
                b1 += weight[b_idx * output_size + i + l1_half];
            } else break;
        }
        
        float l0_w0 = us_val * w0 + them_val * b0;
        float l0_w1 = us_val * w1 + them_val * b1;
        float l0_b0 = us_val * b0 + them_val * w0;
        float l0_b1 = us_val * b1 + them_val * w1;
        
        float clamped_w0 = l0_w0; if (clamped_w0 < 0.0f) clamped_w0 = 0.0f; else if (clamped_w0 > max_ft_act) clamped_w0 = max_ft_act;
        float clamped_w1 = l0_w1; if (clamped_w1 < 0.0f) clamped_w1 = 0.0f; else if (clamped_w1 > max_ft_act) clamped_w1 = max_ft_act;
        float clamped_b0 = l0_b0; if (clamped_b0 < 0.0f) clamped_b0 = 0.0f; else if (clamped_b0 > max_ft_act) clamped_b0 = max_ft_act;
        float clamped_b1 = l0_b1; if (clamped_b1 < 0.0f) clamped_b1 = 0.0f; else if (clamped_b1 > max_ft_act) clamped_b1 = max_ft_act;
        
        float g_l0_w0 = grad_l0[block_idx * l1_size + i] * clamped_w1;
        float g_l0_w1 = grad_l0[block_idx * l1_size + i] * clamped_w0;
        float g_l0_b0 = grad_l0[block_idx * l1_size + l1_half + i] * clamped_b1;
        float g_l0_b1 = grad_l0[block_idx * l1_size + l1_half + i] * clamped_b0;
        
        if (l0_w0 <= 0.0f || l0_w0 >= max_ft_act) g_l0_w0 = 0.0f;
        if (l0_w1 <= 0.0f || l0_w1 >= max_ft_act) g_l0_w1 = 0.0f;
        if (l0_b0 <= 0.0f || l0_b0 >= max_ft_act) g_l0_b0 = 0.0f;
        if (l0_b1 <= 0.0f || l0_b1 >= max_ft_act) g_l0_b1 = 0.0f;
        
        float g_w0 = us_val * g_l0_w0 + them_val * g_l0_b0;
        float g_b0 = them_val * g_l0_w0 + us_val * g_l0_b0;
        float g_w1 = us_val * g_l0_w1 + them_val * g_l0_b1;
        float g_b1 = them_val * g_l0_w1 + us_val * g_l0_b1;
        
        atomicAdd(&grad_bias[i], g_w0 + g_b0);
        atomicAdd(&grad_bias[i + l1_half], g_w1 + g_b1);
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx != -1) {
                atomicAdd(&grad_weight[w_idx * output_size + i], g_w0);
                atomicAdd(&grad_weight[w_idx * output_size + i + l1_half], g_w1);
            } else break;
        }
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                atomicAdd(&grad_weight[b_idx * output_size + i], g_b0);
                atomicAdd(&grad_weight[b_idx * output_size + i + l1_half], g_b1);
            } else break;
        }
    }
    
    if (threadIdx.x == 0) {
        int64_t p_idx = psqt_indices[block_idx];
        float gw_psqt = grad_wpsqt[block_idx];
        float gb_psqt = grad_bpsqt[block_idx];
        
        atomicAdd(&grad_bias[l1_size + p_idx], gw_psqt + gb_psqt);
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int w_idx = w_idx_row[k];
            if (w_idx != -1) {
                atomicAdd(&grad_weight[w_idx * output_size + l1_size + p_idx], gw_psqt);
            } else break;
        }
        
        for(int k=0; k<""" + str(max_active_indices) + r"""; ++k) {
            int b_idx = b_idx_row[k];
            if (b_idx != -1) {
                atomicAdd(&grad_weight[b_idx * output_size + l1_size + p_idx], gb_psqt);
            } else break;
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
