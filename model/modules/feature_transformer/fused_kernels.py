import cupy as cp
import torch

_fused_nnue_forward_cache = dict()

@torch.compiler.disable(recursive=False)
def make_fused_nnue_forward_kernel(max_active_indices: int, L1: int, num_psqt_buckets: int):
    # H is the half-size of L1, determining the split for the SCReLU
    H = L1 // 2
    total_output_cols = H + num_psqt_buckets
    num_threads = min(512, (total_output_cols + 31) // 32 * 32)

    key = (max_active_indices, L1, num_psqt_buckets)
    if key not in _fused_nnue_forward_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void fused_nnue_forward(
    const int32_t* const w_indices,
    const float* const w_values,
    const int32_t* const b_indices,
    const float* const b_values,
    const float* const weight,
    const float* const bias,
    const float* const us_tensor,
    const float* const them_tensor,
          float* const out_l0,
          float* const out_wpsqt,
          float* const out_bpsqt,
    const float          ft_max_val
) {{
    const uint32_t b = blockIdx.x;

    // Total columns of weight matrix
    const uint32_t OUT_SIZE = {L1} + {num_psqt_buckets};

    const int32_t* const w_idx_row = w_indices + b * {max_active_indices};
    const float* const w_val_row = w_values  + b * {max_active_indices};
    const int32_t* const b_idx_row = b_indices + b * {max_active_indices};
    const float* const b_val_row = b_values  + b * {max_active_indices};

    const float us_val = us_tensor[b];
    const float them_val = them_tensor[b];

    for (uint32_t col = threadIdx.x; col < {total_output_cols}; col += blockDim.x)
    {{
        if (col < {H})
        {{
            // --- GLU Activation Path ---
            float acc_w0 = bias[col];
            float acc_w1 = bias[col + {H}];
            float acc_b0 = acc_w0; // Bias is shared
            float acc_b1 = acc_w1;

            for (uint32_t k = 0; k < {max_active_indices}; ++k)
            {{
                const int32_t w_idx = w_idx_row[k];
                if (w_idx == -1) break;
                const float w_val = w_val_row[k];
                const float* w_row = weight + w_idx * OUT_SIZE;
                acc_w0 += w_row[col]       * w_val;
                acc_w1 += w_row[col + {H}] * w_val;
            }}

            for (uint32_t k = 0; k < {max_active_indices}; ++k)
            {{
                const int32_t b_idx = b_idx_row[k];
                if (b_idx == -1) break;
                const float b_val = b_val_row[k];
                const float* w_row = weight + b_idx * OUT_SIZE;
                acc_b0 += w_row[col]       * b_val;
                acc_b1 += w_row[col + {H}] * b_val;
            }}

            // Apply perspective mixing
            float l0_0 = us_val * acc_w0 + them_val * acc_b0;
            float l0_1 = us_val * acc_w1 + them_val * acc_b1;
            float l0_2 = us_val * acc_b0 + them_val * acc_w0;
            float l0_3 = us_val * acc_b1 + them_val * acc_w1;

            // Clamp (ReLU)
            l0_0 = l0_0 < 0.0f ? 0.0f : (l0_0 > ft_max_val ? ft_max_val : l0_0);
            l0_1 = l0_1 < 0.0f ? 0.0f : (l0_1 > ft_max_val ? ft_max_val : l0_1);
            l0_2 = l0_2 < 0.0f ? 0.0f : (l0_2 > ft_max_val ? ft_max_val : l0_2);
            l0_3 = l0_3 < 0.0f ? 0.0f : (l0_3 > ft_max_val ? ft_max_val : l0_3);

            // Multiply and write to concatenated output
            out_l0[b * {L1} + col]       = l0_0 * l0_1;
            out_l0[b * {L1} + {H} + col] = l0_2 * l0_3;
        }}
        else
        {{
            // --- PSQT Path ---
            const uint32_t psqt_col = {L1} + (col - {H});
            float acc_w = bias[psqt_col];
            float acc_b = acc_w;

            for (uint32_t k = 0; k < {max_active_indices}; ++k)
            {{
                const int32_t w_idx = w_idx_row[k];
                if (w_idx == -1) break;
                acc_w += weight[w_idx * OUT_SIZE + psqt_col] * w_val_row[k];
            }}

            for (uint32_t k = 0; k < {max_active_indices}; ++k)
            {{
                const int32_t b_idx = b_idx_row[k];
                if (b_idx == -1) break;
                acc_b += weight[b_idx * OUT_SIZE + psqt_col] * b_val_row[k];
            }}

            const uint32_t out_psqt_idx = b * {num_psqt_buckets} + (col - {H});
            out_wpsqt[out_psqt_idx] = acc_w;
            out_bpsqt[out_psqt_idx] = acc_b;
        }}
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                L1=L1,
                H=H,
                num_psqt_buckets=num_psqt_buckets,
                total_output_cols=total_output_cols
            ),
            "fused_nnue_forward",
        )
        kernel.compile()
        _fused_nnue_forward_cache[key] = (kernel, num_threads)

    return _fused_nnue_forward_cache[key]


_fused_nnue_backward_cache = dict()

@torch.compiler.disable(recursive=False)
def make_fused_nnue_backward_kernel(max_active_indices: int, L1: int, extra: int):
    H = L1 // 2
    total_output_cols = H + extra
    threads_per_block_y = min(256, (total_output_cols + 31) // 32 * 32)

    key = (max_active_indices, L1, extra)
    if key not in _fused_nnue_backward_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void fused_nnue_backward(
          float* const weight_grad,
          float* const bias_grad,
    const int32_t* const w_indices,
    const float* const w_values,
    const int32_t* const b_indices,
    const float* const b_values,
    const float* const weight,
    const float* const bias,
    const float* const us_tensor,
    const float* const them_tensor,
    const float* const grad_out_l0,
    const float* const grad_out_wpsqt,
    const float* const grad_out_bpsqt,
    const float          ft_max_val,
    const uint32_t       batch_size,
    const uint32_t       batch_chunk_size
) {{
    const uint32_t batch_chunk_idx = blockIdx.x;
    const uint32_t col_offset = blockIdx.y * blockDim.x;
    const uint32_t col = col_offset + threadIdx.x;

    if (col >= {total_output_cols}) return;

    const uint32_t OUT_SIZE = {L1} + {extra};
    const uint32_t batch_start = batch_chunk_idx * batch_chunk_size;
    const uint32_t batch_end = min(batch_start + batch_chunk_size, batch_size);

    float local_bias_acc_0 = 0.0f;
    float local_bias_acc_1 = 0.0f;

    for (uint32_t b = batch_start; b < batch_end; ++b)
    {{
        const int32_t* const w_idx_row = w_indices + b * {max_active_indices};
        const float* const w_val_row = w_values  + b * {max_active_indices};
        const int32_t* const b_idx_row = b_indices + b * {max_active_indices};
        const float* const b_val_row = b_values  + b * {max_active_indices};

        if (col < {H})
        {{
            // --- 1. Recompute Forward State WITH BIAS ---
            const float b_val_static = bias[col];
            const float b_val_h_static = bias[col + {H}];

            // Initialize accumulators with bias to ensure l0 is mathematically identical to forward pass
            float w0 = b_val_static, w1 = b_val_h_static;
            float b0 = b_val_static, b1 = b_val_h_static;

            for (uint32_t k = 0; k < {max_active_indices}; ++k) {{
                const int32_t idx = w_idx_row[k];
                if (idx == -1) break;
                const float val = w_val_row[k];
                w0 += weight[idx * OUT_SIZE + col]       * val;
                w1 += weight[idx * OUT_SIZE + col + {H}] * val;
            }}
            for (uint32_t k = 0; k < {max_active_indices}; ++k) {{
                const int32_t idx = b_idx_row[k];
                if (idx == -1) break;
                const float val = b_val_row[k];
                b0 += weight[idx * OUT_SIZE + col]       * val;
                b1 += weight[idx * OUT_SIZE + col + {H}] * val;
            }}

            const float us = us_tensor[b];
            const float them = them_tensor[b];

            float l0_0 = us * w0 + them * b0;
            float l0_1 = us * w1 + them * b1;
            float l0_2 = us * b0 + them * w0;
            float l0_3 = us * b1 + them * w1;

            float c0 = l0_0 < 0.0f ? 0.0f : (l0_0 > ft_max_val ? ft_max_val : l0_0);
            float c1 = l0_1 < 0.0f ? 0.0f : (l0_1 > ft_max_val ? ft_max_val : l0_1);
            float c2 = l0_2 < 0.0f ? 0.0f : (l0_2 > ft_max_val ? ft_max_val : l0_2);
            float c3 = l0_3 < 0.0f ? 0.0f : (l0_3 > ft_max_val ? ft_max_val : l0_3);

            // --- 2. Upstream Gradients ---
            const float g_out_0 = grad_out_l0[b * {L1} + col];
            const float g_out_1 = grad_out_l0[b * {L1} + col + {H}];

            // --- 3. Derivative through SCReLU ---
            float g_c0 = g_out_0 * c1;
            float g_c1 = g_out_0 * c0;
            float g_c2 = g_out_1 * c3;
            float g_c3 = g_out_1 * c2;

            float g_l0_0 = (l0_0 > 0.0f && l0_0 < ft_max_val) ? g_c0 : 0.0f;
            float g_l0_1 = (l0_1 > 0.0f && l0_1 < ft_max_val) ? g_c1 : 0.0f;
            float g_l0_2 = (l0_2 > 0.0f && l0_2 < ft_max_val) ? g_c2 : 0.0f;
            float g_l0_3 = (l0_3 > 0.0f && l0_3 < ft_max_val) ? g_c3 : 0.0f;

            // --- 4. Derivative through Perspective Mixing ---
            float g_w0 = us * g_l0_0 + them * g_l0_2;
            float g_b0 = them * g_l0_0 + us * g_l0_2;
            float g_w1 = us * g_l0_1 + them * g_l0_3;
            float g_b1 = them * g_l0_1 + us * g_l0_3;

            // --- 5. Accumulate into registers and VRAM ---
            local_bias_acc_0 += (g_w0 + g_b0);
            local_bias_acc_1 += (g_w1 + g_b1);

            for (uint32_t k = 0; k < {max_active_indices}; ++k) {{
                const int32_t idx = w_idx_row[k];
                if (idx == -1) break;
                const float val = w_val_row[k];
                if (g_w0 != 0.0f) atomicAdd(&weight_grad[idx * OUT_SIZE + col],       g_w0 * val);
                if (g_w1 != 0.0f) atomicAdd(&weight_grad[idx * OUT_SIZE + col + {H}], g_w1 * val);
            }}

            for (uint32_t k = 0; k < {max_active_indices}; ++k) {{
                const int32_t idx = b_idx_row[k];
                if (idx == -1) break;
                const float val = b_val_row[k];
                if (g_b0 != 0.0f) atomicAdd(&weight_grad[idx * OUT_SIZE + col],       g_b0 * val);
                if (g_b1 != 0.0f) atomicAdd(&weight_grad[idx * OUT_SIZE + col + {H}], g_b1 * val);
            }}
        }}
        else
        {{
            // --- PSQT Path Gradients ---
            const uint32_t psqt_col = {L1} + (col - {H});
            const uint32_t out_psqt_idx = b * {extra} + (col - {H});

            const float g_w = grad_out_wpsqt[out_psqt_idx];
            const float g_b = grad_out_bpsqt[out_psqt_idx];

            local_bias_acc_0 += (g_w + g_b);

            if (g_w != 0.0f) {{
                for (uint32_t k = 0; k < {max_active_indices}; ++k) {{
                    const int32_t idx = w_idx_row[k];
                    if (idx == -1) break;
                    atomicAdd(&weight_grad[idx * OUT_SIZE + psqt_col], g_w * w_val_row[k]);
                }}
            }}

            if (g_b != 0.0f) {{
                for (uint32_t k = 0; k < {max_active_indices}; ++k) {{
                    const int32_t idx = b_idx_row[k];
                    if (idx == -1) break;
                    atomicAdd(&weight_grad[idx * OUT_SIZE + psqt_col], g_b * b_val_row[k]);
                }}
            }}
        }}
    }}

    if (col < {H}) {{
        if (local_bias_acc_0 != 0.0f && bias_grad != nullptr) atomicAdd(&bias_grad[col], local_bias_acc_0);
        if (local_bias_acc_1 != 0.0f && bias_grad != nullptr) atomicAdd(&bias_grad[col + {H}], local_bias_acc_1);
    }} else {{
        const uint32_t psqt_col = {L1} + (col - {H});
        if (local_bias_acc_0 != 0.0f && bias_grad != nullptr) atomicAdd(&bias_grad[psqt_col], local_bias_acc_0);
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                L1=L1,
                H=H,
                extra=extra,
                total_output_cols=total_output_cols
            ),
            "fused_nnue_backward",
        )
        kernel.compile()
        _fused_nnue_backward_cache[key] = (kernel, threads_per_block_y)

    return _fused_nnue_backward_cache[key]
