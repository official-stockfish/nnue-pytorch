import torch
import cupy as cp

def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)
    return f

_sparse_input_linear_forward_kernel_cache = dict()

@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_forward_kernel(max_active_indices: int, output_size: int):
    # A standard block size of 256 or 512 is generally optimal.
    # The block-stride loop handles any output_size automatically.
    num_threads = min(512, (output_size + 31) // 32 * 32)
    key = (max_active_indices, output_size)

    if key not in _sparse_input_linear_forward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void sparse_input_linear_forward(
    const int32_t* const input_indices,
    const float* const input_values,
    const float* const weight,
    const float* const bias,
          float* const output
) {{
    const uint32_t block_idx = blockIdx.x;
    const int32_t* const input_index_row = input_indices + block_idx * {max_active_indices};
    const float* const input_value_row = input_values  + block_idx * {max_active_indices};

    // Block-stride loop ensures perfectly coalesced memory access
    for (uint32_t col = threadIdx.x; col < {output_size}; col += blockDim.x)
    {{
        // Accumulate in a fast thread-local register
        float acc = bias[col];

        for (uint32_t k = 0; k < {max_active_indices}; ++k)
        {{
            const int32_t input_index = input_index_row[k];
            if (input_index == -1) break;

            const float input_value = input_value_row[k];
            acc += weight[input_index * {output_size} + col] * input_value;
        }}

        output[block_idx * {output_size} + col] = acc;
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                output_size=output_size,
            ),
            "sparse_input_linear_forward",
        )
        kernel.compile()
        _sparse_input_linear_forward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_forward_kernel_cache[key]

_sparse_input_linear_backward_cache = dict()

@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(max_active_indices: int, output_size: int):
    threads_per_block_y = min(256, (output_size + 31) // 32 * 32)
    key = (max_active_indices, output_size)

    if key not in _sparse_input_linear_backward_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void sparse_input_linear_backward(
          float* const weight_grad,
          float* const bias_grad,
    const int32_t* const input_indices,
    const float* const input_values,
    const float* const output_grad,
    const uint32_t batch_size,
    const uint32_t batch_chunk_size
) {{
    const uint32_t batch_chunk_idx = blockIdx.x;
    const uint32_t col_offset = blockIdx.y * blockDim.x;
    const uint32_t col = col_offset + threadIdx.x;

    if (col >= {output_size}) return;

    const uint32_t batch_start = batch_chunk_idx * batch_chunk_size;
    const uint32_t batch_end = min(batch_start + batch_chunk_size, batch_size);

    float local_bias_acc = 0.0f;

    for (uint32_t b = batch_start; b < batch_end; ++b)
    {{
        const float out_g = output_grad[b * {output_size} + col];

        if (out_g != 0.0f)
        {{
            local_bias_acc += out_g;

            const int32_t* const input_index_row = input_indices + b * {max_active_indices};
            const float* const input_value_row = input_values  + b * {max_active_indices};

            for (uint32_t k = 0; k < {max_active_indices}; ++k)
            {{
                const int32_t input_index = input_index_row[k];
                if (input_index == -1) break;

                const float input_value = input_value_row[k];
                atomicAdd(&weight_grad[input_index * {output_size} + col], out_g * input_value);
            }}
        }}
    }}

    if (local_bias_acc != 0.0f && bias_grad != nullptr)
    {{
        atomicAdd(&bias_grad[col], local_bias_acc);
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                output_size=output_size,
            ),
            "sparse_input_linear_backward",
        )
        kernel.compile()
        _sparse_input_linear_backward_cache[key] = (kernel, threads_per_block_y)

    return _sparse_input_linear_backward_cache[key]
