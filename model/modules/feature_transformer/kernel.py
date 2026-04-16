import cupy as cp
import torch

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


_sparse_input_linear_backward_kernel_cache = dict()

@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(max_active_indices: int, output_size: int):
    num_threads = min(512, (output_size + 31) // 32 * 32)
    key = (max_active_indices, output_size)

    if key not in _sparse_input_linear_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void sparse_input_linear_backward(
    const int32_t* const input_indices,
    const float* const input_values,
          float* const weight_grad,
          float* const bias_grad,
    const float* const output_grad
) {{
    const uint32_t block_idx = blockIdx.x;
    const int32_t* const input_index_row = input_indices + block_idx * {max_active_indices};
    const float* const input_value_row = input_values  + block_idx * {max_active_indices};

    for (uint32_t col = threadIdx.x; col < {output_size}; col += blockDim.x)
    {{
        // Load gradient into register once per thread
        const float out_g = output_grad[block_idx * {output_size} + col];

        if (out_g != 0.0f)
        {{
            atomicAdd(&bias_grad[col], out_g);

            for (uint32_t k = 0; k < {max_active_indices}; ++k)
            {{
                const int32_t input_index = input_index_row[k];
                if (input_index == -1) break;

                const float input_value = input_value_row[k];
                atomicAdd(&weight_grad[input_index * {output_size} + col], out_g * input_value);
            }}
        }}
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                output_size=output_size,
            ),
            "sparse_input_linear_backward",
        )
        kernel.compile()
        _sparse_input_linear_backward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_backward_kernel_cache[key]
