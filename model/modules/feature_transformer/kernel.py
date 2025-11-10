import cupy as cp
import torch


def _find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value + 1):
        if value % i == 0:
            divisors.append((i, abs(target - i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


_num_threads_forward_cache = dict()


def _get_num_threads_for_forward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_forward_cache[output_size]


_num_threads_backward_cache = dict()


def _get_num_threads_for_backward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_backward_cache[output_size]


def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)

    return f


_feature_transformer_slice_forward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_feature_transformer_slice_forward_kernel(max_active_features, output_size):
    """
    @param: max_active_features
        The maximum number of features that are active
        (non-zero) for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.
    """
    num_threads = _get_num_threads_for_forward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_features, output_size, num_threads)
    if key not in _feature_transformer_slice_forward_kernel_cache:
        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type float32.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type float32.
*/
void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {{
    __shared__
          float          shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          float*   const output_slice        = output + block_idx * {output_size} + slice_offset;
    const float*   const bias_slice          = bias                               + slice_offset;
          float*         shared_output_slice = shared_output                      + slice_offset;

    const int32_t* const feature_index_row   = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row   = feature_values  + block_idx * {max_active_features};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = bias_slice[s];
    }}

    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {{
            const float* const weight_slice = weight + feature_index * {output_size} + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                shared_output_slice[s] += weight_slice[s] * feature_value;
            }}
        }} else break;
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = shared_output_slice[s];
    }}
}}

""".format(
                max_active_features=max_active_features,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
            ),
            "feature_transformer_slice_forward",
        )
        kernel.compile()
        _feature_transformer_slice_forward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _feature_transformer_slice_forward_kernel_cache[key]


_feature_transformer_slice_backward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_feature_transformer_slice_backward_kernel(max_active_features, output_size):
    """
    @param: max_active_features
        The maximum number of features that are active
        (non-zero) for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.
    """
    num_threads = _get_num_threads_for_backward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_features, output_size, num_threads)
    if key not in _feature_transformer_slice_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight_grad
        The weight gradient matrix of shape (NUM_INPUTS, output_size).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Weights must be of type float32.

    @param: bias_grad
        The bias gradient vector of shape (output_size,).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Bias values must be of type float32.

    @param: output_grad
        An output gradient matrix of shape (BATCH_SIZE, output_size).
        Output values must have type float32.
*/
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad
) {{
    __shared__
          float          shared_output_grad[{output_size}];

    const uint32_t       block_idx                = blockIdx.x;
    const uint32_t       slice_offset             = threadIdx.x * {output_thread_slice_size};

    const float*   const output_grad_slice        = output_grad + block_idx * {output_size} + slice_offset;
          float*   const bias_grad_slice          = bias_grad                               + slice_offset;
          float*         shared_output_grad_slice = shared_output_grad                      + slice_offset;

    const int32_t* const feature_index_row        = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row        = feature_values  + block_idx * {max_active_features};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_grad_slice[s] = output_grad_slice[s];
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        const float sog = shared_output_grad_slice[s];
        if (sog != 0.0f)
        {{
            atomicAdd(&bias_grad_slice[s], sog);
        }}
    }}

    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {{
            float* const weight_grad_slice = weight_grad + feature_index * {output_size} + slice_offset;
            #pragma unroll
            for (int s = 0; s < {output_thread_slice_size}; ++s)
            {{
                const float sog = shared_output_grad_slice[s];
                if (sog != 0.0f)
                {{
                    atomicAdd(&weight_grad_slice[s], sog * feature_value);
                }}
            }}
        }} else break;
    }}
}}

""".format(
                max_active_features=max_active_features,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
            ),
            "feature_transformer_slice_backward",
        )
        kernel.compile()
        _feature_transformer_slice_backward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _feature_transformer_slice_backward_kernel_cache[key]
