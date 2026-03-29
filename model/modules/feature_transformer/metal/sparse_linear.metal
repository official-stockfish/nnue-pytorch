#include <metal_stdlib>
using namespace metal;

// Function constants — specialized at pipeline creation time.
constant uint FC_MAX_ACTIVE  [[function_constant(0)]];
constant uint FC_OUTPUT_SIZE [[function_constant(1)]];
constant uint FC_SLICE_SIZE  [[function_constant(2)]];

// ---------------------------------------------------------------------------
// Forward: output[b] = sum_k(weight[indices[b,k]] * values[b,k]) + bias
//
// One threadgroup per batch element.  Each thread accumulates FC_SLICE_SIZE
// consecutive output elements in threadgroup memory, preserving the
// cache-friendly weight access pattern (contiguous reads per feature).
// Threadgroup memory is sized at dispatch time via setThreadgroupMemoryLength.
// ---------------------------------------------------------------------------
kernel void sparse_input_linear_forward(
    device const int*   input_indices [[buffer(0)]],
    device const float* input_values  [[buffer(1)]],
    device const float* weight        [[buffer(2)]],
    device const float* bias          [[buffer(3)]],
    device float*       output        [[buffer(4)]],
    threadgroup float*  tg_mem        [[threadgroup(0)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint t_pos  [[thread_position_in_threadgroup]]
) {
    const uint block_idx    = tg_pos;
    const uint slice_offset = t_pos * FC_SLICE_SIZE;

    device float*       out_slice  = output + block_idx * FC_OUTPUT_SIZE + slice_offset;
    device const float* bias_slice = bias + slice_offset;
    threadgroup float*  acc        = tg_mem + t_pos * FC_SLICE_SIZE;

    device const int*   idx_row = input_indices + block_idx * FC_MAX_ACTIVE;
    device const float* val_row = input_values  + block_idx * FC_MAX_ACTIVE;

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        acc[s] = bias_slice[s];

    for (uint k = 0; k < FC_MAX_ACTIVE; ++k) {
        const int   idx = idx_row[k];
        const float val = val_row[k];
        if (idx == -1) break;
        device const float* w_slice = weight + idx * FC_OUTPUT_SIZE + slice_offset;
        for (uint s = 0; s < FC_SLICE_SIZE; ++s)
            acc[s] += w_slice[s] * val;
    }

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        out_slice[s] = acc[s];
}
