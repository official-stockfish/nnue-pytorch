#include <metal_stdlib>
using namespace metal;

// Backward: weight_grad only. Bias_grad is computed separately via .sum(dim=0).
// Metal 3.0+ (M2+) — uses hardware-native atomic<float>.

constant uint FC_MAX_ACTIVE  [[function_constant(0)]];
constant uint FC_OUTPUT_SIZE [[function_constant(1)]];
constant uint FC_SLICE_SIZE  [[function_constant(2)]];

kernel void sparse_input_linear_backward(
    device const int*      input_indices [[buffer(0)]],
    device const float*    input_values  [[buffer(1)]],
    device atomic<float>*  weight_grad   [[buffer(2)]],
    device const float*    output_grad   [[buffer(3)]],
    threadgroup float*     tg_mem        [[threadgroup(0)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint t_pos  [[thread_position_in_threadgroup]]
) {
    const uint block_idx    = tg_pos;
    const uint slice_offset = t_pos * FC_SLICE_SIZE;

    device const float* og_slice = output_grad + block_idx * FC_OUTPUT_SIZE + slice_offset;
    device const int*   idx_row  = input_indices + block_idx * FC_MAX_ACTIVE;
    device const float* val_row  = input_values  + block_idx * FC_MAX_ACTIVE;
    threadgroup float*  cached_grad = tg_mem + t_pos * FC_SLICE_SIZE;

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        cached_grad[s] = og_slice[s];

    for (uint k = 0; k < FC_MAX_ACTIVE; ++k) {
        const int   idx = idx_row[k];
        const float val = val_row[k];
        if (idx == -1) break;
        device atomic<float>* wg_slice = weight_grad + idx * FC_OUTPUT_SIZE + slice_offset;
        for (uint s = 0; s < FC_SLICE_SIZE; ++s) {
            float sog = cached_grad[s];
            if (sog != 0.0f)
                atomic_fetch_add_explicit(&wg_slice[s], sog * val, memory_order_relaxed);
        }
    }
}
