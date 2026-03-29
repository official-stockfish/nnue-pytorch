#include <metal_stdlib>
using namespace metal;

// Backward: weight_grad only. Bias_grad is computed separately via .sum(dim=0).
// Metal 2.x (M1) — uses CAS-based float atomic add.

constant uint FC_MAX_ACTIVE  [[function_constant(0)]];
constant uint FC_OUTPUT_SIZE [[function_constant(1)]];
constant uint FC_SLICE_SIZE  [[function_constant(2)]];

inline void atomic_add_f32(device float* addr, float val) {
    device atomic_uint* a = (device atomic_uint*)addr;
    uint expected = atomic_load_explicit(a, memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(expected);
        uint desired = as_type<uint>(cur + val);
        if (atomic_compare_exchange_weak_explicit(
                a, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

kernel void sparse_input_linear_backward(
    device const int*   input_indices [[buffer(0)]],
    device const float* input_values  [[buffer(1)]],
    device float*       weight_grad   [[buffer(2)]],
    device const float* output_grad   [[buffer(3)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint t_pos  [[thread_position_in_threadgroup]]
) {
    const uint block_idx    = tg_pos;
    const uint slice_offset = t_pos * FC_SLICE_SIZE;

    device const float* og_slice = output_grad + block_idx * FC_OUTPUT_SIZE + slice_offset;
    device const int*   idx_row  = input_indices + block_idx * FC_MAX_ACTIVE;
    device const float* val_row  = input_values  + block_idx * FC_MAX_ACTIVE;

    float cached_grad[8];
    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        cached_grad[s] = og_slice[s];

    for (uint k = 0; k < FC_MAX_ACTIVE; ++k) {
        const int   idx = idx_row[k];
        const float val = val_row[k];
        if (idx == -1) break;
        device float* wg_slice = weight_grad + idx * FC_OUTPUT_SIZE + slice_offset;
        for (uint s = 0; s < FC_SLICE_SIZE; ++s) {
            float sog = cached_grad[s];
            if (sog != 0.0f)
                atomic_add_f32(&wg_slice[s], sog * val);
        }
    }
}
