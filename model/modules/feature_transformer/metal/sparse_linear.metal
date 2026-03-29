#include <metal_stdlib>
using namespace metal;

// Function constants — specialized at pipeline creation time.
// Same configuration-caching pattern as the CuPy CUDA kernels in kernel.py.
constant uint FC_MAX_ACTIVE  [[function_constant(0)]];
constant uint FC_OUTPUT_SIZE [[function_constant(1)]];
constant uint FC_SLICE_SIZE  [[function_constant(2)]];

// ---------------------------------------------------------------------------
// CAS-based float atomic add (Metal 2.x / M1 compatible).
// Metal 3.0+ (M2+) has native atomic<float>; this fallback works everywhere.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Forward: output[b] = sum_k(weight[indices[b,k]] * values[b,k]) + bias
//
// One threadgroup per batch element.  Each thread handles FC_SLICE_SIZE
// consecutive elements of the output vector.
// ---------------------------------------------------------------------------
kernel void sparse_input_linear_forward(
    device const int*   input_indices [[buffer(0)]],
    device const float* input_values  [[buffer(1)]],
    device const float* weight        [[buffer(2)]],
    device const float* bias          [[buffer(3)]],
    device float*       output        [[buffer(4)]],
    threadgroup float*  shared_output [[threadgroup(0)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint t_pos  [[thread_position_in_threadgroup]]
) {
    const uint block_idx    = tg_pos;
    const uint slice_offset = t_pos * FC_SLICE_SIZE;

    device float*       out_slice  = output + block_idx * FC_OUTPUT_SIZE + slice_offset;
    device const float* bias_slice = bias + slice_offset;
    threadgroup float*  sh_slice   = shared_output + slice_offset;

    device const int*   idx_row = input_indices + block_idx * FC_MAX_ACTIVE;
    device const float* val_row = input_values  + block_idx * FC_MAX_ACTIVE;

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        sh_slice[s] = bias_slice[s];

    for (uint k = 0; k < FC_MAX_ACTIVE; ++k) {
        const int   idx = idx_row[k];
        const float val = val_row[k];
        if (idx == -1) break;
        device const float* w_slice = weight + idx * FC_OUTPUT_SIZE + slice_offset;
        for (uint s = 0; s < FC_SLICE_SIZE; ++s)
            sh_slice[s] += w_slice[s] * val;
    }

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        out_slice[s] = sh_slice[s];
}

// ---------------------------------------------------------------------------
// Backward: accumulates weight_grad and bias_grad from output_grad.
//
// Same threadgroup layout as forward.  Gradients are accumulated with
// atomic_add_f32 because multiple batch elements may reference the same
// weight row.
// ---------------------------------------------------------------------------
kernel void sparse_input_linear_backward(
    device const int*   input_indices [[buffer(0)]],
    device const float* input_values  [[buffer(1)]],
    device float*       weight_grad   [[buffer(2)]],
    device float*       bias_grad     [[buffer(3)]],
    device const float* output_grad   [[buffer(4)]],
    threadgroup float*  shared_grad   [[threadgroup(0)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint t_pos  [[thread_position_in_threadgroup]]
) {
    const uint block_idx    = tg_pos;
    const uint slice_offset = t_pos * FC_SLICE_SIZE;

    device const float* og_slice = output_grad + block_idx * FC_OUTPUT_SIZE + slice_offset;
    device float*       bg_slice = bias_grad + slice_offset;
    threadgroup float*  sh_slice = shared_grad + slice_offset;

    device const int*   idx_row = input_indices + block_idx * FC_MAX_ACTIVE;
    device const float* val_row = input_values  + block_idx * FC_MAX_ACTIVE;

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        sh_slice[s] = og_slice[s];

    for (uint s = 0; s < FC_SLICE_SIZE; ++s) {
        float sog = sh_slice[s];
        if (sog != 0.0f)
            atomic_add_f32(&bg_slice[s], sog);
    }

    for (uint k = 0; k < FC_MAX_ACTIVE; ++k) {
        const int   idx = idx_row[k];
        const float val = val_row[k];
        if (idx == -1) break;
        device float* wg_slice = weight_grad + idx * FC_OUTPUT_SIZE + slice_offset;
        for (uint s = 0; s < FC_SLICE_SIZE; ++s) {
            float sog = sh_slice[s];
            if (sog != 0.0f)
                atomic_add_f32(&wg_slice[s], sog * val);
        }
    }
}
