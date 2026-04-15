#include <metal_stdlib>
using namespace metal;

// Function constants — specialized at pipeline creation time.
constant uint FC_MAX_ACTIVE  [[function_constant(0)]];
constant uint FC_OUTPUT_SIZE [[function_constant(1)]];
constant uint FC_SLICE_SIZE  [[function_constant(2)]];

// ---------------------------------------------------------------------------
// Forward: output[b] = sum_k(weight[indices[b,k]]) + bias
//
// Feature values are always 1.0 where active, so the multiply is elided.
// One threadgroup per batch element.  Each thread accumulates FC_SLICE_SIZE
// consecutive output elements in threadgroup memory, preserving the
// cache-friendly weight access pattern (contiguous reads per feature).
// Threadgroup memory is sized at dispatch time via setThreadgroupMemoryLength.
// ---------------------------------------------------------------------------
kernel void sparse_input_linear_forward(
    device const int*   input_indices [[buffer(0)]],
    device const float* weight        [[buffer(1)]],
    device const float* bias          [[buffer(2)]],
    device float*       output        [[buffer(3)]],
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

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        acc[s] = bias_slice[s];

    for (uint k = 0; k < FC_MAX_ACTIVE; ++k) {
        const int idx = idx_row[k];
        if (idx == -1) break;
        device const float* w_slice = weight + idx * FC_OUTPUT_SIZE + slice_offset;
        for (uint s = 0; s < FC_SLICE_SIZE; ++s)
            acc[s] += w_slice[s];
    }

    for (uint s = 0; s < FC_SLICE_SIZE; ++s)
        out_slice[s] = acc[s];
}

// ---------------------------------------------------------------------------
// Fused weight merge: cat(weight_a, weight_b + virtual_w) → merged
//
// Writes weight_a rows directly, then weight_b rows with virtual_w added
// on the fly. Eliminates the intermediate merged tensor from
// HalfKav2Hm.merged_weight() + torch.cat().
//
// Grid: (num_a + num_b) threadgroups, FC_OUTPUT_SIZE / 4 threads each.
// Thread t writes 4 consecutive elements of one output row.
// ---------------------------------------------------------------------------
kernel void fused_weight_merge(
    device const float* weight_a   [[buffer(0)]],
    device const float* weight_b   [[buffer(1)]],
    device const float* virtual_w  [[buffer(2)]],
    device float*       merged     [[buffer(3)]],
    constant uint&      num_a      [[buffer(4)]],
    constant uint&      vw_period  [[buffer(5)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint col = tid * 4;
    if (col >= FC_OUTPUT_SIZE) return;
    const uint n = min(FC_OUTPUT_SIZE - col, 4u);

    device float* out_row = merged + tg * FC_OUTPUT_SIZE + col;

    if (tg < num_a) {
        device const float* src = weight_a + tg * FC_OUTPUT_SIZE + col;
        for (uint i = 0; i < n; ++i)
            out_row[i] = src[i];
    } else {
        uint local = tg - num_a;
        device const float* wb = weight_b  + local * FC_OUTPUT_SIZE + col;
        device const float* vw = virtual_w + (local % vw_period) * FC_OUTPUT_SIZE + col;
        for (uint i = 0; i < n; ++i)
            out_row[i] = wb[i] + vw[i];
    }
}
