#include <metal_stdlib>
using namespace metal;

// Function constants (specialized at pipeline creation time):
//   FC_L1   = feature transformer L1 width (e.g. 1024)
//   FC_H    = L1 / 2 (e.g. 512)
//   FC_PSQT = number of PSQT buckets (e.g. 8)
//   FC_OUT  = L1 + PSQT (e.g. 1032)
constant uint FC_L1   [[function_constant(0)]];
constant uint FC_H    [[function_constant(1)]];
constant uint FC_PSQT [[function_constant(2)]];
constant uint FC_OUT  [[function_constant(3)]];

// ---------------------------------------------------------------------------
// Fused L0 mixing forward.
//
// Python equivalent:
//   wo, wpsqt = wp.split(L1, dim=1)
//   bo, bpsqt = bp.split(L1, dim=1)
//   l0 = (us * cat([wo,bo])) + (them * cat([bo,wo]))
//   l0 = clamp(l0, 0, 1)
//   s0,s1,s2,s3 = split(l0, H, dim=1)
//   l0 = cat(s0*s1, s2*s3) * (127/128)
//
// One threadgroup per batch element, num_threads = FC_H.
// Thread t processes output indices t and t+FC_H.
// ---------------------------------------------------------------------------
kernel void l0_mixing_forward(
    device const float* wp     [[buffer(0)]],   // (B, OUT)
    device const float* bp     [[buffer(1)]],   // (B, OUT)
    device const float* us     [[buffer(2)]],   // (B, 1)
    device const float* them   [[buffer(3)]],   // (B, 1)
    device float* l0           [[buffer(4)]],   // (B, L1)
    device float* wpsqt_out    [[buffer(5)]],   // (B, PSQT)
    device float* bpsqt_out    [[buffer(6)]],   // (B, PSQT)
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint b = tg;
    device const float* wp_b = wp + b * FC_OUT;
    device const float* bp_b = bp + b * FC_OUT;
    const float u = us[b];
    const float t = them[b];
    const float K = 127.0f / 128.0f;

    // Each thread handles two paired indices: tid and tid+FC_H
    // First pair: c_a from first half of us*[wo]+them*[bo]
    //             c_b from second half
    float x_a = u * wp_b[tid]       + t * bp_b[tid];
    float x_b = u * wp_b[tid + FC_H] + t * bp_b[tid + FC_H];
    float c_a = clamp(x_a, 0.0f, 1.0f);
    float c_b = clamp(x_b, 0.0f, 1.0f);

    device float* l0_b = l0 + b * FC_L1;
    l0_b[tid] = c_a * c_b * K;

    // Second pair: from the second L1 block (us*[bo]+them*[wo])
    float x_c = u * bp_b[tid]       + t * wp_b[tid];
    float x_d = u * bp_b[tid + FC_H] + t * wp_b[tid + FC_H];
    float c_c = clamp(x_c, 0.0f, 1.0f);
    float c_d = clamp(x_d, 0.0f, 1.0f);

    l0_b[tid + FC_H] = c_c * c_d * K;

    // First FC_PSQT threads also copy the PSQT slices
    if (tid < FC_PSQT) {
        wpsqt_out[b * FC_PSQT + tid] = wp_b[FC_L1 + tid];
        bpsqt_out[b * FC_PSQT + tid] = bp_b[FC_L1 + tid];
    }
}

// ---------------------------------------------------------------------------
// Fused L0 mixing backward.
//
// Recomputes intermediate clamped values from wp, bp, us, them to avoid
// storing 2*L1 activation tensor.
// ---------------------------------------------------------------------------
kernel void l0_mixing_backward(
    device const float* grad_l0     [[buffer(0)]],  // (B, L1)
    device const float* grad_wpsqt  [[buffer(1)]],  // (B, PSQT)
    device const float* grad_bpsqt  [[buffer(2)]],  // (B, PSQT)
    device const float* wp          [[buffer(3)]],   // (B, OUT) saved from fwd
    device const float* bp          [[buffer(4)]],   // (B, OUT)
    device const float* us          [[buffer(5)]],   // (B, 1)
    device const float* them        [[buffer(6)]],   // (B, 1)
    device float*       grad_wp     [[buffer(7)]],   // (B, OUT)
    device float*       grad_bp     [[buffer(8)]],   // (B, OUT)
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint b = tg;
    device const float* wp_b = wp + b * FC_OUT;
    device const float* bp_b = bp + b * FC_OUT;
    device const float* gl_b = grad_l0 + b * FC_L1;
    device float* gwp_b = grad_wp + b * FC_OUT;
    device float* gbp_b = grad_bp + b * FC_OUT;

    const float u = us[b];
    const float t = them[b];
    const float K = 127.0f / 128.0f;

    // Recompute clamped intermediates for the first L1 block (us*wo + them*bo)
    float x_a = u * wp_b[tid]       + t * bp_b[tid];
    float x_b = u * wp_b[tid + FC_H] + t * bp_b[tid + FC_H];
    float c_a = clamp(x_a, 0.0f, 1.0f);
    float c_b = clamp(x_b, 0.0f, 1.0f);

    // Gradient through product: l0[tid] = c_a * c_b * K
    float gl0 = gl_b[tid];
    float gc_a = gl0 * c_b * K * float(x_a > 0.0f && x_a < 1.0f);
    float gc_b = gl0 * c_a * K * float(x_b > 0.0f && x_b < 1.0f);

    // Recompute clamped intermediates for the second L1 block (us*bo + them*wo)
    float x_c = u * bp_b[tid]       + t * wp_b[tid];
    float x_d = u * bp_b[tid + FC_H] + t * wp_b[tid + FC_H];
    float c_c = clamp(x_c, 0.0f, 1.0f);
    float c_d = clamp(x_d, 0.0f, 1.0f);

    float gl1 = gl_b[tid + FC_H];
    float gc_c = gl1 * c_d * K * float(x_c > 0.0f && x_c < 1.0f);
    float gc_d = gl1 * c_c * K * float(x_d > 0.0f && x_d < 1.0f);

    // grad_wp[j] = us * grad_c_first[j] + them * grad_c_second[j]
    // grad_bp[j] = them * grad_c_first[j] + us * grad_c_second[j]
    gwp_b[tid]        = u * gc_a + t * gc_c;
    gwp_b[tid + FC_H] = u * gc_b + t * gc_d;
    gbp_b[tid]        = t * gc_a + u * gc_c;
    gbp_b[tid + FC_H] = t * gc_b + u * gc_d;

    // PSQT gradient passthrough
    if (tid < FC_PSQT) {
        gwp_b[FC_L1 + tid] = grad_wpsqt[b * FC_PSQT + tid];
        gbp_b[FC_L1 + tid] = grad_bpsqt[b * FC_PSQT + tid];
    }
}

// ---------------------------------------------------------------------------
// bias_grad = grad_wp.sum(dim=0) + grad_bp.sum(dim=0)
//
// One threadgroup per output column, each with TG_THREADS threads that
// cooperatively reduce across the batch dimension using shared memory.
// ---------------------------------------------------------------------------
kernel void bias_grad_sum(
    device const float* grad_a     [[buffer(0)]],  // (B, FC_OUT) - grad_wp
    device const float* grad_b     [[buffer(1)]],  // (B, FC_OUT) - grad_bp
    device float*       bias_grad  [[buffer(2)]],  // (FC_OUT,)
    constant uint&      batch_size [[buffer(3)]],
    uint tg_pos  [[threadgroup_position_in_grid]],
    uint t_pos   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint col = tg_pos;
    float sum = 0.0f;

    for (uint i = t_pos; i < batch_size; i += tg_size) {
        sum += grad_a[i * FC_OUT + col] + grad_b[i * FC_OUT + col];
    }

    shared[t_pos] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (t_pos < stride) {
            shared[t_pos] += shared[t_pos + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t_pos == 0) {
        bias_grad[col] = shared[0];
    }
}
