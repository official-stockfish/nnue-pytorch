#include <metal_stdlib>
using namespace metal;

constant uint FC_IN_SIZE  [[function_constant(0)]];
constant uint FC_OUT_SIZE [[function_constant(1)]];

// ---------------------------------------------------------------------------
// Indexed stacked linear forward.
//
// For each batch element b, computes only the weight block selected by
// indices[b], avoiding the 'count'-x wasted compute of the full stacked
// linear followed by select_output.
//
//   output[b] = x[b] @ weight[k*out:(k+1)*out].T + bias[k*out:(k+1)*out]
//   where k = indices[b]
//
// weight is (count*out_size, in_size), bias is (count*out_size,).
// One threadgroup per batch element, FC_OUT_SIZE threads.
// Thread t computes output element t.
// ---------------------------------------------------------------------------
kernel void indexed_stacked_linear_forward(
    device const float* x       [[buffer(0)]],   // (B, in_size)
    device const float* weight  [[buffer(1)]],   // (count*out_size, in_size)
    device const float* bias    [[buffer(2)]],   // (count*out_size,)
    device const int*   indices [[buffer(3)]],   // (B,) values in [0, count)
    device float*       output  [[buffer(4)]],   // (B, out_size)
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint b = tg;
    const uint idx = static_cast<uint>(indices[b]);
    device const float* x_b = x + b * FC_IN_SIZE;
    device const float* w_row = weight + (idx * FC_OUT_SIZE + tid) * FC_IN_SIZE;

    float acc = bias[idx * FC_OUT_SIZE + tid];
    uint i = 0;
    for (; i + 3 < FC_IN_SIZE; i += 4) {
        acc += x_b[i]     * w_row[i];
        acc += x_b[i + 1] * w_row[i + 1];
        acc += x_b[i + 2] * w_row[i + 2];
        acc += x_b[i + 3] * w_row[i + 3];
    }
    for (; i < FC_IN_SIZE; ++i)
        acc += x_b[i] * w_row[i];

    output[b * FC_OUT_SIZE + tid] = acc;
}

// ---------------------------------------------------------------------------
// Indexed stacked linear backward for grad_x.
//
//   grad_x[b] = grad_output[b] @ weight[k*out:(k+1)*out]
//   where k = indices[b]
//
// One threadgroup per batch element, FC_IN_SIZE threads (one per input
// element).  Each thread computes one element of grad_x.
// ---------------------------------------------------------------------------
kernel void indexed_stacked_linear_backward_x(
    device const float* grad_output  [[buffer(0)]],  // (B, out_size)
    device const float* weight       [[buffer(1)]],  // (count*out_size, in_size)
    device const int*   indices      [[buffer(2)]],  // (B,)
    device float*       grad_x       [[buffer(3)]],  // (B, in_size)
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint b = tg;
    const uint idx = static_cast<uint>(indices[b]);
    device const float* go_b = grad_output + b * FC_OUT_SIZE;
    device const float* w_block = weight + idx * FC_OUT_SIZE * FC_IN_SIZE;

    float acc = 0.0f;
    for (uint o = 0; o < FC_OUT_SIZE; ++o)
        acc += go_b[o] * w_block[o * FC_IN_SIZE + tid];

    grad_x[b * FC_IN_SIZE + tid] = acc;
}

// ---------------------------------------------------------------------------
// Fused squared-clamp-relu activation for layer stack l1 output.
//
// Given l1c of shape (B, L2+1):
//   l1x[b, t]      = clamp(l1c[b, t]^2 * (255/256), 0, 1)   for t < L2
//   l1x[b, L2 + t] = clamp(l1c[b, t], 0, 1)                 for t < L2
//   l1x_out[b]     = l1c[b, L2]
//
// FC_IN_SIZE = L2, FC_OUT_SIZE = 2*L2.
// One threadgroup per batch element, L2 threads.
// ---------------------------------------------------------------------------
kernel void sqr_crelu_forward(
    device const float* l1c     [[buffer(0)]],   // (B, L2+1)
    device float*       l1x     [[buffer(1)]],   // (B, 2*L2)
    device float*       l1x_out [[buffer(2)]],   // (B,)
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint b = tg;
    const uint in_stride = FC_IN_SIZE + 1;
    float x = l1c[b * in_stride + tid];

    l1x[b * FC_OUT_SIZE + tid]             = clamp(x * x * (255.0f / 256.0f), 0.0f, 1.0f);
    l1x[b * FC_OUT_SIZE + FC_IN_SIZE + tid] = clamp(x, 0.0f, 1.0f);

    if (tid == 0)
        l1x_out[b] = l1c[b * in_stride + FC_IN_SIZE];
}

// ---------------------------------------------------------------------------
// Backward for fused squared-clamp-relu.
//
// grad_l1c[b, t] = grad_l1x[b, t] * 2*x*(255/256) * (0 < x^2*255/256 < 1)
//                 + grad_l1x[b, L2+t] * (0 < x < 1)
// grad_l1c[b, L2] = grad_l1x_out[b]
// ---------------------------------------------------------------------------
kernel void sqr_crelu_backward(
    device const float* grad_l1x     [[buffer(0)]],   // (B, 2*L2)
    device const float* grad_l1x_out [[buffer(1)]],   // (B,)
    device const float* l1c          [[buffer(2)]],   // (B, L2+1)
    device float*       grad_l1c     [[buffer(3)]],   // (B, L2+1)
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint b = tg;
    const uint in_stride = FC_IN_SIZE + 1;
    float x = l1c[b * in_stride + tid];
    float sq = x * x * (255.0f / 256.0f);

    float g_sq  = grad_l1x[b * FC_OUT_SIZE + tid];
    float g_lin = grad_l1x[b * FC_OUT_SIZE + FC_IN_SIZE + tid];

    float grad = g_sq * 2.0f * x * (255.0f / 256.0f) * float(sq > 0.0f && sq < 1.0f)
               + g_lin * float(x > 0.0f && x < 1.0f);

    grad_l1c[b * in_stride + tid] = grad;

    if (tid == 0)
        grad_l1c[b * in_stride + FC_IN_SIZE] = grad_l1x_out[b];
}
