#include <metal_stdlib>
using namespace metal;

// Per-element Adam-like step that replicates the effective Ranger21
// behaviour when all optional features are disabled:
//   variance_ma  = beta2 * variance_ma + (1-beta2) * grad^2
//   grad_ma      = beta1^2 * grad_ma + (1-beta1^2) * grad
//   denom        = sqrt(variance_ma) / sqrt(bias_correction2) + eps
//   param       -= (lr / bias_correction1 / noise_norm) * grad_ma / denom

struct StepParams {
    float beta1_sq;            // beta1^2
    float one_minus_beta1_sq;  // 1 - beta1^2
    float beta2;
    float one_minus_beta2;     // 1 - beta2
    float inv_sqrt_bc2;        // 1 / sqrt(1 - beta2^step)
    float step_size;           // lr / bias_correction1 / noise_norm
    float eps;
};

kernel void fused_adam_step(
    device float*       param       [[buffer(0)]],
    device const float* grad        [[buffer(1)]],
    device float*       grad_ma     [[buffer(2)]],
    device float*       variance_ma [[buffer(3)]],
    constant StepParams& sp         [[buffer(4)]],
    constant uint& num_elements     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    float g  = grad[tid];
    float vm = variance_ma[tid] * sp.beta2 + sp.one_minus_beta2 * g * g;
    variance_ma[tid] = vm;

    float gm = grad_ma[tid] * sp.beta1_sq + sp.one_minus_beta1_sq * g;
    grad_ma[tid] = gm;

    float denom = sqrt(vm) * sp.inv_sqrt_bc2 + sp.eps;
    param[tid] -= sp.step_size * gm / denom;
}
