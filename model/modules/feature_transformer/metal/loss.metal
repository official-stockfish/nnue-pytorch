#include <metal_stdlib>
using namespace metal;

struct LossParams {
    float in_offset;
    float in_scaling;
    float out_offset;
    float out_scaling;
    float actual_lambda;
    float pow_exp;
    float qp_asymmetry;
    float w1_factor;   // precomputed: 2^w1 - 1
    float w2;
};

// ---------------------------------------------------------------------------
// Forward: element-wise loss + weight, then threadgroup partial reduction.
// Produces partial_wloss[num_tg] and partial_weights[num_tg] that the host
// sums to get the final scalar loss = sum(wloss) / sum(weights).
// ---------------------------------------------------------------------------
kernel void loss_forward(
    device const float* scorenet        [[buffer(0)]],
    device const float* score           [[buffer(1)]],
    device const float* outcome         [[buffer(2)]],
    device float*       partial_wloss   [[buffer(3)]],
    device float*       partial_weights [[buffer(4)]],
    constant LossParams& params         [[buffer(5)]],
    constant uint& batch_size           [[buffer(6)]],
    uint tg_pos  [[threadgroup_position_in_grid]],
    uint t_pos   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid     [[thread_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    threadgroup float* s_wloss  = shared;
    threadgroup float* s_weight = shared + tg_size;

    float wloss_val  = 0.0f;
    float weight_val = 0.0f;

    if (tid < batch_size) {
        float sn  = scorenet[tid];
        float sc  = score[tid];
        float out = outcome[tid];

        float q      = (sn - params.in_offset) / params.in_scaling;
        float qm     = (-sn - params.in_offset) / params.in_scaling;
        float sig_q  = 1.0f / (1.0f + exp(-q));
        float sig_qm = 1.0f / (1.0f + exp(-qm));
        float qf     = 0.5f * (1.0f + sig_q - sig_qm);

        float s      = (sc - params.out_offset) / params.out_scaling;
        float sm     = (-sc - params.out_offset) / params.out_scaling;
        float sig_s  = 1.0f / (1.0f + exp(-s));
        float sig_sm = 1.0f / (1.0f + exp(-sm));
        float pf     = 0.5f * (1.0f + sig_s - sig_sm);

        float pt = pf * params.actual_lambda + out * (1.0f - params.actual_lambda);

        float abs_diff = abs(pt - qf);
        float loss     = pow(abs_diff, params.pow_exp);

        if (params.qp_asymmetry != 0.0f) {
            loss *= select(1.0f, params.qp_asymmetry + 1.0f, qf > pt);
        }

        float pfc    = pf - 0.5f;
        float w_base = pfc * pfc * pf * (1.0f - pf);
        float weight = 1.0f + params.w1_factor * pow(w_base, params.w2);

        wloss_val  = loss * weight;
        weight_val = weight;
    }

    s_wloss[t_pos]  = wloss_val;
    s_weight[t_pos] = weight_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (t_pos < stride) {
            s_wloss[t_pos]  += s_wloss[t_pos + stride];
            s_weight[t_pos] += s_weight[t_pos + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t_pos == 0) {
        partial_wloss[tg_pos]  = s_wloss[0];
        partial_weights[tg_pos] = s_weight[0];
    }
}

// ---------------------------------------------------------------------------
// Backward: computes grad_scorenet[i] = grad_scale * weight[i] * dL/d(sn[i])
//
// grad_scale = grad_output / weights_sum  (precomputed scalar on host)
// ---------------------------------------------------------------------------
kernel void loss_backward(
    device const float* scorenet       [[buffer(0)]],
    device const float* score          [[buffer(1)]],
    device const float* outcome        [[buffer(2)]],
    device float*       grad_scorenet  [[buffer(3)]],
    constant LossParams& params        [[buffer(4)]],
    device const float* grad_scale_buf [[buffer(5)]],
    constant uint& batch_size          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_size) return;

    float grad_scale = grad_scale_buf[0];
    float sn  = scorenet[tid];
    float sc  = score[tid];
    float out = outcome[tid];

    float q      = (sn - params.in_offset) / params.in_scaling;
    float qm     = (-sn - params.in_offset) / params.in_scaling;
    float sig_q  = 1.0f / (1.0f + exp(-q));
    float sig_qm = 1.0f / (1.0f + exp(-qm));
    float qf     = 0.5f * (1.0f + sig_q - sig_qm);

    float s      = (sc - params.out_offset) / params.out_scaling;
    float sm     = (-sc - params.out_offset) / params.out_scaling;
    float sig_s  = 1.0f / (1.0f + exp(-s));
    float sig_sm = 1.0f / (1.0f + exp(-sm));
    float pf     = 0.5f * (1.0f + sig_s - sig_sm);

    float pt = pf * params.actual_lambda + out * (1.0f - params.actual_lambda);

    float pfc    = pf - 0.5f;
    float w_base = pfc * pfc * pf * (1.0f - pf);
    float weight = 1.0f + params.w1_factor * pow(w_base, params.w2);

    float diff     = pt - qf;
    float abs_diff = abs(diff);
    float sign_d   = (diff >= 0.0f) ? 1.0f : -1.0f;

    // d(|pt-qf|^p)/d(qf) = p * |pt-qf|^(p-1) * (-sign(pt-qf))
    float dloss_dqf = params.pow_exp * pow(abs_diff, params.pow_exp - 1.0f) * (-sign_d);

    if (params.qp_asymmetry != 0.0f) {
        dloss_dqf *= select(1.0f, params.qp_asymmetry + 1.0f, qf > pt);
    }

    // d(qf)/d(sn) = 0.5/in_scaling * (sig_q*(1-sig_q) + sig_qm*(1-sig_qm))
    float dqf_dsn = 0.5f * (sig_q * (1.0f - sig_q) + sig_qm * (1.0f - sig_qm))
                    / params.in_scaling;

    grad_scorenet[tid] = grad_scale * weight * dloss_dqf * dqf_dsn;
}
