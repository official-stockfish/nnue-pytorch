#include <torch/extension.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#include <mutex>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Metal state (singleton, lazily initialized on first call).
// ---------------------------------------------------------------------------
namespace {

id<MTLDevice>  g_device  = nil;
id<MTLLibrary> g_fwd_library = nil;
id<MTLLibrary> g_bwd_library = nil;
bool           g_has_native_float_atomics = false;
std::mutex     g_mutex;

struct PipelineKey {
    std::string func_name;
    uint32_t    max_active;
    uint32_t    output_size;
    bool operator==(const PipelineKey& o) const {
        return func_name == o.func_name
            && max_active == o.max_active
            && output_size == o.output_size;
    }
};

struct PipelineKeyHash {
    size_t operator()(const PipelineKey& k) const {
        size_t h = std::hash<std::string>{}(k.func_name);
        h ^= std::hash<uint32_t>{}(k.max_active)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.output_size)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

std::unordered_map<PipelineKey, id<MTLComputePipelineState>, PipelineKeyHash>
    g_pipelines;

uint32_t find_nearest_divisor(uint32_t value, uint32_t target) {
    uint32_t best = 1;
    uint32_t best_dist = target - 1;
    for (uint32_t i = 1; i <= value; ++i) {
        if (value % i == 0) {
            uint32_t dist = (i >= target) ? i - target : target - i;
            if (dist < best_dist) {
                best = i;
                best_dist = dist;
            }
        }
    }
    return best;
}

constexpr uint32_t kSparseThreadTarget = 128;

id<MTLLibrary> compile_shader(const std::string& source, MTLLanguageVersion ver) {
    NSString*          src  = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = ver;
    NSError* error = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:src options:opts error:&error];
    TORCH_CHECK(lib, "Metal shader compilation failed: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
    return lib;
}

void ensure_device() {
    if (g_device) return;
    g_device = at::mps::MPSDevice::getInstance()->device();
    TORCH_CHECK(g_device, "No Metal device found");
    g_has_native_float_atomics = [g_device supportsFamily:MTLGPUFamilyMetal3];
}

void ensure_init(const std::string& fwd_src,
                 const std::string& bwd_cas_src,
                 const std::string& bwd_native_src) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_fwd_library) return;

    ensure_device();

    g_fwd_library = compile_shader(fwd_src, MTLLanguageVersion2_4);

    if (g_has_native_float_atomics) {
        g_bwd_library = compile_shader(bwd_native_src, MTLLanguageVersion3_0);
    } else {
        g_bwd_library = compile_shader(bwd_cas_src, MTLLanguageVersion2_4);
    }
}

id<MTLComputePipelineState> get_pipeline(
        id<MTLLibrary> library,
        const std::string& func_name,
        uint32_t max_active,
        uint32_t output_size,
        uint32_t slice_size) {
    PipelineKey key{func_name, max_active, output_size};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(key);
    if (it != g_pipelines.end()) return it->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&max_active  type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&output_size type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&slice_size  type:MTLDataTypeUInt atIndex:2];

    NSString* name = [NSString stringWithUTF8String:func_name.c_str()];
    NSError*  error = nil;
    id<MTLFunction> func =
        [library newFunctionWithName:name constantValues:constants error:&error];
    TORCH_CHECK(func, "Metal function not found: ", func_name,
        error ? (std::string(" — ") + [[error localizedDescription] UTF8String]) : "");

    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, "Pipeline creation failed for: ", func_name);

    g_pipelines[key] = pso;
    return pso;
}

void set_buffer(id<MTLComputeCommandEncoder> enc,
                const torch::Tensor& t, NSUInteger idx) {
    [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
            offset:t.storage_offset() * t.element_size()
           atIndex:idx];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
torch::Tensor sparse_linear_forward_metal(
        torch::Tensor input_indices,
        torch::Tensor input_values,
        torch::Tensor weight,
        torch::Tensor bias,
        const std::string& fwd_src,
        const std::string& bwd_cas_src,
        const std::string& bwd_native_src) {

    ensure_init(fwd_src, bwd_cas_src, bwd_native_src);

    input_indices = input_indices.contiguous();
    input_values  = input_values.contiguous();
    weight        = weight.contiguous();
    bias          = bias.contiguous();

    const int64_t  batch_size  = input_indices.size(0);
    const uint32_t max_active  = static_cast<uint32_t>(input_indices.size(1));
    const uint32_t output_size = static_cast<uint32_t>(weight.size(1));

    if (batch_size == 0 || output_size == 0)
        return torch::empty({batch_size, static_cast<int64_t>(output_size)},
                            input_indices.options().dtype(torch::kFloat32));

    uint32_t num_threads = find_nearest_divisor(output_size, kSparseThreadTarget);
    uint32_t slice_size  = output_size / num_threads;

    auto pipeline = get_pipeline(
        g_fwd_library, "sparse_input_linear_forward",
        max_active, output_size, slice_size);

    auto output = torch::empty(
        {batch_size, static_cast<int64_t>(output_size)},
        input_indices.options().dtype(torch::kFloat32));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        [enc setThreadgroupMemoryLength:num_threads * slice_size * sizeof(float) atIndex:0];
        set_buffer(enc, input_indices, 0);
        set_buffer(enc, input_values,  1);
        set_buffer(enc, weight,        2);
        set_buffer(enc, bias,          3);
        set_buffer(enc, output,        4);
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
    }

    return output;
}

// ---------------------------------------------------------------------------
// Backward — returns weight_grad (bias_grad computed in Python via .sum(dim=0))
// ---------------------------------------------------------------------------
torch::Tensor sparse_linear_backward_metal(
        torch::Tensor input_indices,
        torch::Tensor input_values,
        torch::Tensor grad_output,
        int64_t       num_inputs,
        const std::string& fwd_src,
        const std::string& bwd_cas_src,
        const std::string& bwd_native_src) {

    ensure_init(fwd_src, bwd_cas_src, bwd_native_src);

    input_indices = input_indices.contiguous();
    input_values  = input_values.contiguous();
    grad_output   = grad_output.contiguous();

    const int64_t  batch_size  = input_indices.size(0);
    const uint32_t max_active  = static_cast<uint32_t>(input_indices.size(1));
    const uint32_t output_size = static_cast<uint32_t>(grad_output.size(1));

    auto opts = input_indices.options().dtype(torch::kFloat32);

    if (batch_size == 0 || output_size == 0) {
        return torch::zeros(
            {num_inputs, static_cast<int64_t>(output_size)}, opts);
    }

    uint32_t num_threads = find_nearest_divisor(output_size, kSparseThreadTarget);
    uint32_t slice_size  = output_size / num_threads;

    auto pipeline = get_pipeline(
        g_bwd_library, "sparse_input_linear_backward",
        max_active, output_size, slice_size);

    auto weight_grad = torch::empty(
        {num_inputs, static_cast<int64_t>(output_size)}, opts).fill_(0);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        [enc setThreadgroupMemoryLength:num_threads * slice_size * sizeof(float) atIndex:0];
        set_buffer(enc, input_indices, 0);
        set_buffer(enc, input_values,  1);
        set_buffer(enc, weight_grad,   2);
        set_buffer(enc, grad_output,   3);
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
    }

    return weight_grad;
}

// ---------------------------------------------------------------------------
// L0 mixing — fused element-wise ops for the feature transformer output.
// Replaces ~7 separate MPS dispatches with a single Metal kernel.
// ---------------------------------------------------------------------------
namespace {
    id<MTLLibrary> g_l0_library = nil;
}

void ensure_l0_init(const std::string& l0_src) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_l0_library) return;
    ensure_device();
    g_l0_library = compile_shader(l0_src, MTLLanguageVersion2_4);
}

id<MTLComputePipelineState> get_l0_pipeline(
        const std::string& func_name,
        uint32_t L1, uint32_t H, uint32_t psqt, uint32_t out) {
    PipelineKey key{func_name, L1, out};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(key);
    if (it != g_pipelines.end()) return it->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&L1   type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&H    type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&psqt type:MTLDataTypeUInt atIndex:2];
    [constants setConstantValue:&out  type:MTLDataTypeUInt atIndex:3];

    NSString* name = [NSString stringWithUTF8String:func_name.c_str()];
    NSError*  error = nil;
    id<MTLFunction> func =
        [g_l0_library newFunctionWithName:name constantValues:constants error:&error];
    TORCH_CHECK(func, "L0 Metal function not found: ", func_name,
        error ? (std::string(" — ") + [[error localizedDescription] UTF8String]) : "");

    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, "L0 pipeline creation failed for: ", func_name);

    g_pipelines[key] = pso;
    return pso;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
l0_mixing_forward_metal(
        torch::Tensor wp, torch::Tensor bp,
        torch::Tensor us, torch::Tensor them,
        int64_t L1, int64_t psqt,
        const std::string& l0_src) {

    ensure_l0_init(l0_src);

    wp = wp.contiguous();
    bp = bp.contiguous();
    us = us.contiguous();
    them = them.contiguous();

    const int64_t batch = wp.size(0);
    const uint32_t uL1 = static_cast<uint32_t>(L1);
    const uint32_t uH  = uL1 / 2;
    const uint32_t uPSQT = static_cast<uint32_t>(psqt);
    const uint32_t uOUT = uL1 + uPSQT;

    auto opts = wp.options();
    auto l0 = torch::empty({batch, L1}, opts);
    auto wpsqt = torch::empty({batch, psqt}, opts);
    auto bpsqt = torch::empty({batch, psqt}, opts);

    auto pipeline = get_l0_pipeline("l0_mixing_forward", uL1, uH, uPSQT, uOUT);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pipeline];
        set_buffer(enc, wp, 0);
        set_buffer(enc, bp, 1);
        set_buffer(enc, us, 2);
        set_buffer(enc, them, 3);
        set_buffer(enc, l0, 4);
        set_buffer(enc, wpsqt, 5);
        set_buffer(enc, bpsqt, 6);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(uH, 1, 1)];
    }

    return std::make_tuple(l0, wpsqt, bpsqt);
}

std::tuple<torch::Tensor, torch::Tensor>
l0_mixing_backward_metal(
        torch::Tensor grad_l0, torch::Tensor grad_wpsqt, torch::Tensor grad_bpsqt,
        torch::Tensor wp, torch::Tensor bp,
        torch::Tensor us, torch::Tensor them,
        const std::string& l0_src) {

    ensure_l0_init(l0_src);

    grad_l0    = grad_l0.contiguous();
    grad_wpsqt = grad_wpsqt.contiguous();
    grad_bpsqt = grad_bpsqt.contiguous();

    const int64_t batch = wp.size(0);
    const int64_t out   = wp.size(1);
    const int64_t psqt  = grad_wpsqt.size(1);
    const int64_t L1    = out - psqt;
    const uint32_t uL1 = static_cast<uint32_t>(L1);
    const uint32_t uH  = uL1 / 2;
    const uint32_t uPSQT = static_cast<uint32_t>(psqt);
    const uint32_t uOUT = static_cast<uint32_t>(out);

    auto opts = wp.options();
    auto grad_wp = torch::empty({batch, out}, opts);
    auto grad_bp = torch::empty({batch, out}, opts);

    auto pipeline = get_l0_pipeline("l0_mixing_backward", uL1, uH, uPSQT, uOUT);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pipeline];
        set_buffer(enc, grad_l0, 0);
        set_buffer(enc, grad_wpsqt, 1);
        set_buffer(enc, grad_bpsqt, 2);
        set_buffer(enc, wp, 3);
        set_buffer(enc, bp, 4);
        set_buffer(enc, us, 5);
        set_buffer(enc, them, 6);
        set_buffer(enc, grad_wp, 7);
        set_buffer(enc, grad_bp, 8);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(uH, 1, 1)];
    }

    return std::make_tuple(grad_wp, grad_bp);
}

// ---------------------------------------------------------------------------
// Double-perspective forward — dispatches both perspectives in one C++→Metal
// call, sharing the same command encoder session.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
sparse_linear_double_forward_metal(
        torch::Tensor w_indices, torch::Tensor w_values,
        torch::Tensor b_indices, torch::Tensor b_values,
        torch::Tensor weight,    torch::Tensor bias,
        const std::string& fwd_src,
        const std::string& bwd_cas_src,
        const std::string& bwd_native_src) {

    ensure_init(fwd_src, bwd_cas_src, bwd_native_src);

    w_indices = w_indices.contiguous();
    w_values  = w_values.contiguous();
    b_indices = b_indices.contiguous();
    b_values  = b_values.contiguous();
    weight    = weight.contiguous();
    bias      = bias.contiguous();

    const int64_t  batch_size  = w_indices.size(0);
    const uint32_t max_active  = static_cast<uint32_t>(w_indices.size(1));
    const uint32_t output_size = static_cast<uint32_t>(weight.size(1));

    auto opts = w_indices.options().dtype(torch::kFloat32);
    if (batch_size == 0 || output_size == 0) {
        auto z = torch::empty({batch_size, static_cast<int64_t>(output_size)}, opts);
        return std::make_tuple(z, z.clone());
    }

    uint32_t num_threads = find_nearest_divisor(output_size, kSparseThreadTarget);
    uint32_t slice_size  = output_size / num_threads;

    auto pipeline = get_pipeline(
        g_fwd_library, "sparse_input_linear_forward",
        max_active, output_size, slice_size);

    auto wp = torch::empty({batch_size, static_cast<int64_t>(output_size)}, opts);
    auto bp = torch::empty({batch_size, static_cast<int64_t>(output_size)}, opts);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        [enc setThreadgroupMemoryLength:num_threads * slice_size * sizeof(float) atIndex:0];
        set_buffer(enc, weight, 2);
        set_buffer(enc, bias,   3);

        set_buffer(enc, w_indices, 0);
        set_buffer(enc, w_values,  1);
        set_buffer(enc, wp,        4);
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];

        set_buffer(enc, b_indices, 0);
        set_buffer(enc, b_values,  1);
        set_buffer(enc, bp,        4);
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
    }

    return std::make_tuple(wp, bp);
}

// ---------------------------------------------------------------------------
// Double-perspective backward — dispatches both perspectives into ONE shared
// weight_grad tensor, eliminating the 96 MB allocation + element-wise
// addition that autograd would otherwise perform.
// ---------------------------------------------------------------------------
torch::Tensor sparse_linear_double_backward_metal(
        torch::Tensor w_indices, torch::Tensor w_values,
        torch::Tensor b_indices, torch::Tensor b_values,
        torch::Tensor grad_wp,   torch::Tensor grad_bp,
        int64_t       num_inputs,
        const std::string& fwd_src,
        const std::string& bwd_cas_src,
        const std::string& bwd_native_src) {

    ensure_init(fwd_src, bwd_cas_src, bwd_native_src);

    w_indices = w_indices.contiguous();
    w_values  = w_values.contiguous();
    b_indices = b_indices.contiguous();
    b_values  = b_values.contiguous();
    grad_wp   = grad_wp.contiguous();
    grad_bp   = grad_bp.contiguous();

    const int64_t  batch_size  = w_indices.size(0);
    const uint32_t max_active  = static_cast<uint32_t>(w_indices.size(1));
    const uint32_t output_size = static_cast<uint32_t>(grad_wp.size(1));

    auto opts = w_indices.options().dtype(torch::kFloat32);

    if (batch_size == 0 || output_size == 0) {
        return torch::zeros(
            {num_inputs, static_cast<int64_t>(output_size)}, opts);
    }

    uint32_t num_threads = find_nearest_divisor(output_size, kSparseThreadTarget);
    uint32_t slice_size  = output_size / num_threads;

    auto pipeline = get_pipeline(
        g_bwd_library, "sparse_input_linear_backward",
        max_active, output_size, slice_size);

    auto weight_grad = torch::empty(
        {num_inputs, static_cast<int64_t>(output_size)}, opts).fill_(0);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        [enc setThreadgroupMemoryLength:num_threads * slice_size * sizeof(float) atIndex:0];
        set_buffer(enc, weight_grad, 2);

        set_buffer(enc, w_indices, 0);
        set_buffer(enc, w_values,  1);
        set_buffer(enc, grad_wp,   3);
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];

        set_buffer(enc, b_indices, 0);
        set_buffer(enc, b_values,  1);
        set_buffer(enc, grad_bp,   3);
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
    }

    return weight_grad;
}

// ---------------------------------------------------------------------------
// Fused backward: L0 mixing backward → double sparse backward → bias_grad,
// all chained in one C++ call.  Returns (weight_grad, bias_grad).
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
fused_backward_metal(
        torch::Tensor grad_l0,    torch::Tensor grad_wpsqt,
        torch::Tensor grad_bpsqt, torch::Tensor wp,
        torch::Tensor bp,         torch::Tensor us,
        torch::Tensor them,
        torch::Tensor w_indices,  torch::Tensor w_values,
        torch::Tensor b_indices,  torch::Tensor b_values,
        int64_t       num_inputs,
        const std::string& fwd_src,
        const std::string& bwd_cas_src,
        const std::string& bwd_native_src,
        const std::string& l0_src) {

    ensure_init(fwd_src, bwd_cas_src, bwd_native_src);
    ensure_l0_init(l0_src);

    grad_l0    = grad_l0.contiguous();
    grad_wpsqt = grad_wpsqt.contiguous();
    grad_bpsqt = grad_bpsqt.contiguous();
    w_indices  = w_indices.contiguous();
    w_values   = w_values.contiguous();
    b_indices  = b_indices.contiguous();
    b_values   = b_values.contiguous();

    const int64_t batch  = wp.size(0);
    const int64_t out    = wp.size(1);
    const int64_t psqt   = grad_wpsqt.size(1);
    const int64_t L1     = out - psqt;
    const uint32_t uL1   = static_cast<uint32_t>(L1);
    const uint32_t uH    = uL1 / 2;
    const uint32_t uPSQT = static_cast<uint32_t>(psqt);
    const uint32_t uOUT  = static_cast<uint32_t>(out);

    auto opts = w_indices.options().dtype(torch::kFloat32);

    // L0 backward outputs
    auto grad_wp = torch::empty({batch, out}, wp.options());
    auto grad_bp = torch::empty({batch, out}, wp.options());

    const uint32_t max_active  = static_cast<uint32_t>(w_indices.size(1));
    const uint32_t output_size = static_cast<uint32_t>(out);
    uint32_t num_threads_bwd = find_nearest_divisor(output_size, kSparseThreadTarget);
    uint32_t slice_size_bwd  = output_size / num_threads_bwd;

    auto weight_grad = torch::zeros(
        {num_inputs, static_cast<int64_t>(output_size)}, opts);

    auto l0_pipeline = get_l0_pipeline("l0_mixing_backward", uL1, uH, uPSQT, uOUT);
    auto bwd_pipeline = get_pipeline(
        g_bwd_library, "sparse_input_linear_backward",
        max_active, output_size, slice_size_bwd);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        // L0 mixing backward
        [enc setComputePipelineState:l0_pipeline];
        set_buffer(enc, grad_l0,    0);
        set_buffer(enc, grad_wpsqt, 1);
        set_buffer(enc, grad_bpsqt, 2);
        set_buffer(enc, wp,         3);
        set_buffer(enc, bp,         4);
        set_buffer(enc, us,         5);
        set_buffer(enc, them,       6);
        set_buffer(enc, grad_wp,    7);
        set_buffer(enc, grad_bp,    8);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(uH, 1, 1)];

        // Sparse backward — single shared weight_grad buffer, two
        // dispatches (white + black perspectives).
        [enc setComputePipelineState:bwd_pipeline];
        [enc setThreadgroupMemoryLength:num_threads_bwd * slice_size_bwd * sizeof(float) atIndex:0];
        set_buffer(enc, weight_grad, 2);
        set_buffer(enc, w_indices,   0);
        set_buffer(enc, w_values,    1);
        set_buffer(enc, grad_wp,     3);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads_bwd, 1, 1)];

        set_buffer(enc, b_indices, 0);
        set_buffer(enc, b_values,  1);
        set_buffer(enc, grad_bp,   3);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads_bwd, 1, 1)];
    }

    auto bias_grad = grad_wp.sum(0).add_(grad_bp.sum(0));

    return std::make_tuple(weight_grad, bias_grad);
}

// ---------------------------------------------------------------------------
// Fused loss computation — replaces ~20 element-wise MPS dispatches with
// two Metal kernel invocations (forward + backward).
// ---------------------------------------------------------------------------
namespace {
    id<MTLLibrary> g_loss_library = nil;

    struct LossParamsGPU {
        float in_offset;
        float in_scaling;
        float out_offset;
        float out_scaling;
        float actual_lambda;
        float pow_exp;
        float qp_asymmetry;
        float w1_factor;
        float w2;
    };
}

void ensure_loss_init(const std::string& loss_src) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loss_library) return;
    ensure_device();
    g_loss_library = compile_shader(loss_src, MTLLanguageVersion2_4);
}

id<MTLComputePipelineState> get_loss_pipeline(const std::string& func_name) {
    PipelineKey key{func_name, 0, 0};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(key);
    if (it != g_pipelines.end()) return it->second;

    NSString* name = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> func = [g_loss_library newFunctionWithName:name];
    TORCH_CHECK(func, "Loss Metal function not found: ", func_name);

    NSError* error = nil;
    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, "Loss pipeline creation failed for: ", func_name);

    g_pipelines[key] = pso;
    return pso;
}

std::tuple<torch::Tensor, torch::Tensor>
loss_forward_metal(
        torch::Tensor scorenet, torch::Tensor score, torch::Tensor outcome,
        double in_offset, double in_scaling, double out_offset, double out_scaling,
        double actual_lambda, double pow_exp, double qp_asymmetry,
        double w1_factor, double w2,
        const std::string& loss_src) {

    ensure_loss_init(loss_src);

    scorenet = scorenet.contiguous().view(-1);
    score    = score.contiguous().view(-1);
    outcome  = outcome.contiguous().view(-1);

    const int64_t B = scorenet.size(0);
    constexpr uint32_t kTgThreads = 256;
    uint32_t num_tg = (static_cast<uint32_t>(B) + kTgThreads - 1) / kTgThreads;
    uint32_t batch_u32 = static_cast<uint32_t>(B);

    auto opts = scorenet.options();
    auto partial_wloss   = torch::empty({static_cast<int64_t>(num_tg)}, opts);
    auto partial_weights = torch::empty({static_cast<int64_t>(num_tg)}, opts);

    LossParamsGPU params;
    params.in_offset     = static_cast<float>(in_offset);
    params.in_scaling    = static_cast<float>(in_scaling);
    params.out_offset    = static_cast<float>(out_offset);
    params.out_scaling   = static_cast<float>(out_scaling);
    params.actual_lambda = static_cast<float>(actual_lambda);
    params.pow_exp       = static_cast<float>(pow_exp);
    params.qp_asymmetry  = static_cast<float>(qp_asymmetry);
    params.w1_factor     = static_cast<float>(w1_factor);
    params.w2            = static_cast<float>(w2);

    auto pipeline = get_loss_pipeline("loss_forward");

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        set_buffer(enc, scorenet,        0);
        set_buffer(enc, score,           1);
        set_buffer(enc, outcome,         2);
        set_buffer(enc, partial_wloss,   3);
        set_buffer(enc, partial_weights, 4);
        [enc setBytes:&params     length:sizeof(LossParamsGPU) atIndex:5];
        [enc setBytes:&batch_u32  length:sizeof(uint32_t)      atIndex:6];
        [enc setThreadgroupMemoryLength:2 * kTgThreads * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kTgThreads, 1, 1)];
    }

    return std::make_tuple(partial_wloss, partial_weights);
}

torch::Tensor loss_backward_metal(
        torch::Tensor scorenet, torch::Tensor score, torch::Tensor outcome,
        torch::Tensor grad_scale,
        double in_offset, double in_scaling, double out_offset, double out_scaling,
        double actual_lambda, double pow_exp, double qp_asymmetry,
        double w1_factor, double w2,
        const std::string& loss_src) {

    ensure_loss_init(loss_src);

    scorenet   = scorenet.contiguous().view(-1);
    score      = score.contiguous().view(-1);
    outcome    = outcome.contiguous().view(-1);
    grad_scale = grad_scale.contiguous().view(-1);

    const int64_t B = scorenet.size(0);
    constexpr uint32_t kTgThreads = 256;
    uint32_t num_tg = (static_cast<uint32_t>(B) + kTgThreads - 1) / kTgThreads;
    uint32_t batch_u32 = static_cast<uint32_t>(B);

    auto grad_scorenet = torch::empty_like(scorenet);

    LossParamsGPU params;
    params.in_offset     = static_cast<float>(in_offset);
    params.in_scaling    = static_cast<float>(in_scaling);
    params.out_offset    = static_cast<float>(out_offset);
    params.out_scaling   = static_cast<float>(out_scaling);
    params.actual_lambda = static_cast<float>(actual_lambda);
    params.pow_exp       = static_cast<float>(pow_exp);
    params.qp_asymmetry  = static_cast<float>(qp_asymmetry);
    params.w1_factor     = static_cast<float>(w1_factor);
    params.w2            = static_cast<float>(w2);

    auto pipeline = get_loss_pipeline("loss_backward");

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        set_buffer(enc, scorenet,       0);
        set_buffer(enc, score,          1);
        set_buffer(enc, outcome,        2);
        set_buffer(enc, grad_scorenet,  3);
        [enc setBytes:&params     length:sizeof(LossParamsGPU) atIndex:4];
        set_buffer(enc, grad_scale,     5);
        [enc setBytes:&batch_u32  length:sizeof(uint32_t)      atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kTgThreads, 1, 1)];
    }

    return grad_scorenet;
}

// ---------------------------------------------------------------------------
// Fused optimizer step — one Metal dispatch per parameter tensor.
// ---------------------------------------------------------------------------
namespace {
    id<MTLLibrary> g_opt_library = nil;

    struct StepParamsGPU {
        float beta1_sq;
        float one_minus_beta1_sq;
        float beta2;
        float one_minus_beta2;
        float inv_sqrt_bc2;
        float step_size;
        float eps;
    };
}

void ensure_opt_init(const std::string& opt_src) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_opt_library) return;
    ensure_device();
    g_opt_library = compile_shader(opt_src, MTLLanguageVersion2_4);
}

id<MTLComputePipelineState> get_opt_pipeline(const std::string& func_name) {
    PipelineKey key{func_name, 1, 1};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(key);
    if (it != g_pipelines.end()) return it->second;

    NSString* name = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> func = [g_opt_library newFunctionWithName:name];
    TORCH_CHECK(func, "Optimizer Metal function not found: ", func_name);

    NSError* error = nil;
    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, "Optimizer pipeline creation failed for: ", func_name);

    g_pipelines[key] = pso;
    return pso;
}

void fused_adam_step_multi_metal(
        std::vector<torch::Tensor> params,
        std::vector<torch::Tensor> grads,
        std::vector<torch::Tensor> grad_mas,
        std::vector<torch::Tensor> variance_mas,
        double beta1_sq, double one_minus_beta1_sq,
        double beta2, double one_minus_beta2,
        double inv_sqrt_bc2, double step_size, double eps,
        const std::string& opt_src) {

    const size_t n = params.size();
    TORCH_CHECK(n == grads.size() && n == grad_mas.size() && n == variance_mas.size(),
                "All tensor lists must have the same length");
    if (n == 0) return;

    ensure_opt_init(opt_src);

    StepParamsGPU sp;
    sp.beta1_sq            = static_cast<float>(beta1_sq);
    sp.one_minus_beta1_sq  = static_cast<float>(one_minus_beta1_sq);
    sp.beta2               = static_cast<float>(beta2);
    sp.one_minus_beta2     = static_cast<float>(one_minus_beta2);
    sp.inv_sqrt_bc2        = static_cast<float>(inv_sqrt_bc2);
    sp.step_size           = static_cast<float>(step_size);
    sp.eps                 = static_cast<float>(eps);

    constexpr uint32_t kTgThreads = 256;
    auto pipeline = get_opt_pipeline("fused_adam_step");

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        [enc setBytes:&sp length:sizeof(StepParamsGPU) atIndex:4];

        for (size_t i = 0; i < n; ++i) {
            uint32_t num_elements = static_cast<uint32_t>(params[i].numel());
            if (num_elements == 0) continue;

            uint32_t num_tg = (num_elements + kTgThreads - 1) / kTgThreads;

            set_buffer(enc, params[i],       0);
            set_buffer(enc, grads[i],        1);
            set_buffer(enc, grad_mas[i],     2);
            set_buffer(enc, variance_mas[i], 3);
            [enc setBytes:&num_elements length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kTgThreads, 1, 1)];
        }
    }
}

// ---------------------------------------------------------------------------
// Indexed stacked linear — avoids computing unused output channels.
// ---------------------------------------------------------------------------
namespace {
    id<MTLLibrary> g_stacked_library = nil;
}

void ensure_stacked_init(const std::string& src) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_stacked_library) return;
    ensure_device();
    g_stacked_library = compile_shader(src, MTLLanguageVersion2_4);
}

id<MTLComputePipelineState> get_stacked_pipeline(
        const std::string& func_name,
        uint32_t in_size, uint32_t out_size) {
    PipelineKey key{func_name, in_size, out_size};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(key);
    if (it != g_pipelines.end()) return it->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&in_size  type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&out_size type:MTLDataTypeUInt atIndex:1];

    NSString* name = [NSString stringWithUTF8String:func_name.c_str()];
    NSError* error = nil;
    id<MTLFunction> func =
        [g_stacked_library newFunctionWithName:name constantValues:constants error:&error];
    TORCH_CHECK(func, "Stacked linear Metal function not found: ", func_name,
        error ? (std::string(" — ") + [[error localizedDescription] UTF8String]) : "");

    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, "Stacked linear pipeline creation failed: ", func_name);

    g_pipelines[key] = pso;
    return pso;
}

torch::Tensor indexed_stacked_linear_forward_metal(
        torch::Tensor x,           // (B, in_size)
        torch::Tensor weight,      // (count*out_size, in_size)
        torch::Tensor bias,        // (count*out_size,)
        torch::Tensor indices,     // (B,) int32
        int64_t out_size,
        const std::string& src) {

    ensure_stacked_init(src);

    x       = x.contiguous();
    weight  = weight.contiguous();
    bias    = bias.contiguous();
    indices = indices.contiguous();

    const int64_t batch    = x.size(0);
    const uint32_t in_u    = static_cast<uint32_t>(x.size(1));
    const uint32_t out_u   = static_cast<uint32_t>(out_size);

    auto output = torch::empty({batch, out_size}, x.options());
    if (batch == 0) return output;

    auto pipeline = get_stacked_pipeline(
        "indexed_stacked_linear_forward", in_u, out_u);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();
        [enc setComputePipelineState:pipeline];
        set_buffer(enc, x,       0);
        set_buffer(enc, weight,  1);
        set_buffer(enc, bias,    2);
        set_buffer(enc, indices, 3);
        set_buffer(enc, output,  4);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(out_u, 1, 1)];
    }
    return output;
}

torch::Tensor indexed_stacked_linear_backward_x_metal(
        torch::Tensor grad_output,  // (B, out_size)
        torch::Tensor weight,       // (count*out_size, in_size)
        torch::Tensor indices,      // (B,) int32
        const std::string& src) {

    ensure_stacked_init(src);

    grad_output = grad_output.contiguous();
    weight      = weight.contiguous();
    indices     = indices.contiguous();

    const int64_t batch  = grad_output.size(0);
    const uint32_t out_u = static_cast<uint32_t>(grad_output.size(1));
    const uint32_t in_u  = static_cast<uint32_t>(weight.size(1));
    const uint32_t num_threads = in_u / 4;

    auto grad_x = torch::empty({batch, static_cast<int64_t>(in_u)},
                                grad_output.options());
    if (batch == 0) return grad_x;

    auto pipeline = get_stacked_pipeline(
        "indexed_stacked_linear_backward_x", in_u, out_u);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();
        [enc setComputePipelineState:pipeline];
        set_buffer(enc, grad_output, 0);
        set_buffer(enc, weight,      1);
        set_buffer(enc, indices,     2);
        set_buffer(enc, grad_x,     3);
        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
    }
    return grad_x;
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_linear_forward",  &sparse_linear_forward_metal,
          "Sparse linear forward pass (Metal)");
    m.def("sparse_linear_backward", &sparse_linear_backward_metal,
          "Sparse linear backward pass (Metal)");
    m.def("sparse_linear_double_forward",  &sparse_linear_double_forward_metal,
          "Double-perspective sparse linear forward (Metal)");
    m.def("sparse_linear_double_backward", &sparse_linear_double_backward_metal,
          "Double-perspective sparse linear backward (Metal)");
    m.def("l0_mixing_forward",  &l0_mixing_forward_metal,
          "Fused L0 mixing forward (Metal)");
    m.def("l0_mixing_backward", &l0_mixing_backward_metal,
          "Fused L0 mixing backward (Metal)");
    m.def("fused_backward", &fused_backward_metal,
          "Fused L0 backward + double sparse backward (Metal)");
    m.def("loss_forward",  &loss_forward_metal,
          "Fused loss forward (Metal)");
    m.def("loss_backward", &loss_backward_metal,
          "Fused loss backward (Metal)");
    m.def("fused_adam_step_multi", &fused_adam_step_multi_metal,
          "Multi-tensor fused Adam step — one C++ call for all params (Metal)");
    m.def("indexed_stacked_linear_forward", &indexed_stacked_linear_forward_metal,
          "Indexed stacked linear forward (Metal)");
    m.def("indexed_stacked_linear_backward_x", &indexed_stacked_linear_backward_x_metal,
          "Indexed stacked linear backward for grad_x (Metal)");
}
