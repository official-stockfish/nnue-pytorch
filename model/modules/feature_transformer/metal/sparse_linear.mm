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
id<MTLLibrary> g_library = nil;
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

// Mirrors kernel.py: _find_nearest_divisor(output_size, 512)
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

void ensure_init(const std::string& shader_source) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_device) return;

    g_device = at::mps::MPSDevice::getInstance()->device();
    TORCH_CHECK(g_device, "No Metal device found");

    NSString*          src  = [NSString stringWithUTF8String:shader_source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion2_4;

    NSError* error = nil;
    g_library = [g_device newLibraryWithSource:src options:opts error:&error];
    TORCH_CHECK(g_library,
        "Metal shader compilation failed: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
}

id<MTLComputePipelineState> get_pipeline(
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
        [g_library newFunctionWithName:name constantValues:constants error:&error];
    TORCH_CHECK(func, "Metal function not found: ", func_name,
        error ? (std::string(" — ") + [[error localizedDescription] UTF8String]) : "");

    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso, "Pipeline creation failed for: ", func_name);

    g_pipelines[key] = pso;
    return pso;
}

// Encode a buffer argument: uses the tensor's underlying MTLBuffer directly
// (zero-copy) and accounts for storage_offset.
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
        const std::string& shader_source) {

    ensure_init(shader_source);

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

    uint32_t num_threads = find_nearest_divisor(output_size, 512);
    uint32_t slice_size  = output_size / num_threads;

    auto pipeline = get_pipeline(
        "sparse_input_linear_forward", max_active, output_size, slice_size);

    auto output = torch::empty(
        {batch_size, static_cast<int64_t>(output_size)},
        input_indices.options().dtype(torch::kFloat32));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        set_buffer(enc, input_indices, 0);
        set_buffer(enc, input_values,  1);
        set_buffer(enc, weight,        2);
        set_buffer(enc, bias,          3);
        set_buffer(enc, output,        4);
        [enc setThreadgroupMemoryLength:output_size * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }

    return output;
}

// ---------------------------------------------------------------------------
// Backward — returns (weight_grad, bias_grad)
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> sparse_linear_backward_metal(
        torch::Tensor input_indices,
        torch::Tensor input_values,
        torch::Tensor grad_output,
        int64_t       num_inputs,
        const std::string& shader_source) {

    ensure_init(shader_source);

    input_indices = input_indices.contiguous();
    input_values  = input_values.contiguous();
    grad_output   = grad_output.contiguous();

    const int64_t  batch_size  = input_indices.size(0);
    const uint32_t max_active  = static_cast<uint32_t>(input_indices.size(1));
    const uint32_t output_size = static_cast<uint32_t>(grad_output.size(1));

    if (batch_size == 0 || output_size == 0) {
        auto opts = input_indices.options().dtype(torch::kFloat32);
        return std::make_tuple(
            torch::zeros({num_inputs, static_cast<int64_t>(output_size)}, opts),
            torch::zeros({static_cast<int64_t>(output_size)}, opts));
    }

    uint32_t num_threads = find_nearest_divisor(output_size, 512);
    uint32_t slice_size  = output_size / num_threads;

    auto pipeline = get_pipeline(
        "sparse_input_linear_backward", max_active, output_size, slice_size);

    auto opts = input_indices.options().dtype(torch::kFloat32);
    auto weight_grad = torch::zeros(
        {num_inputs, static_cast<int64_t>(output_size)}, opts);
    auto bias_grad = torch::zeros(
        {static_cast<int64_t>(output_size)}, opts);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto enc    = stream->commandEncoder();

        [enc setComputePipelineState:pipeline];
        set_buffer(enc, input_indices, 0);
        set_buffer(enc, input_values,  1);
        set_buffer(enc, weight_grad,   2);
        set_buffer(enc, bias_grad,     3);
        set_buffer(enc, grad_output,   4);
        [enc setThreadgroupMemoryLength:output_size * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }

    return std::make_tuple(weight_grad, bias_grad);
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_linear_forward",  &sparse_linear_forward_metal,
          "Sparse linear forward pass (Metal)");
    m.def("sparse_linear_backward", &sparse_linear_backward_metal,
          "Sparse linear backward pass (Metal)");
}
