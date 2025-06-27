import math
import torch
import triton
import triton.language as tl

from torch import nn
from torch import autograd


@triton.autotune(
    configs=[
        triton.Config({"OUTPUT_BLOCK_SIZE": 32}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 512}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 1024}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 2048}),
    ],
    key=["max_active_features", "output_size"]
)
@triton.jit
def _feature_transformer_slice_forward_kernel(
        feature_indices,
        feature_values,
        weight,
        bias,
        output,
        max_active_features: tl.constexpr,
        output_size: tl.constexpr,
        OUTPUT_BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    output_block_idx = tl.program_id(1)

    output_offsets = OUTPUT_BLOCK_SIZE * output_block_idx + tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = output_offsets < output_size

    feature_indices_slice = feature_indices + batch_idx * max_active_features
    feature_values_slice = feature_values + batch_idx * max_active_features
    output_slice = output + batch_idx * output_size

    acc = tl.load(bias + output_offsets, mask=output_mask, other=0.0)
    acc = acc.to(tl.float32)

    past_active_features = False
    for k in range(max_active_features):
        if not past_active_features:
            feature_idx = tl.load(feature_indices_slice + k)
            if feature_idx == -1:
                past_active_features = True
            else:
                curr_feature_values = tl.load(feature_values_slice + k)
                curr_weight_values = tl.load(weight + feature_idx * output_size + output_offsets, mask=output_mask, other=0.0)
                acc += curr_weight_values * curr_feature_values

    tl.store(output_slice + output_offsets, acc, mask=output_mask)


def feature_transformer_slice_forward(
        feature_indices,
        feature_values,
        weight,
        bias,
        output,
        batch_size,
        max_active_features,
        output_size
):
    def grid(meta):
        return (batch_size, triton.cdiv(output_size, meta["OUTPUT_BLOCK_SIZE"]))

    _feature_transformer_slice_forward_kernel[grid](
        feature_indices=feature_indices,
        feature_values=feature_values,
        weight=weight,
        bias=bias,
        output=output,
        max_active_features=max_active_features,
        output_size=output_size,
    )


@triton.autotune(
    configs=[
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 512}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 1024}),
    ],
    key=["batch_size", "output_size"]
)
@triton.jit
def _feature_transformer_slice_bias_backward_kernel(
        bias_grad,
        output_grad,
        batch_size: tl.constexpr,
        output_size: tl.constexpr,
        OUTPUT_BLOCK_SIZE: tl.constexpr
):
    output_block_idx = tl.program_id(0)

    output_offsets = OUTPUT_BLOCK_SIZE * output_block_idx + tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = output_offsets < output_size

    acc = tl.zeros((OUTPUT_BLOCK_SIZE,), dtype=tl.float32)

    for k in range(batch_size):
        output_grad_slice = output_grad + k * output_size
        output_grad_values = tl.load(output_grad_slice + output_offsets, mask=output_mask, other=0.0)
        acc += output_grad_values

    bias_update_mask = output_mask & (acc != 0.0)
    tl.atomic_add(bias_grad + output_offsets, acc, mask=bias_update_mask)


@triton.autotune(
    configs=[
        triton.Config({"OUTPUT_BLOCK_SIZE": 32}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 64}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 128}),
        triton.Config({"OUTPUT_BLOCK_SIZE": 256}),
    ],
    key=["max_active_features", "output_size"]
)
@triton.jit
def _feature_transformer_slice_weight_backward_kernel(
        feature_indices,
        feature_values,
        weight_grad,
        output_grad,
        max_active_features: tl.constexpr,
        output_size: tl.constexpr,
        OUTPUT_BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    output_block_idx = tl.program_id(1)

    output_offsets = OUTPUT_BLOCK_SIZE * output_block_idx + tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = output_offsets < output_size

    feature_indices_slice = feature_indices + batch_idx * max_active_features
    feature_values_slice = feature_values + batch_idx * max_active_features

    output_grad_slice = output_grad + batch_idx * output_size
    output_grad_values = tl.load(output_grad_slice + output_offsets, mask=output_mask, other=0.0)

    past_active_features = False
    for k in range(max_active_features):
        if not past_active_features:
            feature_idx = tl.load(feature_indices_slice + k)
            if feature_idx == -1:
                past_active_features = True
            else:
                curr_feature_values = tl.load(feature_values_slice + k)
                curr_weight_grad_values = output_grad_values * curr_feature_values
                curr_weight_update_mask = output_mask & (curr_weight_grad_values != 0)
                tl.atomic_add(
                    weight_grad + feature_idx * output_size + output_offsets,
                    curr_weight_grad_values,
                    mask=curr_weight_update_mask
                )


def feature_transformer_slice_backward(
        feature_indices,
        feature_values,
        bias_grad,
        weight_grad,
        output_grad,
        batch_size,
        max_active_features,
        output_size
):
    def grid_bias(meta):
        return (triton.cdiv(output_size, meta['OUTPUT_BLOCK_SIZE']),)

    _feature_transformer_slice_bias_backward_kernel[grid_bias](
        bias_grad=bias_grad,
        output_grad=output_grad,
        batch_size=batch_size,
        output_size=output_size
    )

    def grid_weights(meta):
        return (
            batch_size,
            triton.cdiv(output_size, meta['OUTPUT_BLOCK_SIZE'])
        )

    _feature_transformer_slice_weight_backward_kernel[grid_weights](
        feature_indices=feature_indices,
        feature_values=feature_values,
        weight_grad=weight_grad,
        output_grad=output_grad,
        max_active_features=max_active_features,
        output_size=output_size
    )


class FeatureTransformerSliceFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(
            (batch_size, output_size),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        feature_transformer_slice_forward(
            feature_indices=feature_indices,
            feature_values=feature_values,
            weight=weight,
            bias=bias,
            output=output,
            batch_size=batch_size,
            max_active_features=max_active_features,
            output_size=output_size
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(
            weight.shape[0], weight.shape[1], dtype=torch.float32, device=device
        )
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        output_thread_slice_size = 128
        grid = (batch_size, triton.cdiv(output_size, output_thread_slice_size))

        feature_transformer_slice_backward[grid](
            feature_indices=feature_indices,
            feature_values=feature_values,
            weight_grad=weight_grad,
            bias_grad=bias_grad,
            output_grad=grad_output,
            max_active_features=max_active_features,
            output_size=output_size,
            OUTPUT_BLOCK_SIZE=output_thread_slice_size
        )

        return None, None, weight_grad, bias_grad


class DoubleFeatureTransformerSliceFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_indices_0,
        feature_values_0,
        feature_indices_1,
        feature_values_1,
        weight,
        bias,
    ):
        ctx.save_for_backward(
            feature_indices_0,
            feature_values_0,
            feature_indices_1,
            feature_values_1,
            weight,
            bias,
        )

        assert len(feature_indices_0.shape) == 2
        assert len(feature_values_0.shape) == 2
        assert feature_indices_0.shape[0] == feature_values_0.shape[0]
        assert feature_indices_0.shape[1] == feature_values_0.shape[1]
        assert feature_indices_0.dtype == torch.int32
        assert feature_values_0.dtype == torch.float32

        assert len(feature_indices_1.shape) == 2
        assert len(feature_values_1.shape) == 2
        assert feature_indices_1.shape[0] == feature_values_1.shape[0]
        assert feature_indices_1.shape[1] == feature_values_1.shape[1]
        assert feature_indices_1.dtype == torch.int32
        assert feature_values_1.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices_0.is_cuda
        assert feature_values_0.is_cuda
        assert feature_indices_1.is_cuda
        assert feature_values_1.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values_0.device == feature_indices_0.device
        assert feature_values_1.device == feature_indices_1.device
        assert feature_indices_0.device == feature_indices_1.device
        assert weight.device == feature_indices_0.device
        assert bias.device == feature_indices_0.device

        assert feature_indices_0.is_contiguous()
        assert feature_values_0.is_contiguous()
        assert feature_indices_1.is_contiguous()
        assert feature_values_1.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        output_0 = torch.empty(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
        )
        output_1 = torch.empty(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
        )

        feature_transformer_slice_forward(
            feature_indices=feature_indices_0,
            feature_values=feature_values_0,
            weight=weight,
            bias=bias,
            output=output_0,
            batch_size=batch_size,
            max_active_features=max_active_features,
            output_size=output_size
        )

        feature_transformer_slice_forward(
            feature_indices=feature_indices_1,
            feature_values=feature_values_1,
            weight=weight,
            bias=bias,
            output=output_1,
            batch_size=batch_size,
            max_active_features=max_active_features,
            output_size=output_size
        )

        return output_0, output_1

    @staticmethod
    def backward(ctx, grad_output_0, grad_output_1):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output_0 = grad_output_0.contiguous()
        grad_output_1 = grad_output_1.contiguous()

        (
            feature_indices_0,
            feature_values_0,
            feature_indices_1,
            feature_values_1,
            weight,
            bias,
        ) = ctx.saved_tensors

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(
            weight.shape[0], weight.shape[1], dtype=torch.float32, device=device
        )
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        feature_transformer_slice_backward(
            feature_indices=feature_indices_0,
            feature_values=feature_values_0,
            weight_grad=weight_grad,
            bias_grad=bias_grad,
            output_grad=grad_output_0,
            batch_size=batch_size,
            max_active_features=max_active_features,
            output_size=output_size,
        )

        feature_transformer_slice_backward(
            feature_indices=feature_indices_1,
            feature_values=feature_values_1,
            weight_grad=weight_grad,
            bias_grad=bias_grad,
            output_grad=grad_output_1,
            batch_size=batch_size,
            max_active_features=max_active_features,
            output_size=output_size,
        )

        return None, None, None, None, weight_grad, bias_grad


class FeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1 / num_inputs)
        self.weight = nn.Parameter(
            torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma)
            - sigma
        )
        self.bias = nn.Parameter(
            torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma
        )

    def forward(self, feature_indices, feature_values):
        return FeatureTransformerSliceFunction.apply(
            feature_indices, feature_values, self.weight, self.bias
        )


class DoubleFeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DoubleFeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1 / num_inputs)
        self.weight = nn.Parameter(
            torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma)
            - sigma
        )
        self.bias = nn.Parameter(
            torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma
        )

    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        return DoubleFeatureTransformerSliceFunction.apply(
            feature_indices_0,
            feature_values_0,
            feature_indices_1,
            feature_values_1,
            self.weight,
            self.bias,
        )


if __name__ == "__main__":
    import time

    def FeatureTransformerSliceFunctionEmulate(
        feature_indices, feature_values, weight, bias
    ):
        batch_size = feature_indices.shape[0]
        num_inputs = weight.shape[0]
        max_active_features = feature_indices.shape[1]
        inputs = torch.zeros(
            batch_size, num_inputs, dtype=torch.float32, device=weight.device
        )
        for i in range(batch_size):
            for j in range(max_active_features):
                feature = feature_indices[i, j]
                value = feature_values[i, j]
                inputs[i, feature] += value

        return torch.mm(inputs, weight) + bias

    def test():
        BATCH_SIZE = 16
        INPUT_SIZE = 768
        MAX_ACTIVE_FEATURES = 32
        STRIDE = 128
        MAX_ERROR = 1e-4

        torch.manual_seed(0)
        weight0 = torch.rand(
            INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True
        )
        bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        torch.manual_seed(0)
        weight1 = torch.rand(
            INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True
        )
        bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(
            dtype=torch.int32
        )
        indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(
            dtype=torch.int32
        )
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

        output00 = FeatureTransformerSliceFunctionEmulate(
            indices0.clone(), values0.clone(), weight0, bias0
        )
        output01 = FeatureTransformerSliceFunctionEmulate(
            indices1.clone(), values1.clone(), weight0, bias0
        )
        # output10 = FeatureTransformerSliceFunction.apply(indices0.clone().cuda(), values0.clone().cuda(), weight1.cuda(), bias1.cuda())
        # output11 = FeatureTransformerSliceFunction.apply(indices1.clone().cuda(), values1.clone().cuda(), weight1.cuda(), bias1.cuda())
        output10, output11 = DoubleFeatureTransformerSliceFunction.apply(
            indices0.clone().cuda(),
            values0.clone().cuda(),
            indices1.clone().cuda(),
            values1.clone().cuda(),
            weight1.cuda(),
            bias1.cuda(),
        )

        assert torch.max(output00.cpu() - output10.cpu()) < MAX_ERROR
        assert torch.max(output01.cpu() - output11.cpu()) < MAX_ERROR
        (output00 - output01).sum().backward()
        (output10 - output11).sum().backward()
        assert torch.max(weight0.grad.cpu() - weight1.grad.cpu()) < MAX_ERROR
        assert torch.max(bias0.grad.cpu() - bias1.grad.cpu()) < MAX_ERROR
        print("Tests passed.")

    def bench():
        INPUT_SIZE = 40960
        BATCH_SIZE = 8192
        ITERS = 64
        STRIDE = 264
        MAX_ACTIVE_FEATURES = 64

        layer = DoubleFeatureTransformerSlice(INPUT_SIZE, STRIDE).cuda()
        layer = torch.compile(layer, fullgraph=True, mode="max-autotune")
        indices0 = torch.cat(
            [
                torch.sort(
                    (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4) * INPUT_SIZE),
                    dim=1,
                )[0].to(dtype=torch.int32),
                torch.full(
                    (BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        ).cuda()
        values0 = torch.rand(
            BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32
        ).cuda()
        indices1 = torch.cat(
            [
                torch.sort(
                    (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4)) * INPUT_SIZE,
                    dim=1,
                )[0].to(dtype=torch.int32),
                torch.full(
                    (BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        ).cuda()
        values1 = torch.rand(
            BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32
        ).cuda()

        # Warmup
        output0, output1 = layer(indices0, values0, indices1, values1)
        output0 = torch.clamp(output0, 0.0, 1.0)
        output0 = output0.clone()
        output1 = output1.clone()
        output1 = torch.clamp(output1, 0.0, 1.0)
        g = ((output0 - output1) ** 2).mean()
        g.backward()
        torch.cuda.synchronize()

        for _ in range(ITERS):
            torch.compiler.cudagraph_mark_step_begin()
            output0, output1 = layer(indices0, values0, indices1, values1)
            output0 = output0.clone()
            output1 = output1.clone()
            output0 = torch.clamp(output0, 0.0, 1.0)
            output1 = torch.clamp(output1, 0.0, 1.0)

            g = ((output0 - output1) ** 2).mean()
            g.backward()

            torch.cuda.synchronize()

        output0, output1 = layer(indices0, values0, indices1, values1)

        start = time.time()

        for _ in range(ITERS):
            torch.compiler.cudagraph_mark_step_begin()
            output0, output1 = layer(indices0, values0, indices1, values1)
            output0 = output0.clone()
            output1 = output1.clone()
            output0 = torch.clamp(output0, 0.0, 1.0)
            output1 = torch.clamp(output1, 0.0, 1.0)

            g = ((output0 - output1) ** 2).mean()
            g.backward()

            torch.cuda.synchronize()

        end = time.time()

        # for param in layer.parameters():
        #    print(param.grad)

        print("{} pos/s".format((ITERS * BATCH_SIZE) / (end - start)))

    test()
    bench()
