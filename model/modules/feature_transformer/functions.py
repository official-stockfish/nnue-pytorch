import math
import torch
from torch import autograd

from .kernel import (
    make_sparse_input_linear_forward_kernel,
    make_sparse_input_linear_backward_kernel_hybrid,
)


_USE_CUSTOM_SPARSE_KERNEL = True

def set_use_custom_sparse_kernel(use_custom: bool):
    global _USE_CUSTOM_SPARSE_KERNEL
    _USE_CUSTOM_SPARSE_KERNEL = use_custom


def sparse_linear_op(feature_indices, feature_values, weight, bias):
    if _USE_CUSTOM_SPARSE_KERNEL:
        return SparseLinearFunction.apply(feature_indices, feature_values, weight, bias)

    batch_size = feature_indices.shape[0]
    num_inputs = weight.shape[0]

    minus_one = torch.tensor(-1, dtype=feature_indices.dtype, device=feature_indices.device)
    valid_mask = feature_indices != minus_one

    safe_indices = feature_indices.clone()
    safe_indices.masked_fill_(~valid_mask, 0)
    safe_indices = safe_indices.to(torch.int64)

    safe_values = feature_values.clone()
    safe_values.masked_fill_(~valid_mask, 0.0)

    dense_input = torch.zeros(
        (batch_size, num_inputs),
        dtype=weight.dtype,
        device=weight.device
    )

    dense_input.scatter_add_(1, safe_indices, safe_values)

    return torch.matmul(dense_input, weight) + bias


_autotune_chunk_cache = dict()

def _get_optimal_chunk_size(
    batch_size: int,
    max_active_features: int,
    output_size: int,
    kernel,
    threads_per_block_y: int,
    feature_indices: torch.Tensor,
    feature_values: torch.Tensor,
    weight_grad: torch.Tensor,
    bias_grad: torch.Tensor,
    grad_output: torch.Tensor
) -> int:
    key = (batch_size, max_active_features, output_size)
    if key in _autotune_chunk_cache:
        return _autotune_chunk_cache[key]

    # Candidate chunk sizes to search
    candidates = [32, 64, 128, 256, 512, 1024]
    best_time = float('inf')
    best_chunk = 128

    # Prevent benchmarking overhead from initial CUDA context switches
    warmup_runs = 3
    eval_runs = 5

    ptr_indices = feature_indices.data_ptr()
    ptr_values = feature_values.data_ptr()
    ptr_w_grad = weight_grad.data_ptr() if weight_grad is not None else 0
    ptr_b_grad = bias_grad.data_ptr() if bias_grad is not None else 0
    ptr_out_grad = grad_output.data_ptr()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for chunk in candidates:
        if chunk > batch_size and chunk != candidates[0]:
            continue # Skip unnecessarily large chunks if batch is small

        grid_x = math.ceil(batch_size / chunk)
        grid_y = math.ceil(output_size / threads_per_block_y)
        grid = (grid_x, grid_y)
        block = (threads_per_block_y,)
        args = (ptr_indices, ptr_values, ptr_w_grad, ptr_b_grad, ptr_out_grad, batch_size, chunk)

        # Warmup
        for _ in range(warmup_runs):
            if weight_grad is not None: weight_grad.zero_()
            if bias_grad is not None: bias_grad.zero_()
            kernel(grid=grid, block=block, args=args)

        torch.cuda.synchronize()
        start_event.record()

        # Benchmark
        for _ in range(eval_runs):
            if weight_grad is not None: weight_grad.zero_()
            if bias_grad is not None: bias_grad.zero_()
            kernel(grid=grid, block=block, args=args)

        end_event.record()
        torch.cuda.synchronize()

        time_ms = start_event.elapsed_time(end_event) / eval_runs
        if time_ms < best_time:
            best_time = time_ms
            best_chunk = chunk

    # Clean tensors for the actual execution
    if weight_grad is not None: weight_grad.zero_()
    if bias_grad is not None: bias_grad.zero_()

    _autotune_chunk_cache[key] = best_chunk
    return best_chunk

class SparseLinearFunction(autograd.Function):
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
            batch_size, output_size, dtype=torch.float32, device=device
        )

        kernel = make_sparse_input_linear_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr(),
            ),
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

        # Bug fixed: Only allocate requested gradients
        needs_weight_grad = ctx.needs_input_grad[2]
        needs_bias_grad = ctx.needs_input_grad[3]

        weight_grad = torch.zeros_like(weight) if needs_weight_grad else None
        bias_grad = torch.zeros_like(bias) if needs_bias_grad else None

        if not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None

        kernel, threads_per_block_y = make_sparse_input_linear_backward_kernel_hybrid(
            max_active_features, output_size
        )

        chunk_size = _get_optimal_chunk_size(
            batch_size, max_active_features, output_size, kernel, threads_per_block_y,
            feature_indices, feature_values, weight_grad, bias_grad, grad_output
        )

        grid_x = math.ceil(batch_size / chunk_size)
        grid_y = math.ceil(output_size / threads_per_block_y)

        kernel(
            grid=(grid_x, grid_y),
            block=(threads_per_block_y,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr() if weight_grad is not None else 0,
                bias_grad.data_ptr() if bias_grad is not None else 0,
                grad_output.data_ptr(),
                batch_size,
                chunk_size
            ),
        )

        return None, None, weight_grad, bias_grad
