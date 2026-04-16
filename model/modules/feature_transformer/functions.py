import math
import torch
from torch import autograd

from .kernel import (
    make_sparse_input_linear_forward_kernel,
    make_sparse_input_linear_backward_kernel,
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


def _get_optimal_chunk_size(
    batch_size: int,
    max_active_indices: int,
    logical_y_threads: int,
    kernel,
    threads_per_block_y: int,
    kernel_args: list,
    weight_grad,
    bias_grad,
    autotune_chunk_cache,
) -> int:
    rounded_max_indices = 1 << (max_active_indices - 1).bit_length()
    key = (batch_size, rounded_max_indices, logical_y_threads)
    if key in autotune_chunk_cache:
        return autotune_chunk_cache[key]

    # Create shallow copy
    kernel_args = kernel_args.copy()
    # Candidate chunk sizes to search
    candidates = [32, 64, 128, 256, 512, 1024]
    best_time = float('inf')
    best_chunk = 128

    warmup_runs = 2
    eval_runs = 3

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    tmp_w_grad = torch.zeros_like(weight_grad) if weight_grad is not None else None
    tmp_b_grad = torch.zeros_like(bias_grad) if bias_grad is not None else None

    for chunk in candidates:
        if chunk > batch_size and chunk != candidates[0]:
            continue # Skip unnecessarily large chunks if batch is small

        grid_x = math.ceil(batch_size / chunk)
        grid_y = math.ceil(logical_y_threads / threads_per_block_y)

        grid = (grid_x, grid_y)
        block = (threads_per_block_y,)

        kernel_args[0] = tmp_w_grad.data_ptr() if tmp_w_grad is not None else 0
        kernel_args[1] = tmp_b_grad.data_ptr() if tmp_b_grad is not None else 0
        kernel_args[-1] = chunk

        # Warmup
        for _ in range(warmup_runs):
            kernel(grid=grid, block=block, args=tuple(kernel_args))

        torch.cuda.synchronize()
        start_event.record()

        # Benchmark
        for _ in range(eval_runs):
            kernel(grid=grid, block=block, args=tuple(kernel_args))

        end_event.record()
        torch.cuda.synchronize()

        time_ms = start_event.elapsed_time(end_event) / eval_runs
        if time_ms < best_time:
            best_time = time_ms
            best_chunk = chunk

        if tmp_w_grad is not None:
            tmp_w_grad.zero_()
        if tmp_b_grad is not None:
            tmp_b_grad.zero_()

    autotune_chunk_cache[key] = best_chunk
    return best_chunk

_autotune_chunk_cache = dict()

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
        max_active_indices = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(
            batch_size, output_size, dtype=torch.float32, device=device
        )

        kernel = make_sparse_input_linear_forward_kernel(max_active_indices, output_size)
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
        max_active_indices = feature_indices.shape[1]
        output_size = weight.shape[1]

        # Bug fixed: Only allocate requested gradients
        needs_weight_grad = ctx.needs_input_grad[2]
        needs_bias_grad = ctx.needs_input_grad[3]

        weight_grad = torch.zeros_like(weight) if needs_weight_grad else None
        bias_grad = torch.zeros_like(bias) if needs_bias_grad else None

        if not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None

        kernel, threads_per_block_y = make_sparse_input_linear_backward_kernel(
            max_active_indices, output_size
        )

        weight_grad_ptr = weight_grad.data_ptr() if weight_grad is not None else 0
        bias_grad_ptr = bias_grad.data_ptr() if bias_grad is not None else 0

        kernel_args = [
                weight_grad_ptr,
                bias_grad_ptr,
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                grad_output.data_ptr(),
                batch_size,
                128, # save default value for chunk_size
        ]

        chunk_size = _get_optimal_chunk_size(
            batch_size, max_active_indices, output_size, kernel, threads_per_block_y,
            kernel_args, weight_grad, bias_grad, _autotune_chunk_cache
        )
        kernel_args[-1] = chunk_size

        grid_x = math.ceil(batch_size / chunk_size)
        grid_y = math.ceil(output_size / threads_per_block_y)

        kernel(
            grid=(grid_x, grid_y),
            block=(threads_per_block_y,),
            args=tuple(kernel_args),
        )

        return None, None, weight_grad, bias_grad
