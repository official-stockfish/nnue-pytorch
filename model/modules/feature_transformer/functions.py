import torch
import torch.nn.functional as F
from torch import autograd

_HAS_CUPY_KERNELS = False
try:
    from .kernel import (
        make_sparse_input_linear_forward_kernel,
        make_sparse_input_linear_backward_kernel,
    )

    _HAS_CUPY_KERNELS = True
except (ImportError, ModuleNotFoundError):
    pass


def sparse_linear(feature_indices, feature_values, weight, bias):
    """Sparse linear: output[b] = sum_k(weight[indices[b,k]] * values[b,k]) + bias.

    Dispatches to a hand-tuned CuPy CUDA kernel when available, otherwise falls
    back to a pure-PyTorch implementation that works on any device.
    """
    if _HAS_CUPY_KERNELS and feature_indices.is_cuda:
        return _CudaSparseLinearFunction.apply(
            feature_indices, feature_values, weight, bias
        )
    return _torch_sparse_linear(feature_indices, feature_values, weight, bias)


def _torch_sparse_linear(feature_indices, feature_values, weight, bias):
    """Pure PyTorch implementation using F.embedding_bag for memory efficiency.

    Instead of materialising the full (batch, max_active, output_size) gathered
    tensor, embedding_bag fuses the lookup, per-sample weighting, and reduction
    into a single call with O(batch * output_size) memory.
    """
    batch_size, max_active = feature_indices.shape

    mask = feature_indices >= 0
    safe_indices = feature_indices.clamp(min=0).long().reshape(-1)
    per_sample_weights = (feature_values * mask).reshape(-1)

    offsets = torch.arange(
        0,
        batch_size * max_active,
        max_active,
        device=feature_indices.device,
    )

    output = F.embedding_bag(
        safe_indices,
        weight,
        offsets,
        mode="sum",
        per_sample_weights=per_sample_weights,
    )
    return output + bias


class _CudaSparseLinearFunction(autograd.Function):
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
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        kernel = make_sparse_input_linear_forward_kernel(
            max_active_features, output_size
        )
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

        weight_grad = torch.zeros(
            weight.shape[0], weight.shape[1], dtype=torch.float32, device=device
        )
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_sparse_input_linear_backward_kernel(
            max_active_features, output_size
        )
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr(),
            ),
        )

        return None, None, weight_grad, bias_grad


# Legacy alias
SparseLinearFunction = _CudaSparseLinearFunction
